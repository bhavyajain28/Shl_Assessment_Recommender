# Approach

## The catalog data problem (read this first)

The assignment points at `shl.com/solutions/products/product-catalog/` for the
Individual Test Solutions table. As of testing this (2026-07), that page no
longer serves that table -- verified via live fetch, headless-Chromium
render (cookie-consent accepted, in case a CMP was blocking the widget's
JS), and by hitting known legacy detail URLs (`/view/opq32r/`,
`/view/verify-numerical-reasoning/`), all of which now 301-redirect to one
of ~10 consolidated marketing pages with no per-item metadata. SHL appears
to have redesigned the site since this assignment was written.

Rather than fabricate data or ship a coarse ~10-item catalog, I
reconstructed the classic ~360-item structure (name, URL, test type,
duration, remote testing, adaptive/IRT) from the same public snapshot this
exact assignment has produced before (several public repos scraped this
catalog while it was live), then **re-verified every URL against the live
site right now** via redirect-following (`app/scraper.py::resolve_live_urls`)
-- every URL in `catalog.json` returns HTTP 200 today. `scraper.py` still
tries the live paginated table first and only falls back to the historical
seed if that structural scrape comes back empty, so the code is correct for
the scenario the assignment describes and resilient to the one I actually
found (disclosed in `README.md` and the seed file's provenance). Job
Solution bundles (named "`<Role> Solution`", e.g. "Entry Level Sales
Solution") are filtered out, leaving 358 Individual Test Solutions.

## Architecture

`scraper.py` → `catalog.json` → `build_vectorstore.py` → FAISS index, loaded
once at API startup by `retriever.py`. `agent.py` is pure, stateless
conversation logic: given the full message history, it re-derives every
constraint from scratch every call. `api.py` is a thin FastAPI shell with a
catch-all exception handler that guarantees schema-compliant responses even
on internal failure.

Deliberately **not** LangChain/LangGraph. With ~360 catalog items and an
8-turn, 30-second-per-call budget, a framework's implicit orchestration adds
cold-start weight and a layer to debug without buying anything raw FAISS +
the `openai` SDK don't already give. Explicit Python control flow over LLM
output is also more directly defensible than "the graph decided to loop" --
every branch (ask/retrieve/answer/refuse) is a visible `if`.

## Retrieval

Hybrid semantic + lexical, fused by reciprocal rank (`retriever.py`):
- **Semantic**: `all-MiniLM-L6-v2` (local, no API key, 384-dim) embeddings
  over `name | description | test_type_label | skills | job_levels`, FAISS
  `IndexFlatIP` (cosine, since vectors are L2-normalized).
- **Lexical**: BM25 over the same text, tokenized on whitespace/`-`/`/`.

Pure semantic search on a small local model under-ranks exact-name queries
("OPQ32r") against semantically-adjacent-but-wrong items; pure BM25 misses
paraphrase ("stakeholder management" → no assessment literally says that
phrase). Fusing both, `score = 1/(1+rank_semantic) + 1/(1+rank_lexical)`,
covers both failure modes without hand-tuned weights.

`get_by_name` (used for comparisons) does exact match → substring/acronym-
expansion match (`OPQ`→`occupational personality questionnaire`, `GSA`→
`global skills assessment`) → character-similarity fallback at a 0.72
threshold (real near-matches score ≥0.9; unrelated names sharing a common
word like "assessment" score ~0.56 -- 0.72 cleanly separates them). Returns
`None` rather than a weak guess, so the agent gives an honest "not in my
catalog" instead of a hallucinated comparison.

## Prompt design & agent workflow

The LLM does exactly two jobs, both in `prompts.py`:

1. **Turn understanding** (`EXTRACTION_SYSTEM_PROMPT`): one JSON-mode call per
   turn, reading the *entire* history, returning `in_scope`, `prompt_injection`,
   `intent`, `comparison_targets`, and cumulative constraints (role, skills,
   seniority, add/remove test types, duration, remote, count,
   "no preference" signal).
2. **Comparison synthesis** (`COMPARISON_SYSTEM_PROMPT`): given only the two
   matched catalog descriptions, write a grounded 3-5 sentence answer, with
   an explicit "don't invent facts not in this text" instruction.

Everything else -- clarify questions, recommend/refine replies, refusals --
is a **Python f-string template**, not a third LLM call: the single biggest
design decision, since it means the most common turns can never hallucinate
a sentence, are unit-testable by exact string match, and cost zero extra
latency/tokens.

Control flow (`Agent.handle_chat`): `regex guardrail (injection/off-topic) →
LLM extraction (JSON) → sanitize against a rule-based fallback → intent
branch (compare/recommend)`. Guardrails run as regex *before* the LLM call
and are OR'd with its verdict -- injected text can bias what extraction
reports, but can't make the agent skip a refusal, since the regex path
doesn't depend on that call succeeding or being honest.

**Statelessness is what makes refinement "just work".** The extraction
prompt re-reads the *whole* history every call, so "Actually, add
personality tests" three turns later is re-derived fresh each time; the
agent's own past replies (fixed template strings) let it detect "have I
already recommended?" / "already asked seniority?" with no external state
store -- just checking whether an earlier assistant message starts with
`"Here are"` or equals a canned clarify string.

Clarify budget is capped at 2 questions (role/skills, then seniority) before
forcing a recommendation regardless, and a "no preference" answer
permanently retires that dimension -- guards against both the 8-turn cap
and looping if the evaluator's simulated user won't give a preference.

## Deployment

Azure App Service (Linux, B1), GitHub Actions CI/CD via Azure's Deployment
Center (push to `main` auto-redeploys). Two non-obvious issues surfaced only
at deployment time, worth recording since they shaped the final config:

- **F1 (free) tier doesn't fit.** `torch` (via `sentence-transformers`)
  installs at ~530MB alone, and pip's default wheel also pulls several
  NVIDIA CUDA packages on Linux (invisible on Windows, where I developed,
  since those deps are conditional on `platform_system == "Linux"`) --
  enough to blow past F1's 1GB disk during the Oryx build. Fix: pin the
  CPU-only wheel (`--extra-index-url https://download.pytorch.org/whl/cpu`,
  `torch==2.12.1+cpu`), which drops the NVIDIA deps and fits on B1.
- **Groq blocks the deployment region.** Every Groq call 403'd once
  deployed to Azure's East Asia (Hong Kong) region, despite the same key
  working locally -- consistent with Groq's export-control geo-restrictions.
  Rather than migrate the App Service to another region, I switched to
  OpenRouter's free auto-router, needing only environment-variable changes
  since the LLM client is provider-agnostic by design.

## Error handling & degraded mode

Every LLM call retries twice with backoff, then returns `None`; the agent
falls back to a small regex/keyword extractor (`rule_based_extract`) rather
than crashing -- a provider outage degrades the agent to keyword-only
recognition rather than a 500, verified by a test that monkeypatches the
LLM client to raise mid-call. `api.py` also has a top-level exception
handler that returns a schema-valid 200 on `/chat` even if something below
it throws unexpectedly, since schema compliance matters more than an honest
500 here.

## Evaluation

26 pytest tests, all offline (LLM mocked via `StubLLMClient`, no API key
needed): clarify-before-recommend, seniority-then-recommend, no-preference
short-circuit, refine adds the requested test type, compare grounds on real
snippets *and never calls the LLM at all* if either name isn't in the
catalog (`agent.llm.compare_calls == []`, proving it was never given the
chance to invent an answer), off-topic/injection refusal (including that the
regex guardrail fires even if the stubbed LLM is wrong, and that a flaky
`in_scope=false` is overridden mid a legitimate flow), and schema compliance
under normal and simulated-crash conditions.

Live manual testing against the deployed app caught something mocks
couldn't: comparing OPQ32r vs. GSA, the free OpenRouter model stated "GSA
uses adaptive scoring" -- directly contradicting our own record
(`adaptive_irt: false`). A mocked LLM can't produce this class of bug, since
it returns whatever the test tells it to. Fix (`_handle_compare`):
duration/remote/adaptive-IRT/type now render as a deterministic fact line
straight from the `Assessment` record; the LLM is restricted to qualitative
commentary only and told not to restate those fields -- removing its
ability to invert a yes/no value, rather than just asking more firmly.
Lesson: mocked tests validate control flow, but only a live model surfaces
prompt-following failures like this.

What I didn't get to: a real Recall@K harness against the 10 provided
conversation traces (I did not have access to fetch them in this
environment) -- I validated retrieval quality by hand-inspecting
`retriever.search()` output for a handful of realistic queries instead. If I
had the traces, I'd score Recall@10 per trace and use it to tune the
semantic/lexical fusion weights and the "how many results before we relax
duration/remote filters" logic in `_handle_recommend`.

## AI tool usage

Built interactively with Claude Code: it wrote the initial draft of each
module, but every design decision above (data-source fallback, retrieval
fusion, deterministic-templates-over-LLM-call, statelessness-enables-
refinement, the fuzzy-match threshold) was discussed, verified against real
output, and adjusted before moving on -- e.g. the first cut of
`get_by_name`'s threshold (0.45) produced a false positive under test,
which is why the final value and its justification are in this document
rather than a number typed in on faith.
