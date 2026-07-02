# Approach

## The catalog data problem (read this first)

The assignment points at `shl.com/solutions/products/product-catalog/` for the
Individual Test Solutions table. As of testing this (2026-07), that page no
longer serves that table: I verified with a live HTTP fetch, then with a
headless-Chromium render (with cookie-consent accepted, in case a CMP was
blocking the widget's JS), then by hitting known legacy detail URLs
(`/product-catalog/view/opq32r/`, `/view/verify-numerical-reasoning/`, etc.).
Every legacy detail URL now 301-redirects to one of ~10 consolidated
marketing category pages with no per-item metadata. SHL appears to have
redesigned the site since this assignment was written.

Rather than fabricate catalog data or silently ship a coarse ~10-item
catalog, I reconstructed the classic ~360-item catalog structure (name, URL,
test type, duration, remote testing, adaptive/IRT) from the same public
snapshot this exact assignment has produced many times before (several
public repos scraped this identical catalog while it was live), then
**re-verified every single URL against the live site right now** by
following redirects with `requests` (`app/scraper.py::resolve_live_urls`).
Every URL `catalog.json` contains returns HTTP 200 today. `scraper.py`
still tries the live paginated table first (`fetch_live_catalog_table`) and
only falls back to the historical seed if that structural scrape comes back
empty -- so the code is correct for the scenario the assignment describes,
and resilient to the scenario I actually found. This is disclosed in
`README.md` and in the seed file's provenance comment. Job Solutions bundles
(items literally named "`<Role> Solution`", e.g. "Entry Level Sales
Solution") are filtered out (`_is_job_solution_bundle`), leaving 358 Individual
Test Solutions.

## Architecture

`scraper.py` → `catalog.json` → `build_vectorstore.py` → FAISS index, loaded
once at API startup by `retriever.py`. `agent.py` is pure, stateless
conversation logic: given the full message history, it re-derives every
constraint from scratch every call. `api.py` is a thin FastAPI shell with a
catch-all exception handler that guarantees schema-compliant responses even
on internal failure.

Deliberately **not** LangChain/LangGraph. With ~360 catalog items and an 8-turn,
30-second-per-call budget, a framework's implicit orchestration adds cold-start
weight and an extra layer to debug without buying anything raw FAISS +
`openai`-SDK calls don't already give me. Explicit Python control flow over
LLM output is also more directly defensible than "the graph decided to loop"
-- every branch (ask/retrieve/answer/refuse) is a visible `if`.

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
expansion match (`OPQ`→`occupational personality questionnaire opq`, `GSA`→
`global skills assessment`, etc.) → character-similarity fallback with a
0.72 threshold (empirically: real near-matches score ≥0.9, unrelated names
sharing a common word like "assessment" score ~0.56 -- 0.72 cleanly
separates them). Returns `None` rather than a weak guess, which the agent
turns into an honest "not in my catalog" rather than a hallucinated
comparison.

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
is a **Python f-string template**, not a third LLM call. This was the single
biggest design decision: it means the two most common turn types
(clarify, recommend) can never hallucinate a sentence, are unit-testable by
exact string match, and cost zero extra latency/tokens. `agent.py`'s
docstring states this tradeoff explicitly because I expect to be asked about
it.

Control flow (`Agent.handle_chat`):
`regex guardrail (injection/off-topic) → LLM extraction (JSON) → sanitize
against a rule-based fallback → intent branch (compare / recommend)`.
Guardrails run as regex *before* the LLM call and are OR'd with the LLM's own
verdict -- injected text can bias what the extraction call reports, but it
cannot make the agent skip a refusal, since the regex path doesn't depend on
that call succeeding or being honest.

**Statelessness is what makes refinement "just work".** Because the
extraction prompt re-reads the *whole* history every call, "Actually, add
personality tests" three turns later is re-derived fresh each time, and the
Python layer's own past replies (fixed template strings) are what let it
detect "have I already recommended?" and "have I already asked about
seniority?" without any external state store -- I check whether an earlier
assistant message starts with `"Here are"` / equals a canned clarify string.

Clarify budget is capped at 2 questions (role/skills, then seniority) before
forcing a recommendation regardless, and a user's "no preference" answer
permanently retires that dimension for the rest of the conversation --
both guard against the 8-turn cap and against looping if the simulated
evaluator user won't give a preference.

## Error handling & degraded mode

Every LLM call (`agent.LLMClient`) retries twice with backoff, then returns
`None`; the agent falls back to a small regex/keyword extractor
(`rule_based_extract`) rather than crashing. This means a provider outage
degrades the agent to "can only recognize duration/remote/seniority/
personality-request keywords" rather than "returns a 500" -- verified by a
test that monkeypatches the LLM client to raise mid-call
(`test_malformed_llm_json_falls_back_gracefully`). `api.py` has a top-level
exception handler that still returns a schema-valid 200 on `/chat` even if
something below it throws unexpectedly, since the hard-eval schema check
matters more than an honest 500 here.

## Evaluation

25 pytest tests, all offline (LLM mocked via `StubLLMClient` in
`tests/conftest.py`, so no API key is needed to run them): clarify-before-
recommend, seniority-then-recommend, no-preference short-circuit, refine
adds the requested test type, compare grounds on real snippets *and never
calls the LLM at all* if either name isn't in the catalog (a fabricated pair
returns a template refusal and `agent.llm.compare_calls == []`, asserting
the LLM was never given the chance to invent an answer), off-topic and
injection refusal (including one test asserting the regex guardrail still
fires if the stubbed LLM is wrong about `prompt_injection`), and schema
compliance under both normal and simulated-crash conditions.

What I didn't get to: a real Recall@K harness against the 10 provided
conversation traces (I did not have access to fetch them in this
environment) -- I validated retrieval quality by hand-inspecting
`retriever.search()` output for a handful of realistic queries instead. If I
had the traces, I'd score Recall@10 per trace and use it to tune the
semantic/lexical fusion weights and the "how many results before we relax
duration/remote filters" logic in `_handle_recommend`.

## AI tool usage

Built interactively with Claude Code: it wrote the initial draft of each
module, but every design decision above (data-source fallback design,
hybrid retrieval fusion, deterministic-templates-over-second-LLM-call,
statelessness-enables-refinement, the fuzzy-match threshold) was discussed,
verified against real output, and adjusted before moving on -- e.g. the
first cut of `get_by_name`'s fuzzy threshold (0.45) produced a false
positive under test, which is why the threshold and its justification are
in this document rather than a number I typed in on faith.
