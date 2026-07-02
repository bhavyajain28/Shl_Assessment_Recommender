# SHL Assessment Recommender

A conversational agent that recommends SHL Individual Test Solutions from a
free-text hiring need, over a stateless `POST /chat` API. See
[`approach.md`](approach.md) for the design writeup (architecture, retrieval,
prompts, evaluation, tradeoffs) and important context on the catalog data
source.

## Project layout

```
app/
  api.py         FastAPI app: GET /health, POST /chat
  agent.py       Stateless conversational logic (clarify/recommend/refine/compare/refuse)
  prompts.py     LLM prompts + deterministic reply templates
  retriever.py   Hybrid FAISS + BM25 retrieval over the catalog
  scraper.py     Builds the catalog (live scrape, with historical fallback)
  embeddings.py  Local sentence-transformers embeddings (no API key needed)
  models.py      Pydantic schemas (API contract + internal catalog record)
  utils.py       Shared helpers (logging, parsing, keyword tagging)
  config.py      Settings from environment / .env
data/
  catalog_seed_historical.csv   Historical catalog snapshot (see approach.md)
  catalog.json                  Built catalog (generated, committed for deploys)
  vectorstore/                  FAISS index + metadata (generated, committed)
scripts/
  scrape_catalog.py     Rebuild data/catalog.json
  build_vectorstore.py  Rebuild data/vectorstore/
tests/                   pytest suite (no API key required -- LLM is mocked)
```

## Setup

Requires Python 3.11+ (developed and tested on 3.11.9; the spec suggests
3.12+, which should also work but wasn't the environment used here).

Only run `python -m venv .venv` if `.venv/` doesn't already exist in your copy
of the project -- running it again while that same venv is activated fails
with a permission error on Windows (the running `python.exe` is locked).

**Git Bash / macOS / Linux:**
```bash
python -m venv .venv          # skip if .venv/ already exists
source .venv/Scripts/activate  # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in LLM_API_KEY
```

**PowerShell:**
```powershell
python -m venv .venv          # skip if .venv/ already exists
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env     # then fill in LLM_API_KEY
```

Get a free API key from one provider:

- **Groq** (recommended -- fast, generous free tier): https://console.groq.com/keys
- **OpenRouter**: https://openrouter.ai/keys
- **OpenAI**: https://platform.openai.com/api-keys

Set `LLM_PROVIDER` and `LLM_API_KEY` in `.env` accordingly. Any other
OpenAI-compatible endpoint works too via `LLM_BASE_URL` / `LLM_MODEL`.

## Rebuilding the data layer (optional -- already committed)

`data/catalog.json` and `data/vectorstore/` are committed so the service runs
immediately after `pip install`. To rebuild them from scratch:

```bash
python scripts/scrape_catalog.py     # writes data/catalog.json
python scripts/build_vectorstore.py  # writes data/vectorstore/
```

`scrape_catalog.py` first tries to scrape the live paginated SHL catalog
table; if that page's structure isn't found (true as of 2026-07, see
approach.md), it falls back to `data/catalog_seed_historical.csv` and
re-resolves every URL against the live site via HTTP redirect-following, so
every URL ever written to `catalog.json` is confirmed live at build time.

## Running locally

```bash
python run.py
# or: uvicorn app.api:app --reload
```

Then:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## Tests

```bash
pytest tests/ -v
```

The LLM is mocked in every test (`tests/conftest.py::StubLLMClient`), so the
suite runs offline with no API key.

## Deployment

**Render**: connect the repo; `render.yaml` is picked up automatically. Set
`LLM_API_KEY` in the service's environment variables (marked `sync: false`
in `render.yaml` so it's not committed). First `/health` call after a cold
start may take up to ~1-2 minutes while the container boots and the
embedding model loads.

**Railway**: connect the repo; `railway.json` + `Procfile` configure the
build/start commands. Set `LLM_API_KEY` (and `LLM_PROVIDER` if not Groq) in
the service's Variables tab.
