# AGENTS.md

> Instructions for AI coding assistants working on this project. Read this file first before making changes.

---

## Project: ValoCoach

A CLI-based Valorant tactical coaching chatbot. The user describes a match situation in natural language; the system responds with a tactical plan grounded in three sources:

1. **The player's historical performance** (stats from past matches)
2. **Current Valorant meta** (live-scraped from the web per query)
3. **Tactical knowledge** (RAG over scraped guides, pro VOD transcripts, patch notes)

The CLI is being built **first**. A web tracker frontend will be added later, sharing the same database and stats engine.

---

## Core architectural decisions (do not change without asking)

These are locked in. If you think one is wrong, surface the concern — do not silently refactor.

| Decision | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Stats math, RAG, LLM ecosystem all Python-native |
| Data source | HenrikDev unofficial API (`api.henrikdev.xyz`) | Only viable option without Riot production key |
| Database | SQLite (WAL mode) → PostgreSQL later | Local-first; schema designed for clean migration |
| ORM | SQLAlchemy 2.0 + Alembic | Dialect-agnostic from day one |
| LLM provider | Ollama local now, Claude API later | Abstracted via LiteLLM — swap is one config line |
| LLM model | Qwen3 (8B/14B/32B by VRAM) | Best local reasoning + 40K context + system prompt support |
| Embeddings | nomic-embed-text via Ollama | 8K context, 86% top-5 retrieval accuracy, free, local |
| Vector store | ChromaDB (separate from main SQLite) | Persistence + metadata filtering + embedded |
| CLI framework | Typer + Rich + prompt_toolkit | Streaming markdown, autocomplete, REPL |
| Content extraction | Trafilatura → readability-lxml → BS4 fallback chain | Layout-agnostic, best F1 on benchmarks |

**Two databases, deliberately separate:**
- `data/valocoach.db` — SQLite for structured data (matches, stats, cache metadata)
- `data/chroma/` — ChromaDB for vector embeddings

Do not try to merge these. Keep concerns separate.

---

## Project structure

```
valorant-coach/
├── src/valocoach/
│   ├── cli/          # Typer commands, Rich display, REPL
│   ├── core/         # Config (pydantic-settings), models, exceptions
│   ├── data/         # API client, SQLAlchemy models, sync logic, Alembic migrations
│   ├── stats/        # Stat formulas, analyzer, formatters
│   ├── retrieval/    # Scraper, embedder, ChromaDB interface, RAG pipeline
│   └── llm/          # LiteLLM wrapper, prompts, coaching orchestrator, memory
├── tests/
├── data/             # Runtime, gitignored
├── pyproject.toml
└── README.md
```

**Module boundary rules:**
- `cli/` may import from any other module
- `llm/` may import from `stats/`, `retrieval/`, `data/`, `core/`
- `retrieval/` may import from `data/`, `core/` only
- `stats/` may import from `data/`, `core/` only
- `data/` may import from `core/` only
- `core/` imports from nothing project-internal

If you find yourself wanting to violate these, stop and ask.

---

## Coding conventions

- **Type hints required** on all function signatures. Use `from __future__ import annotations` at the top of every file.
- **Pydantic v2** for all data models that cross module boundaries (API responses, LLM I/O, config).
- **SQLAlchemy 2.0 style** (`Mapped[...]`, `mapped_column(...)`), not legacy declarative.
- **Async** for all I/O: `httpx.AsyncClient`, async DB sessions where practical. CLI entry points use `asyncio.run()`.
- **No `print()`** outside `cli/display.py`. Use `rich.console.Console` or the `logging` module.
- **No bare `except:`**. Catch specific exceptions. Re-raise as project-defined exceptions from `core/exceptions.py` when crossing module boundaries.
- **Format**: `ruff format` (line length 100). **Lint**: `ruff check`. Both must pass before commit.
- **Docstrings**: Google style for any non-trivial function. One-liners are fine for obvious functions.

---

## Critical correctness rules

These prevent silent bugs that corrupt user-visible stats:

1. **Always filter by queue mode.** Mixing competitive, deathmatch, and swiftplay corrupts every stat. Default filter is `competitive` unless explicitly overridden.
2. **Never recompute stats Riot already provides.** If the API returns `score`, store it raw and divide by `rounds_played` for ACS at query time. Don't reverse-engineer it.
3. **Match data is immutable.** Once a match is synced, never re-fetch it. Use `INSERT OR IGNORE` / upsert by `match_id`.
4. **Timestamps are UTC ISO8601 TEXT in the DB.** Convert to user's local timezone only at display time, in `cli/display.py`.
5. **Side calculation:** rounds 0–11 = first half (Red attacks, Blue defends); rounds 12+ = swap. Overtime alternates. There is a helper for this — use it, don't inline the logic.
6. **Sample-size warnings are mandatory.** Any stat displayed with fewer matches than its threshold (see `stats/calculator.py` constants) must show a `⚠️` warning. Do not silently show unreliable numbers.
7. **AFK rounds skew averages.** Check `afkRounds` and `stayedInSpawnRounds` fields. Document whether each metric includes or excludes them.

---

## LLM and RAG rules

- **Never let the LLM invent agent abilities, map callouts, or patch-specific numbers.** The system prompt enforces "only reference what appears in CONTEXT." Do not weaken this rule.
- **Always inject the current patch version** into the system prompt so the model knows what era it's reasoning about.
- **Cache TTLs are tiered:**
  - Stable (map callouts, agent ability text): 30 days
  - Semi-stable (general strategies): 3–7 days
  - Volatile (tier lists, pick rates, pro comps): 4–24 hours
  - On patch change: delete all volatile, mark semi-stable for refresh
- **RAG retrieval target: top-5 chunks, ~256–512 tokens each.** Adjust down to top-3 if context budget is tight.
- **Streaming is non-negotiable for coaching responses.** The user sees tokens as they arrive via `Rich.Live` + `Markdown`. Do not introduce blocking calls in the response path.
- **Temperature 0.4–0.6** for coaching. Higher = more hallucination. Lower = generic advice.

---

## Testing requirements

- **Stats math gets golden-dataset tests.** For every formula in `stats/calculator.py`, there must be at least one test using a real recorded match where the expected output is known. Compute ACS from raw fields and assert it matches the API-reported value within ±1.
- **API client tests use `pytest-httpx`** to replay recorded responses. Never hit the live API in tests.
- **LLM pipeline tests mock the LLM call.** Test the orchestration (prompt assembly, context retrieval, output parsing), not the model itself.
- **LLM output quality** is evaluated separately via an eval harness (DeepEval / RAGAS) against a fixed scenario set — not in unit tests.
- **Run before every commit:** `ruff check && ruff format --check && pytest`

---

## What "done" looks like for a feature

A feature is not done until:

1. Code is written, type-hinted, and passes `ruff check`.
2. Tests exist and pass.
3. Affected commands have updated `--help` text.
4. If user-facing behavior changed, the README example section is updated.
5. If a new dependency was added, it's pinned in `pyproject.toml` with a justification comment.
6. If the database schema changed, an Alembic migration exists and has been tested both forward and backward.

---

## Things to never do

- **Never commit API keys, riot IDs, or `.env` files.** `.env.example` only.
- **Never bypass rate limits** on HenrikDev or scraped sites. The default is 1 req / 2 sec for scrapes, and the API client respects HenrikDev's documented limits with exponential backoff.
- **Never bypass CAPTCHAs or anti-bot protections.** If a site blocks scraping, drop it from the source list — don't escalate.
- **Never store raw scraped HTML long-term.** Extract the text, store the text, discard the HTML.
- **Never hardcode the LLM provider.** Always go through the LiteLLM wrapper in `llm/provider.py`. The provider swap is a single config change — preserve that property.
- **Never expose the raw HenrikDev response shape to the CLI layer.** Map it through Pydantic models in `data/` first.
- **Never compute stats in SQL when they're needed in Python anyway.** Aggregations (SUM, COUNT) belong in SQL; ratios and formulas belong in `stats/calculator.py` where they can be unit-tested.
- **Never assume the user's timezone.** Read it from config or system, never hardcode.

---

## Project memory — context that persists across sessions

This section is the project's long-term memory. Update it when significant decisions are made.

### Resolved decisions

- **2026-04**: Chose HenrikDev API over local client API and official Riot API. Reason: only option that works without a production key and without the game running.
- **2026-04**: Chose Qwen3 over DeepSeek-R1 for local reasoning. Reason: DeepSeek-R1 officially recommends against system prompts, which breaks our grounding strategy.
- **2026-04**: Chose ChromaDB over sqlite-vec, FAISS, LanceDB. Reason: built-in persistence + metadata filtering + embedded mode + actively maintained.
- **2026-04**: Chose LiteLLM over a custom provider abstraction. Reason: zero-cost abstraction that makes the eventual Claude API swap trivial.
- **2026-04**: Chose to keep ChromaDB and SQLite as separate stores rather than using sqlite-vss. Reason: separation of concerns, sqlite-vss is deprecated.

### Open questions (decide as project progresses)

- Whether to add Riot production key path as a fallback once the project is mature enough to apply.
- Whether to support multiple players per database (current schema supports it; CLI assumes single player).
- Whether the web tracker frontend should be a separate repo or a sibling package in this monorepo.

### Known gotchas discovered during development

_(Add entries here as they are found. Format: date, issue, resolution.)_

- _Empty so far — add as bugs are encountered and fixed._

---

## When you're unsure

- Ask before introducing a new top-level dependency.
- Ask before changing the database schema (any change requires an Alembic migration).
- Ask before changing the system prompt structure (the grounding rules are load-bearing).
- Ask before adding a new data source for RAG (each source has scraping, caching, and ToS implications).
- For everything else, follow the conventions above and propose changes via PR with reasoning.

---

## Reference docs

- HenrikDev API: https://docs.henrikdev.xyz/valorant
- Valorant static assets: https://valorant-api.com/
- Ollama: https://ollama.com/library
- LiteLLM: https://docs.litellm.ai/
- ChromaDB: https://docs.trychroma.com/
- Trafilatura: https://trafilatura.readthedocs.io/
- Typer: https://typer.tiangolo.com/
- Rich: https://rich.readthedocs.io/

The full project research and build plan lives in the project root as `BUILD_PLAN.md` — read it for the "why" behind any decision listed above.