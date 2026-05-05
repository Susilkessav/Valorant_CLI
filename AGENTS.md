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
- `data/valocoach.db` — SQLite for structured data (matches, stats, scrape cache rows with TTL)
- `data/chroma/` — ChromaDB for vector embeddings

Do not try to merge these. Keep concerns separate.

**Two ChromaDB collections, also deliberately separate:**
- `valocoach_static` — corpus content (agents, maps, concepts, patch notes, YouTube transcripts). Indexed once via `valocoach index`; idempotent re-runs are upserts.
- `valocoach_live` — per-query scraped meta (metasrc and similar). Populated lazily by `_fetch_live_meta` and TTL-evicted via the SQLite `meta_cache` table.

`valocoach ingest --clear` clears both. `retrieve_static` queries both per multi-query and dedupes results across them. Do not push live-scraped content into the static collection — it pollutes the index that `valocoach index` is supposed to own.

---

## Project structure

```
valorant-coach/
├── src/valocoach/
│   ├── cli/
│   │   ├── app.py            # Typer entry, command wiring
│   │   ├── display.py        # Rich console + streaming panel helpers
│   │   └── commands/         # one file per command: coach, stats, sync,
│   │                         #   profile, meta, ingest, interactive, config
│   ├── core/
│   │   ├── config.py         # pydantic-settings + TomlConfigSettingsSource
│   │   ├── exceptions.py
│   │   ├── parser.py         # Situation: Pydantic model + parse_situation()
│   │   │                     #   regex-first against canonical JSON name lists
│   │   ├── context_budget.py # count_tokens, trim_text_to_tokens, fit_prompt
│   │   └── memory.py         # ConversationMemory (max_turns + max_tokens eviction)
│   ├── data/
│   │   ├── api_client.py     # HenrikDev async client (httpx + tenacity)
│   │   ├── api_models.py     # Pydantic models for raw API shapes
│   │   ├── models.py         # Pydantic models for cross-module use
│   │   ├── orm_models.py     # SQLAlchemy 2.0 ORM (matches, players,
│   │   │                     #   match_players, meta_cache, patch_versions)
│   │   ├── database.py       # Async engine, session_scope, ensure_db
│   │   ├── repository.py     # Query helpers
│   │   ├── mapper.py         # API → ORM mapping
│   │   └── sync.py           # Match sync orchestration
│   ├── stats/                # Calculator, round analyzer, formatter, filters
│   ├── coach/                # Stats-context builder for the coach prompt
│   ├── retrieval/
│   │   ├── data/             # Bundled JSON: agents, maps, meta
│   │   ├── agents.py / maps.py / meta.py   # JSON loaders + format_*_context
│   │   ├── chunker.py        # tiktoken-accurate chunk_markdown + Chunk
│   │   ├── embedder.py       # nomic-embed-text via Ollama (embed/embed_one)
│   │   ├── vector_store.py   # ChromaDB client + STATIC/LIVE collections
│   │   ├── ingester.py       # ingest_text + ingest_knowledge_base
│   │   ├── searcher.py       # search() + collection_stats() (both buckets)
│   │   ├── retriever.py      # RetrievalResult + retrieve_static (multi-query,
│   │   │                     #   both collections) + async retrieve_context
│   │   ├── cache.py          # async TTL cache over meta_cache table
│   │   ├── patch_tracker.py  # PatchVersion writes + get_current_patch
│   │   └── scrapers/         # ScrapedContent + web (trafilatura) + youtube
│   └── llm/
│       └── provider.py       # LiteLLM streaming wrapper
├── alembic/versions/         # Migrations (auto-stamped by ensure_db)
├── corpus/
│   ├── agents/ maps/ meta/   # generated via scripts/build_corpus.py
│   └── concepts/             # hand-written reference markdown
│                             #   (economy, fundamentals, executes, retakes, roles)
├── tests/                    # 678 tests, asyncio_mode=auto
├── data/                     # Runtime, gitignored — valocoach.db + chroma/
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

- **`parse_situation()` is the first step in `run_coach`, before any LLM call.** It fills in `agent`, `map`, `side`, and other fields from the situation text so the retrieval path gets accurate queries even when the user doesn't pass CLI flags. The parser is regex-first (sub-millisecond) against `list_agent_names()` / `list_map_names()` — not LLM-backed. Never replace it with an LLM call in the hot path; the latency is unacceptable.
- **CLI flags always beat the parser.** `_resolve_fields(situation, agent, map_, side)` returns `(parsed, agent or parsed.primary_agent, map_ or parsed.map, side or parsed.side)`. The user-message metadata block is built from the *resolved* values so it always reflects what the system actually used for retrieval.
- **Never let the LLM invent agent abilities, map callouts, or patch-specific numbers.** The system prompt enforces "only reference what appears in GROUNDED CONTEXT." Do not weaken this rule.
- **JSON facts always go first in the grounded context.** `retrieve_static` orders chunks: structured agent/map/meta JSON, then vector hits. The deterministic JSON portion is the load-bearing part of grounding — vector hits are supplemental.
- **Multi-query retrieval, not single-query.** `build_retrieval_queries` generates 1–4+ focused queries (situation + map callouts + map+side strategies + per-agent ability queries). Concatenating everything into one big query measurably loses recall. Don't refactor back to a single query.
- **Cache TTLs are tiered (in `cache.TTL_HOURS`):**
  - `stable` (corpus / static knowledge): 30 days
  - `semi_stable` (patch notes, meta articles): 5 days
  - `volatile` (live pick rates, scraped tier lists): 12 hours
  - On patch change: `invalidate_volatile()` runs from `patch_tracker.check_patch_update`. Semi-stable refresh is opportunistic via TTL, not forced.
- **Vector search defaults: `n_results=3` per query, `max_distance=0.45`, `doc_types=["patch_note","youtube","web","concept"]`.** Tighter `max_distance` than the 0.5 default keeps weak matches out of the LLM context.
- **Coach output: `n_results=5` final chunks** after dedup across queries and collections. Reduce to 3 if you ever see context-window pressure (Qwen3's 40K leaves plenty of headroom today).
- **`fit_prompt()` is called in `run_coach` after retrieval and before `_build_system_prompt`.** It trims the grounded/stats blocks so the combined prompt stays within the 24 K hard limit. The system base and user message are immovable. Do not call it before retrieval (you don't have the context blocks yet) or after building the system prompt (too late to trim).
- **`ConversationMemory` is used only in interactive mode.** One-shot `coach` invocations do not use it — they are stateless by design. The `run_coach` function accepts `conversation_history` as an optional parameter; one-shot callers pass `None` (the default).
- **Token counting uses `cl100k_base` everywhere** (context_budget + chunker). Don't introduce a second encoding — the consistent tokeniser is deliberate.
- **Streaming is non-negotiable for coaching responses.** The user sees tokens as they arrive via `Rich.Live` + `Markdown`. Do not introduce blocking calls in the response path.
- **Temperature 0.4–0.6** for coaching. Default in config is `llm_temperature = 0.6`. `llm_max_tokens = 3000` is also in `Settings`; bump in `~/.valocoach/config.toml` if responses truncate.

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
- **2026-04**: Split ChromaDB into two collections (`valocoach_static`, `valocoach_live`). Reason: `valocoach ingest --clear` was nuking live-scraped meta along with the corpus, and live scraping was polluting the static index. Static is owned by `valocoach index`; live is TTL-managed via the SQLite `meta_cache` table. `retrieve_static` queries both per multi-query and dedupes.
- **2026-04**: Chose `chunk_markdown` (tiktoken-accurate, heading-boundary aware) over character-count chunking. Reason: char-count chunkers regularly overflow the embedder's 8K token window on dense markdown sections; tiktoken matches the embedder's actual tokenizer.
- **2026-04**: Cache, patch tracker, and meta_cache schema are async (SQLAlchemy `AsyncSession`). Reason: the engine is async everywhere else; mixing sync sessions on the same engine deadlocks.
- **2026-04**: `ensure_db` calls `Base.metadata.create_all` and then stamps Alembic head idempotently. Reason: a CLI-created DB without an alembic_version row makes the first `alembic upgrade head` try to recreate every table and explode. The stamp keeps CLI bootstrap and migration paths compatible.
- **2026-05**: Added `src/valocoach/core/parser.py` — `Situation` Pydantic model + `parse_situation()` using regex against canonical agent/map JSON (not a hardcoded list). Vocabulary is `@lru_cache`-backed so JSON loads once per process. `(?<!\w)NAME(?!\w)` lookarounds replace `\b` so KAY/O's slash works. CLI flags win over parser output via `_resolve_fields()`; the user-message metadata block is built from the *resolved* fields so `--agent Jett` appears in the block even when "Jett" isn't in the situation text.
- **2026-05**: Added `src/valocoach/core/context_budget.py` — tiktoken-based token counter and `fit_prompt()` three-stage trimmer. Hard limit 24 K tokens (40 K Qwen3 window − 16 K reserve). Trimming priority: grounded_context truncated to 2 K first, stats dropped second, grounded truncated further third. system_base and user_msg are never modified. In normal operation this never fires (typical prompt is 3–5 K tokens); it's a safety net for extreme inputs.
- **2026-05**: Added `src/valocoach/core/memory.py` — `ConversationMemory` with max_turns + max_tokens dual eviction. Drops oldest complete user+assistant exchange; handles orphaned assistant turn at head. `messages` property returns a copy for direct injection into LiteLLM's `messages` list. No LLM-based summarisation (stretch goal).
- **2026-05**: `stream_completion` in `llm/provider.py` now accepts `conversation_history: list[dict] | None` — prior turns inserted between system message and the current user message. Backward-compatible default (None).
- **2026-05**: `valocoach interactive` REPL implemented in `cli/commands/interactive.py`. prompt_toolkit session with FileHistory, WordCompleter (agents + maps + slash commands), complete_while_typing=False. Slash commands: /help /clear /memory /stats /quit. Memory stored as raw situation text (not the formatted metadata+situation message) to keep it compact.

### Open questions (decide as project progresses)

- Whether to add Riot production key path as a fallback once the project is mature enough to apply.
- Whether to support multiple players per database (current schema supports it; CLI assumes single player).
- Whether the web tracker frontend should be a separate repo or a sibling package in this monorepo.

### Known gotchas discovered during development

_(Add entries here as they are found. Format: date, issue, resolution.)_

- **2026-04** — Trafilatura returns nothing useful on JS-rendered sites (Blitz.gg, tracker.gg). Resolution: scraper returns `None` softly; callers (`_do_url`, `_fetch_live_meta`) treat `None` as a non-fatal skip. Stick to server-rendered sources (playvalorant.com, VLR.gg, metasrc.com); add Playwright as an optional fallback only if a JS-only source becomes critical.
- **2026-04** — Sample retrieval-pipeline tests proposed a `VectorStore` class API. Resolution: rejected. Three callers (`ingester.py`, `searcher.py`, `cli/commands/ingest.py`) depend on the functional API (`get_collection`, `ingest_text`, `search`). The class would have required rewriting all three. The functional API stays.
- **2026-04** — Patches generated as `revision = '<id>'` (single-quoted) broke `tests/test_database_ensure._alembic_head()` which used `line.split('"')[1]`. Resolution: parser now uses `re.search(r'["\']([a-f0-9]+)["\']', line)` and tolerates either quote style. If you add a migration with neither, fix the regex.
- **2026-04** — First-run corpus indexing through Ollama `nomic-embed-text` takes 30–60 s for ~100 chunks (model warm-up + batch embed). Subsequent single-query embeds are ~100 ms. Don't add a "this is broken" timeout under 90 s.
- **2026-05** — `_SIDE_DEFENSE` regex `defen[cs](?:e|ing|ders?)?` matched "defence/defense" but not "defending" (stem is `defend`, not `defens`/`defenc`). Fixed to `defen(?:d(?:ing|ers?)?|[cs](?:e|ing|ders?)?)`. If you extend this regex, verify "defending", "defenders", "defence", "defense" all still match and "defendant" doesn't.
- **2026-05** — `Situation.to_metadata_block()` renders the agents line as `Agent(s): Jett` (always with the `(s)` suffix regardless of count). Any assertion checking for `"Agent: Jett"` in the user message must use `"Agent(s): Jett"` instead.

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