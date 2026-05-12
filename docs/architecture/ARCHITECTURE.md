# ValoCoach Architecture

ValoCoach is a local-first Valorant coaching CLI. It combines locally synced
match history, static tactical knowledge, optional live patch/meta context, and
a streaming LLM response to produce grounded coaching advice from the terminal.

## Runtime Flow

```text
Typer CLI
  -> config/preflight
  -> command module
  -> data/retrieval/stats services
  -> Rich renderer
```

Core commands:

- `coach`: parses a situation, retrieves agent/map/meta facts, adds player
  context when available, trims the prompt to fit budget, and streams an LLM
  answer.
- `interactive`: wraps `coach` in a prompt-toolkit REPL with sliding-window
  conversation memory and persisted sessions.
- `sync`: fetches HenrikDev account, MMR, stored match IDs, and v4 match details
  into local SQLite.
- `stats` and `profile`: load local player data, apply filters, compute
  aggregates and round-level metrics, then render Rich dashboards.
- `ingest` and `index`: populate ChromaDB from built-in JSON, the markdown
  corpus, URLs, or YouTube transcripts.
- `meta`, `meta-refresh`, and `patch`: inspect and refresh bundled or scraped
  Valorant patch/meta context.
- `notes` and `sessions`: manage local coaching sessions and action items.

## Package Map

```text
src/valocoach/
  cli/          Typer app, command entrypoints, Rich display and formatters
  coach/        prompt context, intent classification, templates, session bridge
  core/         config, parsing, memory, context budgeting, preflight checks
  data/         HenrikDev client, Pydantic API models, SQLAlchemy ORM, sync
  llm/          LiteLLM streaming wrapper
  retrieval/    JSON knowledge, ChromaDB ingestion/search, patch/meta scraping
  stats/        aggregate stats, filters, baseline comparison, round analysis
```

## Persistent State

SQLite lives under `~/.valocoach/data` by default and is managed with Alembic.
The schema stores:

- players and MMR snapshots
- matches and per-player match rows
- rounds, per-round players, and kill events
- sync logs
- coaching sessions and notes
- meta cache and patch versions

ChromaDB is separate from SQLite. It uses two collections:

- `valocoach_static`: durable tactical corpus and bundled JSON knowledge
- `valocoach_live`: TTL-managed scraped patch/meta content

## Coaching Pipeline

```text
run_coach
  -> parse_situation
  -> classify_intent
  -> retrieve_static
       -> exact JSON facts
       -> static and live vector hits
  -> build_stats_context
       -> load local player data
       -> compute aggregate stats
       -> analyze rounds
       -> compare baseline
  -> load last-match and open-note context
  -> fit_prompt
  -> stream_completion
  -> display.stream_to_panel
```

Facts from the JSON knowledge base are inserted before vector hits because they
carry exact ability costs, map callouts, economy thresholds, and curated meta.
Vector search is supplemental and can fail without blocking coaching.

## Data Sync Pipeline

```text
sync_player_matches
  -> HenrikClient
  -> SyncOrchestrator.run
       -> resolve account and MMR
       -> open sync log and detect interrupted syncs
       -> discover new stored match IDs
       -> fetch v4 details
       -> map API data to ORM rows
       -> upsert each match
       -> close sync log
```

Sync is incremental by default and stops at already-stored matches unless
`--full` or resume mode is active. Per-match failures are collected without
aborting the entire run; fatal identity or match-list failures raise `SyncError`.

## Boundaries

- `cli/` owns command-line argument handling and terminal rendering.
- `data/` owns API, ORM, repository, and sync semantics. It should not grow new
  presentation behavior.
- `stats/` functions should remain pure over ORM/domain objects where practical.
- `retrieval/` owns knowledge loading, scraping, caching, embeddings, and search.
- `llm/` should stay provider-thin: build LiteLLM arguments, stream tokens, and
  avoid command-specific prompt logic.

## Quality Gates

Before merging or releasing:

```bash
uv run --extra dev ruff check
uv run --extra dev ruff format --check
uv run --extra dev pytest
```

Live Ollama tests are opt-in with the `live` marker and are skipped by default
unless explicitly enabled.
