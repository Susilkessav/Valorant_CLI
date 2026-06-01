# ValoCoach — Pitch & Architecture Document

> A local-first, privacy-respecting Valorant tactical coaching CLI that turns your
> own match history plus a grounded tactical knowledge base into Immortal-level,
> situation-specific advice — all running on a local LLM, no cloud required.

---

## 1. The Pitch

### 1.1 The problem

Improving at Valorant is hard for three reasons:

1. **Generic advice doesn't transfer.** YouTube guides and tier lists are written for
   the average player on the average map. They don't know that *you* lose 70% of your
   Ascent attack rounds, that your Jett entry trades are negative, or that your win
   rate craters after the third game of a session.
2. **Personal data is locked away.** Your match history lives behind Riot's API and a
   patchwork of third-party trackers. There's no tool that fuses *your* numbers with
   *tactical knowledge* and turns the combination into a coaching conversation.
3. **LLMs hallucinate game facts.** Ask a general chatbot "what does Fade's smoke do?"
   and it will confidently invent an ability Fade doesn't have. For a coaching tool,
   fabricated ability names and wrong costs are worse than no answer.

### 1.2 The solution

ValoCoach is a terminal-native coach that:

- **Runs entirely on your machine.** A local Ollama LLM (default `qwen3:8b`) does the
  reasoning. The only optional network dependency is the HenrikDev API for pulling your
  match history. Your gameplay data never leaves your laptop.
- **Grounds every answer in real facts.** Agent abilities, map callouts, and economy
  thresholds come from a curated JSON knowledge base and a vector store (RAG), not the
  model's memory. Exact ability costs are injected verbatim.
- **Personalises with your own stats.** Synced match history feeds a stats engine that
  computes win rates, K/D, KAST, clutch conversion, trade efficiency, per-map ATK/DEF
  splits, weapon head-shot rates, and session-tilt detection — then ships a compact
  summary into the prompt.
- **Refuses to hallucinate where it matters.** Meta/tier-list answers bypass the LLM
  entirely and are rendered deterministically from data. Every non-meta answer is
  fact-checked after streaming: fabricated abilities, wrong-agent attributions, and
  misquoted personal stats are flagged below the response.

### 1.3 Who it's for

Competitive Valorant players (Diamond → Immortal+ mindset) who are comfortable in a
terminal, care about data privacy, and want coaching that references their actual play
rather than generic theory.

---

## 2. System Architecture

### 2.1 Layered design

```
src/valocoach/
  cli/          Typer app, command entrypoints, Rich display & formatters
  coach/        prompt context, intent classification, templates, session bridge,
                fact-check sanitizers, deterministic meta responses
  core/         config, situation parser, conversation memory, context budgeting,
                preflight checks, session persistence
  data/         HenrikDev API client, Pydantic API models, SQLAlchemy ORM,
                repository, sync orchestrator, loader
  llm/          LiteLLM streaming wrapper (provider-thin)
  retrieval/    JSON knowledge, ChromaDB ingestion/search, embeddings,
                patch/meta scraping, lineups, agent roster sync
  stats/        aggregate stats, filters, baseline comparison, round analysis,
                post-game analyzers, zones/constants
```

**Boundary contract** (enforced by convention and tests):

- `cli/` owns argument handling and terminal rendering only.
- `data/` owns API, ORM, repository, and sync semantics — no presentation logic.
- `stats/` functions are pure over ORM/domain objects where practical.
- `retrieval/` owns knowledge loading, scraping, caching, embeddings, and search.
- `llm/` stays provider-thin: builds LiteLLM arguments, streams tokens, no
  command-specific prompt logic.

### 2.2 Runtime flow (high level)

```
Typer CLI
  → config / preflight
  → command module
  → data / retrieval / stats services
  → Rich renderer (streaming panel)
```

### 2.3 Persistent state

```
~/.valocoach/
├── config.toml          Settings (Riot identity, API keys, model config)
├── history              REPL command history (prompt_toolkit)
└── data/
    ├── valocoach.db      SQLite — players, matches, rounds, kills, sessions, notes,
    │                     meta cache, patch versions, MMR history
    └── chroma/           ChromaDB — static + live vector collections
```

**SQLite** is managed with Alembic migrations. Stored entities:

- players and MMR snapshots / MMR history
- matches and per-player match rows
- rounds, per-round players, kill events, ability casts, damage events (spatial columns)
- sync logs (with interrupted-sync detection)
- coaching sessions and notes
- meta cache (TTL-bounded) and patch versions

**ChromaDB** uses two independently managed collections:

| Collection | Contents | TTL |
|------------|----------|-----|
| `valocoach_static` | Agent abilities, map callouts, economy facts (from JSON) | Permanent |
| `valocoach_live` | Patch notes, YouTube transcripts, web articles, scraped meta stats | 30–60 days |

Live docs carry an `expires_at_unix` metadata key so expired content is filtered out at
query time (`$gt now`) — a second line of defence even if the SQLite cache-invalidation
hook didn't run.

### 2.4 Technology stack

| Layer | Choice | Notes |
|-------|--------|-------|
| CLI framework | Typer | sub-commands, shell completion |
| Terminal UI | Rich | streaming panels, themed Valorant palette |
| REPL | prompt_toolkit | history, tab-completion, auto-suggest |
| LLM runtime | Ollama (local) | swappable to any OpenAI-compatible endpoint |
| LLM gateway | LiteLLM | provider-agnostic streaming |
| Coaching model | `qwen3:8b` (default) | `qwen3:14b` / larger on bigger RAM |
| Embeddings | `nomic-embed-text` (768-dim) | via Ollama |
| Vector store | ChromaDB (persistent, cosine/HNSW) | static + live collections |
| Relational store | SQLite + SQLAlchemy 2.0 async (aiosqlite) | WAL, Alembic migrations |
| Token budgeting | tiktoken (`cl100k_base`) | budget math only |
| Match data | HenrikDev API (v4) | optional, for sync |
| Web scraping | trafilatura + BeautifulSoup, optional Tavily | meta-refresh / URL ingest |
| Validation | Pydantic / pydantic-settings | API models + settings |

---

## 3. The Coaching Pipeline (deep dive)

Every `coach` call passes through a deterministic, mostly-LLM-free preprocessing
pipeline before the model sees anything:

```
run_coach(situation)
  → load_settings + preflight (Riot ID, vector store health)
  → parse_situation        regex cascade → agent, map, side, score, clutch, econ, phase
  → classify_intent        rule-based 9-class classifier
  → [elicitation]          intent-aware follow-up questions (≤3, TTY only)
  → [meta short-circuit]   meta intent → deterministic panel, return (no LLM)
  → build grounded context retrieve_static() → JSON facts + vector hits
       + auto-inject the player's top-played agent ability blocks
  → build stats context    load player data → aggregate + round + baseline analysis
  → load last-match + open-notes context
  → fit_prompt             tiktoken budget enforcer trims low-priority chunks
  → stream_completion      LiteLLM → Ollama streaming
  → display.stream_to_panel
  → post-stream sanitizers ability fact-check + numeric stat fact-check
  → stale-meta warning (meta-sensitive intents)
```

### 3.1 Stage 1 — Parse (`core/parser.py`)

A regex cascade extracts agent, map, side, score, clutch state, economy level, and round
phase from the free-text situation string in **< 1 ms, no LLM**. This means flags
(`--agent`, `--map`, `--side`) are rarely needed — the situation text usually carries
everything.

### 3.2 Stage 2 — Classify intent (`coach/intent.py`)

A rule-based classifier assigns one of nine intents, with priority ordering
(`clutch > post_plant > retake > economy > ...`). The intent selects:

- which **prompt template** the LLM receives (`coach/templates.py`)
- which **fields are worth eliciting**
- whether to **bypass the LLM** (meta intent)

| Intent | Triggered by | Questions asked | Template |
|--------|-------------|-----------------|----------|
| `tactical` | map + side present | map, side, agent | Execute / strategy |
| `clutch` | "1v3", "clutch" | side, agent | Clutch decision tree |
| `post_plant` | "post-plant" / spike down | map, agent, side | Spike timer + position |
| `retake` | "retake" / phase | map, agent, side | Retake route advice |
| `economy` | "eco", "save", "buy" | side, score | Buy-round decision |
| `agent_info` | "how does", "abilities", "kit" | agent | Ability breakdown |
| `meta` | "meta", "tier list", "best agent" | *(none)* | Deterministic panel |
| `stat_analysis` | "my stats", "my KD" | *(none)* | Stats-first template |
| `general` | everything else | agent, map | General coaching |

### 3.3 Stage 3 — Intent-aware elicitation (`coach/elicitation.py`)

When a situation is underspecified *for its intent*, ValoCoach asks at most 3 targeted
follow-up questions — only about fields that matter for that intent. A meta question
asks nothing; an economy question only asks for side (and optionally score). Answers are
remembered for the rest of a REPL session (persisted into `SessionMatchContext`), so the
same field is never re-asked. Elicitation is TTY-only — piped/CI invocations skip it.

### 3.4 Stage 4 — Grounded RAG (`retrieval/`)

Retrieval is **JSON-first, vector-second**:

1. **Structured JSON facts** (`format_agent_context`, `format_map_context`,
   `format_meta_context`) are inserted first — they carry exact ability costs, map
   callouts, economy thresholds, and curated meta tiers.
2. **Multi-query vector search** generates several focused queries (situation +
   "{map} callouts", "{map} {side} strategies", "{agent} abilities utility") because
   multiple focused queries activate different embedding dimensions better than one
   concatenated string. Both the static and live collections are searched; hits are
   deduplicated and tagged with a unified `[SOURCE: kind/name]` provenance label.
3. **Auto-injected agent blocks**: even for agent-less questions ("how do I rank up?"),
   the player's top-played agents' ability lists are prepended — without this, small
   models hallucinate abilities for whatever agent name they pick from training data.

Vector search is supplemental and **fails open** — a ChromaDB or embedding error logs a
warning and falls back to the JSON facts rather than blocking coaching.

### 3.5 Stage 5 — Personal stats context (`coach/context.py`, `stats/`)

`build_stats_context` loads recent matches and produces a compact (~150–200 token) block
covering: record / WR / ACS / K/D / KDA / HS% / ADR / econ, entry first-bloods vs
first-deaths, round-level KAST / clutch / trade efficiency, per-map ATK/DEF splits, top
agents and maps with reliability tagging ("low sample" / "thin sample"), weapon HS%, and
an 8-hour rolling **session-tilt detector** (flags a ≥20pp win-rate drop in the back half
of a session). Thin-sample metrics are *tagged, not omitted*, so the LLM treats them
loosely rather than as ground truth.

### 3.6 Stage 6 — Context budgeting (`core/context_budget.py`)

A tiktoken-based counter enforces a hard token limit (~24,000). A 3-stage trimmer drops
the lowest-priority content first (vector hits, then stats) so the prompt always fits the
model's context window without losing the high-value JSON facts.

### 3.7 Stage 7 — Streaming completion (`llm/provider.py`)

A thin LiteLLM wrapper streams tokens from Ollama. The model is selected purely by
config string — prefix-detection routes `ollama/`, `anthropic/`, or `openai/` models, so
swapping to a cloud endpoint is a one-line config change. Conversation history (from the
REPL's sliding window) is inserted between the system prompt and the current user message.

### 3.8 Stage 8 — Post-stream fact-checking (`coach/sanitizer.py`, `coach/stats_sanitizer.py`)

After the answer streams, two deterministic auditors run:

- **Ability fact-check**: cross-references every ability name in the response against
  `agents.json`, bucketing failures into *fabricated abilities*, *wrong-agent
  attributions*, *weapons mis-cast as abilities*, and *generic descriptors*.
- **Numeric stat fact-check**: verifies the model didn't misquote the player's real
  K/D, ACS, ADR, HS%, or win rate from the injected PLAYER CONTEXT.

The answer itself is never modified — warnings are printed below it. This is framed as a
model limitation, not a bug, and nudges the user to verify against in-game tooltips.

---

## 4. Feature Catalogue (all commands)

### 4.1 `valocoach` (hub)
No-argument invocation on a TTY shows a live dashboard: current patch, player sync
status, stale-meta warning, and quick-navigation hints. Non-interactive contexts get a
banner + one-liner only.

### 4.2 `coach`
The core command. With a situation argument → one-shot grounded advice. Without
arguments → the interactive REPL (see §5). Flags: `--agent`, `--map`, `--side`,
`--with-stats/--no-stats`, `--no-elicit`.

### 4.3 `post-game`
Deterministic debrief of your most recent (or a specified) match. **Ten analyzers** run
before any LLM call: first-contact, economy decisions, utility efficiency, round timing,
traded deaths, ATK/DEF split, clutch outcomes, death-location clusters, engagement
distance, plant/defuse distribution. The top three findings (collapsed by root cause) are
injected as ground truth so every claim ties to a real number. A `low_utility` finding on
a kit-heavy agent also pulls a relevant lineup. Offers to continue in the REPL with the
match context pre-loaded.

### 4.4 `lineup`
Searches the local lineup library (seed entries + ingested YouTube/URL lineup chunks).
Filter by agent, map, site, or free-text query. Values are canonical-case normalised on
both read and write.

### 4.5 `stats`
Performance dashboard from synced matches. Flags: `--period` (7d/30d/90d/all),
`--agent`, `--map`, `--result` (win/loss), `--json`.

### 4.6 `sync`
Incremental fetch of competitive match history from HenrikDev → local SQLite. Stops at
already-stored matches unless `--full`. Per-match failures are collected without aborting;
fatal identity/match-list failures raise `SyncError`. Each run also sweeps expired
retrieval cache rows. Flags: `--limit`, `--full`, `--mode`.

### 4.7 `profile`
Compact player identity card + recent performance summary. Can look up arbitrary players
via `--name`/`--tag`. Flags: `--limit`, `--json`.

### 4.8 `meta`
Deterministic tier list / agent abilities / map callouts / economy thresholds from the
knowledge base — shown instantly, no LLM. Flags: `--agent`, `--map`, `--json`.

### 4.9 `meta-refresh`
Automated meta-sync pipeline (§6). Flags: `--force`, `--dry-run`, `--watch`,
`--install-cron`, `--youtube`.

### 4.10 `agents-refresh`
Diffs local `agents.json` against the Liquipedia agents API; reports new agents and
gaps in `meta.json`. `--extract-kits` parses wikitext templates (no LLM) into agent kit
data; `--auto-stub-meta` appends C-tier placeholders.

### 4.11 `ingest`
Populates / updates the vector store. `--seed` embeds the built-in JSON; `--corpus`
embeds markdown; `--url` runs the full classify pipeline (lineup chunks → lineup DB, web
chunks → 30-day TTL); `--youtube` fetches + classifies + previews a transcript;
`--preview`, `--force`, `--clear`, `--stats`.

### 4.12 `patch`
Shows the locally stored patch version. `--check` compares against the live HenrikDev
version.

### 4.13 `notes`
Capture and track coaching action items: `list`, `add <text>`, `resolve <id>`. Also
available as REPL slash commands.

### 4.14 `sessions`
Manage saved coaching session histories: `list`, `close <id>`.

### 4.15 `config`
`config init` writes/reset `~/.valocoach/config.toml`; `config show` prints effective
settings.

---

## 5. The Interactive REPL

Entered via `valocoach coach` (no args, TTY). Built on prompt_toolkit with persistent
history (`~/.valocoach/history`), tab-completion (agents, maps, slash commands), and
auto-suggest.

- **`SessionMatchContext`** — a mutable bag (agent, map, side, score, enemies, last
  result, econ) written by slash commands and merged into every `run_coach` call. Once
  set, elicitation is skipped.
- **`ConversationMemory`** — a sliding window (20 turns / 3,000 tokens) injected between
  the system prompt and the current message, giving the model full multi-turn context.

**Match-context slash commands:** `/agent`, `/map`, `/side`, `/score`, `/won`, `/lost`,
`/eco`, `/enemy`, `/half` (flip side at half-time), `/context`, `/reset`.

**Session commands:** `/help`, `/clear`, `/memory`, `/save`, `/sessions`, `/stats`,
`/note`, `/notes`, `/resolve`, `/quit`.

---

## 6. The Meta Pipeline (deterministic by design)

The display side of meta is **never** produced by an LLM — the tier list comes straight
from `meta.json` + `agents.json`, so there are no hallucinated abilities and no
temperature variance.

`meta-refresh` runs the data-update pipeline:

```
1. Patch detection     HenrikDev API → compare to stored version
2. Patch notes scrape   playvalorant.com (auto-constructed URL)
3. Patch diff           per-agent changes → patch_changes/
4. Stats scrape         tracker.gg (Diamond+) + vlr.gg (pro/VCT)
5. YouTube ingest       optional; classify → lineup or web chunks
6. LLM tier regen       LLM outputs raw pick/win-rate floats only
7. Write meta.json      stamped sync_in_progress until step 8 completes
8. Re-ingest KB         meta + agents + maps re-embedded into ChromaDB
```

**Deterministic tier scoring** is the key design choice. The LLM is *not* asked to assign
S/A/B/C — it only emits numeric `pick_rate_pct` and `win_rate_pct`. A formula computes the
tier:

```
score = win_rate_pct + log1p(pick_rate_pct) × 0.5
```

| Tier | Score threshold |
|------|----------------|
| S | ≥ 53.5 |
| A | ≥ 51.5 |
| B | ≥ 50.0 |
| C | < 50.0 |

Same inputs always produce the same tiers — no "flapping" across refreshes from LLM
temperature. The LLM still owns the prose `reason` field, where language models add value.

---

## 7. The Data Sync Pipeline

```
sync_player_matches
  → HenrikClient
  → SyncOrchestrator.run
       → resolve account + MMR
       → open sync log, detect interrupted syncs
       → discover new stored match IDs
       → fetch v4 match details
       → map API data to ORM rows
       → upsert each match
       → close sync log
```

Sync is incremental by default; `--full` or resume mode forces a wider sweep. The mapper
(`data/mapper.py`) converts validated HenrikDev v4 Pydantic models into ORM rows including
rounds, per-round players, kills, ability casts, and damage events with spatial columns.

---

## 8. Design Principles & Differentiators

1. **Local-first / privacy.** The LLM and all data live on-device. Only opt-in match
   sync touches the network.
2. **Deterministic where correctness matters; LLM where prose matters.** Tier lists,
   fact-checks, and post-game findings are deterministic; only the explanatory writing
   is delegated to the model.
3. **Ground, then generate.** Exact facts are injected verbatim before the model
   reasons, and audited after it answers.
4. **Fail open.** Vector search, stats, last-match, and notes context are each wrapped so
   any single failure degrades gracefully rather than blocking coaching.
5. **Provider-agnostic.** A one-line config change swaps the local model for any
   OpenAI-compatible endpoint.
6. **Branded, polished terminal UX.** Unified command-frame wrappers, themed Valorant
   palette, and consistent visual hierarchy throughout.

---

## 9. Performance Profile & Latency Roadmap

The dominant wall-clock cost on any `coach` call is local LLM token generation
(`qwen3:8b`), which is inherent to running a model on-device. But the **pre-LLM overhead**
(time-to-first-token) currently carries avoidable redundancy. The identified hotspots and
proposed fixes:

### 9.1 Redundant database initialisation (highest impact)

On a single stats-enabled `coach` call the pipeline issues **~5 independent
`asyncio.run()` boundaries** — `get_top_played_agents`, `build_stats_context`,
`get_last_match`, `get_player_puuid`, and `list_open_notes`. Each one:

- spins up a fresh event loop,
- calls `ensure_db()`, which **recreates the async engine** (`init_engine`, `NullPool`),
  runs `CREATE TABLE IF NOT EXISTS` for every table, and runs an Alembic stamp check
  (importing Alembic and building a `ScriptDirectory`).

**Fix:** initialise the engine once per process and guard `ensure_db` so the
schema-create + Alembic-stamp work runs at most once; consolidate the per-call data needs
into a single loader pass (one event loop, one `PlayerData` bundle reused by all five
consumers). Expected savings: several DB round-trips and repeated Alembic imports per call.

### 9.2 ChromaDB client re-instantiation

`retrieve_static` loops over `len(queries) × 2 collections` `search()` calls, and **every**
`search()` calls `get_collection → get_client → chromadb.PersistentClient(...)`, which
re-opens the persistent store and reloads the HNSW index. A typical map+side+agent query
creates **~8 PersistentClient instances per coach call**.

**Fix:** memoise the client per `data_dir` (module-level singleton / `lru_cache`). The
client is reusable across queries and collections.

### 9.3 Redundant query embeddings

Each query string is embedded **once per collection** (static + live), so the same text
is embedded twice; and every `embed_one` is a serial Ollama HTTP round-trip
(~8 serial calls per coach call).

**Fix:** embed each unique query exactly once and reuse the vector across both
collections; batch all unique queries into a single `ollama.embed(input=[...])` call
(the `embed()` helper already supports batch input) to collapse N serial round-trips into
one.

### 9.4 Uncached settings load

`load_settings()` re-parses `config.toml` + environment on every call and is invoked from
~21 sites. Per-call cost is small but it compounds.

**Fix:** cache the resolved `Settings` for the process lifetime (`functools.lru_cache`),
with an explicit invalidation hook for `config init`.

### 9.5 Summary of recommended changes

| Hotspot | Current | Target |
|---------|---------|--------|
| DB inits per coach call | ~5 engines + 5 schema-creates + 5 Alembic checks | 1 engine, 1 schema-create, consolidated loader |
| Chroma client builds per call | ~8 | 1 (cached) |
| Query embed round-trips | ~8 serial, duplicated per collection | 1 batched call, deduped |
| Settings parses | per call (~21 sites) | 1 cached |

None of these touch the model itself, so they reduce time-to-first-token without changing
answer quality. They are independent and can be landed incrementally; §9.1 and §9.2 carry
the most benefit.

---

## 10. Quality Gates

```bash
uv run --extra dev ruff check
uv run --extra dev ruff format --check
uv run --extra dev pytest
```

The test suite covers parsing, intent classification, retrieval, stats, sanitizers,
sync, meta generation, the REPL, and CLI integration (60+ test modules). Live Ollama
tests are opt-in via the `live` marker and skipped by default.
