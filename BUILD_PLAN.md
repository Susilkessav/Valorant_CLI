# Valorant CLI tactical coaching chatbot: complete build plan

**This system is buildable in six weeks using HenrikDev's API for match data, Qwen3 via Ollama for local inference, ChromaDB for vector search, and a shared SQLite schema designed for future web tracker expansion.** The architecture chains three data streams — the player's historical stats, live-scraped meta knowledge, and structured game-situation parsing — into a single coaching prompt. Every component below is concrete and dependency-ordered so you can execute sequentially. The hardest risk is HenrikDev API reliability; the highest-value early validation is streaming Ollama output through Rich's Live markdown renderer, which proves the core UX in week one.

---

## 1. HenrikDev is the only viable data source right now

**Riot's official Valorant API requires a production key** that demands a working application with RSO (Riot Sign On) integration and weeks of review — a chicken-and-egg problem for a new project. Personal/development keys cannot access `val-match-v1` at all. The local client API (lockfile-based at `127.0.0.1:{port}`) provides the richest data but only while the game is running, making it useless for async coaching sessions.

HenrikDev's unofficial API (`api.henrikdev.xyz`) wraps Riot's internal game-client endpoints into a clean REST interface. A basic API key (instant via their dashboard, requires joining their Discord) gives **30 requests/minute**; an enhanced key (reviewed in 1–2 weeks) bumps this to **90 req/min**. The v4 match endpoint returns everything needed: per-round player stats, kill events with millisecond timestamps, damage breakdowns by hit region, economy data per round, and agent/map metadata.

The key endpoints for this project:

- `/valorant/v4/matches/{region}/{platform}/{name}/{tag}` — match history (filter by `?mode=competitive`)
- `/valorant/v4/match/{region}/{matchid}` — full match details with round-by-round data
- `/valorant/v1/stored-matches/{region}/{name}/{tag}` — lightweight historical match data
- `/valorant/v2/mmr/{region}/{name}/{tag}` — current rank and RR
- `/valorant/v1/version/{region}` — current game version (critical for patch detection)

**Mode filtering is essential.** Always filter to `competitive` for coaching stats. Deathmatch has no rounds or teams. Spike Rush uses random weapons. Swiftplay runs a shorter format (first to 5). Mixing modes corrupts every metric. The `stored-matches` endpoint accepts `?mode=competitive` directly; for full match details, check `metadata.queue.id` in the response.

The long-term play is to build the CLI using HenrikDev data, then apply for a Riot production key once you have a working product to demonstrate during review.

---

## 2. Every stat formula, derived from raw match JSON

### ACS (Average Combat Score)

ACS is the single most-referenced Valorant performance metric and **Riot has never published its exact formula**. The community-verified consensus, confirmed across multiple data-mining efforts, breaks combat score per round into four additive components:

| Component | Points | Details |
|-----------|--------|---------|
| Damage | **1 point per damage dealt** | 1:1 mapping — 150 damage = 150 score |
| Kill bonus | **70–150 per kill** (scales by enemies alive) | 5 alive: 150 · 4 alive: 130 · 3 alive: 110 · 2 alive: 90 · 1 alive: 70 |
| Multi-kill bonus | **+50 per additional kill in same round** | 2K: +50 · 3K: +100 · 4K: +150 · Ace: +200 |
| Non-damaging assist | **+25 per assist** | Sage heal, Sova recon, KAY/O suppress, flash assists |

`ACS = Total Combat Score across all rounds ÷ Rounds Played`, rounded to nearest integer. Note that first blood is inherently worth the most kill points (150) because all 5 enemies are alive. The API returns `stats.score` per player per match, which is the total combat score — divide by `roundsPlayed` to get ACS. Store the raw score; compute ACS at query time.

### Other core metrics

**ADR** = `damage_dealt ÷ rounds_played`. All rounds count in the denominator, including spike-detonation rounds and rounds where the player died instantly. Self-damage and friendly fire are excluded. A benchmark of **~150 ADR** means you're averaging one full kill's worth of damage per round.

**KAST%** = percentage of rounds where the player achieved at least one of: Kill, Assist, Survived, or was Traded. The trade window is **~5 seconds** (community consensus; sources vary from 3–5s, but 5s is what most trackers use). This is a binary per-round metric — multiple contributions in a single round don't stack beyond 1. Above **70% is good; above 75% is excellent**.

**HS%** = `headshots ÷ (headshots + bodyshots + legshots) × 100`. This is per-bullet-hit, not per-kill. Missed shots are excluded. Valorant HS% runs much lower than CS-style kill-headshot% — a 25% HS% in Valorant is roughly equivalent to 50%+ in CS terms. The API provides `stats.headshots`, `stats.bodyshots`, `stats.legshots` directly.

**First Blood / First Death Rate** = per-round. For each round, find the kill event with the lowest `kill_time_in_round` (milliseconds). The killer gets FB credit; the victim gets FD credit. Compute from the `kills` array in round data.

**Clutch Rate** = clutch rounds won ÷ total clutch situations. A clutch is a 1vN scenario — track when a player becomes last alive by examining kill timestamps within each round, count remaining enemies at that moment, then check if the round was won.

**Trade Efficiency** = traded deaths ÷ total deaths. When a player dies at time T, check if any teammate kills that player's killer within **5 seconds**. This requires comparing `kill_time_in_round` across events in the same round.

**Econ Rating** = `(damage_dealt ÷ credits_spent) × 1000`. Credits spent includes weapons, shields, and abilities purchased. Buying weapons for teammates does not affect your econ rating. Average is **55–75**; above 100 is excellent.

**Entry Success Rate** = rounds with first kill ÷ rounds with (first kill OR first death). Only counts rounds where the player was involved in the round's opening engagement. If they weren't part of the first duel, that round is excluded from the denominator.

### Critical computation pitfalls

Watch for **AFK rounds** — the API provides `afkRounds` and `stayedInSpawnRounds` fields. These inflate round counts and deflate per-round averages. Consider excluding them from denominators or flagging matches with high AFK counts. Always check `assists` — non-damaging assists (flash, recon, heal) produce 0 damage but 25 combat score and count for KAST's "A" component.

### Side splits

Standard Valorant assigns sides by round number: **rounds 0–11 are the first half** (Red team attacks, Blue defends); **rounds 12+ swap sides**. Overtime alternates. Determine a player's side per round by cross-referencing `match_players.team` with `rounds.round_number`:

```sql
CASE
  WHEN (round_number < 12 AND team = 'Red') OR (round_number >= 12 AND team = 'Blue')
  THEN 'attack' ELSE 'defense'
END
```

### Rolling baselines and anomaly detection

Compute 7-day and 30-day moving averages per stat. The z-score formula `z = (today_value − rolling_mean) / rolling_stddev` flags sessions where performance deviates by **≥2 standard deviations**. For per-match metrics (ACS, ADR, K/D), compute z-scores per match against the 30-day window. For binary per-round metrics (KAST%, HS%), aggregate to match level first.

### Sample-size thresholds for statistical reliability

| Metric | Minimum matches | Minimum rounds | Why |
|--------|----------------|----------------|-----|
| ACS / ADR | 10–15 | ~200 | Per-round, stabilizes quickly |
| K/D ratio | 15–20 | ~300 | High variance by role |
| HS% | 20–30 | ~400 | Needs many bullet samples |
| KAST% | 10–15 | ~200 | Binary per round |
| First blood rate | 20–30 | ~400 | Rare per-round event |
| Clutch rate | **30–50** | ~600+ | Very rare; small samples are noise |
| Win rate per map/agent | **30+** per split | N/A | Need sufficient per-category games |

Display confidence warnings when data falls below these thresholds. Fall back to general rank-level benchmarks when personal data is insufficient.

---

## 3. SQLite schema for shared CLI and web tracker

The schema below stores raw counts and computes derived ratios at query time, with one exception: `daily_aggregates` pre-computes sums for fast dashboard reads. Key design decisions: TEXT for all UUIDs (Riot returns UUID strings); ISO8601 TEXT for timestamps (sortable as text, converts cleanly to PostgreSQL's TIMESTAMPTZ); `started_at` denormalized into `match_players` to avoid expensive JOINs on the most common query pattern.

### Core tables (15 total)

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE players (
    puuid           TEXT PRIMARY KEY,
    riot_name       TEXT NOT NULL,
    riot_tag        TEXT NOT NULL,
    region          TEXT NOT NULL,
    platform        TEXT DEFAULT 'pc',
    current_tier    INTEGER,
    current_tier_name TEXT,
    current_rr      INTEGER,
    last_match_at   TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE matches (
    match_id        TEXT PRIMARY KEY,
    map_id          TEXT,
    map_name        TEXT NOT NULL,
    queue_id        TEXT NOT NULL,
    is_ranked       INTEGER NOT NULL DEFAULT 0,
    game_version    TEXT,
    game_length_ms  INTEGER,
    season_short    TEXT,
    region          TEXT,
    rounds_played   INTEGER,
    red_score       INTEGER,
    blue_score      INTEGER,
    winning_team    TEXT,
    started_at      TEXT NOT NULL,
    synced_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE match_players (
    id              INTEGER PRIMARY KEY,
    match_id        TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    puuid           TEXT NOT NULL REFERENCES players(puuid) ON DELETE CASCADE,
    agent_id        TEXT,
    agent_name      TEXT NOT NULL,
    team            TEXT NOT NULL CHECK (team IN ('Red','Blue')),
    won             INTEGER NOT NULL DEFAULT 0,
    score           INTEGER NOT NULL DEFAULT 0,
    kills           INTEGER NOT NULL DEFAULT 0,
    deaths          INTEGER NOT NULL DEFAULT 0,
    assists         INTEGER NOT NULL DEFAULT 0,
    rounds_played   INTEGER NOT NULL DEFAULT 0,
    headshots       INTEGER NOT NULL DEFAULT 0,
    bodyshots       INTEGER NOT NULL DEFAULT 0,
    legshots        INTEGER NOT NULL DEFAULT 0,
    damage_dealt    INTEGER NOT NULL DEFAULT 0,
    damage_received INTEGER NOT NULL DEFAULT 0,
    first_bloods    INTEGER NOT NULL DEFAULT 0,
    first_deaths    INTEGER NOT NULL DEFAULT 0,
    plants          INTEGER NOT NULL DEFAULT 0,
    defuses         INTEGER NOT NULL DEFAULT 0,
    competitive_tier INTEGER,
    afk_rounds      INTEGER NOT NULL DEFAULT 0,
    started_at      TEXT NOT NULL,
    UNIQUE(match_id, puuid)
);

CREATE TABLE rounds (
    id              INTEGER PRIMARY KEY,
    match_id        TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    round_number    INTEGER NOT NULL,
    winning_team    TEXT NOT NULL,
    result_code     TEXT NOT NULL,
    bomb_planted    INTEGER NOT NULL DEFAULT 0,
    plant_site      TEXT,
    planter_puuid   TEXT,
    bomb_defused    INTEGER NOT NULL DEFAULT 0,
    defuser_puuid   TEXT,
    UNIQUE(match_id, round_number)
);

CREATE TABLE round_players (
    id              INTEGER PRIMARY KEY,
    round_id        INTEGER NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
    match_id        TEXT NOT NULL,
    puuid           TEXT NOT NULL,
    team            TEXT NOT NULL,
    loadout_value   INTEGER NOT NULL DEFAULT 0,
    spent           INTEGER NOT NULL DEFAULT 0,
    score           INTEGER NOT NULL DEFAULT 0,
    damage_dealt    INTEGER NOT NULL DEFAULT 0,
    kills_this_round INTEGER NOT NULL DEFAULT 0,
    headshots       INTEGER NOT NULL DEFAULT 0,
    bodyshots       INTEGER NOT NULL DEFAULT 0,
    legshots        INTEGER NOT NULL DEFAULT 0,
    survived        INTEGER NOT NULL DEFAULT 1,
    UNIQUE(round_id, puuid)
);

CREATE TABLE kills (
    id              INTEGER PRIMARY KEY,
    round_id        INTEGER NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
    match_id        TEXT NOT NULL,
    round_number    INTEGER NOT NULL,
    time_in_round_ms INTEGER,
    time_in_match_ms INTEGER,
    killer_puuid    TEXT NOT NULL,
    victim_puuid    TEXT NOT NULL,
    damage_weapon_name TEXT,
    damage_type     TEXT,
    is_headshot     INTEGER NOT NULL DEFAULT 0,
    killer_x        INTEGER,
    killer_y        INTEGER,
    victim_x        INTEGER,
    victim_y        INTEGER,
    assistants_json TEXT DEFAULT '[]'
);
```

Additional tables include `agents` (reference data from valorant-api.com), `maps` (callouts as JSON), `daily_aggregates` (pre-computed per player/day/agent/map), `coaching_sessions` (stored conversation history), `coaching_notes` (categorized strengths/weaknesses with confidence scores and expiry), `meta_cache` (TTL-based cache for scraped content), `patches` (game version tracking), `mmr_history`, and `sync_log`.

### Critical indexes

```sql
CREATE INDEX idx_mp_puuid_started ON match_players(puuid, started_at DESC);
CREATE INDEX idx_mp_puuid_agent ON match_players(puuid, agent_name, started_at DESC);
CREATE INDEX idx_kills_round ON kills(round_id);
CREATE INDEX idx_kills_killer ON kills(killer_puuid, match_id);
CREATE INDEX idx_da_puuid_date ON daily_aggregates(puuid, stat_date DESC);
```

The `idx_mp_puuid_started` index covers the three most common queries: "my stats today vs yesterday," "recent form (last 20 matches)," and "performance over last 30 days." The `idx_mp_puuid_agent` index handles per-agent filtering without a table scan.

### Player profile view for LLM prompt injection

```sql
CREATE VIEW v_player_profile AS
SELECT
    p.puuid,
    p.riot_name || '#' || p.riot_tag AS riot_id,
    p.current_tier_name AS current_rank,
    p.current_rr,
    COUNT(DISTINCT mp.match_id) AS matches_30d,
    ROUND(100.0 * SUM(mp.won) / MAX(COUNT(DISTINCT mp.match_id), 1), 1) AS win_rate_30d,
    ROUND(1.0 * SUM(mp.kills) / MAX(SUM(mp.deaths), 1), 2) AS kd_ratio,
    ROUND(1.0 * SUM(mp.score) / MAX(SUM(mp.rounds_played), 1), 1) AS avg_combat_score,
    ROUND(1.0 * SUM(mp.damage_dealt) / MAX(SUM(mp.rounds_played), 1), 1) AS avg_damage_round,
    ROUND(100.0 * SUM(mp.headshots) / MAX(SUM(mp.headshots)+SUM(mp.bodyshots)+SUM(mp.legshots), 1), 1) AS hs_pct,
    SUM(mp.first_bloods) AS total_first_bloods,
    SUM(mp.first_deaths) AS total_first_deaths
FROM players p
LEFT JOIN match_players mp ON p.puuid = mp.puuid
    AND mp.started_at >= strftime('%Y-%m-%dT%H:%M:%SZ', 'now', '-30 days')
GROUP BY p.puuid;
```

### PostgreSQL migration path

The schema uses only standard SQL types (TEXT, INTEGER, REAL), no SQLite-specific features, and proper foreign keys. `INTEGER PRIMARY KEY` auto-increments in SQLite and becomes `SERIAL` in PostgreSQL. The `strftime()` defaults in DEFAULT clauses need replacement with `NOW()` during migration — handle this through SQLAlchemy's dialect abstraction or a migration script. Use **pgloader** for the actual data migration; it handles type conversion automatically. Using **SQLAlchemy 2.0** with **Alembic** for all database access ensures dialect-agnostic application code from day one.

---

## 4. Live meta scraping and RAG retrieval pipeline

### Source priority and approach

Not all sources are equal. Here's the ranked list with concrete scraping strategies:

**Tier 1 — use these first:**
- **Official patch notes** (playvalorant.com/news): Server-rendered HTML, trivially scrapable with Trafilatura. Predictable URL pattern. The **Antosik/rito-news-feeds** GitHub project generates RSS/Atom feeds from Riot's news, polling every 10 minutes — subscribe to this for patch detection
- **VLR.gg** (pro meta): Server-side rendered, very scrapable with BeautifulSoup. Multiple community APIs exist: **axsddlr/vlrggapi** (REST, Vercel-hosted, 30s–1hr cache) provides endpoints for matches, rankings, player stats, and news
- **YouTube transcripts** (Woohoojin, Sliggy, Thinking Man's Valorant): The `youtube-transcript-api` Python package extracts auto-generated and manual captions without a headless browser. This is the richest source of tactical coaching knowledge that doesn't exist on stats sites

**Tier 2 — supplement with these:**
- **Blitz.gg**: Agent pick/win rates per map and rank. No public API; requires headless browser (Playwright) or intercepting internal API calls
- **Tracker.gg**: Internal API exists (`api.tracker.gg/api/v2/valorant/`) but is not officially available. Heavy Cloudflare protection
- **Reddit r/ValorantCompetitive**: Reddit API for patch discussion threads. Focus on post-patch megathreads

**Tier 3 — stable reference data:**
- **valorant-api.com**: Static asset API with agent abilities, maps, weapons. Cache for 30+ days

### The retrieval pipeline, step by step

```
User Query → [Query Mapper] → [Cache Check] → [Fetch if stale] → [Extract] → [Chunk + Embed] → [Store] → [Retrieve top-k] → [Inject into prompt]
```

**Query mapping**: Use the LLM to generate 2–3 search queries from the user's coaching question. "Losing on Haven attack" becomes `["Haven attack strategies current meta", "Haven A site execute setups", "Haven attacker side agent compositions"]`. Route to appropriate sources based on intent: stats question → Blitz/tracker, pro meta → VLR.gg, strategy → guides/YouTube, patch → official notes.

**Content extraction**: Use a **fallback chain** — Trafilatura first (0.945 F1 on extraction benchmarks, best among open-source), then readability-lxml (0.922 F1), then BeautifulSoup with custom selectors for structured data like tier-list tables. For JS-heavy sites (Blitz, Tracker.gg), use Playwright only when needed — it's 10x slower.

**Chunking**: Split extracted text into **256–512 token chunks with 50-token overlap**, preferring paragraph boundaries for semantic coherence.

### Caching with differentiated TTLs

| Tier (constant in `cache.TTL_HOURS`) | TTL | Examples | Invalidation |
|---|---|---|---|
| `stable` | 30 days (720 h) | Corpus markdown, map callouts, agent ability text | Major game update |
| `semi_stable` | 5 days (120 h) | Patch notes, meta articles | New patch — opportunistic via TTL |
| `volatile` | 12 hours | Live tier lists, pick/win rates | New patch — `invalidate_volatile()` immediately |

**Implemented as `valocoach.retrieval.cache`** (async, on top of the SQLite `meta_cache` table):

- `get_cached(url)` — returns content if `expires_at > now`, else evicts the row and returns `None` (eviction happens on read, not on a separate sweep).
- `store_cached(url, text, source, ttl_tier)` — upsert; refreshes `fetched_at`/`expires_at` on every write so live re-scrapes slide the TTL window forward.
- `invalidate_volatile()` — bulk-delete every `ttl_tier == "volatile"` row. Called by `patch_tracker.check_patch_update` after a new game version is detected.
- `purge_expired()` — full sweep of `expires_at < now` rows; optional housekeeping, not relied on by the read path.

`patch_tracker.check_patch_update` writes a new `patch_versions` row when HenrikDev's `/version` differs from the latest stored row, and only then triggers `invalidate_volatile()`. The HTTP call happens **outside** the DB session so we don't hold a transaction across network I/O.

### Patch detection triggers cache invalidation

Poll HenrikDev's `/valorant/v1/version/{region}` endpoint every 30 minutes. When `game_version` changes: delete all volatile cache, mark semi-stable cache for refresh, scrape the new patch notes page, embed and store the patch note content, and update the stored patch version. Patches typically arrive every two weeks on Tuesdays.

### Embedding model: nomic-embed-text v1.5

**Use nomic-embed-text v1.5**, not the commonly suggested all-MiniLM-L6-v2 (which is outdated as of 2025/2026 with lower quality and only 384 dimensions). Nomic-embed-text delivers **86.2% top-5 accuracy** on retrieval benchmarks, supports **8,192 token context** (critical for longer content chunks), runs efficiently on CPU at 137M parameters, and has **native Ollama support**: `ollama pull nomic-embed-text`. Its Matryoshka architecture lets you use 256–768 dimensions flexibly — use 768 for quality, drop to 256 for storage savings. Apache 2.0 licensed.

### Vector store: ChromaDB

**Use ChromaDB** over FAISS, LanceDB, or sqlite-vss. ChromaDB provides built-in persistence (DuckDB + Parquet backend), metadata filtering (critical for filtering by `content_type`, `patch_version`, `source`), collection-based organization, and a high-level Python API — all in an embedded, no-server-needed package. sqlite-vss is deprecated (replaced by sqlite-vec, which is still early-stage). FAISS is a raw indexing library with no persistence, metadata, or document storage — too much glue code needed.

Keep ChromaDB as a **separate vector store alongside the main SQLite DB**. SQLite handles structured data, cache metadata, and TTLs. ChromaDB handles vector similarity search. This separation of concerns is cleaner than forcing vectors into SQLite.

**Two collections, not one:** the implementation splits the vector store into `valocoach_static` and `valocoach_live` so corpus indexing and per-query live scraping don't trample each other. `valocoach ingest` and `valocoach index` only touch static; `_fetch_live_meta` only touches live; `retrieve_static` queries both per multi-query and dedupes hits across them.

```python
from valocoach.retrieval.vector_store import (
    LIVE_COLLECTION, STATIC_COLLECTION, get_collection
)

# Static corpus — agents, maps, concepts, patch notes
collection = get_collection(data_dir, STATIC_COLLECTION)
collection.upsert(
    ids=["concept:economy.md:0"],
    documents=["Full buy thresholds at 3900 cr…"],
    embeddings=[…],
    metadatas=[{"type": "concept", "name": "economy",
                "source": "corpus/concepts/economy.md",
                "content_type": "concepts", "ttl_tier": "stable"}],
)

# Live meta — per-query scraped content
live = get_collection(data_dir, LIVE_COLLECTION)
live.upsert(ids=…, documents=…, embeddings=…, metadatas=[{"type": "web", …}])
```

**Multi-query retrieval, not single-query.** `build_retrieval_queries(situation, map_name, agents, side)` produces 1–4+ focused queries — situation alone, plus `f"{map} callouts positions"`, `f"{map} {side} strategies"`, `f"{agent} abilities utility usage"` for up to 3 agents. Each query is embedded once, run against both collections at `n_results=3` / `max_distance=0.45`, and results are deduped via a `seen` set keyed on chunk text before being trimmed to the final `n_results` (default 5).

### Legal and ToS considerations

Scraping public, non-personal data for internal analysis (feeding into a chatbot's context, not republishing) sits in the **low-risk zone** post-hiQ v. LinkedIn. Best practices: respect `robots.txt`, use identifying User-Agent strings (`ValorantCoachBot/1.0`), rate-limit to 1 request per 2 seconds minimum, never bypass CAPTCHAs, and cache aggressively to minimize requests. Prefer APIs over scraping wherever possible — HenrikDev API, VLR.gg community APIs, and the rito-news-feeds RSS project together cover most data needs without direct scraping.

---

## 5. Ollama model selection by hardware tier

### Qwen3 dominates every VRAM tier for this use case

The Qwen3 family is the clear recommendation across all tiers because it supports **thinking mode** (chain-of-thought reasoning critical for tactical analysis), has excellent instruction following with complex system prompts, and provides 40K native context — far exceeding the ~10K tokens this application needs. DeepSeek-R1 models are strong reasoners but officially recommend *against* system prompts, which is a dealbreaker for a coaching chatbot relying on a persistent system prompt with grounding rules.

| VRAM | Model | Size (Q4_K_M) | Context | Speed | Pull command |
|------|-------|---------------|---------|-------|-------------|
| **8 GB** | **Qwen3 8B** | ~5.2 GB | 40K | ~40–55 tok/s | `ollama pull qwen3:8b` |
| **16 GB** | **Qwen3 14B** | ~9.3 GB | 40K | ~30–40 tok/s | `ollama pull qwen3:14b` |
| **24 GB** | **Qwen3 32B** | ~20 GB | 40K | ~15–25 tok/s | `ollama pull qwen3:32b` |

The quality jump from 8B to 14B is substantial (significantly better structured reasoning); the jump from 14B to 32B puts output quality competitive with GPT-4o. For the 8 GB tier, Phi-4 Mini (3.8B) was considered but its 16K context ceiling is too restrictive if RAG chunks or conversation history grow. Llama 3.1/3.3 8B is a solid fallback but weaker at structured reasoning than Qwen3.

### Context window budget

The total token budget for a coaching query:

| Component | Tokens | Priority |
|-----------|--------|----------|
| System prompt | 500–1,000 | Never cut |
| Player profile | 500–1,000 | Compress if needed |
| RAG chunks (top-5) | 2,000–4,000 | Reduce to top-3 if tight |
| User query + conversation history | 500–2,000 | Keep last 2–3 turns |
| Output reserve | 1,000–2,000 | Fixed |
| **Total** | **~5,000–10,000** | |

All three Qwen3 models have **40K native context**, providing 4x headroom. Implement adaptive context trimming: if total input approaches the budget, reduce RAG from 5 to 3 chunks, summarize player profile to key stats only, and keep only the last 2 conversation turns.

### LLM provider abstraction with LiteLLM

**Use LiteLLM** rather than building a custom abstraction layer. LiteLLM (~40K GitHub stars) supports Ollama, Anthropic, OpenAI, and 100+ providers through a single `completion()` call with identical streaming behavior. Swapping providers is a one-line config change.

```python
# config.py — change ACTIVE_CONFIG to swap providers
from dataclasses import dataclass

@dataclass
class LLMConfig:
    model: str = "ollama/qwen3:14b"
    api_base: str = "http://localhost:11434"
    api_key: str = None
    temperature: float = 0.6
    max_tokens: int = 2000
    stream: bool = True

OLLAMA_LOCAL = LLMConfig(model="ollama/qwen3:14b", api_base="http://localhost:11434")
CLAUDE_CLOUD = LLMConfig(model="anthropic/claude-sonnet-4-5-20250929", api_key="sk-...")

ACTIVE_CONFIG = OLLAMA_LOCAL  # ← change this one line to swap
```

```python
# llm_client.py — thin wrapper
import litellm
from config import ACTIVE_CONFIG

def coaching_stream(system_prompt: str, messages: list[dict]):
    response = litellm.completion(
        model=ACTIVE_CONFIG.model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=ACTIVE_CONFIG.temperature,
        max_tokens=ACTIVE_CONFIG.max_tokens,
        api_base=ACTIVE_CONFIG.api_base,
        stream=True,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content
```

Install with `pip install litellm ollama rich`.

---

## 6. Prompt engineering for situational coaching

### System prompt

```
You are **ValorantCoach**, an expert tactical coach for Valorant with deep knowledge
of competitive play, agent abilities, map layouts, economy management, and team coordination.

## YOUR ROLE
- Analyze game situations and provide specific, actionable tactical advice
- Adapt recommendations to the player's rank, agent pool, and playstyle
- Explain the "why" behind every recommendation, not just the "what"
- Be encouraging but honest — a good coach pushes improvement

## GROUNDING RULES
- **Only reference agent abilities, map callouts, and strategies that appear in the
  CONTEXT section below.** Do not invent callouts or meta strategies from memory.
- If the CONTEXT does not contain information about a specific topic, say:
  "I don't have current data on [topic] — let me focus on what I can help with."
- **Never fabricate patch-specific information** (weapon damage numbers, ability costs,
  cooldowns). If unsure, say "verify the current numbers in-game."

## PLAYER PROFILE
{player_profile}

## CONTEXT (Retrieved Knowledge)
{rag_context}

## OUTPUT FORMAT
🎯 **Assessment**: What's happening and why (1-2 sentences)
🛠️ **Setup**: Pre-round preparation (buys, positioning, utility plan)
⚔️ **Execute**: Step-by-step tactical plan
🔄 **Fallback**: What to do if the plan fails
📊 **Personalized Note**: Based on the player's stats, specific advice for them
💡 **Key Principle**: The underlying tactical concept to internalize

## TONE
- Direct and coach-like. Use second person ("You should", "Your team needs to")
- Be specific: name exact positions, callouts, and timing windows
- Keep total response under 400 words unless a complex breakdown is requested
```

The **grounding rules** are the most critical section. Without them, local models will hallucinate fake callouts and incorrect ability interactions. The explicit instruction to reference only CONTEXT-provided information, combined with the graceful degradation ("say you don't know"), significantly reduces hallucination in testing.

### Input parsing: hybrid regex-first, LLM-fallback

Pure LLM parsing adds 2–5 seconds of latency for a local model. Most Valorant inputs contain highly structured entities that regex handles in sub-millisecond time:

```python
AGENTS = ["Jett", "Raze", "Phoenix", "Reyna", "Yoru", "Neon", "Iso",
          "Sova", "Breach", "Skye", "KAY/O", "Fade", "Gekko",
          "Omen", "Brimstone", "Astra", "Harbor", "Clove",
          "Killjoy", "Cypher", "Sage", "Chamber", "Deadlock", "Vyse"]
MAPS = ["Bind", "Haven", "Split", "Ascent", "Icebox", "Breeze",
        "Fracture", "Pearl", "Lotus", "Sunset", "Abyss"]

def parse_situation(text: str):
    situation = {}
    # Map: exact match against known names
    for m in MAPS:
        if m.lower() in text.lower():
            situation["map"] = m
            break
    # Side: keyword match
    if re.search(r'\b(attack|attacking|t.side)\b', text, re.I):
        situation["side"] = "attack"
    elif re.search(r'\b(defend|defense|ct.side)\b', text, re.I):
        situation["side"] = "defense"
    # Score: digit-dash-digit pattern
    score = re.search(r'(\d{1,2})\s*[-–:]\s*(\d{1,2})', text)
    if score:
        situation["score"] = (int(score.group(1)), int(score.group(2)))
    # Agents: match against known list
    for a in AGENTS:
        if a.lower() in text.lower():
            situation.setdefault("agents", []).append(a)
    situation["raw"] = text  # Pass full text to LLM regardless
    return situation
```

The full text always passes to the LLM regardless — parsing just enriches the prompt with structured metadata. For completely freeform input ("I keep dying on A site"), the LLM handles it naturally through the system prompt.

### Handling subjective complaints

When a player says "we're losing every 1v1," the system should cross-reference their actual stats before responding. If their duel win rate is 52% (average), the coaching response should gently correct the perception: "Your duel win rate is actually 52%, which is average — the issue may be *which* duels you're taking rather than your mechanics." Add this instruction to the system prompt:

```
When a player describes mechanical struggles ("losing duels", "can't hit shots"):
1. Check their stats — if data contradicts perception, note this respectfully
2. Address TACTICAL solutions first (positioning, utility timing) — these are coachable
3. Acknowledge MECHANICAL factors (crosshair placement, peeking) with practice suggestions
4. Distinguish "fix this now" advice from "practice this over time" advice
```

### Output format recommendation

Use the **structured-headers-with-prose** format shown in the system prompt above. Emoji headers provide instant visual parsing. Prose within sections allows tactical nuance. The structured format (Setup → Execute → Fallback → Why-for-you) mirrors how professional coaches communicate — clear phases with actionable steps.

---

## 7. CLI architecture: Typer + Rich + prompt_toolkit

### Framework stack

**Typer** handles command routing and argument parsing (type-hint-based, auto-generates help docs, wraps Click). **Rich** handles all output formatting — markdown rendering, tables for stats, progress bars for sync, panels for coaching output, and critically, **streaming LLM responses via `Live` + `Markdown`**. **prompt_toolkit** provides the interactive REPL with autocomplete for agent names, map names, and slash commands, plus persistent command history.

### Command structure

```
valocoach coach "situation"              # One-shot coaching
  --agent AGENT --map MAP --side attack|defense
valocoach stats                          # Performance stats
  --agent AGENT --map MAP --period 7d|30d|90d
valocoach sync [--full] [--limit N]      # Sync matches from API
valocoach profile                        # Player profile summary
valocoach meta [--agent X] [--map Y]     # Current meta info
valocoach config init|set|show           # Configuration
valocoach interactive                    # Enter coaching REPL
valocoach patch                          # Current patch info
```

### Streaming LLM output with Rich

This is the core UX moment — coaching responses stream token-by-token with live markdown rendering:

```python
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console

console = Console()

def stream_coaching(system_prompt: str, user_message: str):
    full_text = ""
    with Live(console=console, refresh_per_second=8) as live:
        for token in coaching_stream(system_prompt, [{"role": "user", "content": user_message}]):
            full_text += token
            live.update(Panel(Markdown(full_text), title="🎯 Coach", border_style="green"))
    return full_text
```

Refresh rate of **8 FPS** provides smooth reading without excessive re-rendering. Show a spinner while waiting for the first token (TTFT can be 1–5 seconds for local models). For Qwen3's thinking mode, filter out `<think>...</think>` blocks from the displayed output or render them dimmed.

### Interactive mode with conversation memory

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

session = PromptSession(
    history=FileHistory('.valocoach_history'),
    auto_suggest=AutoSuggestFromHistory()
)

memory = ConversationMemory(max_turns=10, max_tokens=3000)

while True:
    user_input = session.prompt("valocoach> ")
    if user_input.startswith("/"):
        handle_slash_command(user_input)  # /stats, /sync, /quit
        continue
    memory.add_user(user_input)
    response = stream_coaching(build_prompt(memory), user_input)
    memory.add_assistant(response)
```

Keep **last 10 turns** in verbatim context. When history exceeds ~3,000 tokens, summarize older turns into a compressed context block using the LLM itself. Persist sessions to `~/.valocoach/sessions/` as JSON for resumability.

---

## 8. Six-week implementation plan


### Week 1 — Foundation and core UX validation ✅ done

Build the CLI skeleton, config system, and validate the hardest assumption: streaming Ollama output through Rich's Live markdown renderer.

- Initialize project: `pyproject.toml`, src layout, git repo
- Implement Typer CLI with all command stubs
- Set up pydantic-settings for configuration (`~/.valocoach/config.toml`)
- Create Rich display helpers (panels, tables, streaming)
- **Critical validation**: connect to Ollama, send a test prompt, stream response with `Rich.Live` + `Markdown` rendering. If this doesn't work smoothly, the entire UX falls apart
- Set up pytest infrastructure, ruff linting

**Deliverable**: `valocoach coach "test"` streams a formatted response from local Ollama. ✅

### Week 2 — Data layer (API + database) ✅ done

- Implement SQLAlchemy models matching the schema above
- Set up Alembic for migrations
- Build HenrikDev API client with httpx (async, rate-limited with exponential backoff via tenacity)
- Implement match sync pipeline: `valocoach sync` fetches recent matches, upserts into SQLite
- Implement `valocoach profile` directly from ORM queries (no SQL view)
- Create pytest fixtures with recorded API responses (pytest-httpx)

**Deliverable**: `valocoach sync` stores 20 matches. `valocoach profile` shows player summary. ✅

**Notes from implementation:**
- `ensure_db` was added to bootstrap the schema *and* stamp Alembic head atomically — without the stamp, the first `alembic upgrade head` against a CLI-created DB tries to recreate every table and fails. See `tests/test_database_ensure.py`.
- The whole DB stack is async (`AsyncEngine`, `AsyncSession`, `async with session_scope()`). Don't mix sync sessions on the same engine.

### Week 3 — Stats engine ✅ done

- Implement all stat computations (ACS, ADR, HS%, KAST%, FB/FD rates, clutch rate, trade efficiency, econ rating, multi-kill rates)
- Per-agent, per-map, per-side filtering
- Rolling baselines (7-day, 30-day) and z-score anomaly detection
- Sample-size threshold warnings (with `⚠` markers and `(thin sample)` / `(low sample)` annotations baked into `coach.context._format_context` so the LLM sees the reliability signal too)
- **Testing**: golden-dataset tests in `tests/test_golden_stats.py` lock computed values against recorded matches
- Implement `valocoach stats` with all filter options, Rich table output

**Deliverable**: `valocoach stats --agent Jett --period 30d` shows accurate, formatted stats. ✅

### Week 4 — RAG pipeline ✅ done (with deltas from the original plan)

- Define source list and scraping strategies per source
- Implement Trafilatura + BS4 extraction chain in `retrieval/scrapers/web.py`; `ScrapedContent` dataclass shared across scrapers
- Set up `youtube-transcript-api` in `retrieval/scrapers/youtube.py`
- Implement chunking — `chunker.chunk_markdown` uses tiktoken (`cl100k_base`) for exact token counts and splits on `## / ###` headings + blank lines, with token-level overlap on chunk boundaries. (Default 400 tokens / 40 overlap, not 512/50 from the original plan — tighter chunks gave better recall on the corpus.)
- Set up nomic-embed-text via Ollama (`ollama pull nomic-embed-text`)
- Initialize ChromaDB with **two collections**: `valocoach_static` (corpus, indexed once) and `valocoach_live` (per-query scraped meta, TTL-managed). The four-collection split from the original plan (`patch_notes`, `strategies`, `pro_meta`, `coaching_content`) was rejected — `doc_type` metadata + the static/live boundary cover the same use cases without the routing complexity.
- Implement retrieval pipeline as `retriever.retrieve_static`: structured JSON facts first, then multi-query vector search across both collections, deduped via `seen` set
- Implement `RetrievalResult` (static_chunks + meta_chunks + sources + patch_version) with `to_context_string()` for direct LLM injection
- Implement patch detection in `patch_tracker.check_patch_update` (HenrikDev `/version` → `patch_versions` table → `invalidate_volatile()` on change)
- Build TTL-based cache invalidation in `retrieval/cache.py` (async, eviction-on-read)
- Implement `valocoach meta` (with live-patch alert when DB patch ≠ JSON patch) and `valocoach index` / `valocoach ingest` commands
- Hand-write a `corpus/concepts/` reference set (economy, fundamentals, retakes, roles, executes) so the LLM has tactical concepts even when no scrape has run

**Deliverable**: RAG retrieves relevant content for any coaching query. `valocoach meta --map Haven` returns current meta info with a live-patch warning if the meta JSON is stale. ✅

### Week 5 — LLM coaching pipeline ✅ done

- ✅ System prompt (`SYSTEM_PROMPT_STUB` in `coach.py`) with grounding rules, emoji-headed output sections, 350-word cap, and a multi-turn note so the LLM builds on earlier advice instead of repeating itself.
- ✅ Coaching orchestrator (`run_coach`, `_build_grounded_context`, `_build_system_prompt`) chains: parse → retrieve → stats → trim → compose → stream.
- ✅ Input parsing — `src/valocoach/core/parser.py`: regex-first against canonical agent/map JSON. `Situation` Pydantic model extracts `map`, `agents`, `side`, `site`, `score`, `clutch`, `econ`, `phase`. Sub-millisecond, word-boundary-safe (handles KAY/O slash, "isolated" ≠ Iso, "passage" ≠ Sage). 58 tests.
- ✅ CLI-flag merging — `_resolve_fields()` lets CLI flags win over parser output; `run_coach` builds the metadata block from resolved fields so `--agent Jett` appears even when "Jett" isn't in the text.
- ✅ Adaptive context trimming — `src/valocoach/core/context_budget.py`:
  - `count_tokens(text)` — tiktoken `cl100k_base`, `@lru_cache`-backed encoder, consistent with the chunker.
  - `trim_text_to_tokens(text, max_tokens)` — exact token-boundary truncation from the end (preserves the beginning, where highest-priority content lives).
  - `fit_prompt(system_base, grounded_context, stats_context, user_msg, hard_limit=24_000)` — three-stage priority trimmer: (1) trim grounded to `GROUNDED_REDUCED_LIMIT=2_000` tokens, (2) drop stats entirely, (3) trim grounded further to whatever remains. system_base and user_msg are never touched. 21 tests.
  - Called in `run_coach` after retrieval and before `_build_system_prompt`.
- ✅ `ConversationMemory` — `src/valocoach/core/memory.py`: sliding-window memory with dual eviction (max_turns + max_tokens). Drops oldest complete user+assistant exchange first; handles orphaned assistant turns gracefully. `messages` property returns a copy safe for direct injection into LiteLLM's `messages` list. 19 tests.
- ✅ `stream_completion` extended with `conversation_history: list[dict] | None` — prior turns inserted between the system message and the current user message. Backward-compatible (defaults to None).
- ✅ `run_coach` returns `str | None` — the assistant response text. The interactive REPL uses this to store the assistant turn in `ConversationMemory`. One-shot CLI calls ignore the return value, so the change is fully backward-compatible.
- ✅ Interactive REPL — `src/valocoach/cli/commands/interactive.py`:
  - `prompt_toolkit` session with `FileHistory('~/.valocoach/history')` for ↑↓ recall.
  - Tab autocomplete (`WordCompleter`) built from `list_agent_names()` + `list_map_names()` + slash commands.
  - `complete_while_typing=False` so autocomplete doesn't interrupt mid-sentence.
  - Slash commands: `/help`, `/clear`, `/memory`, `/stats`, `/quit`.
  - `/memory` shows current turn count and token usage — helps the player understand what context the LLM sees.
  - Calls `run_coach` directly (no duplicated pipeline) and stores both turns in memory only when a non-empty response is returned, so partial / failed streams don't leave one-sided turns.
  - Graceful exit on Ctrl-C (re-prompts) and Ctrl-D / EOFError (exits cleanly).
- ✅ `valocoach interactive` stub in `app.py` replaced with real call to `run_interactive()`.

**Deliverable**: Full end-to-end coaching with multi-turn REPL. `valocoach interactive` supports persistent conversation memory. ✅

**Notes from implementation:**

- The `_SIDE_DEFENSE` regex was `defen[cs](?:e|ing|ders?)?` — doesn't match "defending" (stem ends in `d`). Fixed to `defen(?:d(?:ing|ers?)?|[cs](?:e|ing|ders?)?)`.
- `to_metadata_block()` renders the agents key as `Agent(s): Jett` (with the `(s)` suffix). Assertions must check for `"Agent(s): Jett"` not `"Agent: Jett"`.
- `fit_prompt` hard limit is 24 K tokens (40 K window − 13 K safety − 3 K output reserve). With Qwen3 8B at normal prompt sizes (~3–5 K tokens), trimming never fires in practice — it's a safety net for extreme inputs.
- Conversation history is stored as the raw situation text (not the formatted metadata-block+situation message) to keep memory compact and natural for the LLM.

### Week 6 — Polish, testing, documentation

- Comprehensive test suite: unit tests for stats (golden dataset validation), integration tests for coaching pipeline (mocked LLM), CLI tests via Typer's CliRunner
- LLM output quality evaluation: create 20 coaching scenarios with expected key points and forbidden hallucinations, score with LLM-as-judge using DeepEval
- RAG evaluation with RAGAS metrics (faithfulness, contextual relevancy)
- Error handling hardening (API failures, rate limits, missing data)
- Shell completion installation (`--install-completion`)
- README with installation guide, usage examples
- Performance optimization: lazy loading, connection pooling, request caching

**Deliverable**: Fully tested, documented, installable CLI.

### Project directory structure (as built)

```
valorant-coach/
├── src/valocoach/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli/
│   │   ├── app.py            # Typer app, command wiring
│   │   ├── display.py        # Rich console + streaming panel
│   │   └── commands/         # one file per command
│   │       ├── coach.py      # coaching orchestrator (uses retrieve_static)
│   │       ├── stats.py
│   │       ├── sync.py
│   │       ├── profile.py
│   │       ├── meta.py       # tier list + agent/map view + live-patch alert
│   │       ├── ingest.py     # corpus / URL / YouTube ingestion + --clear/--stats
│   │       ├── interactive.py # prompt_toolkit REPL + ConversationMemory wiring
│   │       └── config.py
│   ├── core/
│   │   ├── config.py         # pydantic-settings + TomlConfigSettingsSource
│   │   ├── exceptions.py
│   │   ├── parser.py         # Situation parser: regex-first against canonical
│   │   │                     #   agent/map JSON; Situation Pydantic model with
│   │   │                     #   map/agents/side/site/score/clutch/econ/phase
│   │   ├── context_budget.py # count_tokens + trim_text_to_tokens + fit_prompt
│   │   │                     #   (tiktoken cl100k_base; 3-stage priority trimmer)
│   │   └── memory.py         # ConversationMemory: sliding-window turn store
│   │                         #   with dual eviction (max_turns + max_tokens)
│   ├── data/
│   │   ├── api_client.py     # HenrikDev async client (httpx + tenacity)
│   │   ├── api_models.py     # Pydantic models for raw API shapes
│   │   ├── models.py         # Pydantic models used cross-module
│   │   ├── orm_models.py     # SQLAlchemy 2.0 (matches, players,
│   │   │                     #   match_players, meta_cache, patch_versions)
│   │   ├── database.py       # async engine + session_scope + ensure_db
│   │   ├── repository.py     # query helpers
│   │   ├── mapper.py         # API → ORM mapping
│   │   └── sync.py           # match sync orchestration
│   ├── stats/                # calculator, round_analyzer, formatter, filters
│   ├── coach/                # stats-context builder for the coach prompt
│   ├── retrieval/
│   │   ├── data/             # bundled JSON: agents, maps, meta
│   │   ├── agents.py / maps.py / meta.py   # JSON loaders + format_*_context
│   │   ├── chunker.py        # tiktoken chunk_markdown + Chunk dataclass
│   │   ├── embedder.py       # nomic-embed-text via Ollama (embed/embed_one)
│   │   ├── vector_store.py   # ChromaDB client + STATIC/LIVE collections
│   │   ├── ingester.py       # ingest_text + ingest_knowledge_base
│   │   ├── searcher.py       # search() + collection_stats() (both buckets)
│   │   ├── retriever.py      # RetrievalResult + retrieve_static
│   │   │                     #   (multi-query, both collections)
│   │   ├── cache.py          # async TTL cache over meta_cache
│   │   ├── patch_tracker.py  # PatchVersion writes + get_current_patch
│   │   └── scrapers/         # ScrapedContent + web (trafilatura) + youtube
│   └── llm/
│       └── provider.py       # LiteLLM streaming wrapper
├── alembic/
│   ├── env.py
│   └── versions/             # auto-stamped by ensure_db on first run
├── corpus/
│   ├── agents/ maps/ meta/   # generated via scripts/build_corpus.py
│   └── concepts/             # hand-written (economy, fundamentals, executes,
│                             #   retakes, roles)
├── tests/                    # 678 tests, asyncio_mode=auto
├── data/                     # runtime (gitignored) — valocoach.db + chroma/
├── pyproject.toml
└── README.md
```

### Key dependencies (as installed)

```toml
dependencies = [
    "typer>=0.12.0",
    "rich>=13.7.0",
    "prompt-toolkit>=3.0.50",          # interactive REPL (week 5)
    "pydantic>=2.7.0",
    "pydantic-settings>=2.4.0",
    "tomli-w>=1.0.0",                  # writing default config.toml
    "sqlalchemy[asyncio]==2.0.49",     # async ORM, dialect-agnostic
    "aiosqlite==0.22.1",               # async SQLite driver
    "alembic==1.18.4",                 # schema migrations
    "httpx==0.27.2",                   # HenrikDev async HTTP
    "tenacity==9.1.4",                 # retry/backoff for HenrikDev
    "litellm>=1.50.0",                 # provider abstraction (Ollama → Claude)
    "chromadb>=0.5.0",                 # vector store (two collections)
    "trafilatura>=1.12.0",             # primary HTML → text extraction
    "beautifulsoup4>=4.12",            # fallback extraction
    "tiktoken>=0.7",                   # exact-token chunk boundaries
    "youtube-transcript-api",          # YouTube captions
    "python-dotenv>=1.0.0",
]
```

---

## 9. Risks that will actually bite you, and how to survive them

**HenrikDev goes down or rate-limits aggressively.** Cache all fetched match data locally — match data is immutable, so once synced it never needs re-fetching. Implement exponential backoff with jitter. Store raw API responses for replay/debugging. Long-term, apply for a Riot production key as backup. The 30 req/min basic limit means syncing 20 matches (each needing a detail fetch) takes ~40 seconds — tolerable with a progress bar, but build sync to be incremental (only fetch new matches by checking `match_id` against the database).

**Scraped sites change layouts or block you.** Use Trafilatura as primary extractor (layout-agnostic content extraction, not CSS-selector-dependent). Maintain multiple sources for the same information — if Blitz.gg breaks, fall back to tracker.gg or VLR.gg. Content-hash stored documents; if a re-scrape returns a wildly different hash, log a warning rather than silently ingesting garbage. Degrade gracefully: if all scrapers fail, the coaching system still works using the player's stats and the LLM's parametric knowledge — RAG just enriches, it doesn't gate.

**LLM hallucinating fake callouts or wrong agent abilities.** The system prompt's grounding rules ("only reference what appears in CONTEXT") are the first defense. Second: maintain a whitelist of valid agent abilities and map callouts in the `agents` and `maps` reference tables. Post-process LLM output to flag any agent/ability names not in the whitelist — add a disclaimer or strip them. Third: prefer lower temperatures (0.4–0.6) for coaching responses to reduce creative hallucination. Test with a **hallucination eval set**: 20 prompts where the RAG context deliberately omits certain information, verifying the model says "I don't have data on that" rather than fabricating.

**Patch changes invalidate advice.** The patch detection system (polling HenrikDev `/version` every 30 minutes) triggers automatic cache invalidation when a new version appears. Add the current patch version to every coaching response: "Based on patch 12.05 meta." If the system detects it's been more than 3 weeks since the last patch note scrape, append a caveat: "Meta information may be outdated — consider running `valocoach patch` to refresh."

**Not enough matches for meaningful stats.** Display explicit confidence warnings: "⚠️ Based on only 4 matches on Jett — stats may not be representative." When data is below the threshold for a given metric, don't display that metric at all, or fall back to rank-level population benchmarks: "Average Jett HS% at Gold is 22% — you'll need 20+ matches for a reliable personal number." The coaching pipeline should detect low-data situations and shift advice from stats-driven to fundamentals-driven: "Since I don't have enough data to assess your Haven performance specifically, here are the core attack principles that apply at your rank."

---

## Conclusion

The riskiest technical assumption is not the LLM, the RAG pipeline, or the database — it's whether the HenrikDev API data is granular enough to compute all desired stats accurately. **Validate this in week 2** by syncing 10 matches and checking that computed ACS matches the API-reported score within rounding error. If kill timestamps, round-level damage, and economy data are all present and correct, every other stat derivation follows mechanically.

The highest-leverage architectural choice is the **LiteLLM abstraction layer**. It costs zero extra effort upfront (it's just a pip install and a different function call than raw Ollama) but makes the Claude API swap genuinely a one-line change later. The second highest-leverage choice is the **schema design** — getting the shared tables right now prevents a painful migration when the web tracker frontend arrives.

Build the streaming CLI UX in week 1. If it feels good to type a coaching question and watch Rich render a markdown response in a green panel while tokens stream in, the project has a soul. Everything else is plumbing.