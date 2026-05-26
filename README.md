# ValoCoach

A local-first Valorant tactical coaching CLI. Streams Immortal-level advice for any match situation using a local Ollama LLM grounded in a RAG knowledge base (agent abilities, map callouts, game economy) and your own synced match history. No cloud required beyond the optional HenrikDev API for match data.

```bash
valocoach coach "we keep losing 8-12 on Ascent attack as Jett, they stack A"
```

```
🎯 Read — The stacked-A read is late because the default A-push burns 25 seconds
  of spike time before you confirm the stack...

🛠️ Pre-round — Jett's Tailwind (200cr) held for escape, not entry...
  ↳ Omen smokes: A Long at 1:45, A Short at 1:42 — not simultaneous.

⚔️ Execute — (1) Sage wall A Long at 1:50...
```

---

## Contents

- [Prerequisites](#prerequisites)
- [Install](#install)
- [First-run setup](#first-run-setup)
- [How it works](#how-it-works)
- [Command reference](#command-reference)
  - [valocoach (hub)](#valocoach-hub)
  - [coach](#coach)
  - [post-game](#post-game)
  - [lineup](#lineup)
  - [stats](#stats)
  - [sync](#sync)
  - [profile](#profile)
  - [meta](#meta)
  - [meta-refresh](#meta-refresh)
  - [agents-refresh](#agents-refresh)
  - [ingest](#ingest)
  - [patch](#patch)
  - [notes](#notes)
  - [sessions](#sessions)
  - [config](#config)
- [REPL reference](#repl-reference)
- [Shell completion](#shell-completion)
- [Configuration file](#configuration-file)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.11 | 3.13 recommended |
| [uv](https://docs.astral.sh/uv/) | any recent | package manager |
| [Ollama](https://ollama.ai) | any recent | local LLM runtime |
| HenrikDev API key | — | free tier, needed for `sync` |

**Ollama models required:**

```bash
ollama pull qwen3:8b          # coaching LLM (default)
ollama pull nomic-embed-text  # embeddings for the vector store
```

The 8B model runs well on 16 GB RAM. Swap to `qwen3:14b` for better reasoning on 32 GB+.

---

## Install

```bash
git clone https://github.com/Susilkessav/Valorant_CLI
cd Valorant_CLI
uv sync                       # install all dependencies
uv sync --extra dev           # + dev tools (pytest, ruff)
```

Verify the CLI is wired up:

```bash
uv run valocoach --version
```

---

## First-run setup

Complete these steps once before using any data-backed commands (`coach`, `stats`, `sync`).

### 1 — Create config

```bash
uv run valocoach config init
```

This writes `~/.valocoach/config.toml`. Open it and fill in your Riot identity and HenrikDev key:

```toml
riot_name  = "YourName"     # Riot username, no # prefix
riot_tag   = "NA1"          # Tag after the #
riot_region = "na"          # na | eu | ap | kr | latam | br

henrikdev_api_key = "hdev-..."  # https://docs.henrikdev.xyz/authentication
```

### 2 — Start Ollama

```bash
ollama serve                  # keep this running in a separate terminal
```

### 3 — Seed the knowledge base

Embeds the built-in JSON knowledge (agent abilities, map callouts, economy thresholds) into ChromaDB. Run once; re-run after updating the knowledge JSONs.

```bash
uv run valocoach ingest --seed
```

Output:
```
Seeded 342 docs — 22 agents · 14 maps · 8 meta
```

### 4 — Sync your match history

```bash
uv run valocoach sync
```

Calls the HenrikDev API and stores your recent competitive matches in a local SQLite database (`~/.valocoach/data/valocoach.db`). Subsequent runs are incremental — only new matches are fetched.

### 5 — Start coaching

```bash
# Open the interactive REPL (recommended for a session)
uv run valocoach coach

# Or get one-shot advice immediately
uv run valocoach coach "we lose pistol round on Bind defense every time"
```

---

## How it works

Every `coach` call goes through four stages before the LLM sees anything:

1. **Parse** — `parser.py` extracts agent, map, side, score, clutch, econ, and phase from the situation string in < 1 ms using a regex cascade. No LLM needed.

2. **Classify intent** — nine intent types: `tactical`, `clutch`, `post_plant`, `retake`, `economy`, `agent_info`, `meta`, `stat_analysis`, `general`. The intent drives which template the LLM gets and which context fields are worth asking about.

3. **Intent-aware elicitation** — if the situation is underspecified for its intent, ValoCoach asks a targeted follow-up (at most 3 questions, only about fields that matter for that intent). A meta question never triggers any prompts. An economy question only asks for side (and optionally score). A tactical question asks for map, side, and agent. Answers are remembered for the rest of the REPL session.

4. **Grounded RAG** — ChromaDB is queried for the top-k chunks relevant to the situation. Agent ability blocks, map callout blocks, and economy thresholds are prepended to the system prompt. Your recent match stats are appended. A token-budget enforcer trims low-priority chunks to stay under the model's context window.

For **meta intent**, the LLM is bypassed entirely — a deterministic tier-list panel is built from `agents.json` + `meta.json` followed by a personalised takeaway from your stats DB. No hallucinated ability names, no temperature variance.

After every non-meta response, a deterministic **fact-check** scans the output for fabricated abilities, wrong-agent attributions, weapons mis-cast as abilities, and generic descriptors. Warnings are printed below the response.

---

## Command reference

### valocoach (hub)

Running `valocoach` with no arguments on a terminal shows the **hub dashboard** — a live at-a-glance card with the current patch, your player status, a stale-meta warning (if applicable), and quick-navigation hints:

```
╭─────────────────────────────────────────╮
│  ValoCoach  v0.x.x                      │
╰─────────────────────────────────────────╯

  Patch 10.09  ·  meta updated 2025-05
  Player YourName#NA1  ·  match history synced — run valocoach stats to view

  Quick navigation
    valocoach coach            →  Interactive coaching session
    valocoach coach "..."      →  One-shot situational advice
    valocoach post-game        →  Debrief your last match
    valocoach stats            →  Performance dashboard
    valocoach meta             →  Current tier list
    valocoach sync             →  Pull latest match history

  valocoach --help  for all commands
```

On a non-interactive context (pipe, CI, script) only the banner and a one-liner usage hint are printed.

---

### coach

Get tactical coaching for a match situation. The LLM is grounded with relevant abilities, callouts, and your recent performance stats.

**Without arguments — opens the interactive REPL:**

```bash
valocoach coach
```

The REPL maintains sliding-window conversation memory (20 turns / 3 000 tokens) so you can have a full multi-turn coaching session. See the [REPL reference](#repl-reference) for slash commands.

**With a situation argument — one-shot advice:**

```bash
valocoach coach "<situation>"
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--agent`, `-a` | auto-detected | Override the agent (e.g. `Jett`) |
| `--map`, `-m` | auto-detected | Override the map (e.g. `Ascent`) |
| `--side`, `-s` | auto-detected | `attack` or `defense` |
| `--with-stats` / `--no-stats` | on | Include your recent performance stats in the prompt |
| `--no-elicit` | off | Skip context questions and go straight to coaching |

**Agent, map, and side are extracted automatically from the situation text** — you rarely need the flags:

```bash
# All three detected automatically:
valocoach coach "losing 8-12 on Ascent attack as Jett, they always stack A"

# Flags when text is ambiguous:
valocoach coach "keep dying in mid" --agent Omen --map Haven --side defense
```

**Intent-aware context questions**

If you give a short or underspecified query, ValoCoach asks targeted follow-up questions — but only about what's relevant for the *type* of question you asked:

```
# "how do I eco" → economy intent, only asks for side (and optionally score)
valocoach coach "how do I eco"

  » Side? [attack/defense]: attack
  » Score? e.g. 8-12  (Enter to skip): 9-11

[streams advice for an eco round on attack at 9-11...]
```

```
# "what's meta" → meta intent, no questions at all
valocoach coach "what's the current meta"

[shows deterministic tier-list panel immediately]
```

At most **3 questions** are asked per turn. Questions already answered earlier in a REPL session are never re-asked.

**Coaching intent types and what they affect:**

| Intent | Triggered by | Questions asked | Template |
|--------|-------------|-----------------|----------|
| `tactical` | map + side both present | map, side, agent | Execute / strategy |
| `clutch` | "1v3", "clutch" keywords | side, agent | Clutch decision tree |
| `post_plant` | "post-plant" / spike down | map, agent, side | Spike timer + position |
| `retake` | "retake" keyword / phase | map, agent, side | Retake route advice |
| `economy` | "eco", "save", "buy" | side, score | Buy-round decision |
| `agent_info` | "how does", "abilities", "kit" | agent | Ability breakdown |
| `meta` | "meta", "tier list", "best agent" | *(none)* | Deterministic panel |
| `stat_analysis` | "my stats", "my KD" | *(none)* | Stats-first template |
| `general` | everything else | agent, map | General coaching |

**Examples:**

```bash
# Post-plant 1v3
valocoach coach "1v3 post plant B site on Bind, I'm on Viper with the spike on B Long"

# Economy decision
valocoach coach "lost pistol and bonus, eco round next — should I buy on 3k or save?"

# Retake scenario
valocoach coach "I'm playing Sage on Split defense, A site gets hit every round — retake tips"

# Agent info
valocoach coach "how does Cypher's trapwire work, when should I use it?"

# Meta check (deterministic, no LLM)
valocoach coach "what agents are strong this patch?"

# Skip personalised stats for a quick generic answer
valocoach coach "best Jett dash angles on Ascent A site" --no-stats
```

**Meta questions** produce a deterministic **Meta — Current Tier List** panel (from `agents.json` + `meta.json`) plus a **Personalised Takeaway** from your stats DB. No LLM call, no hallucinated ability names.

**Ability fact-check** — for non-meta answers, a post-pass scans the output against `agents.json` and prints a warning for any fabricated ability names, wrong-agent attributions, weapons mis-cast as abilities, or generic descriptors. The answer itself is not modified.

---

### post-game

Run a deterministic finding-based analysis of your most recent stored match and stream a focused debrief: critical pattern, round-cost translation, priority drill, next-match focus.

```bash
valocoach post-game
```

**Options:**

| Flag | Description |
|------|-------------|
| `--match-id` | Analyse a specific match ID instead of the most recent |
| `--no-notes` | Skip the coaching-notes prompt at the end |
| `--no-repl` | Skip the offer to continue in the REPL after the debrief |

**What runs under the hood:**

Ten deterministic analyzers fire against the stored match before any LLM is called:

| Analyzer | What it measures |
|----------|-----------------|
| First-contact | Where and when first kills happen each round |
| Economy decisions | Buy/eco adherence, force-buy timing |
| Utility efficiency | Ability usage rate relative to site takes |
| Round timing | Spike-plant timing, remaining time on takes |
| Traded deaths | How often your deaths are answered by teammates |
| ATK/DEF split | Win rate asymmetry between sides |
| Clutch outcomes | 1vN conversion rate |
| Death location clusters | Where you die repeatedly (potential wallbang spots) |
| Engagement distance | Preference for close/medium/long-range fights |
| Plant/defuse distribution | Which sites you plant on and how often |

The top three findings (collapsed by root cause) are injected as ground truth into the prompt — every claim in the debrief is tied to a real number, not hallucinated.

When a `low_utility` finding fires on a kit-heavy agent (Sova, Viper, KAY/O, Brimstone, Omen, Fade, Cypher, Killjoy), the debrief also pulls a relevant lineup from the local library and the LLM is told to cite it.

After the debrief, you are offered to continue in the interactive REPL with your match context pre-loaded (map, side, agent, score automatically set from the analysed match).

---

### lineup

Search the local lineup library — built-in seed entries plus any YouTube or URL content the ingest pipeline classified as lineup-type.

```bash
valocoach lineup Sova --map Ascent --site A
valocoach lineup Viper --map Bind --query "post-plant B"
valocoach lineup --map Haven --site C
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `<agent>` | — | Positional. Filter to one agent (e.g. `Sova`, `KAY/O`). |
| `--map`, `-m` | — | Filter to one map. |
| `--site`, `-s` | — | Site letter: `A`, `B`, `C`, or `Mid`. |
| `--query`, `-q` | — | Free-text similarity search. |
| `--top`, `-n` | `5` | Maximum results to return. |

Filter values are canonical-case normalised at both write and read time — `sova`, `Sova`, and `SOVA` all match the same entries.

**Grow the library with `ingest`** — ingesting a YouTube lineup guide or a written article automatically extracts lineup-classified chunks and stores them with agent, map, site, ability, and purpose metadata so they appear in `lineup` searches.

---

### stats

Show your performance dashboard built from locally synced matches.

```bash
valocoach stats
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--period`, `-p` | `90d` | Time window: `7d`, `30d`, `90d`, or `all` |
| `--agent`, `-a` | all | Filter to one agent (e.g. `Jett`) |
| `--map`, `-m` | all | Filter to one map (e.g. `Ascent`) |
| `--result`, `-r` | both | `win` or `loss` only |
| `--json` | off | Emit raw JSON. Useful for scripting. |

**Examples:**

```bash
valocoach stats                              # last 90 days (default)
valocoach stats --period 7d --agent Jett     # last 7 days on Jett
valocoach stats --map Bind --result loss     # losses on Bind only
valocoach stats --period all                 # all-time
```

Requires `valocoach sync` to have been run at least once.

---

### sync

Fetch your competitive match history from the HenrikDev API and store it locally.

```bash
valocoach sync
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--limit` | `20` | Maximum matches to inspect per run |
| `--full` | off | Inspect all `--limit` matches even if already stored |
| `--mode` | `competitive` | Game mode: `competitive`, `unrated`, etc. |

**Examples:**

```bash
valocoach sync                        # incremental (stops at already-stored matches)
valocoach sync --full --limit 50      # pull up to 50 regardless of stored state
valocoach sync --mode unrated         # sync unrated games
```

Each sync run also sweeps expired retrieval cache rows (SQLite `meta_cache` + live ChromaDB docs) before fetching, keeping the cache bounded.

Requires `henrikdev_api_key` in config.

---

### profile

Show a compact player identity card plus a recent performance summary.

```bash
valocoach profile
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--name`, `-n` | configured | Riot username (defaults to config value) |
| `--tag`, `-t` | configured | Riot tag (must be paired with `--name`) |
| `--limit`, `-l` | `20` | Number of recent matches to summarise |
| `--json` | off | Emit raw profile JSON. |

```bash
valocoach profile                             # your own profile
valocoach profile --name SomePlayer --tag NA1 # look up another player
valocoach profile --limit 50                  # summarise last 50 matches
```

---

### meta

Show current Valorant meta information from the knowledge base: tier list, agent abilities, map callouts, economy thresholds.

```bash
valocoach meta                    # full tier list overview
valocoach meta --agent Jett       # Jett's abilities + meta standing
valocoach meta --map Ascent       # Ascent callouts + top agents
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--agent`, `-a` | — | Show ability and meta info for one agent |
| `--map`, `-m` | — | Show callouts and meta info for one map |
| `--json` | off | Emit the raw `meta.json` contents (or relevant slice) as JSON |

The tier list is deterministic — it's built directly from `meta.json` and shown instantly without calling the LLM.

---

### meta-refresh

Automated meta-sync pipeline. Detects the current patch via HenrikDev, scrapes patch notes + Diamond+/pro stats, optionally ingests YouTube content, regenerates `meta.json`'s tier list via the local LLM, and re-embeds everything into ChromaDB.

```bash
valocoach meta-refresh
```

**Options:**

| Flag | Description |
|------|-------------|
| `--force`, `-f` | Run the full sync even when no new patch is detected |
| `--dry-run` | Simulate all steps but skip writing `meta.json` and re-ingesting |
| `--watch` | Keep running: check daily, sync on each new patch detected |
| `--install-cron` | Add a daily crontab entry that runs `meta-refresh` automatically |
| `--youtube`, `-y URL` | YouTube video to ingest as supplemental context (repeatable) |

**Pipeline steps (in order):**

1. Patch detection — HenrikDev API → compare against stored version
2. Patch notes scrape — `playvalorant.com` (auto-constructed URL)
3. Patch diff extraction — changes written to `patch_changes/` for coaching context
4. Stats scrape — tracker.gg (Diamond+ ranked) + vlr.gg (pro/VCT)
5. YouTube ingest — optional; transcript → classify → lineup or web chunks
6. LLM tier regeneration — LLM outputs raw pick/win-rate floats; deterministic scorer assigns S/A/B/C
7. Write `meta.json` — stamped with `sync_in_progress` until re-ingest completes
8. Re-ingest knowledge base — meta + agents + maps re-embedded into ChromaDB

**Examples:**

```bash
valocoach meta-refresh                                  # one-shot, only on new patch
valocoach meta-refresh --force                          # force full re-run now
valocoach meta-refresh --dry-run                        # preview without writing
valocoach meta-refresh --watch                          # keep running, sync on each patch
valocoach meta-refresh --install-cron                   # schedule daily via crontab
valocoach meta-refresh -y "https://youtu.be/VIDEO_ID"   # ingest a guide video with refresh
```

**How tiers are computed (Phase C2 — deterministic scoring):**

The LLM is no longer asked to assign S/A/B/C tiers. It outputs raw numeric `pick_rate_pct` and `win_rate_pct` values. A deterministic formula computes a composite score:

```
score = win_rate_pct + log1p(pick_rate_pct) × 0.5
```

Score thresholds (calibrated against Diamond+ patch history):

| Tier | Score threshold | Typical values |
|------|----------------|----------------|
| S | ≥ 53.5 | 52% WR, 30% PR |
| A | ≥ 51.5 | 51% WR, 15% PR |
| B | ≥ 50.0 | 50% WR, 20% PR |
| C | < 50.0 | < 50% WR |

Same inputs always produce the same tiers — no "flapping" across refreshes caused by LLM temperature. The LLM is still responsible for the `reason` field because prose reasoning is where language models add value.

---

### agents-refresh

Keep `agents.json` in sync with Riot's current roster.

```bash
valocoach agents-refresh
```

Diffs your local `agents.json` against the Liquipedia `Category:Agents` API and reports any new agents plus any agents missing from `meta.json`'s tier list.

**Options:**

| Flag | Description |
|------|-------------|
| `--extract-kits` | Parse Liquipedia wikitext templates with regex (no LLM) and write kit data into `agents.json` |
| `--auto-stub-meta` | Append C-tier placeholders to `meta.json` for agents missing from the tier list |

**Examples:**

```bash
# Discovery only — print gaps, no writes
valocoach agents-refresh

# Full automated import when a new agent drops
valocoach agents-refresh --extract-kits --auto-stub-meta

# Then refresh tier data
valocoach meta-refresh --force
```

Kit extraction is deterministic (no LLM) because Liquipedia exposes structured wikitext templates with named fields. If the format ever drifts, the parser falls back to printing a JSON skeleton + the wiki URL for manual completion.

---

### ingest

Populate or update the vector store used for RAG retrieval during coaching.

```bash
valocoach ingest [flags]
```

**Options:**

| Flag | Description |
|------|-------------|
| `--seed` | Embed the built-in JSON knowledge base (agents, maps, economy) |
| `--corpus`, `-c` | Embed markdown files from `corpus/` |
| `--url`, `-u URL` | Scrape and embed a URL (patch notes, blog post, lineup guide) |
| `--youtube`, `-y URL` | Fetch, classify, preview, and ingest a YouTube video transcript |
| `--preview` | Analyse a YouTube video and show what would be ingested — no writes |
| `--force` | Re-ingest a YouTube video even if already stored |
| `--clear` | Wipe the vector store before ingesting |
| `--stats` | Show what is currently in the vector store |

**Smart URL ingest** — `--url` is not just a plain embed. It runs the same classification pipeline as YouTube ingest:

1. Scrape the page with trafilatura (falls back to BeautifulSoup)
2. Chunk the content
3. Run each chunk through the anchor-based classifier (tactical, lineups, off-topic, etc.)
4. Lineup-classified chunks → LLM metadata extraction → stored as `type=lineup` in ChromaDB (appear in `valocoach lineup` searches)
5. Other relevant chunks → stored as `type=web` with a 30-day TTL
6. Off-topic chunks dropped silently

This means a Dotesports lineup guide or a Gameleap strategy article ingested via `--url` will surface in `valocoach lineup` searches just like a YouTube video would.

**YouTube ingest flow** — when `--youtube` is provided, a preview is shown before writing:

```
Video:    Sova Haven Lineups — Complete Guide
Channel:  ProGuideChannel

Chunks fetched:   42
Chunks kept:      18  (lineup: 6 · regular: 12)
Dropped:          24  (off-topic: 5 · low-relevance: 14 · no-embedding: 5)

Lineup candidates — 6
────────────────────────────────────────────
  1.  03:20  Sova · Recon Bolt · Haven · A site  [pre-round info]
             "stand here at the corner aim at the box..."
  ...

Ingest these 18 chunks? [y/N]
```

Use `--preview` to inspect a video without being prompted. Use `--force` to re-ingest a video already in the database.

**Examples:**

```bash
# First-time setup
valocoach ingest --seed

# Ingest today's patch notes
valocoach ingest --url "https://playvalorant.com/en-us/news/game-updates/valorant-patch-notes-10-09/"

# Ingest a written lineup guide (lineup chunks go to lineup DB automatically)
valocoach ingest --url "https://dotesports.com/valorant/sova-recon-bolt-lineups-haven"

# Preview a YouTube video before committing
valocoach ingest --youtube "https://www.youtube.com/watch?v=..." --preview

# Ingest a YouTube lineup guide
valocoach ingest --youtube "https://www.youtube.com/watch?v=..."

# Re-ingest with updated metadata
valocoach ingest --youtube "https://www.youtube.com/watch?v=..." --force

# Check what's in the store
valocoach ingest --stats

# Wipe and re-seed from scratch
valocoach ingest --clear --seed
```

---

### patch

Show the latest patch version stored locally.

```bash
valocoach patch
```

**Options:**

| Flag | Description |
|------|-------------|
| `--check` | Fetch the current version from the HenrikDev API and compare to local |

`--check` requires `henrikdev_api_key` in config.

---

### notes

Capture and track action items from coaching sessions.

```bash
valocoach notes                          # list open notes (default)
valocoach notes list
valocoach notes add "practice smoke timings on Ascent A"
valocoach notes resolve 3
```

**Sub-commands:**

| Sub-command | Description |
|-------------|-------------|
| `notes list` | Show all open (unresolved) notes |
| `notes add <text>` | Create a new note |
| `notes resolve <id>` | Mark note `<id>` as resolved |

Notes are also accessible from the REPL with `/note <text>`, `/notes`, and `/resolve <id>`.

---

### sessions

Manage saved coaching session histories.

```bash
valocoach sessions list
valocoach sessions close <id>
```

**Sub-commands:**

| Sub-command | Description |
|-------------|-------------|
| `sessions list` | List saved sessions with date, turn count, and topic |
| `sessions close <id>` | Archive a session |

**Options for `sessions list`:**

| Flag | Default | Description |
|------|---------|-------------|
| `--limit`, `-l` | `20` | Number of recent sessions to show (newest first) |

---

### config

Manage the configuration file at `~/.valocoach/config.toml`.

```bash
valocoach config init    # create / reset to defaults
valocoach config show    # print current effective settings
```

---

## REPL reference

The interactive REPL is entered by running `valocoach coach` with no arguments. It uses `prompt_toolkit` for history, tab-completion, and auto-suggest.

**Starting the REPL:**

```bash
valocoach coach
```

```
━━━ Interactive Coaching Mode ━━━

Ask anything about your gameplay.
Set context first: /agent Jett  /map Ascent  /side attack
Then ask: "how do I hold A site better?"

Commands: /help  ·  Ctrl-D to exit

vc > _
```

**Tab completion** works for agent names, map names, and slash commands at the `vc > ` prompt.
**Arrow keys** navigate history, which persists to `~/.valocoach/history` across restarts.

---

### Match context slash commands

Set your match context once — every subsequent question uses it automatically. Context is injected as a structured header into the LLM prompt so the coach always knows the situation.

| Command | Example | What it does |
|---------|---------|--------------|
| `/agent <name>` | `/agent Jett` | Set your agent for this session |
| `/map <name>` | `/map Ascent` | Set the map |
| `/side <attack\|defense>` | `/side attack` | Set your side |
| `/score <a-b>` | `/score 9-11` | Set the current score |
| `/won` | `/won` | Mark the last match as won |
| `/lost` | `/lost` | Mark the last match as lost |
| `/eco <level>` | `/eco half` | Set economy: `eco` · `half` · `full` |
| `/enemy <agent>` | `/enemy Cypher` | Add an enemy agent (repeatable) |
| `/half` | `/half` | Toggle side at half-time (attack ↔ defense) |
| `/context` | `/context` | Print the current match context |
| `/reset` | `/reset` | Clear all match context for this session |

Once context is set, questions skip elicitation entirely — the REPL never re-asks fields you've already provided.

**Example session:**

```
vc > /agent Jett
Agent set: Jett

vc > /map Ascent
Map set: Ascent

vc > /side attack
Side set: attack

vc > /score 8-12
Score set: 8-12

vc > we keep losing A execute
[... coach answers using the Jett + Ascent attack context ...]

vc > /half
Side flipped: attack → defense  (half-time)

vc > how do I hold B from market?
[... coach answers for B defense on Ascent as Jett ...]
```

---

### Session management slash commands

| Command | Description |
|---------|-------------|
| `/help` | List all slash commands |
| `/clear` | Wipe conversation memory, start fresh |
| `/memory` | Show turn count and token usage in the current window |
| `/save` | Save the current session to disk immediately |
| `/sessions` | List previously saved sessions |
| `/stats` | Print your recent stats card |
| `/note <text>` | Add a coaching note |
| `/notes` | List open coaching notes |
| `/resolve <id>` | Resolve note `<id>` |
| `/quit` | Exit the REPL (also Ctrl-D, Ctrl-C) |

---

## Shell completion

`valocoach` supports tab completion for bash, zsh, fish, and PowerShell.

```bash
valocoach --install-completion bash   # adds to ~/.bashrc
valocoach --install-completion zsh    # adds to ~/.zshrc
valocoach --install-completion fish   # adds to ~/.config/fish/completions/
```

Restart your terminal (or `source` your rc file) after running.

To print the script without auto-installing:

```bash
valocoach --show-completion zsh
```

| Context | Completions |
|---------|-------------|
| `valocoach <TAB>` | All sub-commands |
| `coach --agent <TAB>` | Agent names |
| `coach --map <TAB>` | Map names |
| `coach --side <TAB>` | `attack`, `defense` |
| REPL `vc > ` prompt | Agent names, map names, slash commands |

---

## Configuration file

`valocoach config init` creates `~/.valocoach/config.toml`:

```toml
riot_name   = ""         # Riot username (no # prefix)
riot_tag    = ""         # e.g. NA1
riot_region = "na"       # na | eu | ap | kr | latam | br

henrikdev_api_key = ""   # https://docs.henrikdev.xyz/authentication

ollama_model     = "qwen3:8b"
ollama_host      = "http://localhost:11434"
llm_temperature  = 0.6
llm_max_tokens   = 3000
```

**Environment variable overrides** (prefix `VALOCOACH_`):

```bash
export VALOCOACH_OLLAMA_MODEL=qwen3:14b
export VALOCOACH_HENRIKDEV_API_KEY=hdev-...
export VALOCOACH_RIOT_NAME=MyName
```

**Resolution order** (highest priority first): process env → `.env` file → `config.toml` → field defaults.

---

## Troubleshooting

### "LLM call failed" / coach produces no output

Ollama is not running or the model is not pulled.

```bash
ollama list              # check what's available
ollama serve             # start the server
ollama pull qwen3:8b     # pull the default model if missing
```

### "riot_name / riot_tag not configured"

```bash
valocoach config init    # re-create the file, then edit it manually
# or set environment variables:
export VALOCOACH_RIOT_NAME=YourName
export VALOCOACH_RIOT_TAG=NA1
```

### "No local data — run valocoach sync first"

`valocoach stats`, `post-game`, and stats-personalised coaching need synced data:

```bash
valocoach sync
```

If sync fails with an API error, verify `henrikdev_api_key` in config.

### "Agent / map not found in knowledge base"

The vector store hasn't been seeded yet:

```bash
valocoach ingest --seed
```

If the warning persists after seeding, check canonical spelling:

```bash
valocoach meta --agent Kay/o      # shows canonical name KAY/O
```

### Embeddings are slow or failing

```bash
ollama pull nomic-embed-text      # must be pulled before ingest
```

### First `ingest --seed` fails mid-way

The vector store may be in a partial state. Wipe and re-seed:

```bash
valocoach ingest --clear --seed
```

### YouTube transcript: "IP blocked" message

YouTube blocks transcript fetches from cloud IPs. This affects VPS-hosted or CI environments. The CLI surfaces this specifically so you know the transcript was blocked (not missing). Run on a local machine where YouTube doesn't block the IP.

### Meta info feels outdated

Check when the meta data was last refreshed:

```bash
valocoach patch --check           # compare local meta patch vs live patch
valocoach meta-refresh            # one-shot refresh (runs only on new patch)
valocoach meta-refresh --force    # force refresh even with no new patch
```

### HenrikDev rate limiting during sync

```bash
valocoach sync --limit 5          # reduce API calls per run
```

---

## Architecture

```
valocoach coach "situation"
       │
       ├─ parser.py           Regex cascade → agent, map, side, score, clutch,
       │                      econ, phase  (<1 ms, no LLM)
       │
       ├─ intent.py           Rule-based 9-class intent classifier
       │                      (clutch > post_plant > retake > economy > ...)
       │
       ├─ elicitation.py      Intent-aware follow-up questions (≤ 3, only
       │                      relevant fields, TTY-only, skipped in REPL after
       │                      first answer or slash-command set)
       │
       ├─ retrieval/          RAG pipeline
       │   ├─ retrieve_static()   ChromaDB similarity search → top-k chunks
       │   │                      (static: agents, maps; live: web, YouTube)
       │   └─ format_*()          Format agent abilities, callouts, economy
       │
       ├─ coach/              Player context
       │   └─ build_stats_context()   Recent match stats → LLM-readable block
       │
       ├─ context_budget.py   Tiktoken-based token counter + 3-stage trimmer
       │                      Hard limit: 24 000 tokens; vector hits and stats
       │                      trimmed first when over budget
       │
       ├─ llm/provider.py     LiteLLM → Ollama streaming
       │                      (swap to any OpenAI-compatible endpoint in config)
       │
       ├─ coach/sanitizer.py  Post-stream ability fact-checker: cross-checks
       │                      every ability name in the response against agents.json
       │
       └─ cli/display.py      Rich streaming panel
```

**REPL data flow:**

```
valocoach coach (no args, TTY)
       │
       ├─ run_interactive()        prompt_toolkit REPL loop
       │   ├─ SessionMatchContext  Mutable bag: agent, map, side, score, enemies
       │   │                       Written by /agent /map /side etc.
       │   │                       Merged into every run_coach() call as context
       │   │
       │   └─ ConversationMemory   Sliding window: 20 turns or 3 000 tokens
       │                           Injected between system prompt and current
       │                           user message — LLM has full conversation
       │
       └─ run_coach()              Same pipeline as one-shot coach
```

**Meta pipeline (deterministic, no LLM for display):**

```
valocoach meta-refresh
       │
       ├─ 1. Patch check     HenrikDev API → compare to stored version
       ├─ 2. Patch notes     playvalorant.com scrape
       ├─ 3. Patch diff      extract per-agent changes → patch_changes/
       ├─ 4. Stats scrape    tracker.gg (Diamond+) + vlr.gg (pro/VCT)
       ├─ 5. YouTube ingest  classify → lineup or web chunks
       ├─ 6. LLM tier regen  LLM outputs pick/win floats; deterministic scorer
       │                     assigns S/A/B/C via score = WR + log1p(PR) × 0.5
       ├─ 7. Write meta.json stamped sync_in_progress=true until step 8 completes
       └─ 8. Re-ingest KB    meta + agents + maps re-embedded into ChromaDB
```

**Storage layout:**

```
~/.valocoach/
├── config.toml          Settings
├── history              REPL command history (prompt_toolkit)
└── data/
    ├── valocoach.db     SQLite — players, matches, rounds, coaching sessions, notes
    └── chroma/          ChromaDB — static collection (abilities, callouts)
                                  — live collection (web, YouTube, meta stats)
```

**LLM stack:**

| Layer | Default | Swappable |
|-------|---------|-----------|
| Runtime | Ollama | Any OpenAI-compatible endpoint via `ollama_host` |
| Coaching model | `qwen3:8b` | `qwen3:14b`, `llama3.3:70b`, etc. |
| Embedding model | `nomic-embed-text` | Any Ollama embedding model |
| Token counting | tiktoken `cl100k_base` | Fixed — used for budget math only |

**Vector store collections:**

| Collection | Contents | TTL |
|------------|----------|-----|
| `static` | Agent abilities, map callouts, economy facts (from JSON) | Permanent |
| `live` | Patch notes, YouTube transcripts, web articles, meta stats | 30–60 days |
