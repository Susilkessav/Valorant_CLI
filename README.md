# ValoCoach

A local-first Valorant tactical coaching CLI. Streams Immortal-level advice for any match situation using a local Ollama LLM grounded in a RAG knowledge base (agent abilities, map callouts, game economy) and your own synced match history. No cloud required beyond the optional HenrikDev API for match data.

```
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
- [Full CLI guide](docs/CLI_GUIDE.md)
- [Command reference](#command-reference)
  - [coach](#coach)
  - [interactive](#interactive)
  - [post-game](#post-game)
  - [lineup](#lineup)
  - [stats](#stats)
  - [sync](#sync)
  - [profile](#profile)
  - [meta](#meta)
  - [meta-refresh](#meta-refresh)
  - [agents-refresh](#agents-refresh)
  - [ingest](#ingest)
  - [config](#config)
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

The 8B model is the default and runs well on 16 GB RAM. Swap to `qwen3:14b` for better reasoning on 32 GB+.

---

## Install

```bash
git clone https://github.com/you/valocoach
cd valocoach
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

For a detailed end-user walkthrough of every command, data path, and common
troubleshooting case, see the [full CLI guide](docs/CLI_GUIDE.md).

### 1 — Create config

```bash
uv run valocoach config init
```

This writes `~/.valocoach/config.toml` with safe defaults. Open it and fill in your Riot identity and HenrikDev key:

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

This embeds the built-in JSON knowledge base (agent abilities, map callouts, economy thresholds) into the local ChromaDB vector store. Run once; re-run when you update the knowledge JSONs.

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

This calls the HenrikDev API and stores your recent competitive matches in a local SQLite database (`~/.valocoach/data/valocoach.db`). Subsequent runs are incremental — only new matches are fetched.

### 5 — Start coaching

```bash
uv run valocoach coach "we lose pistol round on Bind defense every time"
```

---

## Command reference

### coach

Get tactical coaching for a match situation. The LLM is grounded with relevant abilities, callouts, and your recent performance stats.

```bash
valocoach coach "<situation>"
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--agent`, `-a` | auto-detected | Override the agent (e.g. `Jett`) |
| `--map`, `-m` | auto-detected | Override the map (e.g. `Ascent`) |
| `--side`, `-s` | auto-detected | `attack` or `defense` |
| `--with-stats` / `--no-stats` | on | Include your recent form in the prompt |

**Agent, map, and side are extracted automatically from the situation text** — you rarely need the flags:

```bash
# All three detected from text:
valocoach coach "losing 8-12 on Ascent attack as Jett, they always stack A"

# Explicit flags when the text is ambiguous:
valocoach coach "keep dying in mid" --agent Omen --map Haven --side defense
```

**Examples:**

```bash
# Post-plant 1v3 on B site
valocoach coach "1v3 post plant B site on Bind, I'm on Viper with the spike on B Long"

# Economy question
valocoach coach "lost pistol and bonus, eco round next — should I buy on 3k or save?"

# Retake scenario
valocoach coach "I'm playing Sage on Split defense, A site gets hit every round — retake tips"

# Skip stats for a quick one-off
valocoach coach "best Jett dash angles on Ascent A site" --no-stats
```

**Meta questions skip the LLM entirely.** Asking `"what's the current meta"` /
`"best agent this patch"` produces a deterministic **Meta — Current Tier List**
panel (built straight from `agents.json` + `meta.json`) followed by a
**Personalised Takeaway** panel computed from your local stats. No LLM, no
hallucinated ability names. Every word in those panels comes from JSON files
or your synced match DB.

**Ability fact-check panel** — for non-meta intents (tactical, post-game,
clutch, agent-info) the LLM IS used. After the answer streams, a deterministic
post-pass scans the output against `agents.json` and prints a categorised
warning if it finds fabricated abilities, wrong-agent attributions, weapons
mis-cast as abilities, or generic descriptors used as ability names. The
answer itself isn't rewritten — you see which claims to verify in-game.

---

### interactive

Start a multi-turn coaching session. The REPL maintains sliding-window conversation memory (20 turns / 3 000 tokens) so the LLM can reference earlier exchanges and build on previous advice.

```bash
valocoach interactive
```

```
ValoCoach interactive mode
Type a coaching question, e.g. "losing on Ascent attack as Jett 8-12".
Slash commands: /help  /clear  /memory  /stats  /quit  ·  Ctrl-D / Ctrl-C to quit

valocoach> losing on Ascent A execute
...

valocoach> what if they run two sentinels?
...       ↑ LLM remembers the Ascent A execute context

valocoach> /memory
Memory: 4 turn(s) · 1 247 tokens

valocoach> /clear
Conversation memory cleared.

valocoach> /quit
Bye.
```

**Slash commands:**

| Command | Description |
|---------|-------------|
| `/help` | List slash commands |
| `/clear` | Wipe conversation memory and start fresh |
| `/memory` | Show turn count and token usage |
| `/stats` | Print your recent stats card |
| `/quit` | Exit (also Ctrl-D, Ctrl-C) |

**Tab completion** is available for agent names, map names, and slash commands.  
**Arrow keys** navigate history, which persists across sessions (`~/.valocoach/history`).

---

### post-game

Run a deterministic finding-based analysis of your most recent stored match and stream a focused 4-section debrief: critical pattern, round-cost translation, priority drill, next-match focus.

```bash
valocoach post-game
```

Ten analyzers run against the match — first-contact patterns, economy decisions, utility efficiency, round-timing, traded deaths, ATK/DEF side split, clutches, death-location clustering, engagement distance, plant/defuse site distribution — plus an MMR-trend check over recent ranked games. The top three findings (collapsed by root cause) are injected into the prompt as ground truth, so every claim in the debrief is tied to a number.

When a `low_utility` finding fires on a util-heavy agent (Sova / Viper / KAY/O / Brimstone / Omen / Fade / Cypher / Killjoy), the debrief also pulls a relevant lineup from the local library and the LLM is told to cite it.

Requires at least one synced match. Spatial analyzers (death clusters, engagement distance) degrade gracefully on matches synced before the spatial-data migration.

---

### lineup

Search the local lineup library — hand-verified seed entries plus any YouTube transcript chunks the ingest pipeline classified as `lineups`.

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
| `--query`, `-q` | — | Free-text similarity query. |
| `--top`, `-n` | `5` | Maximum hits to return. |

Filter values are canonical-case normalised at both write and read time, so `sova` / `Sova` / `SOVA` all match the same entries.

---

### stats

Show your performance dashboard built from locally synced matches.

```bash
valocoach stats
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--period`, `-p` | `30d` | Time window: `7d`, `30d`, `90d`, or `all` |
| `--agent`, `-a` | all | Filter to one agent (e.g. `Jett`) |
| `--map`, `-m` | all | Filter to one map (e.g. `Ascent`) |
| `--result`, `-r` | both | `win` or `loss` only |

**Examples:**

```bash
# Last 30 days (default)
valocoach stats

# Last 7 days on Jett only
valocoach stats --period 7d --agent Jett

# See your losses on Bind specifically
valocoach stats --map Bind --result loss

# All-time stats
valocoach stats --period all
```

Requires `valocoach sync` to have been run at least once.

---

### sync

Fetch your recent competitive match history from the HenrikDev API and store it locally.

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

Every sync run also sweeps expired retrieval cache rows (SQLite `meta_cache` + live ChromaDB docs) before fetching, so the cache stays bounded.

```bash
# Standard incremental sync (stops when it hits already-stored matches)
valocoach sync

# Pull up to 50 matches regardless of what's already stored
valocoach sync --full --limit 50

# Sync unrated games
valocoach sync --mode unrated
```

Requires `henrikdev_api_key` in config.

---

### profile

Show a compact player identity card plus recent performance summary.

```bash
valocoach profile
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--name`, `-n` | configured | Riot username (defaults to config value) |
| `--tag`, `-t` | configured | Riot tag (must be paired with `--name`) |
| `--limit`, `-l` | `20` | Number of recent matches to summarise |

**Examples:**

```bash
# Your own profile
valocoach profile

# Look up another player
valocoach profile --name SomePlayer --tag NA1

# Summarise last 50 matches
valocoach profile --limit 50
```

---

### meta

Show current Valorant meta information from the knowledge base: agent tier list, ability details, map callouts, economy thresholds.

```bash
valocoach meta
valocoach meta --agent Jett
valocoach meta --map Ascent
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--agent`, `-a` | — | Show ability and meta info for one agent |
| `--map`, `-m` | — | Show callouts and meta info for one map |

---

### meta-refresh

Automated meta-sync pipeline. Detects the current patch via HenrikDev, scrapes
patch notes + Diamond+/pro stats, optionally ingests YouTube transcripts,
regenerates `meta.json`'s tier list via the local LLM, and re-embeds everything
into ChromaDB.

```bash
valocoach meta-refresh
```

**Options:**

| Flag | Description |
|------|-------------|
| `--force`, `-f` | Run the full sync even when no new patch is detected |
| `--dry-run` | Simulate all steps but do not write `meta.json` or re-ingest |
| `--install-cron` | Add a daily crontab entry that runs `meta-refresh` automatically |
| `--youtube`, `-y URL` | YouTube video ID/URL to ingest as supplemental context (repeatable) |

**Examples:**

```bash
valocoach meta-refresh                                  # one-shot, only on new patch
valocoach meta-refresh --force                          # force a full re-run
valocoach meta-refresh --install-cron                   # schedule daily
valocoach meta-refresh -y "https://youtu.be/VIDEO_ID"   # add a guide video
```

---

### agents-refresh

Keep `agents.json` in sync with Riot's roster. Where `meta-refresh` updates
tier placements and pick/win rates, this command covers the kit data — names,
costs, descriptions, and roles for new agents.

```bash
valocoach agents-refresh
```

It diffs your local `agents.json` against the Liquipedia `Category:Agents` API
and reports any new agents plus any agents that are in `agents.json` but
missing from `meta.json`'s tier list.

**Options:**

| Flag | Description |
|------|-------------|
| `--extract-kits` | Deterministically parse Liquipedia's `{{Infobox agent}}` + `{{AbilityCard}}` wikitext templates with regex (no LLM) and write the kit data straight into `agents.json` |
| `--auto-stub-meta` | Append C-tier placeholders to `meta.json` for known agents missing from the tier list. Re-run `meta-refresh --force` later to replace with real Diamond+/VCT-driven tier data |

**Examples:**

```bash
# Discovery only — print new agents + tier-list gaps, no writes
valocoach agents-refresh

# Full automated import when a new agent drops
valocoach agents-refresh --extract-kits --auto-stub-meta

# Then refresh tier data from scraped patch notes + stats
valocoach meta-refresh --force
```

Why is kit extraction deterministic (no LLM)? Liquipedia exposes structured
wikitext templates with named fields. Regex-parsing those fields produces the
same output every time — no hallucinated ability names. If Liquipedia's format
ever drifts, the parser refuses to write a half-filled entry and falls back to
printing a JSON skeleton + the wiki URL so you can fill it manually.

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
| `--corpus`, `-c` | Embed markdown files from `corpus/` (built by `scripts/build_corpus.py`) |
| `--url`, `-u URL` | Scrape and embed a URL (patch notes, blog post, etc.) |
| `--youtube`, `-y URL` | Fetch and embed a YouTube video transcript |
| `--clear` | Wipe the vector store before ingesting |
| `--stats` | Show what is currently in the vector store |

**Examples:**

```bash
# First-time setup — embed built-in knowledge base
valocoach ingest --seed

# Ingest today's patch notes
valocoach ingest --url "https://playvalorant.com/en-us/news/game-updates/valorant-patch-notes-9-04/"

# Ingest a pro-play breakdown from YouTube
valocoach ingest --youtube "https://www.youtube.com/watch?v=..."

# Check what's in the store
valocoach ingest --stats

# Wipe and re-seed from scratch
valocoach ingest --clear --seed
```

---

### config

Manage the configuration file at `~/.valocoach/config.toml`.

```bash
valocoach config init    # create default config
valocoach config show    # print current effective settings
```

---

## Shell completion

`valocoach` supports tab completion for **bash**, **zsh**, **fish**, and **PowerShell** via Typer's built-in shell completion.

### One-time installation

```bash
# Bash  (~/.bashrc)
valocoach --install-completion bash

# Zsh   (~/.zshrc)
valocoach --install-completion zsh

# Fish  (~/.config/fish/completions/)
valocoach --install-completion fish

# PowerShell  ($PROFILE)
valocoach --install-completion powershell
```

After running the command, **restart your terminal** (or `source` your shell's rc file) to activate.

### Manual / custom installation

Print the completion script to stdout instead of writing it automatically:

```bash
valocoach --show-completion bash   # prints the bash script
valocoach --show-completion zsh    # prints the zsh script
```

Redirect the output into a file or paste it into your rc file to integrate with
a custom completion setup (e.g. a shared dotfiles repo).

### What gets completed

| Context | Completions |
|---------|-------------|
| `valocoach <TAB>` | All sub-commands |
| `coach --agent <TAB>` | Agent names (Jett, Sage, …) |
| `coach --map <TAB>` | Map names (Ascent, Bind, …) |
| `coach --side <TAB>` | `attack`, `defense` |
| `interactive` REPL prompt | Agent names, map names, slash commands |

> **Note** — interactive-mode REPL completion (agent/map/slash-command hints)
> is handled by `prompt_toolkit`'s `WordCompleter` and works independently of
> shell completion.

---

## Configuration file

`valocoach config init` writes this file to `~/.valocoach/config.toml`:

```toml
riot_name   = ""         # fill in your Riot username
riot_tag    = ""         # fill in your tag (e.g. NA1)
riot_region = "na"       # na | eu | ap | kr | latam | br

henrikdev_api_key = ""   # https://docs.henrikdev.xyz/authentication

ollama_model  = "qwen3:8b"
ollama_host   = "http://localhost:11434"
llm_temperature  = 0.6
llm_max_tokens   = 3000
```

**Environment variable overrides** (prefix `VALOCOACH_`, double underscore for nesting):

```bash
export VALOCOACH_OLLAMA_MODEL=qwen3:14b
export VALOCOACH_HENRIKDEV_API_KEY=hdev-...
export VALOCOACH_RIOT_NAME=MyName
```

**Resolution order** (highest priority first): process env → `.env` file → `config.toml` → field defaults.

---

## Troubleshooting

For an architectural deep-dive on design trade-offs and known edge cases
(why meta intent skips the LLM, how the sanitizer classifies hallucinations,
which earlier review findings were verified as false positives), see
[`docs/REVIEW.md`](docs/REVIEW.md).

### "LLM call failed" / coach produces no output

Ollama is not running or the model is not pulled.

```bash
ollama list              # check what's available
ollama serve             # start the server
ollama pull qwen3:8b     # pull the default model if missing
```

### "riot_name / riot_tag not configured"

Edit `~/.valocoach/config.toml` and fill in both fields, or:

```bash
valocoach config init    # re-create the file, then edit it
```

### "No local data — run valocoach sync first"

`valocoach stats` and stats-personalised coaching require at least one sync run:

```bash
valocoach sync
```

If sync fails with an API error, verify your `henrikdev_api_key` in config.

### "Agent / map not found in knowledge base"

The vector store hasn't been seeded yet:

```bash
valocoach ingest --seed
```

If the warning persists, the agent or map name may not match the canonical name in the knowledge base. Check available names:

```bash
valocoach meta --agent Jett    # exact canonical spelling
```

### Embeddings are slow or failing

The embedding model (`nomic-embed-text`) must be pulled via Ollama before `ingest`:

```bash
ollama pull nomic-embed-text
```

### First `ingest --seed` fails mid-way

The vector store may be in a partial state. Wipe and re-seed:

```bash
valocoach ingest --clear --seed
```

### HenrikDev rate limiting during sync

Use `--limit` to reduce the number of API calls per run:

```bash
valocoach sync --limit 5
```

---

## Architecture

```
valocoach coach "..."
       │
       ├─ parser.py          Regex situation parser — extracts agent, map,
       │                     side, score, clutch, econ, phase (<1 ms)
       │
       ├─ retrieval/         RAG pipeline
       │   ├─ retrieve_static()   ChromaDB similarity search → top-k chunks
       │   └─ format_*()          Format abilities, callouts, economy facts
       │
       ├─ coach/             Player context
       │   └─ build_stats_context()   Recent match stats → LLM-readable summary
       │
       ├─ context_budget.py  Tiktoken-based token counter + 3-stage trimmer
       │                     Hard limit: 24 000 tokens; lowest-priority chunks
       │                     (vector hits, then stats) trimmed first
       │
       ├─ llm/provider.py    LiteLLM → Ollama streaming (or any OpenAI-compat)
       │
       └─ cli/display.py     Rich Live panel — streams markdown as it arrives
```

**Data flow for `interactive`:**

```
REPL loop
  │
  ├─ ConversationMemory   Sliding window: 20 turns or 3 000 tokens, whichever
  │                       is hit first. Oldest exchange evicted when full.
  │
  └─ run_coach()          Same pipeline as one-shot coach, plus prior turns
                          are inserted between the system prompt and the
                          current user message so the LLM has full context.
```

**Storage layout:**

```
~/.valocoach/
├── config.toml          Settings (edited by hand or via env)
├── history              REPL command history (prompt_toolkit FileHistory)
└── data/
    ├── valocoach.db     SQLite — players, matches, rounds, performance rows
    └── chroma/          ChromaDB vector store — static + live collections
```

**LLM stack:**

| Layer | Default | Swappable |
|-------|---------|-----------|
| Runtime | Ollama | Any OpenAI-compatible endpoint via LiteLLM |
| Coaching model | `qwen3:8b` | `qwen3:14b`, `llama3.3:70b`, etc. |
| Embedding model | `nomic-embed-text` | Any Ollama embedding model |
| Token counting | tiktoken `cl100k_base` | Fixed — used for budget math only |
