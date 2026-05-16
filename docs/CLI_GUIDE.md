# ValoCoach CLI Guide

This guide covers the ValoCoach CLI — a Valorant-branded coaching tool that runs
entirely from the terminal. It syncs your match history, analyzes your stats, and
gives tactical coaching backed by an LLM and a local knowledge base.

During local development, run commands through `uv run`, for example
`uv run valocoach stats`. If you installed the package as a command, drop the
`uv run` prefix.

## Quick Start

1. Install dependencies.

```bash
uv sync
```

For development tools:

```bash
uv sync --extra dev
```

2. Pull the required Ollama models.

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

3. Start Ollama in a separate terminal.

```bash
ollama serve
```

4. Create the config file.

```bash
uv run valocoach config init
```

Then edit `~/.valocoach/config.toml` and set at least:

```toml
riot_name = "YourName"
riot_tag = "NA1"
riot_region = "na"
henrikdev_api_key = "hdev-..."
```

5. Seed the built-in knowledge base.

```bash
uv run valocoach ingest --seed
```

6. Sync your recent competitive matches.

```bash
uv run valocoach sync
```

7. Ask for coaching.

```bash
uv run valocoach coach "we keep losing 8-12 on Ascent attack as Jett, they stack A"
```

## First Run Experience

Running `valocoach` with no arguments shows a branded banner and quick-start
commands — it does not fall through to the help text.

```bash
uv run valocoach
```

To see the full command list with groupings:

```bash
uv run valocoach --help
```

## Command Groups

Commands are organized into four groups visible in `--help`:

| Group | Commands |
|---|---|
| **Coaching** | `coach`, `interactive`, `notes`, `sessions`, `lineup` |
| **Performance** | `stats`, `profile`, `post-game` |
| **Data** | `sync`, `ingest`, `index` |
| **Game Info** | `meta`, `meta-refresh`, `patch` |

## Configuration

ValoCoach reads settings from environment variables, `.env`, and
`~/.valocoach/config.toml`. The normal user workflow is to edit the TOML file.

### Required user settings

```toml
riot_name = "YourName"
riot_tag = "NA1"
riot_region = "na"
henrikdev_api_key = "hdev-..."
```

| Key | Meaning |
|---|---|
| `riot_name` | Riot username without the `#tag`. |
| `riot_tag` | Riot tag after the `#`, such as `NA1`. |
| `riot_region` | Region slug: `na`, `eu`, `ap`, `kr`, `latam`, or `br`. |
| `henrikdev_api_key` | HenrikDev API key, required for `sync` and `patch --check`. |

### LLM settings

```toml
ollama_model = "qwen3:8b"
ollama_host = "http://localhost:11434"
llm_temperature = 0.6
llm_max_tokens = 3000
```

| Key | Meaning |
|---|---|
| `ollama_model` | Model name passed through LiteLLM. Default is `qwen3:8b`. |
| `ollama_host` | Ollama HTTP endpoint. Default is `http://localhost:11434`. |
| `llm_temperature` | Coaching response randomness. Default is `0.6`. |
| `llm_max_tokens` | Maximum generated response tokens. Default is `3000`. |

### Environment overrides

```bash
export OLLAMA_MODEL=qwen3:14b
export HENRIKDEV_API_KEY=hdev-...
export RIOT_NAME=YourName
export RIOT_TAG=NA1
```

You can also place those keys in a project-local `.env` file.

### Config safety

`valocoach config show` prints the current effective settings and may include
`henrikdev_api_key`. Do not paste full `config show` output into public chats,
bug reports, screenshots, or issue trackers unless you have redacted the key first.

## Coaching

### One-shot coach

Use `coach` for a single situation and one streamed answer.

```bash
uv run valocoach coach "losing 8-12 on Ascent attack as Jett, they always stack A"
```

ValoCoach detects agent, map, side, site, score, and economy state from the text.
Pass explicit flags when the text is ambiguous:

```bash
uv run valocoach coach "keep dying in mid" --agent Omen --map Haven --side defense
```

Options:

| Option | Meaning |
|---|---|
| `--agent`, `-a` | Override the detected agent. |
| `--map`, `-m` | Override the detected map. |
| `--side`, `-s` | Override the detected side: `attack` or `defense`. |
| `--with-stats` | Include recent synced stats in the prompt (default). |
| `--no-stats` | Skip player stats, use only situation + knowledge context. |

The model name is shown in the coach panel subtitle. If the LLM call fails you see
an error with a hint explaining how to fix it (e.g. check Ollama is running).

Examples:

```bash
uv run valocoach coach "1v3 post plant B site on Bind, I'm Viper and spike is B Long"
uv run valocoach coach "lost pistol and bonus, eco next, should I force on 3k?"
uv run valocoach coach "Sage on Split defense, A site gets hit every round"
uv run valocoach coach "best Jett dash angles on Ascent A site" --no-stats
```

### Interactive mode

Use `interactive` for a multi-turn coaching session.

```bash
uv run valocoach interactive
```

The REPL keeps recent conversation turns in memory so follow-up questions build on
prior advice (up to 20 turns or 3000 tokens). Prior sessions can be resumed on
start. Sessions are saved to `~/.valocoach/sessions/` on exit.

Features:

- Up/down arrows use persistent prompt history from `~/.valocoach/history`.
- Tab completion covers agent names, map names, and slash commands.
- Coaching notes can be added and reviewed without leaving the REPL.

Prompt: `vc > `

Slash commands:

| Command | Meaning |
|---|---|
| `/help` | Show slash command help. |
| `/clear` | Clear conversation memory and start fresh. |
| `/memory` | Show current turn count and token usage. |
| `/save` | Save the current session immediately. |
| `/sessions` | List recently saved sessions. |
| `/stats` | Show the same recent stats card as `valocoach stats`. |
| `/note <text>` | Add a coaching note from within the session. |
| `/notes` | List open (unresolved) coaching notes. |
| `/resolve <id>` | Mark a coaching note as resolved by its ID. |
| `/quit` | Exit the REPL. Ctrl-D also exits. |

Example session:

```text
vc > losing on Ascent A execute as Jett
...
vc > what if they run two sentinels?
...
vc > /note work on smoke timings for A default
vc > /memory
vc > /save
vc > /quit
```

## Coaching Notes

Notes let you capture action items from coaching sessions and track them over time.

```bash
uv run valocoach notes list
uv run valocoach notes add "practice smoke timings on Ascent A"
uv run valocoach notes resolve 3
```

Sub-commands:

| Sub-command | Meaning |
|---|---|
| `notes list` | Show all open (unresolved) coaching notes. |
| `notes add <text>` | Create a new note. |
| `notes resolve <id>` | Mark note `<id>` as resolved. |

Notes are also accessible from interactive mode with `/note`, `/notes`, and
`/resolve`.

## Lineups

Search the local lineup library — hand-verified seed entries plus any
YouTube transcript chunks the ingest pipeline classified as `lineups` and
extracted metadata from.

```bash
uv run valocoach lineup Sova --map Ascent --site A
uv run valocoach lineup --map Bind --query "post-plant B"
uv run valocoach lineup Brimstone --map Haven
```

Results show the agent, ability, map, site, purpose, and (for video-sourced
entries) a `📹 channel "title" @ mm:ss` reference so you can find the clip.

Options:

| Option | Meaning |
|---|---|
| `<agent>` | Positional. Filter to one agent (e.g. `Sova`, `Viper`, `KAY/O`). |
| `--map`, `-m` | Filter to one map. |
| `--site`, `-s` | Site letter: `A`, `B`, `C`, or `Mid`. |
| `--query`, `-q` | Free-text similarity query, e.g. `"smoke A Long"`. |
| `--top`, `-n` | Maximum hits to return. Default is `5`. |

Filters are canonical-case normalised on both write (ingest) and read
(query), so `--agent sova`, `--agent Sova`, and `--agent SOVA` all match the
same entries. If no hits return, broaden the filters or omit `--query`.

## Post-Game Debrief

Run a finding-based analysis of your most recent stored match and get a
focused 4-section coaching debrief.

```bash
uv run valocoach post-game
```

The analyzer runs ten deterministic checks on the match — first-contact
patterns, economy decisions, utility efficiency, round-timing, traded
deaths, ATK/DEF side split, clutch conversion, death-location clustering,
engagement distance, plant/defuse site distribution — plus an MMR-trend
check across recent ranked games. The top three findings (collapsed by
root cause) are written into a structured block that the LLM uses as
ground truth for the debrief.

The debrief is laid out as:

1. **🔴 Critical Pattern** — single most damaging habit, with numbers.
2. **📊 What this cost you** — round-outcome translation.
3. **🎯 Priority drill** — one custom-game or DM drill that targets it.
4. **🎮 Next-match focus** — one mindset cue + one rule.

When a `low_utility` finding fires on a util-heavy agent (Sova, Viper,
KAY/O, Brimstone, Omen, Fade, Cypher, Killjoy), the debrief also injects a
LINEUP SUGGESTIONS block pulled from the lineup library and the LLM is
told to quote one with its `[SOURCE: youtube/...]` citation.

Requires at least one synced match. Plant-site analyzer + spatial
analyzers degrade gracefully on matches synced before the spatial-data
schema migration.

## Coaching Sessions

Sessions are the saved conversation histories from `interactive` mode.

```bash
uv run valocoach sessions list
uv run valocoach sessions close <id>
```

Sub-commands:

| Sub-command | Meaning |
|---|---|
| `sessions list` | Show saved sessions with date, turn count, and topic. |
| `sessions close <id>` | Archive a session so it no longer appears in the list. |

## Syncing Match History

```bash
uv run valocoach sync
```

Default behavior:

- Fetches stored matches in `competitive` mode.
- Inspects up to 20 matches.
- Stops early when it reaches an already stored match.
- Stores data in `~/.valocoach/data/valocoach.db`.
- Sweeps expired retrieval cache rows (SQLite `meta_cache` + live ChromaDB
  docs) before fetching, so the cache stays bounded over time.

Options:

| Option | Meaning |
|---|---|
| `--limit N` | Maximum stored matches to inspect. Default is `20`. |
| `--full` | Inspect all `--limit` matches even after finding stored matches. |
| `--mode MODE` | Game mode filter. Default is `competitive`. |

Examples:

```bash
uv run valocoach sync
uv run valocoach sync --limit 5
uv run valocoach sync --full --limit 50
uv run valocoach sync --mode unrated
```

## Stats Dashboard

```bash
uv run valocoach stats
```

Output is wrapped in a branded frame titled **Stats Dashboard** with a subtitle
showing active filters. Inside the frame, sections appear in this order:

1. **Identity line** — name, rank, region, match count
2. **Core Performance** — Combat group (K/D/A, KDA, HS%, ACS, ADR) and Match
   group (record, win%, rounds, FB/FD) side by side
3. **Recent Form** — trend anomalies only, hidden when performance is stable
4. **Win vs Loss** — split comparison, hidden when a result filter is active
5. **Round Mastery** — round-level stats when round data is available
6. **Agent Breakdown** — top agents, hidden when agent filter is active
7. **Map Breakdown** — top maps, hidden when map filter is active

Cells marked with `⚠` are below sample-size thresholds and should be treated as
directional, not definitive. A legend line appears at the bottom when any warnings
fired.

Options:

| Option | Meaning |
|---|---|
| `--period`, `-p` | Time window: `7d`, `30d`, `90d`, or `all`. Default is `30d`. |
| `--agent`, `-a` | Filter to a single agent. |
| `--map`, `-m` | Filter to a single map. |
| `--result`, `-r` | Filter by `win` or `loss`. Omit for both. |

Examples:

```bash
uv run valocoach stats
uv run valocoach stats --period 7d --agent Jett
uv run valocoach stats --map Bind --result loss
uv run valocoach stats --period all
```

## Player Profile

```bash
uv run valocoach profile
```

Output is wrapped in a **Player Profile** frame with the Riot ID as subtitle.
Sections inside the frame:

1. **Identity panel** — rank, RR, peak rank, account level, last match
2. **Rank Progression** — ELO trend chart when rank history is available
3. **Last N Matches** — compact stats card
4. **Round Mastery** — round-level stats when data is available
5. **Top Agents** — per-agent breakdown
6. **Coaching** — open coaching notes + recent sessions (silently skipped if
   the player hasn't been coached yet)

Options:

| Option | Meaning |
|---|---|
| `--name`, `-n` | Riot username. Defaults to configured `riot_name`. |
| `--tag`, `-t` | Riot tag. Must be paired with `--name`. |
| `--limit`, `-l` | Number of recent matches to summarize. Default is `20`. |

Examples:

```bash
uv run valocoach profile
uv run valocoach profile --name SomePlayer --tag NA1
uv run valocoach profile --limit 50
```

The player must already exist in the local DB.

## Meta and Knowledge Base

### Meta lookup

```bash
uv run valocoach meta
uv run valocoach meta --agent Jett
uv run valocoach meta --map Ascent
```

- No flags: shows the global tier list in an **Agent Intel** frame. Tier colors
  follow the Valorant palette (S=red, A=green, B=amber, C=dim).
- `--agent`: shows ability summary and meta standing (tier, pick rate, win rate,
  reason) in a single panel inside an **Agent Intel** frame.
- `--map`: shows callouts and meta standing (top agents, notes) in a **Map Intel**
  frame.

When live patch data differs from the bundled meta, a warning appears showing the
patch mismatch.

Options:

| Option | Meaning |
|---|---|
| `--agent`, `-a` | Show ability and meta info for one agent. |
| `--map`, `-m` | Show callouts and meta info for one map. |

### Meta refresh

Use `meta-refresh` to run the full automated meta sync pipeline. It detects the
current patch, scrapes official patch notes and ranked/pro stats, optionally
ingests YouTube transcripts, regenerates the tier list via LLM, updates
`meta.json`, and re-embeds everything into ChromaDB.

```bash
uv run valocoach meta-refresh
```

Options:

| Option | Meaning |
|---|---|
| `--force` | Run even when no new patch is detected. |
| `--dry-run` | Execute all steps but don't write `meta.json` or re-ingest. |
| `--watch` | Run continuously: check daily, full sync on new patch. |
| `--install-cron` | Write a crontab entry to run daily patch checks at 08:00. |
| `--youtube URL` | Also ingest a YouTube transcript as part of the sync. |

Examples:

```bash
uv run valocoach meta-refresh
uv run valocoach meta-refresh --force
uv run valocoach meta-refresh --dry-run
uv run valocoach meta-refresh --watch
uv run valocoach meta-refresh --install-cron
uv run valocoach meta-refresh --youtube "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Ingest

Use `ingest` to populate or inspect the vector store used for RAG retrieval.

```bash
uv run valocoach ingest --seed
```

Options:

| Option | Meaning |
|---|---|
| `--seed` | Re-embed the built-in JSON knowledge base. |
| `--corpus`, `-c` | Embed markdown files from `corpus/`. |
| `--url`, `-u URL` | Scrape and ingest a URL. |
| `--youtube`, `-y URL` | Fetch and ingest a YouTube transcript. |
| `--clear` | Wipe the vector store. |
| `--stats` | Show current vector store document counts. |

Examples:

```bash
uv run valocoach ingest --seed
uv run valocoach ingest --corpus
uv run valocoach ingest --url "https://playvalorant.com/en-us/news/game-updates/valorant-patch-notes-9-04/"
uv run valocoach ingest --youtube "https://www.youtube.com/watch?v=VIDEO_ID"
uv run valocoach ingest --stats
uv run valocoach ingest --clear
```

Full reset and re-seed:

```bash
uv run valocoach ingest --clear
uv run valocoach ingest --seed
```

### Index

Embeds static markdown corpus files from `corpus/`.

```bash
uv run valocoach index
```

### Patch

Shows the latest patch version stored locally.

```bash
uv run valocoach patch
```

Use `--check` to fetch the current version from HenrikDev:

```bash
uv run valocoach patch --check
```

`patch --check` requires `henrikdev_api_key`.

## Common Goals

| Goal | Command |
|---|---|
| Get advice now | `uv run valocoach coach "retaking B on Bind as Sova"` |
| Start a multi-turn session | `uv run valocoach interactive` |
| Debrief your last match | `uv run valocoach post-game` |
| Look up a lineup | `uv run valocoach lineup Sova --map Ascent --site A` |
| Add a note during session | `/note work on smoke timings` |
| Review open notes | `uv run valocoach notes list` |
| Sync recent ranked matches | `uv run valocoach sync` |
| Sync more history | `uv run valocoach sync --full --limit 50` |
| Check Jett on Ascent | `uv run valocoach stats --agent Jett --map Ascent` |
| Show your profile | `uv run valocoach profile` |
| Inspect Jett knowledge | `uv run valocoach meta --agent Jett` |
| Inspect Ascent callouts | `uv run valocoach meta --map Ascent` |
| Update meta after patch | `uv run valocoach meta-refresh` |
| Refresh patch info | `uv run valocoach patch --check` |
| Check vector store contents | `uv run valocoach ingest --stats` |
| Reset vector store | `uv run valocoach ingest --clear` |
| Re-seed knowledge after reset | `uv run valocoach ingest --seed` |

## Local Data Paths

| Path | Contents |
|---|---|
| `~/.valocoach/config.toml` | User configuration. |
| `~/.valocoach/data/valocoach.db` | SQLite DB for players, matches, rounds, stats, patch versions. |
| `~/.valocoach/data/chroma/` | ChromaDB vector store for static and live retrieval collections. |
| `~/.valocoach/history` | Interactive REPL prompt history. |
| `~/.valocoach/sessions/` | Saved interactive coaching conversations. |

If `data_dir` is set in config, the database and ChromaDB paths move under that
directory.

## Troubleshooting

When something goes wrong, the CLI shows a plain-language error with an actionable
hint directly below it. The hint tells you exactly what to run next.

### `Ollama is not reachable`

```bash
ollama serve
ollama list
```

If you use a non-default host:

```toml
ollama_host = "http://localhost:11434"
```

### `Model 'qwen3:8b' is not pulled`

```bash
ollama pull qwen3:8b
```

If you configured a different model (e.g. `qwen3:14b`), pull that instead.

### `riot_name / riot_tag not configured`

Run: `valocoach config init`

Or edit `~/.valocoach/config.toml` manually:

```toml
riot_name = "YourName"
riot_tag = "NA1"
```

### No local match data

Run: `valocoach sync`

If there are still no matches, verify your Riot ID and region are correct and that
HenrikDev can see recent matches for the account.

### Missing HenrikDev API key

`sync` and `patch --check` require `henrikdev_api_key`:

```toml
henrikdev_api_key = "hdev-..."
```

### Empty vector store

Run: `valocoach ingest --seed`

If seeding fails halfway, reset and re-seed:

```bash
uv run valocoach ingest --clear
uv run valocoach ingest --seed
```

### Scraper or URL ingest returns no content

Some sites are JavaScript-heavy or block scraping. Prefer official Valorant patch
notes or server-rendered articles. If a URL fails, the CLI skips it rather than
trying to bypass anti-bot protections.

### YouTube transcript ingest fails

Possible causes:

- The URL or video ID is invalid.
- The video has no available transcript.
- Transcripts are disabled or unavailable in English.

Try a different video or verify that captions are visible on YouTube.

### HenrikDev rate limiting

Use smaller sync runs:

```bash
uv run valocoach sync --limit 5
```

The default sync path is incremental, so repeated smaller syncs are safe.

## Current Limitations

- `coach` and `interactive` require Ollama preflight to pass.
- Synced stats require HenrikDev data and a configured Riot ID.
- Live scraped meta quality depends on source availability.
- `config show` may print `henrikdev_api_key`; redact before sharing output.
- `config init` may not create every optional key; add missing TOML keys manually.
- Most workflows assume one configured player, although the schema can store
  multiple players.
