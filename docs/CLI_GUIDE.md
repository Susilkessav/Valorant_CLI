# ValoCoach CLI Guide

This guide explains how to use the ValoCoach command line as it works today. It
is written for players who want to install the CLI, sync match history, seed the
knowledge base, and get tactical coaching from the terminal.

During local development, run commands through `uv run`, for example
`uv run valocoach stats`. If you installed the package as a command, you can
drop the `uv run` prefix and use `valocoach stats`.

## Quick Start

1. Install dependencies.

```bash
uv sync
```

For development tools, use:

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

Current behavior: `config init` may not create every documented key. If a key is
missing, add it manually to the TOML file.

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

## Command Overview

| Command | Purpose |
|---|---|
| `valocoach coach` | Get one-shot tactical coaching for a described situation. |
| `valocoach interactive` | Start a multi-turn coaching REPL with conversation memory. |
| `valocoach sync` | Fetch match history from HenrikDev into the local SQLite DB. |
| `valocoach stats` | Show performance stats from synced matches. |
| `valocoach profile` | Show player identity, rank, and recent summary. |
| `valocoach meta` | Show bundled agent, map, and meta information. |
| `valocoach ingest` | Populate or inspect the vector store used for RAG. |
| `valocoach index` | Index the static markdown corpus into the vector store. |
| `valocoach patch` | Show or refresh the known Valorant game version. |
| `valocoach config` | Create or display local configuration. |

Use `--help` on any command to inspect the live options:

```bash
uv run valocoach --help
uv run valocoach coach --help
uv run valocoach stats --help
```

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

Current behavior: `coach` and `interactive` run an Ollama preflight before the
LLM call. Even though the provider wrapper can route prefixed LiteLLM models,
the CLI currently expects Ollama to be reachable for coaching commands.

### Environment overrides

The current settings model uses unprefixed variable names. Examples:

```bash
export OLLAMA_MODEL=qwen3:14b
export HENRIKDEV_API_KEY=hdev-...
export RIOT_NAME=YourName
export RIOT_TAG=NA1
```

You can also place those keys in a project-local `.env` file.

### Config safety

`valocoach config show` prints the current effective settings. Current behavior:
it may include `henrikdev_api_key`. Do not paste full `config show` output into
public chats, bug reports, screenshots, or issue trackers unless you have
redacted the key first.

## Coaching

Use `coach` for one situation and one streamed answer.

```bash
uv run valocoach coach "losing 8-12 on Ascent attack as Jett, they always stack A"
```

ValoCoach parses the situation text before calling the LLM. It tries to detect:

- agent names, such as `Jett`, `Sage`, `KAY/O`
- map names, such as `Ascent`, `Bind`, `Haven`
- side, such as `attack`, `defense`, `T-side`, `CT-side`
- site, score, clutch state, economy state, and phase

When the text is ambiguous, pass explicit flags:

```bash
uv run valocoach coach "keep dying in mid" --agent Omen --map Haven --side defense
```

Options:

| Option | Meaning |
|---|---|
| `--agent`, `-a` | Override the detected agent. |
| `--map`, `-m` | Override the detected map. |
| `--side`, `-s` | Override the detected side: `attack` or `defense`. |
| `--with-stats` | Include recent synced stats in the prompt. This is the default. |
| `--no-stats` | Skip player stats and use only situation plus knowledge context. |

Examples:

```bash
uv run valocoach coach "1v3 post plant B site on Bind, I'm Viper and spike is B Long"
uv run valocoach coach "lost pistol and bonus, eco next, should I force on 3k?"
uv run valocoach coach "Sage on Split defense, A site gets hit every round"
uv run valocoach coach "best Jett dash angles on Ascent A site" --no-stats
```

## Interactive Mode

Use `interactive` for a longer coaching session.

```bash
uv run valocoach interactive
```

The REPL keeps recent conversation turns in memory so follow-up questions can
build on prior advice. It stores up to 20 turns or 3000 tokens, whichever limit
is hit first. Older turns are evicted automatically.

REPL features:

- Up/down arrows use persistent prompt history from `~/.valocoach/history`.
- Tab completion covers agent names, map names, and slash commands.
- Prior sessions can be resumed when the REPL starts.
- Sessions are saved under `~/.valocoach/sessions/` on exit.

Slash commands:

| Command | Meaning |
|---|---|
| `/help` | Show slash command help. |
| `/clear` | Clear conversation memory and start fresh. |
| `/memory` | Show current turn count and token usage. |
| `/save` | Save the current session immediately. |
| `/sessions` | List recently saved sessions. |
| `/stats` | Show the same recent stats card as `valocoach stats`. |
| `/quit` | Exit the REPL. Ctrl-D also exits. Ctrl-C interrupts the current prompt. |

Example session:

```text
valocoach> losing on Ascent A execute as Jett
...
valocoach> what if they run two sentinels?
...
valocoach> /memory
valocoach> /save
valocoach> /quit
```

## Syncing Match History

Use `sync` to fetch recent matches from HenrikDev and store them locally.

```bash
uv run valocoach sync
```

Default behavior:

- Uses your configured `riot_name`, `riot_tag`, and `riot_region`.
- Fetches stored matches with `mode=competitive`.
- Inspects up to 20 matches.
- Stops early when it reaches an already stored match.
- Stores match data in `~/.valocoach/data/valocoach.db`.

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

Stats and personalized coaching only become useful after syncing at least one
competitive match.

## Stats Dashboard

Use `stats` to inspect your recent performance.

```bash
uv run valocoach stats
```

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

The stats output can include:

- overall record, ACS, ADR, K/D, KDA, HS%, first blood/death counts
- win/loss split
- per-agent breakdown
- per-map breakdown
- trend/anomaly notes when enough history exists
- round-level stats when round data is available

Cells marked with a warning symbol are below sample-size thresholds and should
be treated as directional, not definitive.

## Profile

Use `profile` for a compact identity and recent-form card.

```bash
uv run valocoach profile
```

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

The player must already exist in the local DB. To view another player, sync that
player's data first by changing config or using the configured account workflow.

## Meta and Knowledge Base

### Meta lookup

Use `meta` to read bundled knowledge without invoking the LLM.

```bash
uv run valocoach meta
uv run valocoach meta --agent Jett
uv run valocoach meta --map Ascent
```

Options:

| Option | Meaning |
|---|---|
| `--agent`, `-a` | Show ability and meta info for one agent. |
| `--map`, `-m` | Show callouts and meta info for one map. |

### Ingest

Use `ingest` to populate the vector store used by coach retrieval.

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

`--clear` returns after clearing. If you want a full reset and re-seed, run both
commands:

```bash
uv run valocoach ingest --clear
uv run valocoach ingest --seed
```

### Index

Use `index` to embed static markdown corpus files from `corpus/`.

```bash
uv run valocoach index
```

This is equivalent to indexing the static corpus, not to scraping live web
sources.

### Patch

Use `patch` to show the latest patch version recorded locally.

```bash
uv run valocoach patch
```

Use `--check` to fetch the current version from HenrikDev and update the local
patch table:

```bash
uv run valocoach patch --check
```

`patch --check` requires `henrikdev_api_key`.

## Common Goals

| Goal | Command |
|---|---|
| Get advice now | `uv run valocoach coach "retaking B on Bind as Sova"` |
| Start a multi-turn session | `uv run valocoach interactive` |
| Sync recent ranked matches | `uv run valocoach sync` |
| Sync more history | `uv run valocoach sync --full --limit 50` |
| Check Jett on Ascent | `uv run valocoach stats --agent Jett --map Ascent` |
| Show your profile | `uv run valocoach profile` |
| Inspect Jett knowledge | `uv run valocoach meta --agent Jett` |
| Inspect Ascent callouts | `uv run valocoach meta --map Ascent` |
| Refresh patch info | `uv run valocoach patch --check` |
| Check vector store contents | `uv run valocoach ingest --stats` |
| Reset vector store | `uv run valocoach ingest --clear` |
| Re-seed knowledge after reset | `uv run valocoach ingest --seed` |

## Local Data Paths

Default paths:

| Path | Contents |
|---|---|
| `~/.valocoach/config.toml` | User configuration. |
| `~/.valocoach/data/valocoach.db` | SQLite DB for players, matches, rounds, stats, cache rows, patch versions. |
| `~/.valocoach/data/chroma/` | ChromaDB vector store for static and live retrieval collections. |
| `~/.valocoach/history` | Interactive REPL prompt history. |
| `~/.valocoach/sessions/` | Saved interactive conversations. |

If `data_dir` is changed in config, the database and ChromaDB paths move under
that configured directory.

## Troubleshooting

### `Ollama is not reachable`

Start Ollama and verify the host:

```bash
ollama serve
ollama list
```

If you use a non-default host, set it in config:

```toml
ollama_host = "http://localhost:11434"
```

### `Model 'qwen3:8b' is not pulled`

Pull the configured model:

```bash
ollama pull qwen3:8b
```

If you configured `ollama_model = "qwen3:14b"`, pull that model instead.

### `riot_name / riot_tag not configured`

Edit `~/.valocoach/config.toml` and set both fields:

```toml
riot_name = "YourName"
riot_tag = "NA1"
```

### Missing HenrikDev API key

`sync` and `patch --check` require `henrikdev_api_key`:

```toml
henrikdev_api_key = "hdev-..."
```

### Empty vector store

Seed the knowledge base:

```bash
uv run valocoach ingest --seed
```

If seeding fails halfway, reset and seed again:

```bash
uv run valocoach ingest --clear
uv run valocoach ingest --seed
```

### No local match data

Run sync:

```bash
uv run valocoach sync
```

If there are still no matches, check that your Riot ID and region are correct
and that HenrikDev can see recent matches for the account.

### Scraper or URL ingest returns no content

Some sites are JavaScript-heavy or block scraping. Prefer official Valorant
patch notes, server-rendered articles, or the bundled corpus. If a URL fails,
the CLI skips it rather than trying to bypass anti-bot protections.

### YouTube transcript ingest fails

Possible causes:

- the URL or video ID is invalid
- the video has no available transcript
- transcripts are disabled or unavailable in English

Try a different video or verify that captions are visible on YouTube.

### HenrikDev rate limiting

Reduce each sync run:

```bash
uv run valocoach sync --limit 5
```

The default sync path is intentionally incremental, so repeated smaller syncs
are safe.

## Current Limitations

- `coach` and `interactive` currently require Ollama preflight to pass.
- Synced stats require HenrikDev data and a configured Riot ID.
- Live scraped meta quality depends on source availability and extractability.
- `config show` may print `henrikdev_api_key`; redact before sharing output.
- `config init` may not create every optional key; add missing TOML keys
  manually when needed.
- The CLI assumes one configured player for most workflows, although the schema
  can store multiple players.
