# ValoCoach

CLI-based Valorant tactical coaching for Valorant. Week 1 delivers the command skeleton, TOML-backed configuration, Rich terminal helpers, and live Ollama streaming through `Rich.Live` + `Markdown`.

## Status

- `valocoach coach "test"` streams a local Ollama response in a formatted Rich panel.
- `stats`, `sync`, `profile`, `meta`, and `patch` are stubbed with command/help surface in place for later milestones.
- Config lives at `~/.valocoach/config.toml` by default and can be overridden with environment variables.

## Quick start

```bash
uv sync --python 3.13 --extra dev
uv run valocoach config init
ollama serve
ollama pull qwen3:8b
uv run valocoach coach "test"
```

If you prefer a different model, update it with:

```bash
uv run valocoach config set ollama.model qwen3:14b
```

## Commands

```text
valocoach coach "situation"              # Stream one-shot coaching
valocoach stats                          # Stubbed
valocoach sync                           # Stubbed
valocoach profile                        # Stubbed
valocoach meta                           # Stubbed
valocoach config init|show|set|path|env  # Config management
valocoach interactive                    # Minimal REPL
valocoach patch                          # Stubbed
```

## Configuration

The default config file:

```toml
[ollama]
host = "http://127.0.0.1:11434"
model = "qwen3:8b"
request_timeout_seconds = 300.0

[coach]
temperature = 0.5
patch_version = "unknown"

[ui]
stream_refresh_per_second = 8
coach_border_style = "green"
show_thinking = false
```

Environment overrides:

- `VALOCOACH_HOME` relocates the app home directory.
- `VALOCOACH_CONFIG_FILE` points directly to a config file.
- Nested values can be overridden with `VALOCOACH_...` variables such as `VALOCOACH_OLLAMA__MODEL`.

## Validation target

The highest-risk assumption for the project is the week-one UX: local Ollama tokens must stream smoothly through Rich’s live markdown renderer. The `coach` command is wired specifically to validate that path before the rest of the data and RAG layers are built.
