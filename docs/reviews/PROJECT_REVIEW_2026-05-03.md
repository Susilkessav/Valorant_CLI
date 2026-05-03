# ValoCoach Project Review - 2026-05-03

## Scope

Reviewed the active workspace at `/Users/susil/Desktop/Valorant_CLI` against:

- `AGENTS.md`
- `BUILD_PLAN.md`
- `README.md`
- `pyproject.toml`
- `src/valocoach/**`
- `tests/**`
- `alembic/**`
- `scripts/**`

Excluded runtime/generated directories: `data/`, `htmlcov/`, `.venv/`, `.ruff_cache/`,
`.pytest_cache/`, `.uv-cache/`, and `.claude/` worktrees.

## Automated Checks

| Check | Result |
|---|---|
| `uv run pytest` | Passed: 911 passed, 4 skipped |
| `uv run pytest --cov=valocoach --cov-report=term-missing` | Passed, total coverage 87.6% |
| `uv run ruff check` | Failed: 21 lint findings |
| `uv run ruff format --check` | Failed: 26 files would be reformatted |

The project has strong behavioral coverage, especially around stats math, parser
edge cases, retrieval, sync, and CLI integration. The current commit is not
release-ready under the documented "done" rules because both Ruff checks fail.

## Overall Assessment

The architecture is mostly faithful to the build plan: Python 3.11+,
SQLAlchemy/Alembic, SQLite plus ChromaDB, LiteLLM, Typer/Rich, regex-first
situation parsing, tiktoken prompt budgeting, and local Ollama defaults are all
present. The test suite is broad and fast.

The highest-risk findings are not test failures. They are cross-boundary
correctness gaps:

- Live meta invalidation only clears SQLite cache rows, while stale Chroma live
  vectors remain searchable.
- Filtered stats views compute round-level stats from the unfiltered match set.
- The CLI still hard-requires Ollama even though the LLM wrapper supports
  cloud providers.
- Sample-size warnings ignore the round-count side of the project's own
  thresholds in the main formatter path.

Those issues can produce stale grounded context or misleading user-visible
stats, which are the two areas the project instructions mark as load-bearing.

## Review Artifacts

- Findings: `docs/reviews/FINDINGS_2026-05-03.md`
- Verification log: `docs/reviews/VERIFICATION_2026-05-03.md`

