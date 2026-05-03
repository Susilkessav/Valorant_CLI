"""Lightweight pre-flight sanity checks for the coaching pipeline.

Each function is side-effect-free and non-raising — it returns a
``CheckResult`` named tuple so the caller decides whether to abort,
warn, or degrade gracefully.  This design keeps the functions easy to
unit-test without monkey-patching ``sys.exit``.

Typical call pattern (fatal check)::

    result = check_ollama(settings)
    if not result.ok:
        display.error(result.message)
        if result.hint:
            display.warn(result.hint)
        raise typer.Exit(1)

Typical call pattern (non-fatal warning)::

    result = check_vector_store(settings)
    if not result.ok:
        display.warn(f"{result.message}  {result.hint}")
"""

from __future__ import annotations

from typing import NamedTuple

import httpx


class CheckResult(NamedTuple):
    ok: bool
    message: str
    hint: str = ""


# ---------------------------------------------------------------------------
# Ollama connectivity + model availability
# ---------------------------------------------------------------------------


def check_ollama(settings) -> CheckResult:
    """Probe Ollama's HTTP API and verify the configured model is pulled.

    Uses a 3-second timeout so the caller is not blocked for long when
    Ollama is down.  The probe hits ``/api/tags`` (lists pulled models) —
    it does not start any inference.

    Returns a failing ``CheckResult`` when:
    - Ollama is not reachable at ``settings.ollama_host``.
    - The configured ``settings.ollama_model`` has not been pulled yet.
    """
    url = f"{settings.ollama_host}/api/tags"
    try:
        resp = httpx.get(url, timeout=3.0)
        resp.raise_for_status()
    except Exception:
        return CheckResult(
            ok=False,
            message=f"Ollama is not reachable at {settings.ollama_host}.",
            hint="Start it with:  ollama serve",
        )

    # Verify the configured model is pulled.  Ollama stores model names as
    # "qwen3:8b" or "qwen3:8b:latest" — treat them as equivalent by checking
    # the base name prefix.
    model = settings.ollama_model
    tags = resp.json().get("models", [])
    names = {m.get("name", "") for m in tags}

    # Empty means no models at all; non-empty but not matching is also a miss.
    model_available = any(
        n == model or n.startswith(f"{model}:") or model.startswith(f"{n}:") for n in names
    )
    if not names:
        return CheckResult(
            ok=False,
            message=f"No models are pulled in Ollama (configured: '{model}').",
            hint=f"Pull it with:  ollama pull {model}",
        )
    if not model_available:
        return CheckResult(
            ok=False,
            message=f"Model '{model}' is not pulled in Ollama.",
            hint=f"Pull it with:  ollama pull {model}",
        )

    return CheckResult(ok=True, message="Ollama reachable and model available.")


# ---------------------------------------------------------------------------
# Riot identity configuration
# ---------------------------------------------------------------------------


def check_riot_id(settings) -> CheckResult:
    """Verify ``riot_name`` and ``riot_tag`` are both set in config.

    This is a purely local read — no network calls.  Failing is non-fatal:
    coaching still works without a Riot ID; the user only loses personalised
    stats injection.
    """
    if not settings.riot_name or not settings.riot_tag:
        return CheckResult(
            ok=False,
            message="Riot ID not configured (riot_name / riot_tag are empty).",
            hint=(
                "Edit ~/.valocoach/config.toml and fill in riot_name and riot_tag,\n"
                "  or run:  valocoach config init"
            ),
        )
    return CheckResult(ok=True, message="Riot ID configured.")


# ---------------------------------------------------------------------------
# Vector store population
# ---------------------------------------------------------------------------


def check_vector_store(settings) -> CheckResult:
    """Return a failing result when the static vector store has no documents.

    Uses ``collection.count()`` (a single SQLite lookup inside ChromaDB) —
    much cheaper than ``collection_stats()`` which reads all metadata.

    Non-fatal by design: coaching still works via the LLM's training
    knowledge; it just cannot cite specific ability costs or callout names
    from the JSON knowledge base.
    """
    try:
        from valocoach.retrieval.vector_store import STATIC_COLLECTION, get_collection

        coll = get_collection(settings.data_dir, STATIC_COLLECTION)
        if coll.count() == 0:
            return CheckResult(
                ok=False,
                message=(
                    "Vector store is empty — the LLM will not have grounded ability/callout facts."
                ),
                hint="Seed it once with:  valocoach ingest --seed",
            )
    except Exception:
        return CheckResult(
            ok=False,
            message="Could not read vector store (may not be initialised yet).",
            hint="Seed it once with:  valocoach ingest --seed",
        )

    return CheckResult(ok=True, message="Vector store has documents.")
