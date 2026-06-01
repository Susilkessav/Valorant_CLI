"""Patch-staleness warnings — LLM-free.

This lives outside ``cli/commands/coach.py`` on purpose.  ``coach.py`` imports
``stream_completion`` (and therefore LiteLLM) at module import time, so any
command that merely wants the "your meta data is stale" one-liner — ``stats``,
``profile`` — would otherwise drag the whole LLM stack into a pure local-DB
command and print LiteLLM network warnings.  This module depends only on the
patch tracker and the Rich display.
"""

from __future__ import annotations

import logging

from valocoach.cli import display

log = logging.getLogger(__name__)

_PATCH_STALE_THRESHOLD_DAYS: int = 21

_STALE_META_WARNED: bool = False


def maybe_warn_stale_meta(settings, *, once: bool = False) -> None:
    """Print a one-liner if the cached patch is older than the threshold.

    Used by both the LLM coaching path (post-stream warning) and the
    deterministic meta path so the warning fires on every meta-sensitive
    answer regardless of which code path produced it.

    Args:
        once: when True, suppresses the warning after it has fired once in the
              current process.  Lets non-meta entrypoints (``stats``,
              ``profile``, ``coach`` for non-meta intents) show it without
              bombarding the user inside a single interactive session.
    """
    global _STALE_META_WARNED
    if once and _STALE_META_WARNED:
        return
    try:
        from valocoach.retrieval.patch_tracker import get_patch_staleness_days

        stale_days = get_patch_staleness_days(settings.data_dir)
        if stale_days is None or stale_days > _PATCH_STALE_THRESHOLD_DAYS:
            age_str = (
                "never checked" if stale_days is None else f"{stale_days:.0f}d since last check"
            )
            display.console.print(
                f"[muted]! Meta info may be outdated ({age_str}) — "
                "run [info]valocoach patch --check[/info] to refresh.[/muted]"
            )
            _STALE_META_WARNED = True
    except Exception:
        log.debug("patch staleness check failed", exc_info=True)


def warn_stale_meta_once(settings) -> None:
    """Public entry: fire the staleness warning at most once per process.

    Called by non-coach commands (``stats``, ``profile``) so the user sees the
    staleness signal even on workflows that never touch the coach.
    """
    maybe_warn_stale_meta(settings, once=True)
