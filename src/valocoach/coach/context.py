"""Build a compact player-context snippet for the coach LLM prompt.

The coach command appends this to the system prompt so the LLM can tailor
advice to the player's actual recent form instead of giving generic coaching.

Format goals:
    - Compact (~150-200 tokens). The whole point is to be cheap to ship on
      every request; a verbose dump wastes the context window.
    - Machine-readable by a language model but still human-skimmable —
      bullet-plus-dash, consistent ordering, numbers not words.
    - Self-labelled. The LLM sees a PLAYER CONTEXT header so it knows the
      data refers to the user and not some third party the user mentioned.

Separation of concerns:
    _format_context(player, rows, top_n)  — pure, string in / string out.
                                             Everything tested directly.
    build_stats_context(settings, …)      — sync entry point. Handles the
                                             async I/O and returns None on
                                             "no local data", so the caller
                                             can silently proceed.
"""

from __future__ import annotations

import asyncio

from valocoach.core.config import Settings
from valocoach.data.database import ensure_db, session_scope
from valocoach.data.orm_models import MatchPlayer, Player
from valocoach.data.repository import get_player_by_name, get_recent_matches
from valocoach.stats import (
    compute_per_agent,
    compute_per_map,
    compute_player_stats,
    reliability_flags,
)

# Default match window for the context snippet. Mirrors the profile card —
# "at a glance" form, not a deep dive. Override per-call if needed.
DEFAULT_LIMIT = 20

# How many agents/maps to list. More than 3 bloats the prompt with little
# coaching value — the long tail is noise for an LLM anyway.
DEFAULT_TOP_N = 3


# ---------------------------------------------------------------------------
# Formatting  (pure — no I/O, fully tested)
# ---------------------------------------------------------------------------


def _pct(ratio: float) -> str:
    """0.2734 → '27%'. Integer percent — keeps the snippet compact."""
    return f"{round(ratio * 100)}%"


def _format_context(
    player: Player,
    rows: list[MatchPlayer],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> str:
    """Render a compact context block for the LLM prompt.

    Returns a multi-line string. The caller concatenates it to the system
    prompt — no surrounding fences or separators are added here so the
    caller owns the exact wire format.

    Reliability tagging:
        Metrics below their sample-size threshold (from calculator.py) are
        annotated inline so the LLM knows not to treat them as ground truth.
        The approach is (b) — keep thin data, tag it — rather than omitting:
        a player who has played 3 Jett games still *plays* Jett, so silencing
        the split strips real context. The LLM is instructed via the
        SYSTEM_PROMPT_STUB not to dump stats back at the player; tagging the
        thin ones tells it to use them loosely rather than as hard evidence.
    """
    overall = compute_player_stats(rows)
    overall_flags = reliability_flags(overall)
    per_agent = compute_per_agent(rows)
    per_map = compute_per_map(rows)

    # "(low sample)" appended when the overall window is below the strictest
    # threshold — everything is shaky, warn once on the form line rather than
    # individually per metric.
    overall_thin = not all(overall_flags.values())
    thin_note = " (low sample)" if overall_thin else ""

    header = (
        f"PLAYER CONTEXT — {player.riot_name}#{player.riot_tag} "
        f"· {player.current_tier_patched} · {player.region.upper()}"
    )

    lines = [
        header,
        f"Recent form ({overall.matches} competitive match(es)){thin_note}:",
        (
            f"- Record: {overall.wins}-{overall.losses} ({_pct(overall.win_rate)} WR) "
            f"· ACS {overall.acs:.0f} "
            f"· K/D {overall.kd:.2f} "
            f"· KDA {overall.kda:.2f} "
            f"· HS {_pct(overall.hs_pct)} "
            f"· ADR {overall.adr:.0f}"
        ),
        (
            f"- Entry: FB {overall.first_bloods} / FD {overall.first_deaths} "
            f"(diff {overall.fb_diff:+d})"
        ),
    ]

    # Only show the per-agent block when the player has actually played
    # multiple agents — a single-agent list would duplicate the overall line.
    if len(per_agent) >= 2:
        lines.append("Top agents:")
        for a in per_agent[:top_n]:
            s = a.stats
            split_flags = reliability_flags(s, is_split=True)
            split_thin = " (thin sample)" if not all(split_flags.values()) else ""
            lines.append(
                f"- {a.agent} ({s.matches}g{split_thin}): "
                f"{_pct(s.win_rate)} WR · ACS {s.acs:.0f} · K/D {s.kd:.2f}"
            )

    if len(per_map) >= 2:
        lines.append("Top maps:")
        for m in per_map[:top_n]:
            s = m.stats
            split_flags = reliability_flags(s, is_split=True)
            split_thin = " (thin sample)" if not all(split_flags.values()) else ""
            lines.append(
                f"- {m.map_name} ({s.matches}g{split_thin}): "
                f"{_pct(s.win_rate)} WR · ACS {s.acs:.0f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Async worker
# ---------------------------------------------------------------------------


async def _build_stats_context_async(
    settings: Settings,
    *,
    limit: int,
    top_n: int,
) -> str | None:
    """Fetch the player + recent rows and hand them to the formatter.

    Returns None when the player has never been synced OR when the DB has
    no matches for them. Either way there's nothing to personalise with —
    caller should fall back to generic coaching.
    """
    if not settings.riot_name or not settings.riot_tag:
        return None

    await ensure_db(settings.data_dir / "valocoach.db")

    async with session_scope() as session:
        player = await get_player_by_name(session, settings.riot_name, settings.riot_tag)
        if player is None:
            return None
        rows = await get_recent_matches(session, player.puuid, limit=limit)

    if not rows:
        return None

    return _format_context(player, rows, top_n=top_n)


# ---------------------------------------------------------------------------
# Sync entry point
# ---------------------------------------------------------------------------


def build_stats_context(
    settings: Settings,
    *,
    limit: int = DEFAULT_LIMIT,
    top_n: int = DEFAULT_TOP_N,
) -> str | None:
    """Build the context snippet for the coach prompt.

    Returns:
        str: formatted context block ready to concatenate to the system prompt.
        None: no local data (unsynced player, empty DB, or no configured
              identity). Caller should silently proceed without context.

    Designed to be safe to call unconditionally — the caller decides whether
    to enable it; this function just reports honestly on what data exists.
    """
    return asyncio.run(_build_stats_context_async(settings, limit=limit, top_n=top_n))
