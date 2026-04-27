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
from valocoach.data.loader import load_player_data_async
from valocoach.data.orm_models import MatchPlayer, Player
from valocoach.stats import (
    compute_per_agent,
    compute_per_map,
    compute_player_stats,
    reliability_flags,
)
from valocoach.stats.baseline import BaselineComparison, compare_baseline
from valocoach.stats.round_analyzer import (
    RoundAnalysis,
    analyze_rounds,
    clutch_stat,
    kast_stat,
    multi_kill_summary,
    trade_efficiency_stat,
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


def _warn(stat: object) -> str:
    """Return '⚠' when a StatResult is unreliable, '' otherwise."""
    from valocoach.stats.calculator import StatResult  # avoid circular at module level
    if isinstance(stat, StatResult) and not stat.is_reliable:
        return "⚠"
    return ""


def _format_round_line(analysis: RoundAnalysis, matches: int) -> str | None:
    """Render the round-level line for the coach context.

    Returns None when there is no round-level data (empty analysis) so
    the caller can drop it rather than showing zeros the LLM would mistake
    for real signal.

    Reliability comes directly from StatResult.is_reliable — no manual
    threshold math here. Each metric carries its own ⚠ flag.
    """
    if analysis.rounds == 0:
        return None

    kast = kast_stat(analysis, matches)
    clutch = clutch_stat(analysis, matches)
    trade = trade_efficiency_stat(analysis, matches)
    multi = multi_kill_summary(analysis)

    multi_str = f" · {multi}" if multi else ""

    side_str = ""
    if analysis.attack_win_rate is not None and analysis.defense_win_rate is not None:
        side_str = (
            f" · ATK {_pct(analysis.attack_win_rate)}"
            f"/{_pct(analysis.defense_win_rate)} DEF"
        )

    return (
        f"- Round play ({analysis.rounds} rounds): "
        f"KAST {kast.display}{_warn(kast)} "
        f"· Clutch {analysis.clutches_won}/{analysis.clutch_opportunities}{_warn(clutch)} "
        f"· Traded deaths {trade.display}{_warn(trade)}"
        f"{side_str}{multi_str}"
    )


def _format_baseline_lines(comparison: BaselineComparison) -> list[str]:
    """Render the form-trend block for the coach context.

    Returns a non-empty list of strings when there are detected anomalies,
    an empty list otherwise. The caller drops the block when empty.

    Format: one header line + one bullet per anomaly. Significant anomalies
    are annotated with '(!!)' so the LLM can triage severity quickly.
    """
    if not comparison.has_anomalies:
        return []

    lines = [
        f"Form trend (last {comparison.form_matches}g vs {comparison.baseline_matches}g baseline):"
    ]
    for a in comparison.anomalies:
        sev_tag = " (!)" if a.severity == "notable" else " (!!)"
        lines.append(f"- {a.one_liner()}{sev_tag}")
    return lines


def _format_context(
    player: Player,
    rows: list[MatchPlayer],
    *,
    top_n: int = DEFAULT_TOP_N,
    round_analysis: RoundAnalysis | None = None,
    baseline_comparison: BaselineComparison | None = None,
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
            + (f" · Econ {overall.econ_rating:.0f}" if overall.econ_rating is not None else "")
        ),
        (
            f"- Entry: FB {overall.first_bloods} / FD {overall.first_deaths} "
            f"(diff {overall.fb_diff:+d})"
        ),
    ]

    # Round-level line is optional: only appears when the caller supplied
    # analysis (i.e. loaded the heavier full-match fetch). Keeps the
    # snippet cheap for callers that don't need KAST/clutch/trade.
    if round_analysis is not None:
        round_line = _format_round_line(round_analysis, overall.matches)
        if round_line is not None:
            lines.append(round_line)

    # Baseline / anomaly block — only when there is something to flag.
    # Silent when the window is too thin or when performance is stable,
    # so the LLM doesn't see an empty section that wastes tokens.
    if baseline_comparison is not None:
        lines.extend(_format_baseline_lines(baseline_comparison))

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
    data = await load_player_data_async(settings, limit=limit, include_rounds=True)
    if data is None or not data.rows:
        return None

    analysis = analyze_rounds(data.full_matches, data.player.puuid) if data.full_matches else None
    comparison = compare_baseline(data.rows)
    return _format_context(
        data.player, data.rows,
        top_n=top_n,
        round_analysis=analysis,
        baseline_comparison=comparison,
    )


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
