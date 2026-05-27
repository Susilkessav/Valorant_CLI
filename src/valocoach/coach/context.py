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

from datetime import UTC

from valocoach.core.config import Settings
from valocoach.data.loader import load_player_data
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
    WeaponSplit,
    analyze_rounds,
    analyze_rounds_per_map,
    clutch_stat,
    compute_weapon_stats,
    kast_stat,
    multi_kill_summary,
    trade_efficiency_stat,
)


def _detect_tilt(rows: list[MatchPlayer]) -> str | None:
    """E5 — detect WR decline in the back half of the current gaming session.

    Uses a rolling 8-hour window ending at "now" rather than a calendar day.
    This sidesteps the timezone problem: a UTC-based "today" split late-night
    PST sessions across two days and hid genuine tilt.  An 8-hour window
    captures any realistic gaming session in one bucket regardless of where
    the player lives.

    Requires ≥ 6 matches in the window. Returns a warning when WR drops
    ≥ 20pp from the first half to the second half of the window.
    """
    from datetime import datetime as _dt
    from datetime import timedelta

    # Rolling window: anchor to "now" in UTC (started_at is stored as UTC ISO).
    # 8 hours is wide enough to capture an evening session but narrow enough
    # to exclude yesterday's games.
    now = _dt.now(tz=UTC)
    cutoff = (now - timedelta(hours=8)).isoformat()

    def _row_started_at(r: MatchPlayer) -> str:
        # started_at may be str (most rows) or datetime (depending on driver).
        # Normalise to ISO string for comparison; cheap and robust.
        v = r.started_at
        if isinstance(v, _dt):
            return v.isoformat()
        return str(v) if v is not None else ""

    today_rows = sorted(
        [r for r in rows if _row_started_at(r) >= cutoff],
        key=_row_started_at,
    )
    # Threshold lowered from 6 to 4 because most casual ranked sessions are
    # 3-5 matches.  At >=6 the detector basically never fired for non-pro
    # play.  Four is the minimum that still gives a 2 / 2 split with
    # meaningful per-half win-rate signal.
    if len(today_rows) < 4:
        return None

    mid = len(today_rows) // 2
    early = today_rows[:mid]
    late = today_rows[mid:]

    early_wr = sum(1 for r in early if r.won) / len(early)
    late_wr = sum(1 for r in late if r.won) / len(late)

    drop = early_wr - late_wr
    if drop >= 0.20:
        n = len(today_rows)
        e_pct = round(early_wr * 100)
        l_pct = round(late_wr * 100)
        return (
            f"! Session tilt ({n} games in last 8h): "
            f"WR dropped {e_pct}% -> {l_pct}% — consider a break."
        )
    return None


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
    """Return '!' when a StatResult is unreliable, '' otherwise."""
    from valocoach.stats.calculator import StatResult  # avoid circular at module level

    if isinstance(stat, StatResult) and not stat.is_reliable:
        return "!"
    return ""


def _format_round_line(analysis: RoundAnalysis, matches: int) -> str | None:
    """Render the round-level line for the coach context.

    Returns None when there is no round-level data (empty analysis) so
    the caller can drop it rather than showing zeros the LLM would mistake
    for real signal.

    Reliability comes directly from StatResult.is_reliable — no manual
    threshold math here. Each metric carries its own reliability flag.
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
        side_str = f" · ATK {_pct(analysis.attack_win_rate)}/{_pct(analysis.defense_win_rate)} DEF"

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
    are annotated with ``[CRIT]`` and notable anomalies with ``[WATCH]`` so
    the LLM can triage severity quickly using the same labels as the CLI.
    """
    if not comparison.has_anomalies:
        return []

    lines = [
        f"Form trend (last {comparison.form_matches}g vs {comparison.baseline_matches}g baseline):"
    ]
    for a in comparison.anomalies:
        sev_tag = " [WATCH]" if a.severity == "notable" else " [CRIT]"
        lines.append(f"- {a.one_liner()}{sev_tag}")
    return lines


def _format_context(
    player: Player,
    rows: list[MatchPlayer],
    *,
    top_n: int = DEFAULT_TOP_N,
    round_analysis: RoundAnalysis | None = None,
    baseline_comparison: BaselineComparison | None = None,
    per_map_analysis: dict[str, RoundAnalysis] | None = None,   # E1
    weapon_splits: list[WeaponSplit] | None = None,               # E2
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
            # E1: per-map ATK/DEF split
            map_ra = per_map_analysis.get(m.map_name) if per_map_analysis else None
            side_str = ""
            if (
                map_ra is not None
                and map_ra.attack_win_rate is not None
                and map_ra.defense_win_rate is not None
            ):
                side_str = (
                    f" · ATK {_pct(map_ra.attack_win_rate)}"
                    f"/{_pct(map_ra.defense_win_rate)} DEF"
                )
            lines.append(
                f"- {m.map_name} ({s.matches}g{split_thin}): "
                f"{_pct(s.win_rate)} WR · ACS {s.acs:.0f}{side_str}"
            )

    # E2: weapon HS% — only when we have meaningful data
    if weapon_splits:
        top_weapons = weapon_splits[:4]  # at most 4
        weapon_str = " · ".join(
            f"{w.weapon} {_pct(w.hs_pct)}"
            for w in top_weapons
        )
        lines.append(f"Weapon HS%: {weapon_str}")

    # E5: session tilt detector
    tilt_warn = _detect_tilt(rows)
    if tilt_warn:
        lines.append(tilt_warn)

    return "\n".join(lines)


def get_top_played_agents(
    settings: Settings,
    *,
    limit: int = DEFAULT_LIMIT,
    top_n: int = DEFAULT_TOP_N,
) -> list[str]:
    """Return the player's most-played agent names across recent matches.

    Returned in descending match-count order, length up to ``top_n``.  Empty
    list when there is no local data — caller should treat that as "no
    grounded agent context available" rather than guessing.

    Used by ``cli/commands/coach.py`` to auto-inject AGENT context blocks
    for the player's top agents even when the situation text doesn't name
    one.  Without this, asking "how do I rank up?" produced agent-free
    prompts and small LLMs (qwen3:8b) hallucinated abilities wildly.
    """
    data = load_player_data(settings, limit=limit, include_rounds=False)
    if data is None or not data.rows:
        return []
    per_agent = compute_per_agent(data.rows)
    return [a.agent for a in per_agent[:top_n] if a.agent]


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
    data = load_player_data(settings, limit=limit, include_rounds=True)
    if data is None or not data.rows:
        return None

    analysis = analyze_rounds(data.full_matches, data.player.puuid) if data.full_matches else None
    comparison = compare_baseline(data.rows)

    # E1: per-map ATK/DEF split
    per_map_analysis = (
        analyze_rounds_per_map(data.full_matches, data.player.puuid)
        if data.full_matches
        else None
    )
    # E2: weapon HS%
    weapon_splits = (
        compute_weapon_stats(data.full_matches, data.player.puuid)
        if data.full_matches
        else None
    )

    return _format_context(
        data.player,
        data.rows,
        top_n=top_n,
        round_analysis=analysis,
        baseline_comparison=comparison,
        per_map_analysis=per_map_analysis,
        weapon_splits=weapon_splits,
    )
