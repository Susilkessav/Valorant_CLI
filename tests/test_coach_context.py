"""Tests for the coach stats-context builder.

Two layers:
    _format_context(player, rows, top_n)   — pure, covered directly
    _build_system_prompt(base, ctx)        — pure, covered directly
    run_coach integration                  — mocks build_stats_context +
                                             stream_completion, asserts the
                                             context lands in the system prompt
"""

from __future__ import annotations

from unittest.mock import patch

from valocoach.cli.commands.coach import SYSTEM_PROMPT_STUB, _build_system_prompt, run_coach
from valocoach.coach.context import _format_context
from valocoach.core.config import Settings
from valocoach.data.orm_models import Match, MatchPlayer, Player
from valocoach.stats.round_analyzer import RoundAnalysis

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _mp(
    *,
    agent: str = "Jett",
    map_name: str = "Ascent",
    won: bool = True,
    rounds_played: int = 20,
    score: int = 5000,
    kills: int = 20,
    deaths: int = 10,
    assists: int = 5,
    headshots: int = 30,
    bodyshots: int = 60,
    legshots: int = 10,
    damage_dealt: int = 3000,
    first_bloods: int = 3,
    first_deaths: int = 1,
    match_id: str = "m-1",
) -> MatchPlayer:
    match = Match(
        match_id=match_id,
        map_name=map_name,
        queue_id="competitive",
        is_ranked=True,
        game_length_secs=0,
        rounds_played=rounds_played,
        red_score=0,
        blue_score=0,
        started_at="2026-04-19T18:00:00+00:00",
    )
    mp = MatchPlayer(
        match_id=match_id,
        puuid="p-tracked",
        agent_name=agent,
        team="Blue",
        won=won,
        score=score,
        kills=kills,
        deaths=deaths,
        assists=assists,
        rounds_played=rounds_played,
        headshots=headshots,
        bodyshots=bodyshots,
        legshots=legshots,
        damage_dealt=damage_dealt,
        damage_received=0,
        first_bloods=first_bloods,
        first_deaths=first_deaths,
        plants=0,
        defuses=0,
        afk_rounds=0,
        rounds_in_spawn=0,
        started_at="2026-04-19T18:00:00+00:00",
    )
    mp.match = match
    return mp


def _player() -> Player:
    return Player(
        puuid="p-tracked",
        riot_name="Yoursaviour01",
        riot_tag="SK04",
        region="na",
        platform="pc",
        account_level=240,
        current_tier=12,
        current_tier_patched="Gold 1",
        current_rr=25,
        elo=925,
        peak_tier=14,
        peak_tier_patched="Gold 3",
        last_match_at="2026-04-18T18:00:00+00:00",
    )


def _settings() -> Settings:
    return Settings(
        riot_name="Yoursaviour01",
        riot_tag="SK04",
        riot_region="na",
        henrikdev_api_key="fake",
    )


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------


def test_format_context_header_identifies_player() -> None:
    out = _format_context(_player(), [_mp()])
    assert "PLAYER CONTEXT" in out
    assert "Yoursaviour01#SK04" in out
    assert "Gold 1" in out
    assert "NA" in out  # region uppercased


def test_format_context_includes_core_numbers() -> None:
    out = _format_context(_player(), [_mp()])
    assert "1-0" in out  # record
    assert "100% WR" in out
    assert "ACS 250" in out  # 5000/20
    assert "K/D 2.00" in out
    assert "HS 30%" in out  # 30 / (30+60+10)
    assert "ADR 150" in out  # 3000/20


def test_format_context_entry_line_signs_diff() -> None:
    """fb_diff always gets a sign so the LLM can parse +3 vs -3 unambiguously."""
    rows = [_mp(first_bloods=5, first_deaths=2)]
    out = _format_context(_player(), rows)
    assert "FB 5 / FD 2" in out
    assert "diff +3" in out


def test_format_context_omits_agent_block_when_single_agent() -> None:
    """Single-agent block would just echo the overall line — skip it."""
    rows = [_mp(agent="Jett", match_id="a"), _mp(agent="Jett", match_id="b")]
    out = _format_context(_player(), rows)
    assert "Top agents:" not in out


def test_format_context_includes_agent_block_for_multi_agent() -> None:
    rows = [
        _mp(agent="Jett", match_id="a"),
        _mp(agent="Jett", match_id="b"),
        _mp(agent="Reyna", match_id="c"),
    ]
    out = _format_context(_player(), rows)
    assert "Top agents:" in out
    assert "Jett" in out
    assert "Reyna" in out
    # Jett has 2 games, Reyna 1 — Jett should come first (sorted by matches desc).
    # Both are thin (< 30 matches for a split) so "(thin sample)" appears after the
    # game count — index on "Jett (" to stay robust to the appended annotation.
    jett_idx = out.index("Jett (2g")
    reyna_idx = out.index("Reyna (1g")
    assert jett_idx < reyna_idx


def test_format_context_top_n_truncates_agent_block() -> None:
    rows = [_mp(agent=f"Agent{i}", match_id=f"m-{i}") for i in range(10)]
    out = _format_context(_player(), rows, top_n=2)
    # Agent0 and Agent1 appear; Agent9 does not
    assert "Agent0" in out
    assert "Agent1" in out
    assert "Agent9" not in out


def test_format_context_includes_map_block_for_multi_map() -> None:
    rows = [
        _mp(map_name="Ascent", match_id="a"),
        _mp(map_name="Ascent", match_id="b"),
        _mp(map_name="Lotus", match_id="c"),
    ]
    out = _format_context(_player(), rows)
    assert "Top maps:" in out
    assert "Ascent" in out
    assert "Lotus" in out


def test_format_context_omits_map_block_when_single_map() -> None:
    rows = [_mp(map_name="Ascent", match_id="a"), _mp(map_name="Ascent", match_id="b")]
    out = _format_context(_player(), rows)
    assert "Top maps:" not in out


def test_format_context_is_compact() -> None:
    """Rough budget: under ~600 chars for a typical player. Guards against
    accidental bloat that would eat the LLM's context window."""
    rows = [_mp(agent=f"A{i % 4}", map_name=f"M{i % 3}", match_id=f"m-{i}") for i in range(20)]
    out = _format_context(_player(), rows)
    assert len(out) < 600, f"context too long ({len(out)} chars): {out}"


# ---------------------------------------------------------------------------
# _build_system_prompt
# ---------------------------------------------------------------------------


def test_build_system_prompt_with_context_appends_separator() -> None:
    ctx = "PLAYER CONTEXT — fake"
    out = _build_system_prompt("BASE PROMPT", ctx)
    assert out.startswith("BASE PROMPT")
    assert "---" in out
    assert ctx in out


def test_build_system_prompt_without_context_is_passthrough() -> None:
    out = _build_system_prompt("BASE PROMPT", None)
    assert out == "BASE PROMPT"


# ---------------------------------------------------------------------------
# run_coach integration
# ---------------------------------------------------------------------------


def _stream(tokens: list[str]):
    """Minimal generator that yields the given tokens — stand-in for the LLM."""
    return (t for t in tokens)


def test_run_coach_injects_context_into_system_prompt() -> None:
    """When build_stats_context returns a snippet, it MUST land in the system
    prompt that stream_completion sees — otherwise personalisation is dead."""
    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch(
            "valocoach.cli.commands.coach.build_stats_context",
            return_value="PLAYER CONTEXT — injected",
        ),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["hello"]),
        ) as mock_stream,
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
    ):
        run_coach("push A site", agent="Jett", with_stats=True)

    assert mock_stream.call_count == 1
    kwargs = mock_stream.call_args.kwargs
    assert "PLAYER CONTEXT — injected" in kwargs["system_prompt"]
    assert kwargs["system_prompt"].startswith(SYSTEM_PROMPT_STUB)
    assert "Situation: push A site" in kwargs["user_message"]
    assert "Agent: Jett" in kwargs["user_message"]


def test_run_coach_skips_context_when_with_stats_false() -> None:
    """--no-stats means DON'T even call the builder. Faster + offline-safe."""
    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch(
            "valocoach.cli.commands.coach.build_stats_context",
        ) as mock_build,
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["hi"]),
        ) as mock_stream,
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
    ):
        run_coach("any situation", with_stats=False)

    mock_build.assert_not_called()
    assert mock_stream.call_args.kwargs["system_prompt"] == SYSTEM_PROMPT_STUB


def test_run_coach_proceeds_when_context_is_none() -> None:
    """No synced data → build_stats_context returns None → no '---' divider
    appears in the system prompt. Coach still runs."""
    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch(
            "valocoach.cli.commands.coach.build_stats_context",
            return_value=None,
        ),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["ok"]),
        ) as mock_stream,
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
    ):
        run_coach("situation", with_stats=True)

    assert mock_stream.call_count == 1
    assert mock_stream.call_args.kwargs["system_prompt"] == SYSTEM_PROMPT_STUB


def test_run_coach_survives_context_builder_exception() -> None:
    """A stats-side crash must never block coaching. Warn and proceed."""
    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch(
            "valocoach.cli.commands.coach.build_stats_context",
            side_effect=RuntimeError("db exploded"),
        ),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["ok"]),
        ) as mock_stream,
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
        patch("valocoach.cli.commands.coach.display.warn") as mock_warn,
    ):
        run_coach("situation", with_stats=True)

    assert mock_stream.call_count == 1
    # Warning should mention we're continuing without context
    assert mock_warn.called
    warn_msg = mock_warn.call_args.args[0]
    assert "stats context" in warn_msg.lower() or "continuing" in warn_msg.lower()
    # No context injected when builder failed
    assert mock_stream.call_args.kwargs["system_prompt"] == SYSTEM_PROMPT_STUB


# ---------------------------------------------------------------------------
# Reliability tagging in the LLM context — Phase D
# ---------------------------------------------------------------------------
#
# The key contract: thin data stays in the context (so the LLM knows what
# agent/map the player actually plays) but is labelled so the LLM doesn't
# treat a 3-game sample as reliable evidence of a tendency.


def test_format_context_overall_low_sample_note() -> None:
    """When the overall window is below the strictest threshold (30 for HS%),
    the form header line gets a '(low sample)' annotation so the LLM knows
    the entire block is indicative rather than reliable."""
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    out = _format_context(_player(), rows)
    assert "(low sample)" in out
    # The note must be on the 'Recent form' line, not buried inside a metric.
    form_line = next(line for line in out.splitlines() if "Recent form" in line)
    assert "(low sample)" in form_line


def test_format_context_no_overall_note_at_full_reliability() -> None:
    """At 30 matches every threshold clears — no '(low sample)' on the form line."""
    rows = [_mp(match_id=f"m-{i}") for i in range(30)]
    out = _format_context(_player(), rows)
    form_line = next(line for line in out.splitlines() if "Recent form" in line)
    assert "(low sample)" not in form_line


def test_format_context_thin_agent_split_is_tagged() -> None:
    """A per-agent split below the 30-match split threshold gets '(thin sample)'
    appended to its game-count marker so the LLM can discount it."""
    # 27 Jett + 3 Reyna — Jett is thin for splits (< 30), Reyna even more so.
    rows = [_mp(agent="Jett", match_id=f"j-{i}") for i in range(27)] + [
        _mp(agent="Reyna", match_id=f"r-{i}") for i in range(3)
    ]
    out = _format_context(_player(), rows)
    assert "Top agents:" in out
    jett_line = next(line for line in out.splitlines() if "Jett" in line)
    reyna_line = next(line for line in out.splitlines() if "Reyna" in line)
    assert "(thin sample)" in jett_line, f"Jett (27g) should be thin: {jett_line!r}"
    assert "(thin sample)" in reyna_line, f"Reyna (3g) should be thin: {reyna_line!r}"


def test_format_context_reliable_agent_split_not_tagged() -> None:
    """At 30+ games per agent, no '(thin sample)' annotation — the split
    is reliable and the LLM should treat it as ground truth."""
    rows = [_mp(agent="Jett", match_id=f"j-{i}") for i in range(30)]
    # Need a second agent to trigger the block at all.
    rows += [_mp(agent="Reyna", match_id=f"r-{i}") for i in range(30)]
    out = _format_context(_player(), rows)
    assert "Top agents:" in out
    for line in out.splitlines():
        if "Jett" in line or "Reyna" in line:
            assert "(thin sample)" not in line, f"Unexpected thin tag: {line!r}"


def test_format_context_thin_map_split_is_tagged() -> None:
    """Same tagging applies to per-map splits — a 3-game Ascent sample
    is as unreliable as a 3-game agent sample."""
    rows = [_mp(map_name="Ascent", match_id=f"a-{i}") for i in range(3)] + [
        _mp(map_name="Lotus", match_id=f"l-{i}") for i in range(3)
    ]
    out = _format_context(_player(), rows)
    assert "Top maps:" in out
    ascent_line = next(line for line in out.splitlines() if "Ascent" in line)
    assert "(thin sample)" in ascent_line, f"Ascent (3g) should be thin: {ascent_line!r}"


def test_format_context_compact_budget_still_holds_with_tags() -> None:
    """Tags add a few bytes — verify the snippet stays under 700 chars.
    (Budget was 600; +100 headroom for annotations is generous enough to
    cover top_n=3 of thin splits while still being LLM-prompt-cheap.)"""
    rows = [_mp(agent=f"A{i % 4}", map_name=f"M{i % 3}", match_id=f"m-{i}") for i in range(5)]
    out = _format_context(_player(), rows)
    assert len(out) < 700, f"context too long ({len(out)} chars):\n{out}"


# ---------------------------------------------------------------------------
# Round-level line (KAST / clutch / trade)
# ---------------------------------------------------------------------------


def _round_analysis(
    *,
    rounds: int,
    rounds_kast: int,
    clutches_won: int = 0,
    clutch_opportunities: int = 0,
    deaths: int = 0,
    traded_deaths: int = 0,
    triples: int = 0,
    aces: int = 0,
) -> RoundAnalysis:
    return RoundAnalysis(
        rounds=rounds,
        deaths=deaths,
        teammate_deaths=0,
        clutch_opportunities=clutch_opportunities,
        rounds_with_kill=0,
        rounds_with_assist=0,
        rounds_survived=0,
        rounds_traded_death=0,
        rounds_kast=rounds_kast,
        clutches_won=clutches_won,
        traded_deaths=traded_deaths,
        trades_given=0,
        double_kills=0,
        triple_kills=triples,
        quadra_kills=0,
        aces=aces,
    )


def test_format_context_omits_round_line_when_analysis_is_none() -> None:
    """Backwards compat: callers that don't pass analysis keep the old output."""
    out = _format_context(_player(), [_mp()])
    assert "Round play" not in out


def test_format_context_omits_round_line_on_empty_analysis() -> None:
    """Empty analysis (zero rounds) should drop the line — not emit zeros
    that the LLM would treat as real data."""
    analysis = _round_analysis(rounds=0, rounds_kast=0)
    out = _format_context(_player(), [_mp()], round_analysis=analysis)
    assert "Round play" not in out


def test_format_context_includes_round_line_when_analysis_present() -> None:
    """Happy path: KAST + clutch + traded-deaths land on one dash line."""
    analysis = _round_analysis(
        rounds=700,               # above every round floor (max is clutch: 600)
        rounds_kast=490,          # 70 % KAST
        clutches_won=2,
        clutch_opportunities=4,
        deaths=200,
        traded_deaths=100,        # 50 % traded
        triples=3,
        aces=1,
    )
    # 30 rows so match-count thresholds pass; 700 rounds clears every
    # round-count floor → no ⚠ anywhere on this line.
    rows = [_mp(match_id=f"m-{i}") for i in range(30)]
    out = _format_context(_player(), rows, round_analysis=analysis)
    round_line = next(line for line in out.splitlines() if "Round play" in line)
    assert "KAST 70.0%" in round_line
    assert "Clutch 2/4" in round_line
    assert "Traded deaths 50.0%" in round_line
    assert "1xAce" in round_line
    assert "3x3K" in round_line
    assert "⚠" not in round_line


def test_format_context_round_line_tags_thin_sample() -> None:
    """Below matches+rounds floor → individual metrics get a ⚠ tag but
    the line still renders (LLM should see the data, down-weighted)."""
    analysis = _round_analysis(
        rounds=50,        # well below KAST's 200-round floor
        rounds_kast=30,
        clutches_won=0,
        clutch_opportunities=0,
        deaths=10,
    )
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]  # 5 matches — thin
    out = _format_context(_player(), rows, round_analysis=analysis)
    round_line = next(line for line in out.splitlines() if "Round play" in line)
    assert "⚠" in round_line
