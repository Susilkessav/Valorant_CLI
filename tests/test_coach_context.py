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


def test_build_system_prompt_with_grounded_and_stats_context() -> None:
    grounded = "GROUNDED CONTEXT — fake"
    stats = "PLAYER CONTEXT — fake"
    out = _build_system_prompt("BASE PROMPT", grounded, stats)
    assert out.startswith("BASE PROMPT")
    assert "---" in out
    assert grounded in out
    assert stats in out


def test_build_system_prompt_without_contexts_is_passthrough() -> None:
    out = _build_system_prompt("BASE PROMPT", None, None)
    assert out == "BASE PROMPT"


def test_build_system_prompt_grounded_only() -> None:
    grounded = "GROUNDED CONTEXT — fake"
    out = _build_system_prompt("BASE PROMPT", grounded, None)
    assert grounded in out
    assert "PLAYER CONTEXT" not in out


def test_build_system_prompt_stats_only() -> None:
    stats = "PLAYER CONTEXT — fake"
    out = _build_system_prompt("BASE PROMPT", None, stats)
    assert stats in out
    assert "GROUNDED CONTEXT" not in out


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
    assert "Agent(s): Jett" in kwargs["user_message"]


def test_run_coach_skips_context_when_with_stats_false() -> None:
    """--no-stats means DON'T even call the stats builder. Faster + offline-safe.
    Grounded meta context is still injected (it comes from local JSON, not the DB)."""
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
    prompt = mock_stream.call_args.kwargs["system_prompt"]
    assert prompt.startswith(SYSTEM_PROMPT_STUB)
    # Grounded meta context is always injected even without stats
    assert "GROUNDED CONTEXT" in prompt


def test_run_coach_proceeds_when_context_is_none() -> None:
    """No synced data → build_stats_context returns None → no PLAYER CONTEXT block
    in the prompt, but grounded meta context is still injected. Coach still runs."""
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
    prompt = mock_stream.call_args.kwargs["system_prompt"]
    assert prompt.startswith(SYSTEM_PROMPT_STUB)
    assert "GROUNDED CONTEXT" in prompt
    # No stats block appended (stats header includes "—" separator after "PLAYER CONTEXT")
    assert "PLAYER CONTEXT —" not in prompt


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
    # Stats context is absent when builder failed, but grounded meta is still present
    prompt = mock_stream.call_args.kwargs["system_prompt"]
    assert prompt.startswith(SYSTEM_PROMPT_STUB)
    assert "PLAYER CONTEXT —" not in prompt
    assert "GROUNDED CONTEXT" in prompt


def test_run_coach_returns_assistant_text() -> None:
    """run_coach must return whatever stream_to_panel produced — the interactive
    REPL relies on this to capture the assistant turn for ConversationMemory."""
    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["unused — stream_to_panel is mocked"]),
        ),
        patch(
            "valocoach.cli.commands.coach.display.stream_to_panel",
            return_value="full assistant response",
        ),
    ):
        result = run_coach("any situation", with_stats=False)

    assert result == "full assistant response"


def test_run_coach_returns_none_for_empty_stream() -> None:
    """Empty stream_to_panel return ('') must surface as None so callers can
    cleanly skip storing an empty assistant turn."""
    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream([]),
        ),
        patch(
            "valocoach.cli.commands.coach.display.stream_to_panel",
            return_value="",
        ),
    ):
        result = run_coach("situation", with_stats=False)

    assert result is None


def test_run_coach_forwards_conversation_history() -> None:
    """When the REPL passes prior turns, they must reach stream_completion as-is.
    Without this plumbing, multi-turn coaching falls back to single-turn behaviour."""
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["ok"]),
        ) as mock_stream,
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
    ):
        run_coach("follow-up", with_stats=False, conversation_history=history)

    assert mock_stream.call_args.kwargs["conversation_history"] == history


def test_run_coach_default_conversation_history_is_none() -> None:
    """One-shot CLI invocations must pass None — prior_history is REPL-only."""
    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["ok"]),
        ) as mock_stream,
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
    ):
        run_coach("situation", with_stats=False)

    assert mock_stream.call_args.kwargs["conversation_history"] is None


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
        rounds=700,  # above every round floor (max is clutch: 600)
        rounds_kast=490,  # 70 % KAST
        clutches_won=2,
        clutch_opportunities=4,
        deaths=200,
        traded_deaths=100,  # 50 % traded
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
        rounds=50,  # well below KAST's 200-round floor
        rounds_kast=30,
        clutches_won=0,
        clutch_opportunities=0,
        deaths=10,
    )
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]  # 5 matches — thin
    out = _format_context(_player(), rows, round_analysis=analysis)
    round_line = next(line for line in out.splitlines() if "Round play" in line)
    assert "⚠" in round_line


# ---------------------------------------------------------------------------
# Missing branch coverage for coach.py
# ---------------------------------------------------------------------------


def _check_ok():
    from valocoach.core.preflight import CheckResult
    return CheckResult(ok=True, message="OK", hint="")


def _check_fail():
    from valocoach.core.preflight import CheckResult
    return CheckResult(ok=False, message="Riot ID not configured", hint="Run valocoach config init")


def _check_vs_fail():
    from valocoach.core.preflight import CheckResult
    return CheckResult(ok=False, message="Vector store empty", hint="Run valocoach index")


def test_run_coach_warns_when_unknown_agent_in_grounded_context() -> None:
    """_build_grounded_context warns when agent is not in knowledge base (line 66)."""
    warn_calls: list[str] = []

    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["ok"]),
        ),
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
        patch("valocoach.cli.commands.coach.display.warn", side_effect=lambda m: warn_calls.append(m)),
        # Return None for unknown agent → triggers the warn at line 66.
        patch(
            "valocoach.cli.commands.coach.format_agent_context",
            return_value=None,
        ),
    ):
        run_coach("push A", agent="ZZZUnknownAgent", with_stats=False)

    assert any("not found in knowledge base" in m for m in warn_calls)
    assert any("ZZZUnknownAgent" in m for m in warn_calls)


def test_run_coach_warns_when_unknown_map_in_grounded_context() -> None:
    """_build_grounded_context warns when map is not in knowledge base (line 68)."""
    warn_calls: list[str] = []

    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["ok"]),
        ),
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
        patch("valocoach.cli.commands.coach.display.warn", side_effect=lambda m: warn_calls.append(m)),
        patch(
            "valocoach.cli.commands.coach.format_map_context",
            return_value=None,
        ),
    ):
        run_coach("push B", map_="ZZZUnknownMap", with_stats=False)

    assert any("not found in knowledge base" in m for m in warn_calls)
    assert any("ZZZUnknownMap" in m for m in warn_calls)


def test_run_coach_warns_when_riot_id_not_configured() -> None:
    """check_riot_id not ok + with_stats=True → warning at line 131."""
    warn_calls: list[str] = []

    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["ok"]),
        ),
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
        patch("valocoach.cli.commands.coach.display.warn", side_effect=lambda m: warn_calls.append(m)),
        # check_riot_id/check_vector_store are lazily imported inside run_coach's body
        # via `from valocoach.core.preflight import ...`, so patch the source module.
        patch("valocoach.core.preflight.check_riot_id", return_value=_check_fail()),
        patch("valocoach.core.preflight.check_vector_store", return_value=_check_ok()),
    ):
        run_coach("eco round", with_stats=True)

    assert any("Riot ID" in m or "Coaching will proceed" in m for m in warn_calls)


def test_run_coach_warns_when_vector_store_empty() -> None:
    """check_vector_store not ok → warning at line 140→145."""
    warn_calls: list[str] = []

    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            return_value=_stream(["ok"]),
        ),
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
        patch("valocoach.cli.commands.coach.display.warn", side_effect=lambda m: warn_calls.append(m)),
        patch("valocoach.core.preflight.check_riot_id", return_value=_check_ok()),
        patch("valocoach.core.preflight.check_vector_store", return_value=_check_vs_fail()),
    ):
        run_coach("eco round", with_stats=True)

    assert any("Vector store" in m or "empty" in m.lower() for m in warn_calls)


def test_run_coach_raises_when_llm_fails() -> None:
    """stream_completion raises → error + warn displayed, exception re-raised (lines 214-217).

    Note: with_stats=False means check_riot_id is NOT called (it's inside `if with_stats:`),
    but check_vector_store IS always called (outside the guard), so we patch it at the source.
    """
    import pytest

    with (
        patch("valocoach.cli.commands.coach.load_settings", return_value=_settings()),
        patch("valocoach.cli.commands.coach.build_stats_context", return_value=None),
        patch(
            "valocoach.cli.commands.coach.stream_completion",
            side_effect=RuntimeError("ollama is down"),
        ),
        patch("valocoach.cli.commands.coach.display.stream_to_panel"),
        patch("valocoach.cli.commands.coach.display.error") as mock_error,
        patch("valocoach.cli.commands.coach.display.warn") as mock_warn,
        # check_vector_store is always called (not guarded by with_stats).
        # Patch at the source module since it's lazily imported inside run_coach.
        patch("valocoach.core.preflight.check_vector_store", return_value=_check_ok()),
    ):
        with pytest.raises(RuntimeError, match="ollama is down"):
            run_coach("push B", with_stats=False)

    mock_error.assert_called()
    mock_warn.assert_called()
    error_msg = mock_error.call_args.args[0]
    assert "LLM call failed" in error_msg


# ---------------------------------------------------------------------------
# context.py branch coverage — _format_round_line side data (line 96)
# and _format_context baseline_comparison (line 197)
# and build_stats_context / _build_stats_context_async (lines 243-249, 279)
# ---------------------------------------------------------------------------


def _make_player_stats():
    """Full PlayerStats object for BaselineComparison construction."""
    from valocoach.stats.calculator import PlayerStats

    return PlayerStats(
        matches=20,
        rounds=400,
        wins=12,
        losses=8,
        win_rate=0.6,
        acs=250.0,
        adr=150.0,
        kills=400,
        deaths=200,
        assists=100,
        kd=2.0,
        kda=2.5,
        headshots=120,
        bodyshots=240,
        legshots=40,
        hs_pct=0.30,
        first_bloods=40,
        first_deaths=20,
        fb_rate=0.10,
        fd_rate=0.05,
        fb_diff=20,
        plants=10,
        defuses=5,
        econ_rating=None,
    )


def _make_anomaly_obj(*, severity: str = "significant", is_improvement: bool = False):
    """Build a real Anomaly for baseline_comparison tests."""
    from valocoach.stats.baseline import Anomaly

    return Anomaly(
        metric="acs",
        label="ACS",
        baseline_val=250.0,
        form_val=200.0,
        delta=-50.0,
        pct_delta=-0.2,
        direction="down",
        severity=severity,  # type: ignore[arg-type]
        is_improvement=is_improvement,
        fmt=".0f",
        z_score=-2.1,
        baseline_stddev=30.0,
    )


def _make_baseline_comparison(anomalies=None):
    """Build a real BaselineComparison with the given anomalies."""
    from valocoach.stats.baseline import BaselineComparison

    ps = _make_player_stats()
    return BaselineComparison(
        baseline=ps,
        form=ps,
        baseline_matches=15,
        form_matches=5,
        anomalies=anomalies or [],
    )


def test_format_round_line_shows_atk_def_when_side_data_present() -> None:
    """attack_win_rate and defense_win_rate non-None → ATK/DEF appended (line 96).

    Coverage target: context.py line 96 — the side_str f-string.
    """
    from valocoach.coach.context import _format_round_line

    analysis = RoundAnalysis(
        rounds=100,
        deaths=50,
        teammate_deaths=100,
        clutch_opportunities=20,
        rounds_with_kill=60,
        rounds_with_assist=30,
        rounds_survived=40,
        rounds_traded_death=10,
        rounds_kast=75,
        clutches_won=5,
        traded_deaths=20,
        trades_given=15,
        double_kills=5,
        triple_kills=2,
        quadra_kills=1,
        aces=0,
        # Provide side data — non-None → triggers line 96
        attack_rounds=50,
        attack_wins=30,
        defense_rounds=50,
        defense_wins=35,
    )
    result = _format_round_line(analysis, 20)
    assert result is not None
    assert "ATK" in result
    assert "DEF" in result
    assert "60%" in result  # 30/50 = 60% attack win rate
    assert "70%" in result  # 35/50 = 70% defense win rate


def test_format_context_includes_baseline_block_when_comparison_has_anomalies() -> None:
    """baseline_comparison non-None + has_anomalies → extend with trend lines (line 197).

    Coverage target: context.py line 197 — lines.extend(_format_baseline_lines(comparison)).
    """
    anomaly = _make_anomaly_obj(severity="significant", is_improvement=False)
    comparison = _make_baseline_comparison(anomalies=[anomaly])
    rows = [_mp(match_id=f"m-{i}") for i in range(20)]
    out = _format_context(_player(), rows, baseline_comparison=comparison)
    assert "Form trend" in out
    assert "ACS" in out
    assert "(!!)" in out  # significant anomaly marker from _format_baseline_lines


def test_format_baseline_lines_returns_empty_when_no_anomalies() -> None:
    """has_anomalies=False → _format_baseline_lines returns [] (line 116→117).

    Coverage target: context.py line 116 — guard returns early.
    """
    from valocoach.coach.context import _format_baseline_lines

    comparison = _make_baseline_comparison(anomalies=[])
    result = _format_baseline_lines(comparison)
    assert result == []


def test_build_stats_context_returns_none_when_no_data() -> None:
    """load_player_data_async returns None → build_stats_context returns None (line 244).

    Coverage target: context.py lines 243 (await), 244 (if data is None), 279 (asyncio.run).
    """
    from unittest.mock import AsyncMock

    from valocoach.coach.context import build_stats_context

    with patch(
        "valocoach.coach.context.load_player_data_async",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = build_stats_context(_settings())

    assert result is None


def test_build_stats_context_returns_none_when_rows_empty() -> None:
    """data.rows=[] → build_stats_context returns None (line 244 second condition)."""
    from unittest.mock import AsyncMock, MagicMock

    from valocoach.coach.context import build_stats_context

    fake_data = MagicMock()
    fake_data.rows = []

    with patch(
        "valocoach.coach.context.load_player_data_async",
        new_callable=AsyncMock,
        return_value=fake_data,
    ):
        result = build_stats_context(_settings())

    assert result is None


def test_build_stats_context_returns_string_when_data_present() -> None:
    """data with non-empty rows → build_stats_context returns formatted string.

    Coverage target: context.py lines 247 (analysis), 248 (compare_baseline),
    249 (return _format_context), 279 (asyncio.run returns str).
    """
    from unittest.mock import AsyncMock

    from valocoach.coach.context import build_stats_context
    from valocoach.data.loader import PlayerData

    player = _player()
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    fake_data = PlayerData(player=player, rows=rows, full_matches=[])

    with patch(
        "valocoach.coach.context.load_player_data_async",
        new_callable=AsyncMock,
        return_value=fake_data,
    ):
        result = build_stats_context(_settings())

    assert result is not None
    assert isinstance(result, str)
    assert "PLAYER CONTEXT" in result
    assert "Yoursaviour01" in result
