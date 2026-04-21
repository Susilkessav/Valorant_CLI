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
    # Jett has 2 games, Reyna 1 — Jett should come first (sorted by matches desc)
    jett_idx = out.index("Jett (2g)")
    reyna_idx = out.index("Reyna (1g)")
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
