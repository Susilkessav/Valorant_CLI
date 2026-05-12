"""Tests for `valocoach profile` — identity resolution + rendering.

The async DB plumbing is covered by test_repository.py; here we focus on
the parts profile owns: figuring out which player to show (CLI args vs
settings fallback) and that the renderer surfaces the right numbers.

Mock contract:
    profile calls load_player_data (sync), so mocks use
    MagicMock returning PlayerData | None.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock, patch

import click
import pytest
import typer
from rich.console import Console

from valocoach.cli.commands.profile import (
    _resolve_identity,
    run_profile,
)
from valocoach.cli.formatter import (
    WARN_PREFIX,
)
from valocoach.cli.formatter import (
    render_identity_panel as _render_identity_panel,
)
from valocoach.cli.formatter import (
    render_summary_card as _render_summary_card,
)
from valocoach.data.loader import PlayerData
from valocoach.data.orm_models import Match, MatchPlayer, Player

# ---------------------------------------------------------------------------
# Helpers
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


def _player(
    *,
    name: str = "Yoursaviour01",
    tag: str = "SK04",
    region: str = "na",
    level: int = 240,
    current_tier_patched: str = "Gold 1",
    current_rr: int = 25,
    elo: int = 925,
    peak_tier_patched: str = "Gold 3",
    last_match_at: str | None = "2026-04-18T18:00:00+00:00",
) -> Player:
    return Player(
        puuid="p-tracked",
        riot_name=name,
        riot_tag=tag,
        region=region,
        platform="pc",
        account_level=level,
        current_tier=12,
        current_tier_patched=current_tier_patched,
        current_rr=current_rr,
        elo=elo,
        peak_tier=14,
        peak_tier_patched=peak_tier_patched,
        last_match_at=last_match_at,
    )


def _capture_console() -> Console:
    from valocoach.cli.display import THEME
    return Console(file=StringIO(), force_terminal=False, width=120, theme=THEME)


def _player_data(rows: list[MatchPlayer]) -> PlayerData:
    """Convenience: wrap player + rows into a PlayerData with no round data."""
    return PlayerData(player=_player(), rows=rows, full_matches=[])


# ---------------------------------------------------------------------------
# _resolve_identity
# ---------------------------------------------------------------------------


def test_resolve_uses_cli_args_when_given() -> None:
    out = _resolve_identity(
        name="Target",
        tag="TG01",
        settings_name="Self",
        settings_tag="SELF",
    )
    assert out == ("Target", "TG01")


def test_resolve_falls_back_to_settings_when_no_cli_args() -> None:
    out = _resolve_identity(
        name=None,
        tag=None,
        settings_name="Self",
        settings_tag="SELF",
    )
    assert out == ("Self", "SELF")


@pytest.mark.parametrize(
    ("name", "tag"),
    [
        ("OnlyName", None),
        (None, "OnlyTag"),
    ],
)
def test_resolve_rejects_half_given_args(name: str | None, tag: str | None) -> None:
    with pytest.raises(typer.BadParameter):
        _resolve_identity(
            name=name,
            tag=tag,
            settings_name="Self",
            settings_tag="SELF",
        )


def test_resolve_rejects_empty_settings_with_no_cli_args() -> None:
    with pytest.raises(typer.BadParameter):
        _resolve_identity(
            name=None,
            tag=None,
            settings_name="",
            settings_tag="",
        )


def test_resolve_cli_args_override_empty_settings() -> None:
    """User may not have configured a default — explicit --name/--tag still works."""
    out = _resolve_identity(
        name="Target",
        tag="TG01",
        settings_name="",
        settings_tag="",
    )
    assert out == ("Target", "TG01")


# ---------------------------------------------------------------------------
# _render_identity_panel
# ---------------------------------------------------------------------------


def test_identity_panel_shows_name_tag_region_rank() -> None:
    con = _capture_console()
    _render_identity_panel(con, _player())
    out = con.file.getvalue()
    assert "Yoursaviour01" in out
    assert "SK04" in out
    assert "NA" in out
    assert "Gold 1" in out  # current tier
    assert "25 RR" in out
    assert "Gold 3" in out  # peak
    assert "240" in out  # account level


def test_identity_panel_handles_never_synced_last_match() -> None:
    con = _capture_console()
    _render_identity_panel(con, _player(last_match_at=None))
    out = con.file.getvalue()
    assert "never" in out.lower()


# ---------------------------------------------------------------------------
# _render_summary_card
# ---------------------------------------------------------------------------


def test_summary_card_shows_record_and_key_stats() -> None:
    rows = [_mp()]
    con = _capture_console()
    _render_summary_card(con, rows, limit=20)
    out = con.file.getvalue()
    assert "Last 1 Match" in out
    assert "Record" in out
    assert "1-0" in out
    assert "ACS" in out
    assert "250" in out  # 5000/20
    assert "K/D" in out


def test_summary_card_empty_rows_shows_dim_message() -> None:
    con = _capture_console()
    _render_summary_card(con, [], limit=20)
    out = con.file.getvalue()
    assert "No matches" in out


def test_summary_card_limit_clamps_shown_count() -> None:
    """'Last N matches' title uses min(len(rows), limit) — don't claim 20 when 3."""
    rows = [_mp(match_id=f"m-{i}") for i in range(3)]
    con = _capture_console()
    _render_summary_card(con, rows, limit=20)
    out = con.file.getvalue()
    assert "Last 3 Match" in out
    assert "Last 20 Match" not in out


def test_summary_card_includes_fb_diff_with_sign() -> None:
    rows = [_mp(first_bloods=5, first_deaths=2)]
    con = _capture_console()
    _render_summary_card(con, rows, limit=20)
    out = con.file.getvalue()
    assert "+3" in out  # fb_diff with explicit sign


# ---------------------------------------------------------------------------
# Reliability warnings (⚠️)
# ---------------------------------------------------------------------------


def test_summary_card_warns_on_thin_sample() -> None:
    """Default --limit is 20. HS% and FB rate both need 30 matches — at
    20 the card should warn even though ACS (needs 15) and K/D (needs 20)
    are reliable."""
    rows = [_mp(match_id=f"m-{i}") for i in range(20)]
    con = _capture_console()
    any_warn = _render_summary_card(con, rows, limit=20)
    out = con.file.getvalue()
    assert any_warn is True
    assert WARN_PREFIX in out
    # HS% row must be tagged; ACS row must NOT be (20 >= 15).
    hs_line = next(line for line in out.splitlines() if "HS%" in line)
    acs_line = next(line for line in out.splitlines() if "ACS" in line)
    assert WARN_PREFIX.strip() in hs_line, f"HS% should warn at 20 matches: {hs_line!r}"
    assert WARN_PREFIX.strip() not in acs_line, f"ACS should NOT warn at 20 matches: {acs_line!r}"


def test_summary_card_no_warn_at_full_reliability() -> None:
    """At 30+ matches every threshold clears — clean card, no ⚠️."""
    rows = [_mp(match_id=f"m-{i}") for i in range(30)]
    con = _capture_console()
    any_warn = _render_summary_card(con, rows, limit=30)
    assert any_warn is False
    assert WARN_PREFIX not in con.file.getvalue()


def test_summary_card_empty_returns_false() -> None:
    """Empty-row branch returns False so the legend doesn't appear."""
    con = _capture_console()
    assert _render_summary_card(con, [], limit=20) is False


# ---------------------------------------------------------------------------
# run_profile end-to-end  (loader mocked at import point)
# ---------------------------------------------------------------------------


def _fake_settings(tmp_path):
    from valocoach.core.config import Settings

    return Settings(
        riot_name="Tester",
        riot_tag="NA1",
        riot_region="na",
        henrikdev_api_key="fake",
        data_dir=tmp_path,
    )


def test_run_profile_legend_only_when_warnings_fire(tmp_path) -> None:
    """Legend text appears iff any ⚠️ was rendered."""
    fake_settings = _fake_settings(tmp_path)
    thin_rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    thick_rows = [_mp(match_id=f"m-{i}") for i in range(30)]

    for rows, expect_legend in [(thin_rows, True), (thick_rows, False)]:
        data = _player_data(rows)
        con = _capture_console()
        with (
            patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
            patch(
                "valocoach.cli.commands.profile.load_player_data",
                new=MagicMock(return_value=data),
            ),
        ):
            run_profile(name="Tester", tag="NA1", console=con)
        out = con.file.getvalue()
        has_legend = "sample size" in out
        assert has_legend is expect_legend, (
            f"profile legend mismatch (rows={len(rows)}): "
            f"got has_legend={has_legend}, expected {expect_legend}"
        )


def test_run_profile_not_found_warns_and_exits(tmp_path) -> None:
    """None from the loader → warn and exit(1)."""
    fake_settings = _fake_settings(tmp_path)
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch(
            "valocoach.cli.commands.profile.load_player_data",
            new=MagicMock(return_value=None),
        ),
        pytest.raises(click.exceptions.Exit),
    ):
        run_profile(name="Ghost", tag="XX99")


def test_run_profile_passes_resolved_name_to_loader(tmp_path) -> None:
    """The loader must be called with the CLI --name/--tag, not the settings default."""
    fake_settings = _fake_settings(tmp_path)
    data = _player_data([_mp(match_id=f"m-{i}") for i in range(30)])
    mock_loader = MagicMock(return_value=data)

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=mock_loader),
    ):
        run_profile(name="TargetPlayer", tag="TG99", console=con)

    call_kwargs = mock_loader.call_args.kwargs
    assert call_kwargs["name"] == "TargetPlayer"
    assert call_kwargs["tag"] == "TG99"
    assert call_kwargs["include_rounds"] is True


def test_run_profile_uses_settings_identity_when_no_args(tmp_path) -> None:
    """No --name/--tag → loader receives settings.riot_name / riot_tag."""
    fake_settings = _fake_settings(tmp_path)
    data = _player_data([_mp(match_id=f"m-{i}") for i in range(30)])
    mock_loader = MagicMock(return_value=data)

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=mock_loader),
    ):
        run_profile(console=con)  # no name/tag

    call_kwargs = mock_loader.call_args.kwargs
    assert call_kwargs["name"] == "Tester"
    assert call_kwargs["tag"] == "NA1"


def test_run_profile_round_stats_shown_when_full_matches_present(tmp_path) -> None:
    """When PlayerData.full_matches is non-empty with rounds, KAST appears."""
    import json

    from valocoach.data.orm_models import Kill, Round

    fake_settings = _fake_settings(tmp_path)

    # Build a minimal full Match with one round so analyze_rounds finds something.
    from valocoach.data.orm_models import Match as OrmMatch

    full_match = OrmMatch(
        match_id="m-full",
        map_name="Ascent",
        map_id=None,
        queue_id="competitive",
        is_ranked=True,
        game_version=None,
        game_length_secs=0,
        season_short=None,
        region="na",
        rounds_played=1,
        red_score=0,
        blue_score=0,
        winning_team=None,
        started_at="2026-04-19T18:00:00+00:00",
    )
    rnd = Round(
        match_id="m-full",
        round_number=0,
        winning_team="Blue",
        result_code="Elimination",
        bomb_planted=False,
        plant_site=None,
        bomb_defused=False,
    )
    kill = Kill(
        round_id=0,
        match_id="m-full",
        round_number=0,
        time_in_round_ms=5000,
        killer_puuid="p-tracked",
        victim_puuid="p-enemy",
        weapon_name=None,
        is_headshot=True,
        assistants_json=json.dumps([]),
    )
    rnd.kills = [kill]
    full_match.rounds = [rnd]
    full_match.players = [
        MatchPlayer(
            match_id="m-full",
            puuid="p-tracked",
            agent_name="Jett",
            agent_id=None,
            team="Blue",
            won=True,
            score=0,
            kills=1,
            deaths=0,
            assists=0,
            rounds_played=1,
            headshots=1,
            bodyshots=0,
            legshots=0,
            damage_dealt=150,
            damage_received=0,
            first_bloods=1,
            first_deaths=0,
            plants=0,
            defuses=0,
            afk_rounds=0,
            rounds_in_spawn=0,
            competitive_tier=None,
            started_at="2026-04-19T18:00:00+00:00",
        ),
        MatchPlayer(
            match_id="m-full",
            puuid="p-enemy",
            agent_name="Reyna",
            agent_id=None,
            team="Red",
            won=False,
            score=0,
            kills=0,
            deaths=1,
            assists=0,
            rounds_played=1,
            headshots=0,
            bodyshots=0,
            legshots=0,
            damage_dealt=0,
            damage_received=150,
            first_bloods=0,
            first_deaths=0,
            plants=0,
            defuses=0,
            afk_rounds=0,
            rounds_in_spawn=0,
            competitive_tier=None,
            started_at="2026-04-19T18:00:00+00:00",
        ),
    ]

    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[full_match])

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch(
            "valocoach.cli.commands.profile.load_player_data",
            new=MagicMock(return_value=data),
        ),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    out = con.file.getvalue()
    assert "KAST" in out, "Round-level stats section should appear when full_matches present"


def test_run_profile_no_round_stats_when_full_matches_empty(tmp_path) -> None:
    """When full_matches=[], the KAST section is silently skipped."""
    fake_settings = _fake_settings(tmp_path)
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[])

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch(
            "valocoach.cli.commands.profile.load_player_data",
            new=MagicMock(return_value=data),
        ),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    assert "KAST" not in con.file.getvalue()


def test_run_profile_no_round_stats_when_full_matches_have_zero_rounds(tmp_path) -> None:
    """full_matches non-empty but match.rounds=[] → analyze_rounds returns rounds=0
    → the KAST section is silently skipped (line 149->156)."""
    from valocoach.data.orm_models import Match as OrmMatch

    fake_settings = _fake_settings(tmp_path)

    # Match with no rounds — analyze_rounds will return rounds=0.
    empty_match = OrmMatch(
        match_id="m-norounds",
        map_name="Ascent",
        map_id=None,
        queue_id="competitive",
        is_ranked=True,
        game_version=None,
        game_length_secs=0,
        season_short=None,
        region="na",
        rounds_played=0,
        red_score=0,
        blue_score=0,
        winning_team=None,
        started_at="2026-04-19T18:00:00+00:00",
    )
    empty_match.rounds = []  # no rounds → _tally_round never called → acc.rounds=0
    empty_match.players = []

    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[empty_match])

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch(
            "valocoach.cli.commands.profile.load_player_data",
            new=MagicMock(return_value=data),
        ),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    # KAST must NOT appear because round_analysis.rounds == 0.
    assert "KAST" not in con.file.getvalue()


def test_run_profile_breakdown_shown_when_multiple_agents(tmp_path) -> None:
    """When per_agent has ≥ 2 entries, the breakdown section is rendered (lines 158-159)."""
    fake_settings = _fake_settings(tmp_path)

    # Mix of two different agents so compute_per_agent returns 2+ rows.
    rows = [_mp(agent="Jett", match_id=f"m-jett-{i}") for i in range(5)] + [
        _mp(agent="Reyna", match_id=f"m-reyna-{i}") for i in range(5)
    ]
    data = PlayerData(player=_player(), rows=rows, full_matches=[])

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch(
            "valocoach.cli.commands.profile.load_player_data",
            new=MagicMock(return_value=data),
        ),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    out = con.file.getvalue()
    # The "Top agents" breakdown table must appear.
    assert "Top Agents" in out
    # Both agents should appear in the breakdown.
    assert "Jett" in out
    assert "Reyna" in out


# ---------------------------------------------------------------------------
# Phase B — coaching section coverage
# ---------------------------------------------------------------------------


def test_run_profile_coaching_sessions_rendered(tmp_path) -> None:
    """When list_coaching_sessions returns data, the sessions table is shown."""
    from valocoach.coach.session_manager import SessionInfo

    fake_settings = _fake_settings(tmp_path)
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[])

    sessions = [
        SessionInfo(id=3, title="Post-plant drill", started_at="2026-05-06T10:00:00",
                    ended_at="2026-05-06T12:00:00", focus_agent=None, focus_map=None),
    ]

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=MagicMock(return_value=data)),
        patch("valocoach.cli.commands.profile.list_coaching_sessions", return_value=sessions),
        patch("valocoach.cli.commands.profile.list_open_notes", return_value=[]),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    out = con.file.getvalue()
    assert "Post-plant drill" in out or "coaching sessions" in out.lower()


def test_run_profile_open_notes_rendered(tmp_path) -> None:
    """When list_open_notes returns data, the notes table is shown."""
    from valocoach.coach.session_manager import NoteInfo

    fake_settings = _fake_settings(tmp_path)
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[])

    notes = [
        NoteInfo(id=7, body="Work on crossfire at A long", category="tactical",
                 priority=2, created_at="2026-05-06T10:00:00"),
    ]

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=MagicMock(return_value=data)),
        patch("valocoach.cli.commands.profile.list_coaching_sessions", return_value=[]),
        patch("valocoach.cli.commands.profile.list_open_notes", return_value=notes),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    out = con.file.getvalue()
    assert "Work on crossfire" in out or "coaching notes" in out.lower()


def test_run_profile_coaching_section_exception_silently_skipped(tmp_path) -> None:
    """If list_coaching_sessions raises, the profile still renders without crashing."""
    fake_settings = _fake_settings(tmp_path)
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[])

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=MagicMock(return_value=data)),
        patch("valocoach.cli.commands.profile.list_coaching_sessions",
              side_effect=RuntimeError("db down")),
        patch("valocoach.cli.commands.profile.list_open_notes", return_value=[]),
    ):
        run_profile(name="Tester", tag="NA1", console=con)  # must not raise

    # Profile card still rendered
    assert "Yoursaviour01" in con.file.getvalue()


def test_run_profile_invalid_limit_raises(tmp_path) -> None:
    """limit <= 0 raises typer.BadParameter (line 109)."""
    fake_settings = _fake_settings(tmp_path)
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        pytest.raises(typer.BadParameter),
    ):
        run_profile(limit=0)


# ---------------------------------------------------------------------------
# Rank-trend section (Phase C wiring)
# ---------------------------------------------------------------------------


def test_run_profile_rank_trend_rendered_when_history_present(tmp_path) -> None:
    """When get_mmr_trend returns 2+ snapshots, render_rank_trend fires."""
    from valocoach.coach.session_manager import MMRHistoryInfo

    fake_settings = _fake_settings(tmp_path)
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[])

    history = [
        MMRHistoryInfo(tier_patched="Plat I", rr=20, elo=1420, mmr_change=20, recorded_at="2026-05-06"),
        MMRHistoryInfo(tier_patched="Gold II", rr=60, elo=1260, mmr_change=-10, recorded_at="2026-05-01"),
    ]

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=MagicMock(return_value=data)),
        patch("valocoach.cli.commands.profile.get_mmr_trend", return_value=history),
        patch("valocoach.cli.commands.profile.list_coaching_sessions", return_value=[]),
        patch("valocoach.cli.commands.profile.list_open_notes", return_value=[]),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    out = con.file.getvalue()
    # render_rank_trend writes "Rank trend" and the tier names
    assert "Rank Trend" in out
    assert "Plat I" in out or "Gold II" in out


def test_run_profile_rank_trend_skipped_when_no_history(tmp_path) -> None:
    """When get_mmr_trend returns [], the rank trend block is absent."""
    fake_settings = _fake_settings(tmp_path)
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[])

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=MagicMock(return_value=data)),
        patch("valocoach.cli.commands.profile.get_mmr_trend", return_value=[]),
        patch("valocoach.cli.commands.profile.list_coaching_sessions", return_value=[]),
        patch("valocoach.cli.commands.profile.list_open_notes", return_value=[]),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    assert "Rank Trend" not in con.file.getvalue()


def test_run_profile_rank_trend_skipped_with_single_snapshot(tmp_path) -> None:
    """A single MMR snapshot produces no trend output (needs 2+ to compare)."""
    from valocoach.coach.session_manager import MMRHistoryInfo

    fake_settings = _fake_settings(tmp_path)
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[])

    history = [
        MMRHistoryInfo(tier_patched="Gold II", rr=55, elo=1255, mmr_change=None, recorded_at="2026-05-06"),
    ]

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=MagicMock(return_value=data)),
        patch("valocoach.cli.commands.profile.get_mmr_trend", return_value=history),
        patch("valocoach.cli.commands.profile.list_coaching_sessions", return_value=[]),
        patch("valocoach.cli.commands.profile.list_open_notes", return_value=[]),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    assert "Rank Trend" not in con.file.getvalue()


def test_run_profile_rank_trend_elo_delta_shown(tmp_path) -> None:
    """ELO delta is visible in the rank trend line."""
    from valocoach.coach.session_manager import MMRHistoryInfo

    fake_settings = _fake_settings(tmp_path)
    rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    data = PlayerData(player=_player(), rows=rows, full_matches=[])

    history = [
        MMRHistoryInfo(tier_patched="Plat I", rr=10, elo=1410, mmr_change=50, recorded_at="2026-05-06"),
        MMRHistoryInfo(tier_patched="Gold III", rr=80, elo=1280, mmr_change=30, recorded_at="2026-04-20"),
    ]

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=MagicMock(return_value=data)),
        patch("valocoach.cli.commands.profile.get_mmr_trend", return_value=history),
        patch("valocoach.cli.commands.profile.list_coaching_sessions", return_value=[]),
        patch("valocoach.cli.commands.profile.list_open_notes", return_value=[]),
    ):
        run_profile(name="Tester", tag="NA1", console=con)

    out = con.file.getvalue()
    # delta = 1410 - 1280 = +130
    assert "+130" in out


# ---------------------------------------------------------------------------
# Phase D — live lookup (ephemeral, no DB)
# ---------------------------------------------------------------------------


def _api_match(
    *,
    name: str = "TargetPlayer",
    tag: str = "TG99",
    team: str = "Blue",
    won: bool = True,
    rounds_played: int = 20,
    score: int = 5000,
    kills: int = 20,
    deaths: int = 10,
    assists: int = 5,
    headshots: int = 8,
    bodyshots: int = 50,
    legshots: int = 2,
    damage_dealt: int = 3000,
):
    """Build a minimal api_models.MatchData for testing."""
    from valocoach.data.api_models import (
        MatchData,
        MatchMetadata,
        MatchPlayer as ApiMatchPlayer,
        MatchPlayers,
        MatchTeams,
        PlayerStats as ApiPlayerStats,
        TeamResult,
    )

    player = ApiMatchPlayer(
        puuid="p-target",
        name=name,
        tag=tag,
        team=team,
        character="Jett",
        stats=ApiPlayerStats(
            score=score,
            kills=kills,
            deaths=deaths,
            assists=assists,
            headshots=headshots,
            bodyshots=bodyshots,
            legshots=legshots,
            damage_dealt=damage_dealt,
        ),
    )

    red_won = team == "Red" and won
    blue_won = team == "Blue" and won

    return MatchData(
        metadata=MatchMetadata(
            match_id="m-api-1",
            map_name="Ascent",
            queue_id="competitive",
            rounds_played=rounds_played,
        ),
        players=MatchPlayers(all_players=[player]),
        teams=MatchTeams(
            red=TeamResult(has_won=red_won),
            blue=TeamResult(has_won=blue_won),
        ),
    )


# ---------------------------------------------------------------------------
# compute_stats_from_api
# ---------------------------------------------------------------------------


def test_compute_stats_from_api_basic() -> None:
    from valocoach.stats.calculator import compute_stats_from_api

    matches = [_api_match(), _api_match(won=False)]
    stats = compute_stats_from_api(matches, "TargetPlayer", "TG99")

    assert stats.matches == 2
    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.kills == 40
    assert stats.deaths == 20
    assert stats.kd == 2.0
    assert stats.rounds == 40
    assert stats.acs == 250.0  # 10000 / 40


def test_compute_stats_from_api_case_insensitive() -> None:
    from valocoach.stats.calculator import compute_stats_from_api

    matches = [_api_match(name="TargetPlayer", tag="TG99")]
    stats = compute_stats_from_api(matches, "targetplayer", "tg99")
    assert stats.matches == 1


def test_compute_stats_from_api_empty() -> None:
    from valocoach.stats.calculator import compute_stats_from_api

    stats = compute_stats_from_api([], "Nobody", "XX")
    assert stats.matches == 0
    assert stats.acs == 0.0


def test_compute_stats_from_api_player_not_in_match() -> None:
    from valocoach.stats.calculator import compute_stats_from_api

    matches = [_api_match(name="OtherPlayer", tag="OP01")]
    stats = compute_stats_from_api(matches, "TargetPlayer", "TG99")
    assert stats.matches == 0


def test_compute_stats_from_api_damage_from_rounds() -> None:
    """When player-level damage_dealt is 0, damage is summed from round data."""
    from valocoach.data.api_models import RoundData, RoundPlayerStats
    from valocoach.stats.calculator import compute_stats_from_api

    match = _api_match(damage_dealt=0, rounds_played=2)
    match.rounds = [
        RoundData(
            winning_team="Blue",
            end_type="Elimination",
            player_stats=[RoundPlayerStats(puuid="p-target", damage=150)],
        ),
        RoundData(
            winning_team="Red",
            end_type="Elimination",
            player_stats=[RoundPlayerStats(puuid="p-target", damage=120)],
        ),
    ]

    stats = compute_stats_from_api([match], "TargetPlayer", "TG99")
    assert stats.adr == 135.0  # (150 + 120) / 2 rounds


# ---------------------------------------------------------------------------
# render_lookup_identity_panel
# ---------------------------------------------------------------------------


def test_render_lookup_identity_panel_shows_rank_and_region() -> None:
    from valocoach.cli.formatter import render_lookup_identity_panel
    from valocoach.data.api_models import (
        AccountResponse,
        CurrentRankData,
        HighestRank,
        MMRData as ApiMMR,
    )

    account = AccountResponse(
        puuid="p-1", region="na", account_level=150, name="Lookup", tag="LU01",
    )
    mmr = ApiMMR(
        name="Lookup",
        tag="LU01",
        current_data=CurrentRankData(
            currenttierpatched="Platinum 2",
            ranking_in_tier=45,
            elo=1545,
            mmr_change_to_last_game=22,
        ),
        highest_rank=HighestRank(patched_tier="Diamond 1", season="e8a3"),
    )

    con = _capture_console()
    render_lookup_identity_panel(con, account, mmr)
    out = con.file.getvalue()

    assert "Lookup" in out
    assert "LU01" in out
    assert "Platinum 2" in out
    assert "45 RR" in out
    assert "Diamond 1" in out
    assert "NA" in out
    assert "150" in out
    assert "+22" in out


# ---------------------------------------------------------------------------
# render_lookup_summary
# ---------------------------------------------------------------------------


def test_render_lookup_summary_shows_stats() -> None:
    from valocoach.cli.formatter import render_lookup_summary
    from valocoach.stats.calculator import compute_stats_from_api

    matches = [_api_match() for _ in range(3)]
    stats = compute_stats_from_api(matches, "TargetPlayer", "TG99")

    con = _capture_console()
    render_lookup_summary(con, stats)
    out = con.file.getvalue()

    assert "3 Match" in out
    assert "250" in out  # ACS
    assert "2.00" in out  # K/D


def test_render_lookup_summary_returns_warn_flag() -> None:
    from valocoach.cli.formatter import render_lookup_summary
    from valocoach.stats.calculator import compute_stats_from_api

    matches = [_api_match() for _ in range(3)]
    stats = compute_stats_from_api(matches, "TargetPlayer", "TG99")

    con = _capture_console()
    any_warn = render_lookup_summary(con, stats)
    assert isinstance(any_warn, bool)


def test_render_lookup_summary_empty() -> None:
    from valocoach.cli.formatter import render_lookup_summary
    from valocoach.stats.calculator import _zero_stats

    con = _capture_console()
    result = render_lookup_summary(con, _zero_stats())
    out = con.file.getvalue()
    assert "No recent competitive matches" in out
    assert result is False


# ---------------------------------------------------------------------------
# run_profile fallback to run_lookup
# ---------------------------------------------------------------------------


def test_compute_per_agent_from_api() -> None:
    from valocoach.stats.calculator import compute_per_agent_from_api

    matches = [
        _api_match(name="P", tag="T"),
        _api_match(name="P", tag="T"),
    ]
    # Both are Jett (default) → single agent
    agents = compute_per_agent_from_api(matches, "P", "T")
    assert len(agents) == 1
    assert agents[0].agent == "Jett"
    assert agents[0].stats.matches == 2


def test_compute_per_agent_from_api_multiple_agents() -> None:
    from valocoach.data.api_models import (
        MatchData,
        MatchMetadata,
        MatchPlayer as ApiMatchPlayer,
        MatchPlayers,
        MatchTeams,
        PlayerStats as ApiPlayerStats,
        TeamResult,
    )
    from valocoach.stats.calculator import compute_per_agent_from_api

    def _make(agent: str):
        player = ApiMatchPlayer(
            puuid="p-1", name="P", tag="T", team="Blue", character=agent,
            stats=ApiPlayerStats(score=3000, kills=15, deaths=10, assists=3,
                                 headshots=5, bodyshots=30, legshots=2, damage_dealt=2000),
        )
        return MatchData(
            metadata=MatchMetadata(match_id="m-1", map_name="Ascent",
                                   queue_id="competitive", rounds_played=20),
            players=MatchPlayers(all_players=[player]),
            teams=MatchTeams(blue=TeamResult(has_won=True)),
        )

    matches = [_make("Jett"), _make("Jett"), _make("Reyna")]
    agents = compute_per_agent_from_api(matches, "P", "T")
    assert len(agents) == 2
    assert agents[0].agent == "Jett"  # sorted by matches desc
    assert agents[0].stats.matches == 2
    assert agents[1].agent == "Reyna"


def test_run_profile_falls_back_to_lookup_on_db_miss(tmp_path) -> None:
    """When --name/--tag given but no local data, profile falls back to live lookup."""
    from valocoach.cli.commands.profile import run_profile
    from valocoach.data.api_models import (
        AccountResponse,
        CurrentRankData,
        HighestRank,
        MMRData as ApiMMR,
    )

    fake_settings = _fake_settings(tmp_path)
    account = AccountResponse(
        puuid="p-ext", region="na", account_level=100, name="External", tag="EX01",
    )
    mmr = ApiMMR(
        name="External",
        tag="EX01",
        current_data=CurrentRankData(currenttierpatched="Silver 3", ranking_in_tier=30, elo=830),
        highest_rank=HighestRank(patched_tier="Gold 1"),
    )
    api_matches = [_api_match(name="External", tag="EX01")]

    con = _capture_console()
    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=MagicMock(return_value=None)),
        patch("valocoach.cli.commands.profile.asyncio") as mock_asyncio,
    ):
        mock_asyncio.run = MagicMock(return_value=(account, mmr, api_matches, []))
        run_profile(name="External", tag="EX01", console=con)

    out = con.file.getvalue()
    assert "External" in out
    assert "Silver 3" in out
    assert "Player Lookup" in out


def test_run_profile_no_fallback_without_explicit_name_tag(tmp_path) -> None:
    """When no --name/--tag and DB returns None, error as before (no fallback)."""
    fake_settings = _fake_settings(tmp_path)

    with (
        patch("valocoach.cli.commands.profile.load_settings", return_value=fake_settings),
        patch("valocoach.cli.commands.profile.load_player_data", new=MagicMock(return_value=None)),
        pytest.raises(click.exceptions.Exit),
    ):
        run_profile()  # no name/tag — uses settings identity, DB miss → error
