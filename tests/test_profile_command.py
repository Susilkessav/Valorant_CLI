"""Tests for `valocoach profile` — identity resolution + rendering.

The async DB plumbing is covered by test_repository.py; here we focus on
the parts profile owns: figuring out which player to show (CLI args vs
settings fallback) and that the renderer surfaces the right numbers.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest
import typer
from rich.console import Console

from valocoach.cli.commands.profile import (
    _render_identity_panel,
    _render_summary_card,
    _resolve_identity,
    run_profile,
)
from valocoach.cli.commands.stats import WARN_PREFIX
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
    return Console(file=StringIO(), force_terminal=False, width=120)


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
    assert "Last 1 match" in out
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
    assert "Last 3 match" in out
    assert "Last 20 match" not in out


def test_summary_card_includes_fb_diff_with_sign() -> None:
    rows = [_mp(first_bloods=5, first_deaths=2)]
    con = _capture_console()
    _render_summary_card(con, rows, limit=20)
    out = con.file.getvalue()
    assert "+3" in out  # fb_diff with explicit sign


# ---------------------------------------------------------------------------
# Reliability warnings (⚠️) — Phase C
# ---------------------------------------------------------------------------
#
# The profile card shares machinery with `valocoach stats`, but users interact
# with the two differently: stats is a deep dive, profile is at-a-glance. The
# ⚠️ signal matters *more* on profile because a glance is where small-sample
# numbers are most likely to be taken at face value.


def test_summary_card_warns_on_thin_sample() -> None:
    """Default --limit is 20. HS% and FB rate both need 30 matches — at
    20 the card should warn even though ACS (needs 15) and K/D (needs 20)
    are reliable. This is the realistic first-weeks user experience."""
    rows = [_mp(match_id=f"m-{i}") for i in range(20)]
    con = _capture_console()
    any_warn = _render_summary_card(con, rows, limit=20)
    out = con.file.getvalue()
    assert any_warn is True
    assert WARN_PREFIX in out
    # HS% row must be tagged; ACS row must NOT be (20 >= 15).
    hs_line = next(line for line in out.splitlines() if "HS%" in line)
    acs_line = next(line for line in out.splitlines() if "ACS" in line)
    assert "⚠️" in hs_line, f"HS% should warn at 20 matches: {hs_line!r}"
    assert "⚠️" not in acs_line, f"ACS should NOT warn at 20 matches: {acs_line!r}"


def test_summary_card_no_warn_at_full_reliability() -> None:
    """At 30+ matches every threshold clears — clean card, no ⚠️."""
    rows = [_mp(match_id=f"m-{i}") for i in range(30)]
    con = _capture_console()
    any_warn = _render_summary_card(con, rows, limit=30)
    assert any_warn is False
    assert WARN_PREFIX not in con.file.getvalue()


def test_summary_card_empty_returns_false() -> None:
    """Empty-row branch returns False so the legend doesn't appear on an
    otherwise-blank profile (no ⚠️ rendered → nothing to explain)."""
    con = _capture_console()
    assert _render_summary_card(con, [], limit=20) is False


def test_run_profile_legend_only_when_warnings_fire(tmp_path) -> None:
    """End-to-end: legend text appears iff any ⚠️ was rendered. Mirrors
    the same contract as `valocoach stats` so users see consistent
    behaviour across the two commands."""
    from valocoach.core.config import Settings

    fake_settings = Settings(
        riot_name="Tester",
        riot_tag="NA1",
        riot_region="na",
        henrikdev_api_key="fake",
        data_dir=tmp_path,
    )

    thin_rows = [_mp(match_id=f"m-{i}") for i in range(5)]
    thick_rows = [_mp(match_id=f"m-{i}") for i in range(30)]

    async def _fetch_thin(**kwargs: object):
        return _player(), thin_rows

    async def _fetch_thick(**kwargs: object):
        return _player(), thick_rows

    for fetch, expect_legend in [(_fetch_thin, True), (_fetch_thick, False)]:
        con = _capture_console()
        with (
            patch(
                "valocoach.cli.commands.profile.load_settings",
                return_value=fake_settings,
            ),
            patch(
                "valocoach.cli.commands.profile._fetch_profile_data",
                side_effect=fetch,
            ),
        ):
            run_profile(name="Tester", tag="NA1", console=con)
        out = con.file.getvalue()
        has_legend = "sample-size threshold" in out
        assert has_legend is expect_legend, (
            f"profile legend mismatch (fetch={fetch.__name__}): "
            f"got has_legend={has_legend}, expected {expect_legend}"
        )
