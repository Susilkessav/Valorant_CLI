"""Golden dataset tests — confidence that the formulas produce correct numbers.

Unlike the unit tests (which test one behaviour in isolation) these tests
verify the COMPLETE computation pipeline against a fully hand-specified
dataset where every expected value was derived independently.

If any of these tests fail the answer to "is the bug in the test or the code?"
is almost always "the code" — because the expected values below were calculated
by hand using nothing but arithmetic.

Dataset overview
----------------
Three match-player rows (A, B, C) for player ``P_HERO = "p-hero"``:

    Match A — Ascent, win,  20 rounds, Jett,    score=4 000, kills=18, deaths=8,  assists=4
    Match B — Haven,  loss, 25 rounds, Jett,    score=4 500, kills=15, deaths=14, assists=6
    Match C — Ascent, win,  13 rounds, Phoenix, score=2 860, kills=10, deaths=4,  assists=2

Aggregate (A+B+C):
    rounds       = 58          (20+25+13)
    wins         = 2
    acs          = 11360/58    ≈ 195.862   (total score / total rounds)
    adr          = 8580/58     ≈ 147.931   (total damage / total rounds)
    kills        = 43
    deaths       = 26
    kd           = 43/26       ≈ 1.6538
    kda          = 55/26       ≈ 2.1154    ((kills+assists)/deaths)
    hs_pct       = 84/260      ≈ 0.3231    (headshots / total shots)
    first_bloods = 9
    first_deaths = 6
    fb_diff      = 3
    plants       = 3
    defuses      = 1

Round-level golden (Match D — 5 rounds, fully specified):
    KAST    = 4/5 = 80.0 %
    traded  = 1/2 deaths → trade_efficiency = 50 %
    trades_given = 1/1 teammate_deaths → trade_participation = 100 %
    double_kills = 1

Tolerance:  floating-point equality within 1e-6.
"""

from __future__ import annotations

import json

import pytest

from valocoach.data.orm_models import Kill, Match, MatchPlayer, Round
from valocoach.stats.calculator import compute_per_agent, compute_per_map, compute_player_stats
from valocoach.stats.filters import (
    filter_by_agent,
    filter_by_map,
    filter_by_result,
    split_by_result,
)
from valocoach.stats.round_analyzer import analyze_rounds

# ---------------------------------------------------------------------------
# Constants for the golden dataset
# ---------------------------------------------------------------------------

P_HERO = "p-hero"
EPS = 1e-6  # tolerance for float comparisons

TEAMMATES = [f"gold-m{i}" for i in range(1, 5)]  # M1..M4
ENEMIES = [f"gold-e{i}" for i in range(1, 6)]  # E1..E5


# ---------------------------------------------------------------------------
# Match-player fixture builders
# ---------------------------------------------------------------------------


def _mp(
    *,
    match_id: str,
    map_name: str,
    won: bool,
    agent: str,
    rounds_played: int,
    score: int,
    kills: int,
    deaths: int,
    assists: int,
    headshots: int,
    bodyshots: int,
    legshots: int,
    damage_dealt: int,
    first_bloods: int,
    first_deaths: int,
    plants: int,
    defuses: int,
) -> MatchPlayer:
    """Build a fully-specified in-memory MatchPlayer + attached Match."""
    match = Match(
        match_id=match_id,
        map_name=map_name,
        map_id=None,
        queue_id="competitive",
        is_ranked=True,
        game_version=None,
        game_length_secs=0,
        season_short=None,
        region="na",
        rounds_played=rounds_played,
        red_score=0,
        blue_score=0,
        winning_team=None,
        started_at="2026-04-19T18:00:00+00:00",
    )
    mp = MatchPlayer(
        match_id=match_id,
        puuid=P_HERO,
        agent_name=agent,
        agent_id=None,
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
        plants=plants,
        defuses=defuses,
        afk_rounds=0,
        rounds_in_spawn=0,
        competitive_tier=None,
        credits_spent=None,
        started_at="2026-04-19T18:00:00+00:00",
    )
    mp.match = match
    return mp


# ---------------------------------------------------------------------------
# The three golden match-player rows
# ---------------------------------------------------------------------------
#
# All expected values in the test bodies below were calculated by hand.
# Do not compute them from the same formula being tested.

_MATCH_A = _mp(
    match_id="gold-a",
    map_name="Ascent",
    won=True,
    agent="Jett",
    rounds_played=20,
    score=4_000,
    kills=18,
    deaths=8,
    assists=4,
    headshots=36,  # 36/100 shots = 36% HS
    bodyshots=54,
    legshots=10,
    damage_dealt=3_000,
    first_bloods=4,
    first_deaths=1,
    plants=2,
    defuses=0,
)

_MATCH_B = _mp(
    match_id="gold-b",
    map_name="Haven",
    won=False,
    agent="Jett",
    rounds_played=25,
    score=4_500,
    kills=15,
    deaths=14,
    assists=6,
    headshots=28,  # 28/100 shots = 28% HS
    bodyshots=56,
    legshots=16,
    damage_dealt=3_500,
    first_bloods=3,
    first_deaths=5,
    plants=0,
    defuses=1,
)

_MATCH_C = _mp(
    match_id="gold-c",
    map_name="Ascent",
    won=True,
    agent="Phoenix",
    rounds_played=13,
    score=2_860,
    kills=10,
    deaths=4,
    assists=2,
    headshots=20,  # 20/60 shots = 33.3% HS
    bodyshots=30,
    legshots=10,
    damage_dealt=2_080,
    first_bloods=2,
    first_deaths=0,
    plants=1,
    defuses=0,
)

ALL_ROWS = [_MATCH_A, _MATCH_B, _MATCH_C]


# ---------------------------------------------------------------------------
# Round-level fixture builders (for Match D — KAST / trade / clutch)
# ---------------------------------------------------------------------------


def _kill(
    *,
    killer: str,
    victim: str,
    t_ms: int | None,
    assistants: list[str] | None = None,
) -> Kill:
    return Kill(
        round_id=0,
        match_id="gold-d",
        round_number=0,
        time_in_round_ms=t_ms,
        killer_puuid=killer,
        victim_puuid=victim,
        weapon_name=None,
        is_headshot=False,
        assistants_json=json.dumps(assistants or []),
    )


def _round(round_number: int, winning_team: str, kills: list[Kill]) -> Round:
    r = Round(
        match_id="gold-d",
        round_number=round_number,
        winning_team=winning_team,
        result_code="Elimination",
        bomb_planted=False,
        plant_site=None,
        bomb_defused=False,
    )
    r.kills = kills
    return r


def _build_gold_match_d() -> Match:
    """Build Match D — 5 fully-specified rounds for round-level golden tests.

    Round 0 (Blue wins):
        P kills E1 at 1 000ms, P kills E2 at 2 000ms.
        → K=True (rounds_with_kill), double_kill=1, S=True (rounds_survived).

    Round 1 (Red wins):
        E3 kills P at 10 000ms; M1 kills E3 at 13 000ms (3s later, within 5s).
        → T=True (traded_death), KAST via T.

    Round 2 (Blue wins):
        M1 kills E4; P is listed as an assistant; P survives.
        → A=True, S=True.

    Round 3 (Red wins):
        E1 kills P at 5 000ms; no avenge within 5 000ms.
        → K/A/S/T all False → NOT KAST.

    Round 4 (Blue wins):
        E2 kills M2 at 5 000ms; P kills E2 at 7 000ms (2s later, within 5s).
        P survives.
        → S=True, trades_given=1 (P avenges teammate), K=True.

    Aggregate expectations
    ~~~~~~~~~~~~~~~~~~~~~~
    rounds              = 5
    deaths              = 2       (rounds 1 and 3)
    traded_deaths       = 1       (round 1 only)
    rounds_kast         = 4       (rounds 0, 1, 2, 4)
    kast_pct            = 0.80
    trade_efficiency    = 0.50    (1/2 deaths traded)
    teammate_deaths     = 1       (M2 in round 4)
    trades_given        = 1       (P avenges M2)
    trade_participation = 1.00    (1/1 teammate deaths)
    double_kills        = 1       (round 0)
    rounds_with_kill    = 2       (rounds 0 and 4)
    rounds_survived     = 3       (rounds 0, 2, 4)
    rounds_with_assist  = 1       (round 2)
    rounds_traded_death = 1       (round 1)
    """
    rounds = [
        # Round 0: P double-kill, survives
        _round(
            0,
            "Blue",
            [
                _kill(killer=P_HERO, victim=ENEMIES[0], t_ms=1_000),
                _kill(killer=P_HERO, victim=ENEMIES[1], t_ms=2_000),
            ],
        ),
        # Round 1: P dies, M1 avenges within window
        _round(
            1,
            "Red",
            [
                _kill(killer=ENEMIES[2], victim=P_HERO, t_ms=10_000),
                _kill(killer=TEAMMATES[0], victim=ENEMIES[2], t_ms=13_000),
            ],
        ),
        # Round 2: P assists M1's kill, survives
        _round(
            2,
            "Blue",
            [
                _kill(killer=TEAMMATES[0], victim=ENEMIES[3], t_ms=5_000, assistants=[P_HERO]),
            ],
        ),
        # Round 3: P dies, no avenge — NOT KAST
        _round(
            3,
            "Red",
            [
                _kill(killer=ENEMIES[0], victim=P_HERO, t_ms=5_000),
            ],
        ),
        # Round 4: M2 dies, P avenges within window, P survives
        _round(
            4,
            "Blue",
            [
                _kill(killer=ENEMIES[1], victim=TEAMMATES[1], t_ms=5_000),
                _kill(killer=P_HERO, victim=ENEMIES[1], t_ms=7_000),
            ],
        ),
    ]

    m = Match(
        match_id="gold-d",
        map_name="Ascent",
        map_id=None,
        queue_id="competitive",
        is_ranked=True,
        game_version=None,
        game_length_secs=0,
        season_short=None,
        region="na",
        rounds_played=5,
        red_score=0,
        blue_score=0,
        winning_team=None,
        started_at="2026-04-20T18:00:00+00:00",
    )
    # Build player roster: P_HERO + 4 teammates + 5 enemies
    players = [
        MatchPlayer(
            match_id="gold-d",
            puuid=P_HERO,
            agent_name="Jett",
            agent_id=None,
            team="Blue",
            won=True,
            score=0,
            kills=0,
            deaths=0,
            assists=0,
            rounds_played=5,
            headshots=0,
            bodyshots=0,
            legshots=0,
            damage_dealt=0,
            damage_received=0,
            first_bloods=0,
            first_deaths=0,
            plants=0,
            defuses=0,
            afk_rounds=0,
            rounds_in_spawn=0,
            competitive_tier=None,
            started_at="2026-04-20T18:00:00+00:00",
        )
    ]
    for t in TEAMMATES:
        players.append(
            MatchPlayer(
                match_id="gold-d",
                puuid=t,
                agent_name="Sage",
                agent_id=None,
                team="Blue",
                won=True,
                score=0,
                kills=0,
                deaths=0,
                assists=0,
                rounds_played=5,
                headshots=0,
                bodyshots=0,
                legshots=0,
                damage_dealt=0,
                damage_received=0,
                first_bloods=0,
                first_deaths=0,
                plants=0,
                defuses=0,
                afk_rounds=0,
                rounds_in_spawn=0,
                competitive_tier=None,
                started_at="2026-04-20T18:00:00+00:00",
            )
        )
    for e in ENEMIES:
        players.append(
            MatchPlayer(
                match_id="gold-d",
                puuid=e,
                agent_name="Reyna",
                agent_id=None,
                team="Red",
                won=False,
                score=0,
                kills=0,
                deaths=0,
                assists=0,
                rounds_played=5,
                headshots=0,
                bodyshots=0,
                legshots=0,
                damage_dealt=0,
                damage_received=0,
                first_bloods=0,
                first_deaths=0,
                plants=0,
                defuses=0,
                afk_rounds=0,
                rounds_in_spawn=0,
                competitive_tier=None,
                started_at="2026-04-20T18:00:00+00:00",
            )
        )
    m.players = players
    m.rounds = rounds
    return m


MATCH_D = _build_gold_match_d()


# ---------------------------------------------------------------------------
# Section 1 — Overall aggregate stats
# ---------------------------------------------------------------------------


class TestGoldenOverall:
    """Verify aggregate stats from 3 fully-specified matches.

    Every expected value here was calculated by hand — none are derived from
    the same formula used in calculator.py.
    """

    @pytest.fixture(autouse=True)
    def _stats(self) -> None:
        self.s = compute_player_stats(ALL_ROWS)

    # ---- denominators ----

    def test_match_count(self) -> None:
        assert self.s.matches == 3

    def test_round_count(self) -> None:
        # 20 + 25 + 13 = 58
        assert self.s.rounds == 58

    # ---- outcomes ----

    def test_wins_and_losses(self) -> None:
        assert self.s.wins == 2
        assert self.s.losses == 1

    def test_win_rate(self) -> None:
        # 2/3
        assert abs(self.s.win_rate - 2 / 3) < EPS

    # ---- per-round combat ----

    def test_acs(self) -> None:
        # ACS = total score / total rounds = (4000+4500+2860) / 58 = 11360/58
        expected = 11_360 / 58
        assert abs(self.s.acs - expected) < EPS

    def test_adr(self) -> None:
        # ADR = total damage / total rounds = (3000+3500+2080) / 58 = 8580/58
        expected = 8_580 / 58
        assert abs(self.s.adr - expected) < EPS

    # ---- KDA ----

    def test_kills(self) -> None:
        assert self.s.kills == 43  # 18+15+10

    def test_deaths(self) -> None:
        assert self.s.deaths == 26  # 8+14+4

    def test_assists(self) -> None:
        assert self.s.assists == 12  # 4+6+2

    def test_kd(self) -> None:
        # K/D = 43/26
        assert abs(self.s.kd - 43 / 26) < EPS

    def test_kda(self) -> None:
        # KDA = (kills+assists)/deaths = (43+12)/26 = 55/26
        assert abs(self.s.kda - 55 / 26) < EPS

    # ---- accuracy ----

    def test_headshots(self) -> None:
        assert self.s.headshots == 84  # 36+28+20

    def test_bodyshots(self) -> None:
        assert self.s.bodyshots == 140  # 54+56+30

    def test_legshots(self) -> None:
        assert self.s.legshots == 36  # 10+16+10

    def test_hs_pct(self) -> None:
        # HS% = headshots / total_shots = 84 / (84+140+36) = 84/260
        assert abs(self.s.hs_pct - 84 / 260) < EPS

    # ---- impact ----

    def test_first_bloods(self) -> None:
        assert self.s.first_bloods == 9  # 4+3+2

    def test_first_deaths(self) -> None:
        assert self.s.first_deaths == 6  # 1+5+0

    def test_fb_diff(self) -> None:
        assert self.s.fb_diff == 3  # 9-6

    def test_fb_rate(self) -> None:
        # fb_rate = first_bloods / rounds = 9/58
        assert abs(self.s.fb_rate - 9 / 58) < EPS

    def test_fd_rate(self) -> None:
        # fd_rate = first_deaths / rounds = 6/58
        assert abs(self.s.fd_rate - 6 / 58) < EPS

    # ---- objective ----

    def test_plants(self) -> None:
        assert self.s.plants == 3  # 2+0+1

    def test_defuses(self) -> None:
        assert self.s.defuses == 1  # 0+1+0


# ---------------------------------------------------------------------------
# Section 2 — Per-agent groupings
# ---------------------------------------------------------------------------


class TestGoldenPerAgent:
    """Verify per-agent aggregation. Jett (A+B) and Phoenix (C) must come out right."""

    @pytest.fixture(autouse=True)
    def _stats(self) -> None:
        self.by_agent = {a.agent: a.stats for a in compute_per_agent(ALL_ROWS)}

    def test_two_agents_detected(self) -> None:
        assert set(self.by_agent.keys()) == {"Jett", "Phoenix"}

    def test_jett_matches(self) -> None:
        assert self.by_agent["Jett"].matches == 2

    def test_jett_acs(self) -> None:
        # Jett ACS = (4000+4500) / (20+25) = 8500/45
        expected = 8_500 / 45
        assert abs(self.by_agent["Jett"].acs - expected) < EPS

    def test_jett_kd(self) -> None:
        # K/D = (18+15) / (8+14) = 33/22 = 1.5 exactly
        assert abs(self.by_agent["Jett"].kd - 33 / 22) < EPS

    def test_jett_hs_pct(self) -> None:
        # headshots: 36+28=64 / (100+100)=200 shots = 0.32
        assert abs(self.by_agent["Jett"].hs_pct - 64 / 200) < EPS

    def test_jett_win_rate(self) -> None:
        # Jett: 1 win (A), 1 loss (B) = 0.5
        assert abs(self.by_agent["Jett"].win_rate - 0.5) < EPS

    def test_phoenix_matches(self) -> None:
        assert self.by_agent["Phoenix"].matches == 1

    def test_phoenix_acs(self) -> None:
        # Phoenix ACS = 2860/13 = 220.0 exactly
        assert abs(self.by_agent["Phoenix"].acs - 2_860 / 13) < EPS

    def test_phoenix_kd(self) -> None:
        # K/D = 10/4 = 2.5 exactly
        assert abs(self.by_agent["Phoenix"].kd - 2.5) < EPS

    def test_phoenix_hs_pct(self) -> None:
        # 20 / 60 shots = 1/3
        assert abs(self.by_agent["Phoenix"].hs_pct - 20 / 60) < EPS

    def test_jett_sorted_first_by_matches(self) -> None:
        """Jett has 2 matches vs Phoenix's 1 — must appear first."""
        order = [a.agent for a in compute_per_agent(ALL_ROWS)]
        assert order[0] == "Jett"
        assert order[1] == "Phoenix"


# ---------------------------------------------------------------------------
# Section 3 — Per-map groupings
# ---------------------------------------------------------------------------


class TestGoldenPerMap:
    """Ascent (A+C) and Haven (B) exact stats."""

    @pytest.fixture(autouse=True)
    def _stats(self) -> None:
        self.by_map = {m.map_name: m.stats for m in compute_per_map(ALL_ROWS)}

    def test_two_maps_detected(self) -> None:
        assert set(self.by_map.keys()) == {"Ascent", "Haven"}

    def test_ascent_matches(self) -> None:
        assert self.by_map["Ascent"].matches == 2

    def test_ascent_acs(self) -> None:
        # Ascent ACS = (4000+2860) / (20+13) = 6860/33
        assert abs(self.by_map["Ascent"].acs - 6_860 / 33) < EPS

    def test_ascent_wins(self) -> None:
        # Both Ascent matches were wins
        assert self.by_map["Ascent"].wins == 2
        assert self.by_map["Ascent"].losses == 0

    def test_ascent_kd(self) -> None:
        # K/D = (18+10) / (8+4) = 28/12 = 7/3
        assert abs(self.by_map["Ascent"].kd - 28 / 12) < EPS

    def test_ascent_hs_pct(self) -> None:
        # headshots: 36+20=56 / (100+60)=160 shots = 7/20 = 0.35
        assert abs(self.by_map["Ascent"].hs_pct - 56 / 160) < EPS

    def test_haven_acs(self) -> None:
        # Haven ACS = 4500/25 = 180.0 exactly
        assert abs(self.by_map["Haven"].acs - 180.0) < EPS

    def test_haven_adr(self) -> None:
        # Haven ADR = 3500/25 = 140.0 exactly
        assert abs(self.by_map["Haven"].adr - 140.0) < EPS

    def test_haven_hs_pct(self) -> None:
        # 28/100 = 0.28 exactly
        assert abs(self.by_map["Haven"].hs_pct - 0.28) < EPS

    def test_ascent_sorted_first_by_matches(self) -> None:
        order = [m.map_name for m in compute_per_map(ALL_ROWS)]
        assert order[0] == "Ascent"
        assert order[1] == "Haven"


# ---------------------------------------------------------------------------
# Section 4 — Filter pipeline
# ---------------------------------------------------------------------------


class TestGoldenFilters:
    """filter_by_* applied to ALL_ROWS — exact expected subset stats."""

    def test_filter_by_agent_jett_gives_two_matches(self) -> None:
        jett = filter_by_agent(ALL_ROWS, "Jett")
        s = compute_player_stats(jett)
        assert s.matches == 2
        assert s.kills == 33

    def test_filter_by_agent_is_case_insensitive(self) -> None:
        assert len(filter_by_agent(ALL_ROWS, "jett")) == 2
        assert len(filter_by_agent(ALL_ROWS, "JETT")) == 2
        assert len(filter_by_agent(ALL_ROWS, "Jett")) == 2

    def test_filter_by_agent_phoenix_gives_one_match(self) -> None:
        phoenix = filter_by_agent(ALL_ROWS, "Phoenix")
        s = compute_player_stats(phoenix)
        assert s.matches == 1
        assert s.acs == pytest.approx(2_860 / 13, abs=EPS)

    def test_filter_by_map_ascent_gives_two_matches(self) -> None:
        ascent = filter_by_map(ALL_ROWS, "Ascent")
        s = compute_player_stats(ascent)
        assert s.matches == 2
        assert s.wins == 2

    def test_filter_by_map_is_case_insensitive(self) -> None:
        assert len(filter_by_map(ALL_ROWS, "ascent")) == 2
        assert len(filter_by_map(ALL_ROWS, "ASCENT")) == 2

    def test_filter_by_result_wins_gives_two_matches(self) -> None:
        wins = filter_by_result(ALL_ROWS, won=True)
        s = compute_player_stats(wins)
        assert s.matches == 2
        assert s.wins == 2
        assert s.losses == 0

    def test_filter_by_result_losses_gives_one_match(self) -> None:
        losses = filter_by_result(ALL_ROWS, won=False)
        s = compute_player_stats(losses)
        assert s.matches == 1
        assert s.losses == 1
        assert s.wins == 0
        # The single loss is Match B: ACS = 4500/25 = 180 exactly
        assert abs(s.acs - 180.0) < EPS

    def test_filter_by_result_none_is_passthrough(self) -> None:
        all_ = filter_by_result(ALL_ROWS, won=None)
        assert len(all_) == 3

    def test_split_by_result_combined_stats_equal_overall(self) -> None:
        """Wins + losses after split compute to the same aggregate as the full set."""
        wins, losses = split_by_result(ALL_ROWS)
        sw = compute_player_stats(wins)
        sl = compute_player_stats(losses)
        sa = compute_player_stats(ALL_ROWS)
        # Total kills must match
        assert sw.kills + sl.kills == sa.kills
        # Total rounds must match
        assert sw.rounds + sl.rounds == sa.rounds

    def test_stacked_filters_agent_and_result(self) -> None:
        """Jett losses → only Match B."""
        jett_rows = filter_by_agent(ALL_ROWS, "Jett")
        jett_losses = filter_by_result(jett_rows, won=False)
        s = compute_player_stats(jett_losses)
        assert s.matches == 1
        assert s.acs == pytest.approx(4_500 / 25, abs=EPS)

    def test_filter_unknown_agent_gives_empty(self) -> None:
        assert filter_by_agent(ALL_ROWS, "Neon") == []

    def test_filter_unknown_map_gives_empty(self) -> None:
        assert filter_by_map(ALL_ROWS, "Icebox") == []


# ---------------------------------------------------------------------------
# Section 5 — Round-level golden stats (KAST / trade / clutch)
# ---------------------------------------------------------------------------


class TestGoldenRoundLevel:
    """Verify round analyzer against Match D (5 fully-specified rounds).

    Round layout is documented in _build_gold_match_d() docstring above.
    These expected values were derived by hand-tracing the event sequence.
    """

    @pytest.fixture(autouse=True)
    def _analysis(self) -> None:
        self.a = analyze_rounds([MATCH_D], P_HERO)

    # ---- denominators ----

    def test_round_count(self) -> None:
        assert self.a.rounds == 5

    def test_death_count(self) -> None:
        # P dies in rounds 1 and 3 only.
        assert self.a.deaths == 2

    def test_teammate_death_count(self) -> None:
        # M2 dies in round 4 only.
        assert self.a.teammate_deaths == 1

    # ---- KAST components ----

    def test_rounds_with_kill(self) -> None:
        # Kills: round 0 (P kills E1, E2), round 4 (P kills E2)
        assert self.a.rounds_with_kill == 2

    def test_rounds_with_assist(self) -> None:
        # Assist: round 2 only
        assert self.a.rounds_with_assist == 1

    def test_rounds_survived(self) -> None:
        # Survived: rounds 0, 2, 4
        assert self.a.rounds_survived == 3

    def test_rounds_traded_death(self) -> None:
        # Traded: round 1 only (M1 avenges P within 3s)
        assert self.a.rounds_traded_death == 1

    def test_rounds_kast(self) -> None:
        # KAST rounds: 0(K+S), 1(T), 2(A+S), 4(K+S) = 4 rounds
        assert self.a.rounds_kast == 4

    def test_kast_pct(self) -> None:
        # 4/5 = 0.80
        assert abs(self.a.kast_pct - 0.80) < EPS

    # ---- trades ----

    def test_traded_deaths(self) -> None:
        # Round 1: P dies, M1 avenges within 3 000ms < 5 000ms → traded
        assert self.a.traded_deaths == 1

    def test_trade_efficiency(self) -> None:
        # 1 traded out of 2 deaths = 0.50
        assert abs(self.a.trade_efficiency - 0.50) < EPS

    def test_trades_given(self) -> None:
        # Round 4: P kills E2 within 2 000ms of M2's death → 1 trade given
        assert self.a.trades_given == 1

    def test_trade_participation(self) -> None:
        # 1 trade given out of 1 teammate death = 1.0
        assert abs(self.a.trade_participation - 1.0) < EPS

    # ---- multi-kills ----

    def test_double_kills(self) -> None:
        # Round 0: P kills 2 enemies → exactly one double-kill
        assert self.a.double_kills == 1

    def test_no_triple_or_higher(self) -> None:
        assert self.a.triple_kills == 0
        assert self.a.quadra_kills == 0
        assert self.a.aces == 0

    # ---- no clutch scenario in this dataset ----

    def test_no_clutch_opportunities(self) -> None:
        # P is never the last Blue player alive in any of these 5 rounds.
        assert self.a.clutch_opportunities == 0
        assert self.a.clutches_won == 0


# ---------------------------------------------------------------------------
# Section 6 — Single-value reference checks (guard against silent truncation)
# ---------------------------------------------------------------------------


class TestGoldenSpecificValues:
    """Pin exact values for a few key stats so regressions surface immediately.

    These are the numbers a user would see on screen. If they change, a human
    must have consciously changed the formula — not a side effect of refactoring.
    """

    def test_acs_pinned(self) -> None:
        s = compute_player_stats(ALL_ROWS)
        # 11360/58 = 195.862068965517...
        assert abs(s.acs - 195.862_068_965_517) < 1e-9

    def test_hs_pct_pinned(self) -> None:
        s = compute_player_stats(ALL_ROWS)
        # 84/260 = 0.323076923...
        assert abs(s.hs_pct - 0.323_076_923) < 1e-9

    def test_kd_pinned(self) -> None:
        s = compute_player_stats(ALL_ROWS)
        # 43/26 = 1.653846153...
        assert abs(s.kd - 1.653_846_153) < 1e-9

    def test_kast_pct_pinned(self) -> None:
        a = analyze_rounds([MATCH_D], P_HERO)
        # 4/5 = 0.80 exactly
        assert a.kast_pct == pytest.approx(0.80, abs=1e-12)


# ---------------------------------------------------------------------------
# Section 7 — ACS/ADR/HS% sanity check against a real API match
# ---------------------------------------------------------------------------
#
# Source: data/sample_match.json
#   match_id : b0c012f7-9a68-46d1-a527-32783a190a5c
#   map      : Lotus   queue: competitive   rounds: 17
#   result   : Red won 13-4
#
# Expected values were taken DIRECTLY from the raw API JSON — NOT computed
# from the calculator under test.  If any assertion below fails the bug is
# in the calculator (or the data mapping), not in the expected number.
#
# To re-derive: in data/sample_match.json
#   ACS  = stats.score / len(rounds)          = score / 17
#   ADR  = stats.damage.dealt / len(rounds)   = damage_dealt / 17
#   HS%  = stats.headshots / (hs+body+leg)
#
# These formulas match Riot's published definitions:
#   ACS  https://playvalorant.com/en-us/news/game-updates/valorant-systems-health-series-data-and-leaderboards/
#   ADR  damage dealt per round
#   HS%  headshots / total shots fired

_LOTUS_ROUNDS = 17  # len(d['rounds']) from sample_match.json

# (display_name, puuid, team, won, score, kills, deaths, assists,
#  headshots, bodyshots, legshots, damage_dealt, damage_received)
#
# Won  = True for Red players (Red won 13-4), False for Blue players.
_LOTUS_PLAYERS: list[tuple] = [
    (
        "dipp",
        "77135f96-5842-5724-a0e2-606026886cd0",
        "Red",
        True,
        3811,
        12,
        10,
        6,
        6,
        49,
        2,
        2430,
        1897,
    ),
    (
        "VBJ",
        "1941b3d8-5021-5506-b47e-644266397fd6",
        "Red",
        True,
        3422,
        12,
        9,
        5,
        7,
        37,
        4,
        2142,
        1724,
    ),
    (
        "Yoursaviour01",
        "20905543-1b42-5f6f-8435-ab284a0094f8",
        "Blue",
        False,
        4344,
        14,
        14,
        1,
        13,
        32,
        1,
        2843,
        2645,
    ),
    (
        "Alphaxenon",
        "533f3998-fbc2-5b8a-ad16-c89d05495020",
        "Blue",
        False,
        2786,
        8,
        17,
        2,
        7,
        28,
        1,
        2005,
        2643,
    ),
    (
        "sn0w",
        "1b213c58-1258-5ba3-8404-c1e2130305fd",
        "Blue",
        False,
        3178,
        10,
        15,
        2,
        4,
        28,
        0,
        2203,
        2478,
    ),
    (
        "Yoursaviour02",
        "e8523cfa-a69b-553f-960a-2269864cefc7",
        "Blue",
        False,
        2346,
        7,
        15,
        4,
        7,
        27,
        3,
        1632,
        2376,
    ),
    (
        "fairywink",
        "8df719e4-2200-5510-ba82-d9bef41d5d8f",
        "Red",
        True,
        4225,
        15,
        8,
        9,
        9,
        39,
        3,
        2767,
        1987,
    ),
    (
        "Shortbread",
        "11d29861-9334-568d-8ffe-e25088e765c6",
        "Red",
        True,
        3684,
        15,
        8,
        11,
        7,
        40,
        2,
        2236,
        2113,
    ),
    (
        "ElmoHadCrack",
        "9edd92d2-c582-5bf4-8950-7107b74c9450",
        "Red",
        True,
        4900,
        20,
        10,
        5,
        11,
        27,
        3,
        2894,
        2330,
    ),
    (
        "ThunderMahou",
        "759a5eac-2faf-5417-9591-ba8cad376e06",
        "Blue",
        False,
        1846,
        6,
        13,
        1,
        6,
        14,
        2,
        1316,
        2275,
    ),
]


def _lotus_mp(
    puuid: str,
    team: str,
    won: bool,
    score: int,
    kills: int,
    deaths: int,
    assists: int,
    headshots: int,
    bodyshots: int,
    legshots: int,
    damage_dealt: int,
    damage_received: int,
) -> MatchPlayer:
    """Build a single-match MatchPlayer for the Lotus sanity match."""
    match = Match(
        match_id="lotus-b0c012f7",
        map_name="Lotus",
        map_id=None,
        queue_id="competitive",
        is_ranked=True,
        game_version=None,
        game_length_secs=0,
        season_short=None,
        region="na",
        rounds_played=_LOTUS_ROUNDS,
        red_score=13,
        blue_score=4,
        winning_team="Red",
        started_at="2026-04-19T18:00:00+00:00",
    )
    mp = MatchPlayer(
        match_id="lotus-b0c012f7",
        puuid=puuid,
        agent_name="Unknown",  # not relevant for ACS/ADR/HS% calculation
        agent_id=None,
        team=team,
        won=won,
        score=score,
        kills=kills,
        deaths=deaths,
        assists=assists,
        rounds_played=_LOTUS_ROUNDS,
        headshots=headshots,
        bodyshots=bodyshots,
        legshots=legshots,
        damage_dealt=damage_dealt,
        damage_received=damage_received,
        first_bloods=0,
        first_deaths=0,
        plants=0,
        defuses=0,
        afk_rounds=0,
        rounds_in_spawn=0,
        competitive_tier=None,
        credits_spent=None,
        started_at="2026-04-19T18:00:00+00:00",
    )
    mp.match = match
    return mp


@pytest.mark.parametrize(
    "name,puuid,team,won,score,kills,deaths,assists,hs,body,leg,dmg_dealt,dmg_recv",
    _LOTUS_PLAYERS,
    ids=[row[0] for row in _LOTUS_PLAYERS],
)
def test_acs_matches_api_reported_value(
    name: str,
    puuid: str,
    team: str,
    won: bool,
    score: int,
    kills: int,
    deaths: int,
    assists: int,
    hs: int,
    body: int,
    leg: int,
    dmg_dealt: int,
    dmg_recv: int,
) -> None:
    """ACS = score / rounds_played — verified against raw API score for all 10 players.

    This is the single most important pipeline check: if any of these ten
    cases fail, the fundamental ORM → calculator data path is broken and
    every ACS number shown to users is wrong.  Pass means our numbers would
    match what a user sees on tracker.gg for this exact match.
    """
    row = _lotus_mp(
        puuid, team, won, score, kills, deaths, assists, hs, body, leg, dmg_dealt, dmg_recv
    )
    s = compute_player_stats([row])
    expected = score / _LOTUS_ROUNDS
    assert abs(s.acs - expected) < 1e-9, (
        f"{name}: acs={s.acs:.6f} expected={expected:.6f} (score={score} / rounds={_LOTUS_ROUNDS})"
    )


@pytest.mark.parametrize(
    "name,puuid,team,won,score,kills,deaths,assists,hs,body,leg,dmg_dealt,dmg_recv",
    _LOTUS_PLAYERS,
    ids=[row[0] for row in _LOTUS_PLAYERS],
)
def test_adr_matches_api_reported_value(
    name: str,
    puuid: str,
    team: str,
    won: bool,
    score: int,
    kills: int,
    deaths: int,
    assists: int,
    hs: int,
    body: int,
    leg: int,
    dmg_dealt: int,
    dmg_recv: int,
) -> None:
    """ADR = damage_dealt / rounds_played — verified against the API damage.dealt field."""
    row = _lotus_mp(
        puuid, team, won, score, kills, deaths, assists, hs, body, leg, dmg_dealt, dmg_recv
    )
    s = compute_player_stats([row])
    expected = dmg_dealt / _LOTUS_ROUNDS
    assert abs(s.adr - expected) < 1e-9, (
        f"{name}: adr={s.adr:.6f} expected={expected:.6f} "
        f"(dmg={dmg_dealt} / rounds={_LOTUS_ROUNDS})"
    )


@pytest.mark.parametrize(
    "name,puuid,team,won,score,kills,deaths,assists,hs,body,leg,dmg_dealt,dmg_recv",
    _LOTUS_PLAYERS,
    ids=[row[0] for row in _LOTUS_PLAYERS],
)
def test_hs_pct_matches_api_reported_value(
    name: str,
    puuid: str,
    team: str,
    won: bool,
    score: int,
    kills: int,
    deaths: int,
    assists: int,
    hs: int,
    body: int,
    leg: int,
    dmg_dealt: int,
    dmg_recv: int,
) -> None:
    """HS% = headshots / total_shots — verified against the API headshots/bodyshots/legshots fields."""
    row = _lotus_mp(
        puuid, team, won, score, kills, deaths, assists, hs, body, leg, dmg_dealt, dmg_recv
    )
    s = compute_player_stats([row])
    total = hs + body + leg
    expected = hs / total if total else 0.0
    assert abs(s.hs_pct - expected) < 1e-9, (
        f"{name}: hs_pct={s.hs_pct:.6f} expected={expected:.6f} (hs={hs} / total_shots={total})"
    )
