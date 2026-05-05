"""Round-level analyzer — KAST / clutch / trade logic tests.

Builds ORM Match/Round/Kill objects in memory (no DB, same pattern as
test_stats_calculator.py). Each test isolates one code path so a break
points to the exact scenario that regressed.

Conventions:
    P = player under test ("puuid-p")
    M1, M2, M3, M4 = teammates
    E1..E5 = enemies
    Times in ms; we keep them well inside 0..120_000 so trade-window
    arithmetic is unambiguous.
"""

from __future__ import annotations

import json

from valocoach.data.orm_models import Kill, Match, MatchPlayer, Round
from valocoach.stats.constants import (
    FIRST_HALF_END,
    MAX_REGULATION_ROUNDS,
    TRADE_WINDOW_MS,
)
from valocoach.stats.round_analyzer import analyze_rounds, get_side, multi_kill_summary

P = "puuid-p"
TEAMMATES = [f"puuid-m{i}" for i in range(1, 5)]  # M1..M4
ENEMIES = [f"puuid-e{i}" for i in range(1, 6)]  # E1..E5


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _player(puuid: str, team: str) -> MatchPlayer:
    return MatchPlayer(
        match_id="m-1",
        puuid=puuid,
        agent_name="Jett",
        agent_id=None,
        team=team,
        won=False,
        score=0,
        kills=0,
        deaths=0,
        assists=0,
        rounds_played=0,
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
        started_at="2026-04-19T18:00:00+00:00",
    )


def _kill(
    *,
    killer: str,
    victim: str,
    t_ms: int | None,
    assistants: list[str] | None = None,
    round_number: int = 0,
) -> Kill:
    return Kill(
        round_id=0,
        match_id="m-1",
        round_number=round_number,
        time_in_round_ms=t_ms,
        killer_puuid=killer,
        victim_puuid=victim,
        weapon_name=None,
        is_headshot=False,
        assistants_json=json.dumps(assistants or []),
    )


def _round(round_number: int, winning_team: str, kills: list[Kill]) -> Round:
    r = Round(
        match_id="m-1",
        round_number=round_number,
        winning_team=winning_team,
        result_code="Elimination",
        bomb_planted=False,
        plant_site=None,
        bomb_defused=False,
    )
    r.kills = kills
    return r


def _match(rounds: list[Round]) -> Match:
    m = Match(
        match_id="m-1",
        map_name="Ascent",
        map_id=None,
        queue_id="competitive",
        is_ranked=True,
        game_version=None,
        game_length_secs=0,
        season_short=None,
        region="na",
        rounds_played=len(rounds),
        red_score=0,
        blue_score=0,
        winning_team=None,
        started_at="2026-04-19T18:00:00+00:00",
    )
    players: list[MatchPlayer] = [_player(P, "Blue")]
    players.extend(_player(t, "Blue") for t in TEAMMATES)
    players.extend(_player(e, "Red") for e in ENEMIES)
    m.players = players
    m.rounds = rounds
    return m


# ---------------------------------------------------------------------------
# Empty / absent cases
# ---------------------------------------------------------------------------


def test_empty_input_is_zeros() -> None:
    a = analyze_rounds([], P)
    assert a.rounds == 0
    assert a.kast_pct == 0.0
    assert a.clutch_rate == 0.0
    assert a.trade_efficiency == 0.0


def test_puuid_not_in_match_is_skipped() -> None:
    m = _match([_round(0, "Blue", [_kill(killer=ENEMIES[0], victim=TEAMMATES[0], t_ms=1000)])])
    a = analyze_rounds([m], "puuid-stranger")
    assert a.rounds == 0  # we skipped the whole match; puuid isn't there


# ---------------------------------------------------------------------------
# KAST — each letter in isolation
# ---------------------------------------------------------------------------


def test_kast_kill_only() -> None:
    # P kills E1, then dies with no trade → K yes, S no, T no
    r = _round(
        0,
        "Blue",
        [
            _kill(killer=P, victim=ENEMIES[0], t_ms=1000),
            _kill(killer=ENEMIES[1], victim=P, t_ms=90_000),  # no teammate avenge
        ],
    )
    a = analyze_rounds([_match([r])], P)
    assert a.rounds_with_kill == 1
    assert a.rounds_survived == 0
    assert a.rounds_traded_death == 0
    assert a.rounds_kast == 1
    assert a.kast_pct == 1.0


def test_kast_assist_only() -> None:
    # M1 kills E1 with P assisting; P survives (no death event). A+S both fire.
    r = _round(
        0,
        "Blue",
        [
            _kill(killer=TEAMMATES[0], victim=ENEMIES[0], t_ms=1000, assistants=[P]),
        ],
    )
    a = analyze_rounds([_match([r])], P)
    assert a.rounds_with_assist == 1
    assert a.rounds_survived == 1
    assert a.rounds_kast == 1


def test_kast_survive_only_no_combat() -> None:
    # No kills at all (e.g. bomb-timer rounds in the data) — P gets S.
    r = _round(0, "Blue", [])
    a = analyze_rounds([_match([r])], P)
    assert a.rounds_survived == 1
    assert a.rounds_kast == 1


def test_kast_trade_only() -> None:
    # P dies at 10s, M1 kills E1 (the killer) 2s later → T fires.
    r = _round(
        0,
        "Blue",
        [
            _kill(killer=ENEMIES[0], victim=P, t_ms=10_000),
            _kill(killer=TEAMMATES[0], victim=ENEMIES[0], t_ms=12_000),
        ],
    )
    a = analyze_rounds([_match([r])], P)
    assert a.rounds_traded_death == 1
    assert a.rounds_with_kill == 0
    assert a.rounds_survived == 0
    assert a.rounds_kast == 1  # T carries the round
    assert a.traded_deaths == 1


def test_kast_trade_outside_window_does_not_count() -> None:
    # Revenge is TRADE_WINDOW_MS + 1 ms after the death → not traded.
    t_death = 10_000
    r = _round(
        0,
        "Blue",
        [
            _kill(killer=ENEMIES[0], victim=P, t_ms=t_death),
            _kill(killer=TEAMMATES[0], victim=ENEMIES[0], t_ms=t_death + TRADE_WINDOW_MS + 1),
        ],
    )
    a = analyze_rounds([_match([r])], P)
    assert a.rounds_traded_death == 0
    assert a.traded_deaths == 0
    # Only K/A/S/T available: none fire → KAST 0.
    assert a.rounds_kast == 0


def test_kast_trade_boundary_counts() -> None:
    # Exactly TRADE_WINDOW_MS is still a trade (inclusive upper bound).
    t_death = 10_000
    r = _round(
        0,
        "Blue",
        [
            _kill(killer=ENEMIES[0], victim=P, t_ms=t_death),
            _kill(killer=TEAMMATES[0], victim=ENEMIES[0], t_ms=t_death + TRADE_WINDOW_MS),
        ],
    )
    a = analyze_rounds([_match([r])], P)
    assert a.traded_deaths == 1


# ---------------------------------------------------------------------------
# Trades given
# ---------------------------------------------------------------------------


def test_trade_given_counts_when_player_avenges_teammate() -> None:
    # M1 dies to E1 at 5s; P kills E1 at 7s → trade given.
    r = _round(
        0,
        "Blue",
        [
            _kill(killer=ENEMIES[0], victim=TEAMMATES[0], t_ms=5_000),
            _kill(killer=P, victim=ENEMIES[0], t_ms=7_000),
        ],
    )
    a = analyze_rounds([_match([r])], P)
    assert a.trades_given == 1
    assert a.teammate_deaths == 1
    assert a.trade_participation == 1.0


def test_trade_given_not_counted_outside_window() -> None:
    r = _round(
        0,
        "Blue",
        [
            _kill(killer=ENEMIES[0], victim=TEAMMATES[0], t_ms=5_000),
            _kill(killer=P, victim=ENEMIES[0], t_ms=5_000 + TRADE_WINDOW_MS + 1),
        ],
    )
    a = analyze_rounds([_match([r])], P)
    assert a.trades_given == 0


# ---------------------------------------------------------------------------
# Clutches
# ---------------------------------------------------------------------------


def test_clutch_won_1v1() -> None:
    # Set up a clean 1v1 by dropping 4 enemies BEFORE the last teammate
    # dies, so the clutch entry state is genuinely "1 vs 1".
    kills = [
        _kill(killer=TEAMMATES[0], victim=ENEMIES[1], t_ms=1_000),
        _kill(killer=TEAMMATES[0], victim=ENEMIES[2], t_ms=2_000),
        _kill(killer=TEAMMATES[0], victim=ENEMIES[3], t_ms=3_000),
        _kill(killer=TEAMMATES[0], victim=ENEMIES[4], t_ms=4_000),
        # Enemies: only E0 alive now. Team still 5.
        _kill(killer=ENEMIES[0], victim=TEAMMATES[0], t_ms=5_000),
        _kill(killer=ENEMIES[0], victim=TEAMMATES[1], t_ms=6_000),
        _kill(killer=ENEMIES[0], victim=TEAMMATES[2], t_ms=7_000),
        _kill(killer=ENEMIES[0], victim=TEAMMATES[3], t_ms=8_000),
        # Now P is last alive vs E0 → 1v1 entry.
        _kill(killer=P, victim=ENEMIES[0], t_ms=9_000),
    ]
    r = _round(0, "Blue", kills)
    a = analyze_rounds([_match([r])], P)
    assert a.clutch_opportunities == 1
    assert a.clutches_won == 1
    assert a.clutch_breakdown == {1: (1, 1)}


def test_clutch_entered_as_1v3_recorded_as_1v3() -> None:
    # Teammates die while only 2 enemies drop → clutch entered as 1v3.
    kills = [
        _kill(killer=ENEMIES[0], victim=TEAMMATES[0], t_ms=1_000),
        _kill(killer=TEAMMATES[1], victim=ENEMIES[4], t_ms=1_500),
        _kill(killer=ENEMIES[1], victim=TEAMMATES[1], t_ms=2_000),
        _kill(killer=TEAMMATES[2], victim=ENEMIES[3], t_ms=2_500),
        _kill(killer=ENEMIES[2], victim=TEAMMATES[2], t_ms=3_000),
        _kill(killer=ENEMIES[2], victim=TEAMMATES[3], t_ms=4_000),
        # Now P is last alive vs E1, E2, E3 (three enemies) — 1v3.
        _kill(killer=ENEMIES[0], victim=P, t_ms=5_000),  # P dies, loss
    ]
    r = _round(0, "Red", kills)  # P's team (Blue) loses
    a = analyze_rounds([_match([r])], P)
    assert a.clutch_opportunities == 1
    assert a.clutches_won == 0
    assert a.clutch_breakdown == {3: (0, 1)}


def test_no_clutch_when_team_still_alive() -> None:
    # Only two teammates die — P is not the last alive.
    kills = [
        _kill(killer=ENEMIES[0], victim=TEAMMATES[0], t_ms=1_000),
        _kill(killer=ENEMIES[1], victim=TEAMMATES[1], t_ms=2_000),
        _kill(killer=P, victim=ENEMIES[0], t_ms=3_000),
    ]
    r = _round(0, "Blue", kills)
    a = analyze_rounds([_match([r])], P)
    assert a.clutch_opportunities == 0


# ---------------------------------------------------------------------------
# Multi-kill tiering
# ---------------------------------------------------------------------------


def test_multikill_triple_not_also_double() -> None:
    # Three kills by P in one round → one triple, zero doubles.
    kills = [
        _kill(killer=P, victim=ENEMIES[0], t_ms=1_000),
        _kill(killer=P, victim=ENEMIES[1], t_ms=2_000),
        _kill(killer=P, victim=ENEMIES[2], t_ms=3_000),
    ]
    r = _round(0, "Blue", kills)
    a = analyze_rounds([_match([r])], P)
    assert a.triple_kills == 1
    assert a.double_kills == 0
    assert a.rounds_with_kill == 1


def test_ace_tiered_correctly() -> None:
    kills = [_kill(killer=P, victim=e, t_ms=1000 + i * 500) for i, e in enumerate(ENEMIES)]
    r = _round(0, "Blue", kills)
    a = analyze_rounds([_match([r])], P)
    assert a.aces == 1
    assert a.quadra_kills == 0


# ---------------------------------------------------------------------------
# Rates & aggregation across rounds
# ---------------------------------------------------------------------------


def test_kast_pct_across_rounds() -> None:
    # 3 rounds: 2 KAST-eligible (K, S), 1 blank (dies with no trade, no K/A).
    rounds = [
        _round(0, "Blue", [_kill(killer=P, victim=ENEMIES[0], t_ms=1000)]),  # K
        _round(1, "Blue", []),  # S
        _round(2, "Red", [_kill(killer=ENEMIES[0], victim=P, t_ms=1000)]),  # no KAST
    ]
    a = analyze_rounds([_match(rounds)], P)
    assert a.rounds == 3
    assert a.rounds_kast == 2
    assert abs(a.kast_pct - (2 / 3)) < 1e-9


def test_trade_efficiency_rate() -> None:
    # P dies twice; one trade, one not → 50 %.
    rounds = [
        _round(
            0,
            "Blue",
            [
                _kill(killer=ENEMIES[0], victim=P, t_ms=1_000),
                _kill(killer=TEAMMATES[0], victim=ENEMIES[0], t_ms=2_000),
            ],
        ),
        _round(
            1,
            "Red",
            [
                _kill(killer=ENEMIES[1], victim=P, t_ms=1_000),  # no avenge
            ],
        ),
    ]
    a = analyze_rounds([_match(rounds)], P)
    assert a.deaths == 2
    assert a.traded_deaths == 1
    assert a.trade_efficiency == 0.5


# ---------------------------------------------------------------------------
# KAST T — precision guard tests for the teammate-kill-of-killer requirement
# ---------------------------------------------------------------------------
#
# T fires iff ALL three conditions hold simultaneously:
#   (a) the player died this round
#   (b) a *teammate* (not the player themselves, not an enemy) killed the killer
#   (c) that kill happened within TRADE_WINDOW_MS of the player's death
#
# The tests below hand-trace every kill sequence to verify each guard.


class TestKASTTradeCorrectness:
    """Hand-crafted kill sequences verifying the three guards for KAST T."""

    def test_enemy_kills_killer_is_not_t(self) -> None:
        """P dies to E1; E2 (enemy, not a teammate) kills E1 within window → NOT T.

        In casual usage "P got traded" whenever the killer subsequently dies, but
        KAST T specifically requires *a teammate of P* to make the kill.  An
        intra-enemy kill (effectively friendly fire on the opponent side) must not
        count — the T guard checks ``k.killer_puuid in teammates``, excluding all
        enemies regardless of timing.
        """
        r = _round(
            0,
            "Red",
            [
                _kill(killer=ENEMIES[0], victim=P, t_ms=5_000),  # P dies to E1
                _kill(killer=ENEMIES[1], victim=ENEMIES[0], t_ms=7_000),  # E2 kills E1 (wrong side)
            ],
        )
        a = analyze_rounds([_match([r])], P)

        assert a.rounds_traded_death == 0, "enemy killing P's killer must NOT count as T"
        assert a.traded_deaths == 0
        # P has no K, A, S, or T this round → not a KAST round.
        assert a.rounds_kast == 0

    def test_player_killing_own_killer_is_k_not_t(self) -> None:
        """Kills by the player themselves satisfy K, never T.

        P is explicitly excluded from the ``teammates`` set, so even if the
        data contains a kill event where P killed P's own killer (which is
        physically impossible mid-round), that kill must register as K and
        must NOT count as T.  This verifies the ``p != puuid`` exclusion in
        the team map.
        """
        # Intentionally contradictory sequence: P kills E1, then E1 kills P.
        # A dead player can't kill anyone — this tests robustness to malformed
        # data, not a real-game scenario.
        r = _round(
            0,
            "Red",
            [
                _kill(killer=P, victim=ENEMIES[0], t_ms=1_000),  # P kills E1 → K
                _kill(killer=ENEMIES[0], victim=P, t_ms=5_000),  # E1 "kills" P
            ],
        )
        a = analyze_rounds([_match([r])], P)

        assert a.rounds_with_kill == 1, "P's kill of E1 should register as K"
        assert a.rounds_survived == 0, "P died so S must not fire"
        assert a.rounds_traded_death == 0, "P killing own killer must NOT count as T"
        assert a.traded_deaths == 0
        # K carries the KAST round even though T did not fire.
        assert a.rounds_kast == 1

    def test_teammate_kills_wrong_enemy_is_not_t(self) -> None:
        """P dies to E1; teammate T1 kills E2 (not E1) within window → NOT T.

        The trade check requires ``k.victim_puuid == death.killer_puuid`` — a
        kill on a *different* enemy within the window is irrelevant, even if the
        teammate acted fast enough that the timing would have qualified.
        """
        r = _round(
            0,
            "Red",
            [
                _kill(killer=ENEMIES[0], victim=P, t_ms=5_000),  # P dies to E1
                _kill(
                    killer=TEAMMATES[0], victim=ENEMIES[1], t_ms=7_000
                ),  # T1 kills E2 (wrong target)
            ],
        )
        a = analyze_rounds([_match([r])], P)

        assert a.rounds_traded_death == 0, "teammate killing wrong enemy must NOT count as T"
        assert a.traded_deaths == 0
        assert a.rounds_kast == 0

    def test_only_correct_teammate_kill_triggers_t_in_mixed_sequence(self) -> None:
        """Mixed sequence: only the kill that targets *the killer* within the window fires T.

        Timeline (all timestamps in ms):
            4 000  E2 kills T1   — teammate death; trade window for T1 opens
            5 000  E1 kills P    — P dies; trade window for P opens (until 10 000ms)
            6 000  T2 kills E2   — T2 avenges T1, but E2 is NOT P's killer → no T for P
            8 000  T3 kills E1   — T3 kills P's killer 3 000ms after P's death → VALID T

        T fires because T3 is in ``teammates`` and ``k.victim_puuid == E1 == death.killer_puuid``
        and 8 000 - 5 000 = 3 000 ms <= TRADE_WINDOW_MS (5 000 ms).

        The T2→E2 kill does NOT interfere because E2 ≠ E1 (wrong victim).
        P did not personally avenge T1 → trades_given = 0.
        """
        r = _round(
            0,
            "Red",
            [
                _kill(killer=ENEMIES[1], victim=TEAMMATES[0], t_ms=4_000),  # T1 dies to E2
                _kill(killer=ENEMIES[0], victim=P, t_ms=5_000),  # P dies to E1
                _kill(killer=TEAMMATES[1], victim=ENEMIES[1], t_ms=6_000),  # T2 avenges T1 (not P)
                _kill(killer=TEAMMATES[2], victim=ENEMIES[0], t_ms=8_000),  # T3 avenges P ✓
            ],
        )
        a = analyze_rounds([_match([r])], P)

        # T fires for P (T3→E1 within 3 000ms).
        assert a.rounds_traded_death == 1
        assert a.traded_deaths == 1
        assert a.rounds_kast == 1  # T carries the round (no K/A/S)

        # P did not personally avenge any teammate → trades_given is 0.
        assert a.trades_given == 0
        # T1 was the only teammate who died (T2/T3 made kills but weren't killed).
        assert a.teammate_deaths == 1


# ---------------------------------------------------------------------------
# get_side() — overtime path (lines 197-207)
# ---------------------------------------------------------------------------


def test_get_side_overtime_even_index_gives_second_half_attacker() -> None:
    """OT round with even ot_index → same swap as second half (lines 202-206)."""
    # Round MAX_REGULATION_ROUNDS = OT round 0 (ot_index=0, even)
    # attacker_team="Red" → second_half_attacker="blue" → round_attacker="blue" → "defense"
    ot_round_0 = MAX_REGULATION_ROUNDS
    side = get_side(ot_round_0, "Red")
    # In OT round 0 (even index), the team that attacked in the second half attacks again.
    # second_half_attacker = "blue" (since attacker_norm="red"), round_attacker = "blue" → defense
    assert side == "defense"


def test_get_side_overtime_odd_index_gives_first_half_attacker() -> None:
    """OT round with odd ot_index → flips back to first-half attacker (line 206)."""
    ot_round_1 = MAX_REGULATION_ROUNDS + 1
    side = get_side(ot_round_1, "Red")
    # ot_index=1 (odd) → round_attacker = attacker_norm = "red" → "attack"
    assert side == "attack"


def test_get_side_overtime_covers_first_and_second_half_too() -> None:
    """Basic sanity — first half attacker and second half swap still work."""
    # attacker_team="Red" means Red started on attack in first half.
    assert get_side(0, "Red") == "attack"  # round 0, Red attacks
    assert get_side(FIRST_HALF_END + 1, "Red") == "defense"  # second half, Red defends
    # attacker_team="Blue" means Blue started on attack in first half.
    assert get_side(0, "Blue") == "attack"  # round 0, Blue attacks
    assert get_side(FIRST_HALF_END + 1, "Blue") == "defense"  # second half, Blue defends


# ---------------------------------------------------------------------------
# _infer_attacker_team() — second half and OT plant (lines 224-235)
# ---------------------------------------------------------------------------


def _round_with_planter(
    round_number: int,
    winning_team: str,
    kills: list[Kill],
    planter_puuid: str | None = None,
) -> Round:
    """Like _round() but also sets planter_puuid."""
    r = Round(
        match_id="m-1",
        round_number=round_number,
        winning_team=winning_team,
        result_code="Elimination",
        bomb_planted=planter_puuid is not None,
        plant_site="A" if planter_puuid else None,
        bomb_defused=False,
        planter_puuid=planter_puuid,
    )
    r.kills = kills
    return r


def test_infer_attacker_team_second_half_plant(monkeypatch) -> None:
    """Plant in second-half round → return opposite team (lines 229-232)."""
    from valocoach.stats.round_analyzer import _infer_attacker_team

    # Round 12 is in the second half (FIRST_HALF_END=11 < 12 <= SECOND_HALF_END=24).
    # ENEMIES[0] is "Red"; planting in second half means Red was the second-half attacker.
    # So the first-half attacker = "Blue" (Red swapped to attack in second half).
    second_half_round = _round_with_planter(
        round_number=FIRST_HALF_END + 1,  # 12 — first second-half round
        winning_team="Red",
        kills=[],
        planter_puuid=ENEMIES[0],  # ENEMIES[0] is "Red"
    )
    m = _match([second_half_round])
    team_map = {P: "Blue", **dict.fromkeys(TEAMMATES, "Blue"), **dict.fromkeys(ENEMIES, "Red")}

    result = _infer_attacker_team(m, team_map)
    # Planted in second half → Red is second-half attacker → Blue was first-half attacker.
    assert result == "Blue"


def test_infer_attacker_team_ot_plant_skipped_returns_none(monkeypatch) -> None:
    """OT plant is skipped (continue at line 234); if no other plants → return None."""
    from valocoach.stats.round_analyzer import _infer_attacker_team

    # OT round (> SECOND_HALF_END = 24) with a plant → continue (ambiguous).
    ot_round = _round_with_planter(
        round_number=MAX_REGULATION_ROUNDS,  # 25 — first OT round
        winning_team="Red",
        kills=[],
        planter_puuid=ENEMIES[0],
    )
    m = _match([ot_round])
    team_map = {P: "Blue", **dict.fromkeys(TEAMMATES, "Blue"), **dict.fromkeys(ENEMIES, "Red")}

    result = _infer_attacker_team(m, team_map)
    # OT plant is ambiguous — continue skips it; no other rounds → return None.
    assert result is None


def test_infer_attacker_team_first_half_plant() -> None:
    """Plant in first half → return planting team directly (lines 227-228)."""
    from valocoach.stats.round_analyzer import _infer_attacker_team

    first_half_round = _round_with_planter(
        round_number=0,  # round 0 ≤ FIRST_HALF_END
        winning_team="Blue",
        kills=[],
        planter_puuid=ENEMIES[0],  # "Red" team
    )
    m = _match([first_half_round])
    team_map = {P: "Blue", **dict.fromkeys(TEAMMATES, "Blue"), **dict.fromkeys(ENEMIES, "Red")}

    result = _infer_attacker_team(m, team_map)
    assert result == "Red"  # Red planted in first half → Red is the first-half attacker


# ---------------------------------------------------------------------------
# Side accounting (lines 289-298) — requires attacker_team is not None
# ---------------------------------------------------------------------------


def test_side_accounting_tracked_when_attacker_known() -> None:
    """attack_rounds, defense_rounds, and win counts populated when plant present."""
    # Plant in round 0 (first half) → attacker_team = "Red" (ENEMIES are Red).
    # Player P is "Blue" → on defense in first half, attack in second half.
    # Round 0: P's team (Blue) wins while defending → defense_wins += 1
    round0 = _round_with_planter(
        round_number=0,
        winning_team="Blue",  # Blue wins while defending in first half
        kills=[_kill(killer=P, victim=ENEMIES[0], t_ms=5_000)],
        planter_puuid=ENEMIES[0],  # Red plants → Red is first-half attacker
    )
    # Round 12 (second half): Blue now attacks.  Blue wins → attack_wins += 1 (line 294)
    round12 = _round_with_planter(
        round_number=FIRST_HALF_END + 1,
        winning_team="Blue",  # Blue wins while attacking in second half
        kills=[_kill(killer=P, victim=ENEMIES[0], t_ms=5_000)],
        planter_puuid=None,
    )

    # Round 1 (first half): Blue defends and LOSES → defense_rounds += 1, no defense_wins (line 297->299)
    round1 = _round_with_planter(
        round_number=1,
        winning_team="Red",  # Red wins round 1 while attacking → Blue loses on defense
        kills=[_kill(killer=ENEMIES[0], victim=P, t_ms=5_000)],
        planter_puuid=None,
    )
    # Round 13 (second half): Blue attacks and LOSES → attack_rounds += 1, no attack_wins (line 293->299)
    round13 = _round_with_planter(
        round_number=FIRST_HALF_END + 2,
        winning_team="Red",  # Red wins round 13 while defending → Blue loses on attack
        kills=[_kill(killer=ENEMIES[0], victim=P, t_ms=5_000)],
        planter_puuid=None,
    )

    m = _match([round0, round12, round1, round13])
    a = analyze_rounds([m], P)

    # Side accounting should be populated (not None).
    assert a.attack_rounds is not None
    assert a.defense_rounds is not None
    assert a.defense_rounds >= 2  # rounds 0 and 1
    assert a.attack_rounds >= 2  # rounds 12 and 13
    # Blue won round 12 while attacking → attack_wins must be set.
    assert a.attack_wins is not None
    assert a.attack_wins >= 1
    # Blue won round 0 while defending → defense_wins must be set.
    assert a.defense_wins is not None
    assert a.defense_wins >= 1


# ---------------------------------------------------------------------------
# Double kill (line 329) and Quadra kill (line 333)
# ---------------------------------------------------------------------------


def test_double_kill_counted() -> None:
    """Player kills exactly 2 enemies in one round → double_kills += 1 (line 329)."""
    r = _round(
        0,
        "Blue",
        [
            _kill(killer=P, victim=ENEMIES[0], t_ms=1_000),
            _kill(killer=P, victim=ENEMIES[1], t_ms=2_000),
        ],
    )
    a = analyze_rounds([_match([r])], P)
    assert a.double_kills == 1
    assert a.triple_kills == 0
    assert a.quadra_kills == 0


def test_quadra_kill_counted() -> None:
    """Player kills 4 enemies in one round → quadra_kills += 1 (line 333)."""
    r = _round(
        0,
        "Blue",
        [
            _kill(killer=P, victim=ENEMIES[0], t_ms=1_000),
            _kill(killer=P, victim=ENEMIES[1], t_ms=2_000),
            _kill(killer=P, victim=ENEMIES[2], t_ms=3_000),
            _kill(killer=P, victim=ENEMIES[3], t_ms=4_000),
        ],
    )
    a = analyze_rounds([_match([r])], P)
    assert a.quadra_kills == 1
    assert a.triple_kills == 0  # exclusive: only highest tier counted
    assert a.double_kills == 0


# ---------------------------------------------------------------------------
# _tally_clutch continue (line 434) — unknown victim+killer
# ---------------------------------------------------------------------------


def test_clutch_continue_on_unknown_victim_and_killer() -> None:
    """Kill where victim and killer are both NOT P and NOT teammates → continue."""
    # Both are "unknown" puuids — not in the player's team or known enemies list
    # from our perspective (we just use two ENEMIES to simulate unknown-to-us actors).
    # The kill is: E2 kills E3 (two enemies, neither is P or teammates).
    r = _round(
        0,
        "Blue",
        [
            # Teammates are all dead first (team_alive drops to 1 = P).
            _kill(killer=ENEMIES[0], victim=TEAMMATES[0], t_ms=1_000),
            _kill(killer=ENEMIES[0], victim=TEAMMATES[1], t_ms=2_000),
            _kill(killer=ENEMIES[0], victim=TEAMMATES[2], t_ms=3_000),
            _kill(killer=ENEMIES[0], victim=TEAMMATES[3], t_ms=4_000),
            # Now P is alone (1vN); this kill has unknown victim+killer (both "enemies"
            # to us but neither is our player or teammates) → triggers continue.
            # To make victim "not a teammate and not P" AND killer "not P and not teammate",
            # we use two different enemies fighting each other.
            _kill(killer=ENEMIES[1], victim=ENEMIES[2], t_ms=5_000),
            # P finishes the round (survived).
        ],
    )
    a = analyze_rounds([_match([r])], P)
    # The continue skips that kill; P survived the round → KAST S fires.
    assert a.rounds_survived == 1
    assert a.rounds_kast == 1


# ---------------------------------------------------------------------------
# _assistants() invalid JSON (lines 503-504)
# ---------------------------------------------------------------------------


def test_assistants_invalid_json_returns_empty() -> None:
    """Kill with malformed assistants_json → _assistants returns [] without raising."""
    # Build a kill where P is an assistant but with invalid JSON.
    bad_kill = Kill(
        round_id=0,
        match_id="m-1",
        round_number=0,
        time_in_round_ms=5_000,
        killer_puuid=ENEMIES[0],
        victim_puuid=ENEMIES[1],
        weapon_name=None,
        is_headshot=False,
        assistants_json="NOT_VALID_JSON{{{",  # invalid — triggers ValueError
    )
    r = _round(0, "Blue", [bad_kill])
    # Must not raise; the invalid assistants line is treated as []
    a = analyze_rounds([_match([r])], P)
    assert a.rounds == 1  # round was processed


# ---------------------------------------------------------------------------
# _was_death_traded() — death.time_in_round_ms is None (line 520)
# ---------------------------------------------------------------------------


def test_death_with_none_timestamp_not_counted_as_traded() -> None:
    """Player dies with time_in_round_ms=None → _was_death_traded returns False (line 520)."""
    r = _round(
        0,
        "Red",
        [
            # P dies at t_ms=None — no timestamp.
            _kill(killer=ENEMIES[0], victim=P, t_ms=None),
            # Teammate avenges within "trade window" — but death has no timestamp.
            _kill(killer=TEAMMATES[0], victim=ENEMIES[0], t_ms=3_000),
        ],
    )
    a = analyze_rounds([_match([r])], P)
    # Even though the teammate killed the killer quickly, no T fires because
    # the death timestamp is None — _was_death_traded returns False at line 520.
    assert a.traded_deaths == 0
    assert a.rounds_traded_death == 0


# ---------------------------------------------------------------------------
# _within_trade_window() — None revenge timestamp (line 538)
# ---------------------------------------------------------------------------


def test_trade_revenge_with_none_timestamp_not_counted() -> None:
    """Teammate avenges but their kill has t_ms=None → _within_trade_window returns False."""
    r = _round(
        0,
        "Red",
        [
            _kill(killer=ENEMIES[0], victim=P, t_ms=5_000),  # P dies (non-None)
            _kill(killer=TEAMMATES[0], victim=ENEMIES[0], t_ms=None),  # revenge, t_ms=None
        ],
    )
    a = analyze_rounds([_match([r])], P)
    # _within_trade_window(5000, None) → False at line 538 → trade not counted.
    assert a.traded_deaths == 0
    assert a.rounds_traded_death == 0


# ---------------------------------------------------------------------------
# multi_kill_summary() — double_kills and quadra_kills branches (lines 692, 696)
# ---------------------------------------------------------------------------


def _make_analysis(**kwargs) -> RoundAnalysis:  # noqa: F821
    """Build a RoundAnalysis with all required fields, defaulting ints to 0."""
    from valocoach.stats.round_analyzer import RoundAnalysis

    defaults = {
        "rounds": 0,
        "deaths": 0,
        "teammate_deaths": 0,
        "clutch_opportunities": 0,
        "rounds_with_kill": 0,
        "rounds_with_assist": 0,
        "rounds_survived": 0,
        "rounds_traded_death": 0,
        "rounds_kast": 0,
        "clutches_won": 0,
        "traded_deaths": 0,
        "trades_given": 0,
        "double_kills": 0,
        "triple_kills": 0,
        "quadra_kills": 0,
        "aces": 0,
    }
    defaults.update(kwargs)
    return RoundAnalysis(**defaults)


def test_multi_kill_summary_double_kill() -> None:
    """multi_kill_summary with double_kills > 0 renders '2K' line (line 696)."""
    analysis = _make_analysis(rounds=5, double_kills=3)
    result = multi_kill_summary(analysis)
    assert "2K" in result
    assert "3x2K" in result


def test_multi_kill_summary_quadra_kill() -> None:
    """multi_kill_summary with quadra_kills > 0 renders '4K' line (line 692)."""
    analysis = _make_analysis(rounds=5, quadra_kills=2)
    result = multi_kill_summary(analysis)
    assert "4K" in result
    assert "2x4K" in result


def test_multi_kill_summary_all_tiers() -> None:
    """All multi-kill tiers combined in descending display order."""
    analysis = _make_analysis(rounds=20, aces=1, quadra_kills=2, triple_kills=3, double_kills=4)
    result = multi_kill_summary(analysis)
    assert "1xAce" in result
    assert "2x4K" in result
    assert "3x3K" in result
    assert "4x2K" in result


def test_multi_kill_summary_empty_when_no_multikills() -> None:
    """No multi-kills → empty string returned."""
    analysis = _make_analysis(rounds=5)
    result = multi_kill_summary(analysis)
    assert result == ""
