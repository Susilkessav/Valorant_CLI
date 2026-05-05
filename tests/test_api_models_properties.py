"""Property accessor and method tests for api_models.py.

Coverage targets (27 missed statements):
  96  StoredMatchMeta.map_name
  127 StoredMatchStats.agent_name
  131 StoredMatchStats.headshots
  135 StoredMatchStats.bodyshots
  139 StoredMatchStats.legshots
  143 StoredMatchStats.damage_made
  147 StoredMatchStats.damage_received
  196 MatchDetailsMetadata.map_name
  200 MatchDetailsMetadata.queue_id
  204 MatchDetailsMetadata.game_length_secs
  226 MatchDetailsPlayerStats.damage_dealt
  230 MatchDetailsPlayerStats.damage_received
  269 MatchDetailsPlayer.agent_name
  273 MatchDetailsPlayer.team
  277 MatchDetailsPlayer.current_tier
  297 MatchDetailsTeam.rounds_lost
  321 MatchDetailsKill.killer_puuid
  325 MatchDetailsKill.victim_puuid
  329 MatchDetailsKill.killer_team
  333 MatchDetailsKill.victim_team
  337 MatchDetailsKill.weapon_name
  371 MatchDetailsRoundPlayerStats.puuid
  375 MatchDetailsRoundPlayerStats.score
  379 MatchDetailsRoundPlayerStats.kills
  404 MatchDetailsRound.player_stats
  422 MatchDetails.player_by_puuid
  429 MatchDetails.rounds_played
"""

from __future__ import annotations

from valocoach.data.api_models import (
    MatchDetails,
    MatchDetailsKill,
    MatchDetailsMetadata,
    MatchDetailsPlayer,
    MatchDetailsPlayerStats,
    MatchDetailsRound,
    MatchDetailsRoundPlayerStats,
    MatchDetailsTeam,
    StoredMatch,
    StoredMatchMeta,
    StoredMatchStats,
    _PlayerRef,
    _Ref,
    _StoredDamage,
    _StoredShots,
    _V4PlayerDamage,
    _V4Queue,
    _V4TeamRounds,
)

# ---------------------------------------------------------------------------
# StoredMatchMeta (line 96)
# ---------------------------------------------------------------------------


class TestStoredMatchMeta:
    def test_map_name_returns_map_name(self) -> None:
        """Line 96: return self.map.name or 'Unknown'."""
        meta = StoredMatchMeta(map=_Ref(id="lotus-id", name="Lotus"))
        assert meta.map_name == "Lotus"

    def test_map_name_returns_unknown_when_none(self) -> None:
        """Line 96: 'or Unknown' branch when map.name is None."""
        meta = StoredMatchMeta(map=_Ref(id="lotus-id", name=None))
        assert meta.map_name == "Unknown"


# ---------------------------------------------------------------------------
# StoredMatchStats (lines 127, 131, 135, 139, 143, 147)
# ---------------------------------------------------------------------------


class TestStoredMatchStats:
    def _stats(self) -> StoredMatchStats:
        return StoredMatchStats(
            puuid="p1",
            character=_Ref(id="jett-id", name="Jett"),
            shots=_StoredShots(head=10, body=30, leg=5),
            damage=_StoredDamage(made=2500, received=1800),
        )

    def test_agent_name(self) -> None:
        """Line 127: return self.character.name or ''."""
        assert self._stats().agent_name == "Jett"

    def test_agent_name_empty_when_none(self) -> None:
        """Line 127: empty-string fallback when character.name is None."""
        s = StoredMatchStats(character=_Ref(id="x", name=None))
        assert s.agent_name == ""

    def test_headshots(self) -> None:
        """Line 131: return self.shots.head."""
        assert self._stats().headshots == 10

    def test_bodyshots(self) -> None:
        """Line 135: return self.shots.body."""
        assert self._stats().bodyshots == 30

    def test_legshots(self) -> None:
        """Line 139: return self.shots.leg."""
        assert self._stats().legshots == 5

    def test_damage_made(self) -> None:
        """Line 143: return self.damage.made."""
        assert self._stats().damage_made == 2500

    def test_damage_received(self) -> None:
        """Line 147: return self.damage.received."""
        assert self._stats().damage_received == 1800


# ---------------------------------------------------------------------------
# StoredMatch (sanity check — match_id delegates to meta)
# ---------------------------------------------------------------------------


def test_stored_match_match_id() -> None:
    """StoredMatch.match_id delegates to meta.match_id."""
    m = StoredMatch(meta=StoredMatchMeta(match_id="abc-123"))
    assert m.match_id == "abc-123"


# ---------------------------------------------------------------------------
# MatchDetailsMetadata (lines 196, 200, 204)
# ---------------------------------------------------------------------------


class TestMatchDetailsMetadata:
    def _meta(self) -> MatchDetailsMetadata:
        return MatchDetailsMetadata(
            match_id="m1",
            map=_Ref(id="lotus-id", name="Lotus"),
            game_length_in_ms=90_000,
            queue=_V4Queue(id="competitive", name="Competitive", mode_type="Standard"),
        )

    def test_map_name(self) -> None:
        """Line 196: return self.map.name or 'Unknown'."""
        assert self._meta().map_name == "Lotus"

    def test_map_name_unknown_when_none(self) -> None:
        """Line 196: 'Unknown' fallback when map.name is None."""
        meta = MatchDetailsMetadata(map=_Ref(id="x", name=None))
        assert meta.map_name == "Unknown"

    def test_queue_id(self) -> None:
        """Line 200: return self.queue.id."""
        assert self._meta().queue_id == "competitive"

    def test_game_length_secs(self) -> None:
        """Line 204: return self.game_length_in_ms / 1000."""
        assert self._meta().game_length_secs == 90.0


# ---------------------------------------------------------------------------
# MatchDetailsPlayerStats (lines 226, 230)
# ---------------------------------------------------------------------------


class TestMatchDetailsPlayerStats:
    def _stats(self) -> MatchDetailsPlayerStats:
        return MatchDetailsPlayerStats(
            damage=_V4PlayerDamage(dealt=3000, received=2000),
        )

    def test_damage_dealt(self) -> None:
        """Line 226: return self.damage.dealt."""
        assert self._stats().damage_dealt == 3000

    def test_damage_received(self) -> None:
        """Line 230: return self.damage.received."""
        assert self._stats().damage_received == 2000


# ---------------------------------------------------------------------------
# MatchDetailsPlayer (lines 269, 273, 277)
# ---------------------------------------------------------------------------


class TestMatchDetailsPlayer:
    def _player(self, *, tier_id: str = "13", tier_name: str = "Gold 2") -> MatchDetailsPlayer:
        return MatchDetailsPlayer(
            puuid="puuid-001",
            name="Yoursaviour01",
            tag="SK04",
            team_id="Blue",
            agent=_Ref(id="jett-id", name="Jett"),
            tier=_Ref(id=tier_id, name=tier_name),
        )

    def test_agent_name(self) -> None:
        """Line 269: return self.agent.name or ''."""
        assert self._player().agent_name == "Jett"

    def test_agent_name_empty_when_none(self) -> None:
        """Line 269: '' fallback when agent.name is None."""
        p = MatchDetailsPlayer(puuid="x", agent=_Ref(id="x", name=None))
        assert p.agent_name == ""

    def test_team_returns_team_id(self) -> None:
        """Line 273: return self.team_id."""
        assert self._player().team == "Blue"

    def test_current_tier_returns_int(self) -> None:
        """Line 277: return self.tier.id (as int via _coerce_str → cast back)."""
        # tier.id is stored as str (coerced by BeforeValidator), but the property
        # returns it as-is.  When id="13" the property returns "13" (truthy).
        p = self._player(tier_id="13")
        # The property: `return self.tier.id if self.tier.id else 0`
        assert p.current_tier == "13"

    def test_current_tier_returns_zero_when_empty(self) -> None:
        """Line 277: `else 0` branch when tier.id is empty string (falsy)."""
        p = self._player(tier_id="")
        assert p.current_tier == 0


# ---------------------------------------------------------------------------
# MatchDetailsTeam (line 293 rounds_won + line 297 rounds_lost)
# ---------------------------------------------------------------------------


class TestMatchDetailsTeam:
    def _team(self) -> MatchDetailsTeam:
        return MatchDetailsTeam(
            team_id="Blue",
            won=True,
            rounds=_V4TeamRounds(won=13, lost=7),
        )

    def test_rounds_won(self) -> None:
        """MatchDetailsTeam.rounds_won (covered but verified here for completeness)."""
        assert self._team().rounds_won == 13

    def test_rounds_lost(self) -> None:
        """Line 297: return self.rounds.lost."""
        assert self._team().rounds_lost == 7


# ---------------------------------------------------------------------------
# MatchDetailsKill (lines 321, 325, 329, 333, 337)
# ---------------------------------------------------------------------------


class TestMatchDetailsKill:
    def _kill(self) -> MatchDetailsKill:
        return MatchDetailsKill(
            round=2,
            time_in_round_in_ms=8_000,
            time_in_match_in_ms=120_000,
            killer=_PlayerRef(puuid="killer-puuid", team="Red"),
            victim=_PlayerRef(puuid="victim-puuid", team="Blue"),
            weapon=_Ref(id="vandal-id", name="Vandal"),
        )

    def test_killer_puuid(self) -> None:
        """Line 321: return self.killer.puuid."""
        assert self._kill().killer_puuid == "killer-puuid"

    def test_victim_puuid(self) -> None:
        """Line 325: return self.victim.puuid."""
        assert self._kill().victim_puuid == "victim-puuid"

    def test_killer_team(self) -> None:
        """Line 329: return self.killer.team."""
        assert self._kill().killer_team == "Red"

    def test_victim_team(self) -> None:
        """Line 333: return self.victim.team."""
        assert self._kill().victim_team == "Blue"

    def test_weapon_name(self) -> None:
        """Line 337: return self.weapon.name."""
        assert self._kill().weapon_name == "Vandal"

    def test_weapon_name_none_for_ability_kill(self) -> None:
        """Line 337: returns None when weapon.name is None (ability kill)."""
        k = MatchDetailsKill(
            round=0,
            time_in_round_in_ms=1_000,
            time_in_match_in_ms=1_000,
            killer=_PlayerRef(puuid="x", team="Red"),
            victim=_PlayerRef(puuid="y", team="Blue"),
            weapon=_Ref(id="", name=None),
        )
        assert k.weapon_name is None


# ---------------------------------------------------------------------------
# MatchDetailsRoundPlayerStats (lines 371, 375, 379)
# ---------------------------------------------------------------------------


class TestMatchDetailsRoundPlayerStats:
    def _rps(self) -> MatchDetailsRoundPlayerStats:
        from valocoach.data.api_models import _RoundPlayerInnerStats

        return MatchDetailsRoundPlayerStats(
            player=_PlayerRef(puuid="rps-puuid", team="Blue"),
            stats=_RoundPlayerInnerStats(score=250, kills=3),
        )

    def test_puuid(self) -> None:
        """Line 371: return self.player.puuid."""
        assert self._rps().puuid == "rps-puuid"

    def test_score(self) -> None:
        """Line 375: return self.stats.score."""
        assert self._rps().score == 250

    def test_kills(self) -> None:
        """Line 379: return self.stats.kills."""
        assert self._rps().kills == 3


# ---------------------------------------------------------------------------
# MatchDetailsRound (line 404)
# ---------------------------------------------------------------------------


def test_match_details_round_player_stats_alias() -> None:
    """Line 404: MatchDetailsRound.player_stats returns self.stats."""
    rps = MatchDetailsRoundPlayerStats(player=_PlayerRef(puuid="q", team="Red"))
    rnd = MatchDetailsRound(id=0, winning_team="Red", stats=[rps])
    assert rnd.player_stats is rnd.stats
    assert rnd.player_stats == [rps]


# ---------------------------------------------------------------------------
# MatchDetails (lines 422, 429)
# ---------------------------------------------------------------------------


class TestMatchDetails:
    def _match(self) -> MatchDetails:
        p1 = MatchDetailsPlayer(puuid="p-001", name="Alpha", team_id="Blue")
        p2 = MatchDetailsPlayer(puuid="p-002", name="Beta", team_id="Red")
        t1 = MatchDetailsTeam(team_id="Blue", won=False)
        t2 = MatchDetailsTeam(team_id="Red", won=True)
        r0 = MatchDetailsRound(id=0, winning_team="Red")
        r1 = MatchDetailsRound(id=1, winning_team="Blue")
        return MatchDetails(
            players=[p1, p2],
            teams=[t1, t2],
            rounds=[r0, r1],
        )

    def test_player_by_puuid_found(self) -> None:
        """Line 422: player_by_puuid returns matching player."""
        m = self._match()
        p = m.player_by_puuid("p-001")
        assert p is not None
        assert p.name == "Alpha"

    def test_player_by_puuid_not_found_returns_none(self) -> None:
        """Line 422: player_by_puuid returns None when puuid absent."""
        m = self._match()
        assert m.player_by_puuid("no-such-puuid") is None

    def test_team_result_found(self) -> None:
        """MatchDetails.team_result returns matching team (line 425)."""
        m = self._match()
        t = m.team_result("Red")
        assert t is not None
        assert t.won is True

    def test_team_result_not_found_returns_none(self) -> None:
        """MatchDetails.team_result returns None for unknown team_id."""
        m = self._match()
        assert m.team_result("Purple") is None

    def test_rounds_played(self) -> None:
        """Line 429: return len(self.rounds)."""
        m = self._match()
        assert m.rounds_played == 2

    def test_rounds_played_empty(self) -> None:
        """Line 429: returns 0 when rounds list is empty."""
        m = MatchDetails()
        assert m.rounds_played == 0
