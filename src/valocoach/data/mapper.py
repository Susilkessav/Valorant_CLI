"""API shape → ORM shape conversions.  Update this file when HenrikDev changes their schema.

All knowledge of what the HenrikDev API sends (field names, types, nesting,
quirks) is concentrated here.  The repository layer calls these functions and
handles only DB-level concerns (existence checks, session management, error
handling).

Public API
----------
match_from_details(details: MatchDetails) -> Match
    Convert a v4 MatchDetails API object into a fully-populated Match ORM
    tree (players, rounds, kills attached via SQLAlchemy relationships).
    ``session.add(match) + flush()`` is all the caller needs.

player_from_account_mmr(account: AccountData, mmr: MMRData) -> Player
    Convert account + MMR API objects into a Player ORM row ready for
    ``session.merge()``.

V4 API quirks encoded here (do not scatter these facts elsewhere)
-----------------------------------------------------------------
- tier.id arrives as a JSON integer (e.g. 13), not a string.  _Ref.id has a
  BeforeValidator that coerces it to str; _tier_int() converts back for the
  ORM integer column.
- weapon.name is null on ability / ultimate kills.  Stored as NULL in the DB.
- kill events have no is_headshot field — always False in the Kill row.
- player.behavior.afk_rounds and rounds_in_spawn are floats (e.g. 0.0, 1.0);
  the ORM columns are int — truncate with int().
- rounds_played is not in v4 metadata; derive it as len(match.rounds).
- team_id is "Red" / "Blue" (title-case) in v4; the ORM stores it as-is.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass

from valocoach.core.exceptions import MapperError
from valocoach.data.api_models import (
    AccountData,
    MatchDetails,
    MatchDetailsKill,
    MatchDetailsPlayer,
    MMRData,
)
from valocoach.data.orm_models import Kill, Match, OrmMatchPlayer, Player, Round, RoundPlayer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _xy(location: dict) -> tuple[int | None, int | None]:
    """Extract (x, y) integers from an API location dict.

    HenrikDev v4 location shape: ``{"x": int, "y": int}``.
    Returns ``(None, None)`` when the dict is absent or malformed so callers
    can safely unpack and store NULL without crashing.
    """
    if not isinstance(location, dict):
        return None, None
    x = location.get("x")
    y = location.get("y")
    return (
        int(x) if isinstance(x, int | float) else None,
        int(y) if isinstance(y, int | float) else None,
    )


def _engagement_distance(kx: int | None, ky: int | None, vx: int | None, vy: int | None) -> str | None:
    """Return Euclidean distance between killer and victim as a TEXT string.

    Stored as TEXT to avoid float precision noise in SQLite.  Cast to float
    when reading.  Returns None when any coordinate is absent.
    """
    if kx is None or ky is None or vx is None or vy is None:
        return None
    dist = ((kx - vx) ** 2 + (ky - vy) ** 2) ** 0.5
    return f"{dist:.2f}"


def _tier_int(tier_id: str) -> int | None:
    """Convert a tier id string ("13") to int (13).  Returns None on blank / non-numeric."""
    return int(tier_id) if tier_id and tier_id.lstrip("-").isdigit() else None


def _econ_int(economy: dict, *keys: str) -> int | None:
    """Safely traverse a nested economy dict and return an int leaf or None.

    HenrikDev v4 economy shape on MatchDetailsPlayer:
        {"spent": {"overall": 12000, "average": 400},
         "loadout_value": {"overall": 35000, "average": 1166}}

    Example: _econ_int(player.economy, "spent", "overall") → 12000.
    Returns None if any key is missing or the leaf is not an int — old
    rows (synced before this migration) are left as NULL rather than
    crashing the sync.
    """
    cur: object = economy
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return int(cur) if isinstance(cur, int | float) else None


@dataclass(slots=True)
class _ImpactStats:
    """Per-PUUID impact counters derived from kill and round lists."""

    first_bloods: dict[str, int]
    first_deaths: dict[str, int]
    plants: dict[str, int]
    defuses: dict[str, int]


def _compute_impact(details: MatchDetails) -> _ImpactStats:
    """Derive first_bloods, first_deaths, plants, defuses from kill/round lists.

    first_blood  = the killer of the earliest kill in a round (by time_in_round_in_ms).
    first_death  = the victim of that same kill.
    plants       = rounds where round.plant.player.puuid matches the player.
    defuses      = rounds where round.defuse.player.puuid matches the player.
    """
    fb: dict[str, int] = defaultdict(int)
    fd: dict[str, int] = defaultdict(int)
    plants: dict[str, int] = defaultdict(int)
    defuses: dict[str, int] = defaultdict(int)

    # First bloods / deaths — group kills by round, pick the earliest one
    round_kills: dict[int, list[MatchDetailsKill]] = defaultdict(list)
    for kill in details.kills:
        round_kills[kill.round].append(kill)

    for kills in round_kills.values():
        if kills:
            first = min(kills, key=lambda k: k.time_in_round_in_ms)
            fb[first.killer.puuid] += 1
            fd[first.victim.puuid] += 1

    # Plants / defuses — attributed to the player on the bomb event
    for rnd in details.rounds:
        if rnd.plant and rnd.plant.player.puuid:
            plants[rnd.plant.player.puuid] += 1
        if rnd.defuse and rnd.defuse.player.puuid:
            defuses[rnd.defuse.player.puuid] += 1

    return _ImpactStats(
        first_bloods=fb,
        first_deaths=fd,
        plants=plants,
        defuses=defuses,
    )


def _map_player(
    player: MatchDetailsPlayer,
    *,
    match_id: str,
    rounds_played: int,
    started_at: str,
    team_won: bool,
    impact: _ImpactStats,
) -> OrmMatchPlayer:
    """Map one MatchDetailsPlayer to an OrmMatchPlayer row."""
    return OrmMatchPlayer(
        match_id=match_id,
        puuid=player.puuid or None,
        agent_name=player.agent.name or "",
        agent_id=player.agent.id or None,
        team=player.team_id,  # "Red" | "Blue"
        won=team_won,
        score=player.stats.score,
        kills=player.stats.kills,
        deaths=player.stats.deaths,
        assists=player.stats.assists,
        rounds_played=rounds_played,
        headshots=player.stats.headshots,
        bodyshots=player.stats.bodyshots,
        legshots=player.stats.legshots,
        damage_dealt=player.stats.damage.dealt,  # nested {dealt, received}
        damage_received=player.stats.damage.received,
        first_bloods=impact.first_bloods[player.puuid],
        first_deaths=impact.first_deaths[player.puuid],
        plants=impact.plants[player.puuid],
        defuses=impact.defuses[player.puuid],
        afk_rounds=int(player.behavior.afk_rounds),  # float → int
        rounds_in_spawn=int(player.behavior.rounds_in_spawn),  # float → int
        competitive_tier=_tier_int(player.tier.id),  # int JSON → str → int
        started_at=started_at,
        credits_spent=_econ_int(player.economy, "spent", "overall"),
        avg_loadout=_econ_int(player.economy, "loadout_value", "average"),
    )


def _build_round_tree(details: MatchDetails, match_id: str) -> list[Round]:
    """Build Round ORM rows with Kill and RoundPlayer children attached.

    Returns a list in round-index order.  Children are appended to their
    parent Round so SQLAlchemy's unit-of-work resolves FKs automatically
    on flush — no explicit round.id look-up needed.

    V4 notes:
    - round.id  is the zero-based round index (0 … N-1).
    - kill.round is that same index used to match kills to rounds.
    - kill.is_headshot does not exist in v4 — always stored as False.
    - weapon.name is null on ability kills — stored as NULL.
    - round.stats[*].economy is a flat dict: {loadout_value, remaining, weapon, armor}.
    """
    # puuid → team ("Red" | "Blue") — needed for RoundPlayer.team field.
    puuid_to_team: dict[str, str] = {p.puuid: p.team_id for p in details.players}

    # round_number → set of victim PUUIDs — for the survived flag.
    round_victims: dict[int, set[str]] = defaultdict(set)
    for kill in details.kills:
        round_victims[kill.round].add(kill.victim.puuid)

    round_map: dict[int, Round] = {}

    for rnd in details.rounds:
        plant_x, plant_y = _xy(rnd.plant.location) if rnd.plant else (None, None)
        defuse_x, defuse_y = _xy(rnd.defuse.location) if rnd.defuse else (None, None)
        orm_round = Round(
            match_id=match_id,
            round_number=rnd.id,
            winning_team=rnd.winning_team,
            result_code=rnd.ceremony or "",
            bomb_planted=rnd.plant is not None,
            plant_site=rnd.plant.site if rnd.plant else None,
            planter_puuid=rnd.plant.player.puuid if rnd.plant and rnd.plant.player.puuid else None,
            plant_x=plant_x,
            plant_y=plant_y,
            bomb_defused=rnd.defuse is not None,
            defuser_puuid=rnd.defuse.player.puuid
            if rnd.defuse and rnd.defuse.player.puuid
            else None,
            defuse_x=defuse_x,
            defuse_y=defuse_y,
        )
        round_map[rnd.id] = orm_round

        victims_this_round = round_victims.get(rnd.id, set())
        for ps in rnd.stats:
            puuid = ps.puuid
            if not puuid:
                continue  # malformed row — skip
            econ = ps.economy if isinstance(ps.economy, dict) else {}
            damage_dealt = sum(
                int(ev.get("damage", 0)) for ev in ps.damage_events if isinstance(ev, dict)
            )
            # Compact damage events: keep only the fields useful for analysis
            damage_summary = [
                {
                    "receiver": ev.get("receiver"),
                    "damage": int(ev.get("damage", 0)),
                    "headshots": int(ev.get("headshots", 0)),
                    "bodyshots": int(ev.get("bodyshots", 0)),
                    "legshots": int(ev.get("legshots", 0)),
                }
                for ev in ps.damage_events
                if isinstance(ev, dict)
            ]
            ac = ps.ability_casts
            orm_round.round_players.append(
                RoundPlayer(
                    match_id=match_id,
                    puuid=puuid,
                    team=puuid_to_team.get(puuid, ""),
                    score=ps.score,
                    kills=ps.kills,
                    headshots=ps.stats.headshots,
                    bodyshots=ps.stats.bodyshots,
                    legshots=ps.stats.legshots,
                    damage_dealt=damage_dealt,
                    loadout_value=econ.get("loadout_value"),
                    remaining_credits=econ.get("remaining"),
                    survived=puuid not in victims_this_round,
                    was_afk=ps.was_afk,
                    stayed_in_spawn=ps.stayed_in_spawn,
                    ability_casts_grenade=ac.grenade,
                    ability_casts_ability1=ac.ability_1,
                    ability_casts_ability2=ac.ability_2,
                    ability_casts_ultimate=ac.ultimate,
                    damage_events_json=json.dumps(damage_summary) if damage_summary else None,
                )
            )

    for kill in details.kills:
        orm_round = round_map.get(kill.round)
        if orm_round is None:
            # kill references a round index not present in rounds list — skip
            log.debug("kill in unknown round %d — skipped", kill.round)
            continue

        # Extract spatial data from the kill event.
        # kill.location is the victim's position at death.
        # kill.player_locations is a list of {puuid, location: {x, y}, view_radians}
        # for all players at the moment of the kill — we use the killer's entry.
        victim_x, victim_y = _xy(kill.location)
        killer_x: int | None = None
        killer_y: int | None = None
        for pl in kill.player_locations:
            if isinstance(pl, dict) and pl.get("player_id") == kill.killer.puuid:
                killer_x, killer_y = _xy(pl.get("location", {}))
                break

        orm_round.kills.append(
            Kill(
                match_id=match_id,
                round_number=kill.round,
                time_in_round_ms=kill.time_in_round_in_ms,
                killer_puuid=kill.killer.puuid,
                victim_puuid=kill.victim.puuid,
                weapon_name=kill.weapon.name,  # None on ability kills
                is_headshot=False,  # not present in v4
                assistants_json=json.dumps([a.puuid for a in kill.assistants]),
                killer_x=killer_x,
                killer_y=killer_y,
                victim_x=victim_x,
                victim_y=victim_y,
                engagement_distance=_engagement_distance(killer_x, killer_y, victim_x, victim_y),
            )
        )

    return list(round_map.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_from_details(details: MatchDetails) -> Match:
    """Convert a v4 MatchDetails API object into a fully-populated Match ORM tree.

    Returns a ``Match`` with ``.players``, ``.rounds``, and ``.rounds[n].kills``
    populated via SQLAlchemy relationships.  The caller only needs::

        session.add(match)
        await session.flush()

    This function is the single point of contact between the HenrikDev v4
    wire format and the ORM schema.  When the API changes, update here only.
    """
    meta = details.metadata
    match_id = meta.match_id

    # Team outcomes
    red_team = details.team_result("Red")
    blue_team = details.team_result("Blue")
    red_won = red_team.won if red_team else False
    blue_won = blue_team.won if blue_team else False
    winning_team = "Red" if red_won else ("Blue" if blue_won else None)

    rounds_played = len(details.rounds)
    # started_at is the ORDER BY key for every recent-match query and the
    # bucket boundary for daily aggregates.  An empty string sorts before
    # every valid ISO8601 timestamp, so a match with no started_at would
    # silently corrupt "last N matches" results.  Refuse to map.
    if not meta.started_at:
        raise MapperError(
            f"MatchDetails {match_id[:8]}… has no started_at timestamp — "
            "refusing to map (would corrupt ORDER BY started_at queries)."
        )
    started_at = meta.started_at

    # Per-player impact stats derived from the kill + round lists
    impact = _compute_impact(details)

    # Match row
    match = Match(
        match_id=match_id,
        map_name=meta.map.name or "",
        map_id=meta.map.id or None,
        queue_id=meta.queue.id,
        is_ranked=meta.queue.id.lower() == "competitive",
        game_version=meta.game_version,
        game_length_secs=int(meta.game_length_in_ms / 1000),
        season_short=meta.season.name or meta.season.id or None,
        region=meta.region,
        rounds_played=rounds_played,
        red_score=red_team.rounds_won if red_team else 0,
        blue_score=blue_team.rounds_won if blue_team else 0,
        winning_team=winning_team,
        started_at=started_at,
    )

    # One OrmMatchPlayer per participant (all 10 players in a standard match)
    for player in details.players:
        team_won = red_won if player.team_id == "Red" else blue_won
        match.players.append(
            _map_player(
                player,
                match_id=match_id,
                rounds_played=rounds_played,
                started_at=started_at,
                team_won=team_won,
                impact=impact,
            )
        )

    # Round rows with Kill children
    for orm_round in _build_round_tree(details, match_id):
        match.rounds.append(orm_round)

    return match


def player_from_account_mmr(account: AccountData, mmr: MMRData) -> Player:
    """Convert account + MMR API objects into a Player ORM row.

    Caller should use ``session.merge()`` so a second call updates the rank
    snapshot without creating a duplicate row.
    """
    return Player(
        puuid=account.puuid,
        riot_name=account.name,
        riot_tag=account.tag,
        region=account.region,
        account_level=account.account_level,
        current_tier=mmr.current_data.currenttier,
        current_tier_patched=mmr.current_data.currenttierpatched,
        current_rr=mmr.current_data.ranking_in_tier,
        elo=mmr.current_data.elo,
        peak_tier=mmr.highest_rank.tier,
        peak_tier_patched=mmr.highest_rank.patched_tier,
    )
