"""One-off script: fetch one match, dump it, verify required fields exist.

Run BEFORE building the sync pipeline. If fields are missing or named
differently than expected, open data/sample_match.json, find the actual
key names, and update the Pydantic models before writing any sync logic.

Usage:
    python scripts/validate_api_data.py

Exit codes:
    0 — all field checks passed
    1 — one or more fields missing (fix models before writing sync pipeline)

NOTE: Field paths below use RAW API key names (what the JSON actually contains),
not Pydantic field names.  Our models rename several keys via validation_alias:
    matchid       → match_id        (MatchMetadata)
    map           → map_name        (MatchMetadata)
    mode_id       → queue_id        (MatchMetadata)
    game_length   → game_length_secs (MatchMetadata — API sends seconds, not ms)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from valocoach.core.config import load_settings
from valocoach.core.exceptions import APIError
from valocoach.data.api_client import HenrikClient

# ---------------------------------------------------------------------------
# Field checklists  (raw API key names, dot-separated for nesting)
# ---------------------------------------------------------------------------

# Top-level keys on the match object
REQUIRED_MATCH_FIELDS = [
    "metadata",
    "players",
    "teams",
    "rounds",
    "kills",
]

# Keys inside the metadata block (confirmed from sample_match.json 2026-04-19)
REQUIRED_METADATA_FIELDS = [
    "match_id",  # v4 uses match_id (v3 used matchid)
    "map",  # {id, name} object
    "game_version",
    "game_length_in_ms",  # milliseconds (v3 sent game_length in seconds)
    "started_at",  # ISO8601 string (v3 sent game_start unix int)
    "is_completed",
    "queue",  # {id, name, mode_type} object
    "region",
    # NOTE: no mode/mode_id/rounds_played at metadata level in v4
]

# Keys on each player object (raw names, confirmed from sample_match.json)
REQUIRED_PLAYER_FIELDS = [
    "puuid",
    "team_id",  # "Red" or "Blue"  (v4 uses team_id, NOT team)
    "agent",  # {id, name} object (v4 uses agent, NOT character)
    "stats.score",
    "stats.kills",
    "stats.deaths",
    "stats.headshots",
    "stats.bodyshots",
    "stats.legshots",
    "stats.damage.dealt",  # nested: stats.damage = {dealt, received}
    "stats.damage.received",
    "behavior.afk_rounds",
    "behavior.rounds_in_spawn",
]

# Keys on each kill event (v4 uses nested killer/victim objects, NOT flat puuid fields)
REQUIRED_KILL_FIELDS = [
    "round",
    "time_in_round_in_ms",
    "time_in_match_in_ms",
    "killer.puuid",  # nested object  (NOT flat killer_puuid)
    "killer.team",
    "victim.puuid",  # nested object  (NOT flat victim_puuid)
    "victim.team",
    "weapon.name",  # nested object  (NOT flat weapon_name)
    # NOTE: no is_headshot field at kill level in v4
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check(obj: dict, dotted: str) -> bool:
    """Return True if the dotted key path exists and is not None."""
    parts = dotted.split(".")
    cur = obj
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return False
    return cur is not None


def print_checks(label: str, obj: dict, fields: list[str]) -> int:
    """Print ✓/✗ for each field. Returns number of failures."""
    print(f"\n{label}:")
    failures = 0
    for f in fields:
        ok = check(obj, f)
        print(f"  {'✓' if ok else '✗'}  {f}")
        if not ok:
            failures += 1
    return failures


def extract_players_list(players_block: dict | list) -> list[dict]:
    """Handle both v3 shape {all_players: [...]} and a hypothetical flat v4 list."""
    if isinstance(players_block, list):
        return players_block
    if isinstance(players_block, dict):
        # v3 shape: {"all_players": [...], "red": [...], "blue": [...]}
        return players_block.get("all_players", [])
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    settings = load_settings()

    if not settings.riot_name or not settings.riot_tag:
        print("✗  riot_name / riot_tag not set in .env — cannot run validation.")
        return 1

    print(f"Player : {settings.riot_name}#{settings.riot_tag}  region={settings.riot_region}")

    async with HenrikClient(settings) as client:
        # ── 1. Account ────────────────────────────────────────────────
        print(f"\nFetching account {settings.riot_name}#{settings.riot_tag}...")
        try:
            account = await client.get_account(settings.riot_name, settings.riot_tag)
        except APIError as exc:
            print(f"  ✗  get_account failed: {exc}")
            return 1
        print(f"  ✓  puuid={account.puuid}")

        # ── 2. Stored matches (v1) — get most recent match_id ─────────
        print("\nFetching stored match list (v1)...")
        try:
            stored = await client.get_stored_matches(
                settings.riot_region,
                settings.riot_name,
                settings.riot_tag,
                size=1,
            )
        except APIError as exc:
            print(f"  ✗  get_stored_matches failed: {exc}")
            print("  Trying v3 get_matches as fallback...")
            stored = []

        if stored:
            # Fix: stored is list[StoredMatch] (Pydantic), not list[dict]
            match_id = stored[0].match_id
            print(f"  ✓  most recent match_id (v1): {match_id}")
        else:
            # Fallback: try v3 get_matches
            print("\nFetching v3 match list as fallback...")
            try:
                v3_matches = await client.get_matches(
                    settings.riot_region,
                    settings.riot_name,
                    settings.riot_tag,
                    size=1,
                )
            except APIError as exc:
                print(f"  ✗  get_matches fallback also failed: {exc}")
                return 1
            if not v3_matches:
                print("  ✗  No competitive matches found. Play one and re-run.")
                return 1
            match_id = v3_matches[0].metadata.match_id
            print(f"  ✓  most recent match_id (v3 fallback): {match_id}")

        # ── 3. Full match detail (v4) ─────────────────────────────────
        print(f"\nFetching v4 match detail for {match_id}...")
        try:
            details_raw = await client._get(  # intentional: inspect raw response shape
                f"/valorant/v4/match/{settings.riot_region}/{match_id}"
            )
        except APIError as exc:
            print(f"  ✗  v4 match fetch failed: {type(exc).__name__}: {exc}")
            print(
                "  If this is a 404, the match may not be indexed yet. "
                "Try again with a different match_id from sample_match_list.json."
            )
            return 1

        details = details_raw.get("data", {})
        if not details:
            print("  ✗  Response had no 'data' key — API shape may have changed.")
            return 1

        # ── 4. Save full response for offline inspection ──────────────
        out = Path("data/sample_match.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(details, indent=2))
        print(f"  ✓  full response saved → {out}  ({out.stat().st_size // 1024} KB)")
        print("  Open this file to inspect real field names if any checks below show ✗.")

        # ── 5. Top-level field checks ─────────────────────────────────
        total_failures = 0
        total_failures += print_checks("Top-level match fields", details, REQUIRED_MATCH_FIELDS)

        # ── 6. Metadata field checks ──────────────────────────────────
        meta = details.get("metadata", {})
        total_failures += print_checks(
            "metadata block (raw API keys)", meta, REQUIRED_METADATA_FIELDS
        )

        # ── 7. Per-player field checks ────────────────────────────────
        players_list = extract_players_list(details.get("players", {}))
        if players_list:
            p0 = players_list[0]
            total_failures += print_checks(
                f"Per-player fields (player[0]: {p0.get('name', '?')}#{p0.get('tag', '?')})",
                p0,
                REQUIRED_PLAYER_FIELDS,
            )
        else:
            print("\n✗  players list is empty — cannot run per-player checks.")
            total_failures += 1

        # ── 8. Per-kill field checks ──────────────────────────────────
        kills = details.get("kills", [])
        if kills:
            total_failures += print_checks(
                f"Per-kill fields (kill[0] of {len(kills)} total)", kills[0], REQUIRED_KILL_FIELDS
            )
        else:
            print("\n⚠  kills list is empty — match may be unavailable or still processing.")

        # ── 9. Round count ────────────────────────────────────────────
        rounds = details.get("rounds", [])
        print(f"\nRound data: {len(rounds)} rounds found")
        if rounds:
            r0 = rounds[0]
            has_player_stats = bool(r0.get("player_stats"))
            print(f"  round[0] has player_stats: {'✓' if has_player_stats else '✗'}")
            if has_player_stats:
                ps0 = r0["player_stats"][0]
                print(f"  round player_stats keys: {list(ps0.keys())}")

        # ── 10. ACS sanity check ──────────────────────────────────────
        # ACS (Average Combat Score) = total score / rounds played
        # Sane range: roughly 100-450.  If wildly off, score means something else.
        if players_list and rounds:
            p0 = players_list[0]
            score = p0.get("stats", {}).get("score", 0)
            rounds_played = len(rounds)
            acs = score / rounds_played
            sane = 50 <= acs <= 600
            print(
                f"\nACS sanity: {p0.get('name', '?')} "
                f"score={score} / {rounds_played} rounds = {acs:.1f}  "
                f"{'✓ sane' if sane else '✗ UNEXPECTED — score field may mean something else'}"
            )
            if not sane:
                print(
                    "  Open data/sample_match.json and search for the player's score field. "
                    "It may be stored per-round rather than cumulative."
                )
                total_failures += 1

        # ── Summary ───────────────────────────────────────────────────
        print("\n" + "─" * 50)
        if total_failures == 0:
            print("✓  All field checks passed. Safe to write the sync pipeline.")
        else:
            print(
                f"✗  {total_failures} check(s) failed.\n"
                "   Open data/sample_match.json, find the real field names,\n"
                "   and update the Pydantic models before writing sync logic."
            )

    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
