#!/usr/bin/env python
"""Live validation script for all HenrikDev API endpoints.

Run this BEFORE writing the sync pipeline to confirm:
  - Your API key works
  - Every endpoint returns data your Pydantic models can parse
  - StoredMatch and MatchDetails field names match the real API (we guessed those)

Usage:
    # Uses riot_name / riot_tag / riot_region from .env / env vars:
    python scripts/validate_api.py

    # Override any field on the command line:
    python scripts/validate_api.py --name Yoursaviour01 --tag SK04 --region na

Exit codes:
    0 — all checks passed
    1 — one or more checks failed (fix models / key before writing sync logic)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path

# ── bootstrap the package path when running as a script ──────────────────────
# (not needed when installed with `pip install -e .`, but harmless)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from valocoach.core.config import load_settings
from valocoach.core.exceptions import ConfigError
from valocoach.data.api_client import HenrikClient

console = Console()

# ── result tracking ───────────────────────────────────────────────────────────

results: list[tuple[str, bool, str]] = []  # (label, passed, detail)


def _ok(label: str, detail: str) -> None:
    results.append((label, True, detail))
    console.print(f"  [bold green]✓[/bold green]  {detail}")


def _fail(label: str, detail: str) -> None:
    results.append((label, False, detail))
    console.print(f"  [bold red]✗[/bold red]  {detail}")


def _warn(detail: str) -> None:
    console.print(f"  [bold yellow]⚠[/bold yellow]  {detail}")


def _raw(label: str, data: object, max_keys: int = 8) -> None:
    """Print a compact JSON preview — useful for inspecting uncertain models."""
    if isinstance(data, list):
        preview = data[0] if data else {}
        suffix = f"  [dim](+ {len(data)-1} more)[/dim]" if len(data) > 1 else ""
    else:
        preview = data
        suffix = ""

    if isinstance(preview, dict):
        trimmed = dict(list(preview.items())[:max_keys])
        if len(preview) > max_keys:
            trimmed["…"] = f"({len(preview) - max_keys} more keys)"
    else:
        trimmed = preview

    console.print(
        f"  [dim]raw {label}:[/dim]{suffix}\n"
        + "  "
        + json.dumps(trimmed, indent=2, default=str).replace("\n", "\n  ")
    )


# ── individual checks ─────────────────────────────────────────────────────────


async def check_account(client: HenrikClient, name: str, tag: str) -> None:
    label = "get_account"
    console.rule(f"[cyan]{label}[/cyan]")
    try:
        acc = await client.get_account(name, tag)
        _ok(label, f"puuid={acc.puuid}  region={acc.region}  level={acc.account_level}")
    except Exception as exc:
        _fail(label, f"{type(exc).__name__}: {exc}")


async def check_account_force(client: HenrikClient, name: str, tag: str) -> None:
    label = "get_account(force=True)"
    console.rule(f"[cyan]{label}[/cyan]")
    try:
        acc = await client.get_account(name, tag, force=True)
        _ok(label, f"puuid={acc.puuid}  (cache bypassed)")
    except Exception as exc:
        _fail(label, f"{type(exc).__name__}: {exc}")


async def check_mmr(client: HenrikClient, region: str, name: str, tag: str) -> None:
    label = "get_mmr"
    console.rule(f"[cyan]{label}[/cyan]")
    try:
        mmr = await client.get_mmr(region, name, tag)
        cd = mmr.current_data
        hr = mmr.highest_rank
        _ok(
            label,
            f"rank={cd.currenttierpatched}  rr={cd.ranking_in_tier}  "
            f"elo={cd.elo}  peak={hr.patched_tier}",
        )
    except Exception as exc:
        _fail(label, f"{type(exc).__name__}: {exc}")


async def check_mmr_history(client: HenrikClient, region: str, name: str, tag: str) -> None:
    label = "get_mmr_history"
    console.rule(f"[cyan]{label}[/cyan]")
    try:
        history = await client.get_mmr_history(region, name, tag)
        if not history:
            _warn("Empty history list returned — no ranked games on record?")
            results.append((label, True, "empty list"))
            return
        h = history[0]
        _ok(
            label,
            f"{len(history)} entries — latest: {h.currenttierpatched} "
            f"rr={h.ranking_in_tier}  delta={h.mmr_change_to_last_game:+d}",
        )
    except Exception as exc:
        _fail(label, f"{type(exc).__name__}: {exc}")


async def check_matches_v3(client: HenrikClient, region: str, name: str, tag: str) -> str | None:
    """Returns first match_id on success for use in check_match_details."""
    label = "get_matches (v3)"
    console.rule(f"[cyan]{label}[/cyan]")
    try:
        matches = await client.get_matches(region, name, tag, size=3)
        if not matches:
            _warn("Empty match list — no competitive games on record?")
            results.append((label, True, "empty list"))
            return None
        m = matches[0]
        meta = m.metadata
        player = m.player_by_puuid(
            next((p.puuid for p in m.players.all_players if p.name.lower() == name.lower()), "")
        )
        kd = f"  kd={player.stats.kd_ratio}" if player else ""
        _ok(
            label,
            f"{len(matches)} matches — latest: {meta.map_name} "
            f"({meta.queue_id}) rounds={meta.rounds_played}{kd}",
        )
        return meta.match_id
    except Exception as exc:
        _fail(label, f"{type(exc).__name__}: {exc}")
        return None


async def check_stored_matches(
    client: HenrikClient, region: str, name: str, tag: str
) -> str | None:
    """
    This endpoint's response shape was inferred — the raw JSON is printed
    alongside the parsed result so you can verify StoredMatch field names.
    Returns first match_id on success.
    """
    label = "get_stored_matches (v1)"
    console.rule(f"[cyan]{label}[/cyan]")
    try:
        raw_payload = (
            await client._get(  # intentional raw access to inspect the real response shape
                f"/valorant/v1/stored-matches/{region}/{name}/{tag}",
                params={"mode": "competitive", "size": 3},
            )
        )
        raw_items = raw_payload.get("data", [])

        console.print(
            "  [dim]StoredMatch field names are inferred — printing raw JSON to verify:[/dim]"
        )
        _raw("data[0]", raw_items)

        # Now validate through the Pydantic model
        from valocoach.data.api_models import StoredMatch

        parsed = [StoredMatch.model_validate(item) for item in raw_items]

        if not parsed:
            _warn("Empty stored-matches list.")
            results.append((label, True, "empty list"))
            return None

        first = parsed[0]
        _ok(
            label,
            f"{len(parsed)} entries — first match_id={first.match_id!r}  "
            f"map={first.meta.map_name!r}  mode={first.meta.mode!r}",
        )

        # Warn if match_id came back empty (alias mismatch)
        if not first.match_id:
            _warn(
                "match_id is empty — the 'id'/'matchid' alias did not match the real key. "
                "Check the raw JSON above and update StoredMatchMeta.match_id aliases."
            )

        return first.match_id or None

    except Exception as exc:
        _fail(label, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


async def check_match_details(client: HenrikClient, region: str, match_id: str) -> None:
    """
    v4 match endpoint — validates against the real MatchDetails model.
    """
    label = "get_match_details (v4)"
    console.rule(f"[cyan]{label}[/cyan]")
    try:
        from valocoach.data.api_models import MatchDetails

        payload = await client._get(  # intentional raw access
            f"/valorant/v4/match/{region}/{match_id}"
        )
        raw_data = payload.get("data", {})
        detail = MatchDetails.model_validate(raw_data)
        meta = detail.metadata
        _ok(
            label,
            f"map={meta.map_name}  rounds={detail.rounds_played}  "
            f"players={len(detail.players)}  kills={len(detail.kills)}  "
            f"started={meta.started_at}",
        )
    except Exception as exc:
        _fail(label, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


async def check_version(client: HenrikClient, region: str) -> None:
    label = "get_version"
    console.rule(f"[cyan]{label}[/cyan]")
    try:
        ver = await client.get_version(region)
        if not ver:
            _warn("Empty version dict returned.")
            results.append((label, True, "empty dict"))
            return
        # Common keys: version, branch, lastChecked, region, buildDate
        display = {k: ver[k] for k in list(ver)[:4]}
        _ok(label, "  ".join(f"{k}={v}" for k, v in display.items()))
    except Exception as exc:
        _fail(label, f"{type(exc).__name__}: {exc}")


async def check_snapshot(client: HenrikClient, region: str, name: str, tag: str) -> None:
    label = "fetch_player_snapshot"
    console.rule(f"[cyan]{label}[/cyan]")
    try:
        acc, mmr, matches = await client.fetch_player_snapshot(region, name, tag, match_count=3)
        _ok(
            label,
            f"concurrent gather OK — "
            f"account={acc.name}#{acc.tag}  "
            f"rank={mmr.current_data.currenttierpatched}  "
            f"matches={len(matches)}",
        )
    except Exception as exc:
        _fail(label, f"{type(exc).__name__}: {exc}")


# ── summary table ─────────────────────────────────────────────────────────────


def print_summary() -> bool:
    """Print results table. Returns True if all passed."""
    console.print()
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Endpoint", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detail")

    all_passed = True
    for label, passed, detail in results:
        status = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
        if not passed:
            all_passed = False
        table.add_row(label, status, detail)

    console.print(table)

    if all_passed:
        console.print(
            Panel(
                "[bold green]All checks passed.[/bold green]  " "Safe to write the sync pipeline.",
                border_style="green",
            )
        )
    else:
        failed = [label for label, passed, _ in results if not passed]
        console.print(
            Panel(
                "[bold red]Some checks failed:[/bold red] "
                + ", ".join(failed)
                + "\n\nFix the issues above before writing the sync pipeline.",
                border_style="red",
            )
        )

    return all_passed


# ── entry point ───────────────────────────────────────────────────────────────


async def main(name: str, tag: str, region: str) -> bool:
    console.print(
        Panel(
            f"[bold]valocoach API validation[/bold]\n"
            f"player: [cyan]{name}#{tag}[/cyan]  region: [cyan]{region}[/cyan]",
            border_style="blue",
        )
    )

    settings = load_settings()
    # Allow CLI overrides to be used even if settings has defaults
    settings.riot_name = name
    settings.riot_tag = tag
    settings.riot_region = region  # type: ignore[assignment]

    try:
        client = HenrikClient(settings, throttle_seconds=2.0)
    except ConfigError as exc:
        console.print(f"[bold red]Config error:[/bold red] {exc}")
        console.print(
            "Set [bold]HENRIKDEV_API_KEY[/bold] in your [bold].env[/bold] file "
            "or environment and try again."
        )
        return False

    async with client:
        # ── core account / rank ──────────────────────────────────────
        await check_account(client, name, tag)
        await check_account_force(client, name, tag)
        await check_mmr(client, region, name, tag)
        await check_mmr_history(client, region, name, tag)

        # ── match data ───────────────────────────────────────────────
        v3_match_id = await check_matches_v3(client, region, name, tag)
        stored_match_id = await check_stored_matches(client, region, name, tag)

        # Use whichever match_id we got for the v4 detail check
        detail_match_id = v3_match_id or stored_match_id
        if detail_match_id:
            await check_match_details(client, region, detail_match_id)
        else:
            _fail(
                "get_match_details (v4)",
                "Skipped — no match_id available (v3 and v1 both failed)",
            )

        # ── utility ──────────────────────────────────────────────────
        await check_version(client, region)
        await check_snapshot(client, region, name, tag)

    return print_summary()


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    p = argparse.ArgumentParser(description="Validate all HenrikDev API endpoints live.")
    p.add_argument(
        "--name",
        default=settings.riot_name or "Yoursaviour01",
        help="Riot name (no #tag).  Default: from settings.",
    )
    p.add_argument(
        "--tag",
        default=settings.riot_tag or "SK04",
        help="Riot tag.  Default: from settings.",
    )
    p.add_argument(
        "--region",
        default=settings.riot_region or "na",
        help="Region (na/eu/ap/kr/latam/br).  Default: from settings.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    passed = asyncio.run(main(args.name, args.tag, args.region))
    sys.exit(0 if passed else 1)
