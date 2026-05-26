"""Deterministic meta/agent-info response builder.

For ``meta`` and ``agent_info`` intents the LLM consistently fabricates
ability names (Riftwalk, Sonic Blast, Dark Visage, Sonic Boom, …) even with
21 KB of grounded context.  The root cause is that the model's training
prior dominates the prompt at 8B–14B scale.

We solve it structurally: the agent listing, role labels, and ability rosters
come from ``agents.json`` + ``meta.json`` (100% accurate, no LLM), and the
LLM is only asked to write a short personalized takeaway grounded in the
player's stats.  That way the hallucination-prone surface area shrinks from
"recite every agent's kit" to "write 2 sentences about which tier-list
agents match the player's pool".
"""

from __future__ import annotations

from valocoach.retrieval.agents import get_agent, list_agent_names
from valocoach.retrieval.meta import get_meta


def _is_known_agent(name: str) -> bool:
    """True when *name* is in ``agents.json``.

    Returned ``False`` for agents released after the bundled knowledge base
    was last updated.  Callers should still surface unknown agents (the
    user may genuinely play them) but flag the kit/role data as missing.
    """
    return name.casefold() in {n.casefold() for n in list_agent_names()}


_TIER_HEADERS: dict[str, str] = {
    "S": "🏆 S-Tier",
    "A": "🥈 A-Tier",
    "B": "🥉 B-Tier",
    "C": "▫ C-Tier",
}


def _ability_names(agent_name: str) -> list[str]:
    """Return ordered (Q, E, C, X) ability names for *agent_name*."""
    agent = get_agent(agent_name)
    if not agent:
        return []
    return [ab["name"] for ab in agent["abilities"].values()]


def _agent_line(name: str, agent_meta: dict | None) -> str:
    """One bullet line: ``Name (Role) — ability1, ability2, ability3, ability4``.

    Falls back gracefully if the agent isn't in ``agents.json`` (a brand-new
    agent the user pulled before we updated the bundle).
    """
    agent = get_agent(name)
    role = agent["role"] if agent else "?"
    abilities = _ability_names(name)
    abilities_str = ", ".join(abilities) if abilities else "abilities not in bundle"

    line = f"  • {name} ({role}) — {abilities_str}"

    if agent_meta:
        pick = agent_meta.get("pick_rate")
        win = agent_meta.get("win_rate")
        if pick and win:
            line += f"\n    pick {pick} · win {win}"
        reason = agent_meta.get("reason")
        if reason:
            line += f"\n    {reason}"
    return line


def format_tier_list_panel() -> str:
    """Return the full deterministic tier-list block as a multi-line string.

    Layout::

        META — Patch 10.08 · updated 2025-04

        🏆 S-Tier
          • Omen (Controller) — Shrouded Step, Paranoia, Dark Cover, From the Shadows
            pick ~18% · win ~51%
          • Viper (Controller) — Snake Bite, Poison Cloud, Toxic Screen, Viper's Pit
          ...
        🥈 A-Tier
          ...
    """
    meta = get_meta()
    lines: list[str] = []
    lines.append(f"META — Patch {meta.get('patch', '?')} · updated {meta.get('updated', '?')}")

    tier_list = meta.get("tier_list", {}) or {}
    agent_meta_by_name = meta.get("agent_meta", {}) or {}

    for tier_key in ("S", "A", "B", "C"):
        agents = tier_list.get(tier_key, []) or []
        if not agents:
            continue
        lines.append("")
        lines.append(_TIER_HEADERS[tier_key])
        for agent_name in agents:
            # Case-insensitive lookup against agent_meta keys
            meta_key = next(
                (k for k in agent_meta_by_name if k.casefold() == agent_name.casefold()),
                None,
            )
            agent_meta = agent_meta_by_name.get(meta_key) if meta_key else None
            lines.append(_agent_line(agent_name, agent_meta))

    return "\n".join(lines)


def format_player_alignment(top_agents: list[str]) -> str | None:
    """Return a 1-2 line block on how the player's top agents map onto tiers.

    Returns ``None`` when the player has no top-agent data — the LLM
    takeaway prompt will then skip personalisation gracefully.
    """
    if not top_agents:
        return None

    meta = get_meta()
    tier_list = meta.get("tier_list", {}) or {}

    # Build: agent -> tier key (e.g. "Omen" -> "S")
    agent_to_tier: dict[str, str] = {}
    for tier_key in ("S", "A", "B", "C"):
        for name in tier_list.get(tier_key, []) or []:
            agent_to_tier[name.casefold()] = tier_key

    lines = ["YOUR AGENT ALIGNMENT WITH CURRENT META:"]
    for name in top_agents:
        # Skip empty / None entries — match-data corruption occasionally
        # leaves NULL agent_name values that bubble up here.
        if not name or not isinstance(name, str):
            continue
        tier = agent_to_tier.get(name.casefold())
        if tier:
            lines.append(f"  • {name} → {tier}-tier this patch")
        elif _is_known_agent(name):
            lines.append(f"  • {name} → unranked / niche this patch")
        else:
            lines.append(
                f"  • {name} → new agent — not in bundled knowledge base; "
                "ability/role info will be missing until the next data refresh"
            )
    return "\n".join(lines)


def format_full_meta_block(top_agents: list[str] | None = None) -> str:
    """Public entry — returns the full deterministic meta block for printing.

    Composes the tier list + the optional player-alignment block.  This is
    the value the meta-intent code path prints directly (no LLM).
    """
    parts = [format_tier_list_panel()]
    if top_agents:
        alignment = format_player_alignment(top_agents)
        if alignment:
            parts.append("")
            parts.append(alignment)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Deterministic personalised takeaway
# ---------------------------------------------------------------------------

# Per-rate thresholds used in the deterministic stat-summary block.  These are
# the same heuristics the coach used in prose form; pulling them out as
# constants makes the block reproducible and tweakable in one place.
_GOOD_HS_PCT: float = 0.30
_LOW_HS_PCT: float = 0.20
_GOOD_KD: float = 1.10
_LOW_KD: float = 0.85
_MIN_MATCHES_TO_TRUST: int = 5


def format_personalised_takeaway(settings) -> str | None:
    """Return a deterministic personalised takeaway block, or None when the
    player has no synced stats yet.

    The block is built entirely from ``compute_per_agent`` / ``compute_per_map``
    aggregates of the player's match history and the current ``meta.json``
    tier list — no LLM, so zero hallucination risk.  Output is shaped like
    the prose takeaway the LLM used to write, just with verifiable numbers.
    """
    try:
        # Load recent rows the same way build_stats_context does so the
        # numbers shown match what the user already sees in `valocoach stats`.
        from valocoach.data.loader import load_player_data

        data = load_player_data(settings, include_rounds=False)
    except Exception:
        return None

    if data is None or not data.rows:
        return None

    from valocoach.stats.calculator import (
        compute_per_agent,
        compute_per_map,
        compute_player_stats,
    )

    rows = data.rows
    overall = compute_player_stats(rows)
    per_agent = compute_per_agent(rows)
    per_map = compute_per_map(rows)

    meta = get_meta()
    tier_list = meta.get("tier_list", {}) or {}
    tier_of: dict[str, str] = {
        name.casefold(): tier
        for tier in ("S", "A", "B", "C")
        for name in tier_list.get(tier, []) or []
    }

    lines: list[str] = ["PERSONALISED TAKEAWAY (from your synced match history):"]

    # 1. Agent-pool tier alignment summary.  We include every agent the
    #    player has played — including agents released after the bundled
    #    ``agents.json`` was last refreshed (they'll show as "new"). Drop
    #    rows with empty/None agent names (match-data corruption).
    canonical_set = {n.casefold() for n in list_agent_names()}
    per_agent = [a for a in per_agent if a.agent and isinstance(a.agent, str)]
    pool_top = [a for a in per_agent if a.stats.matches >= 1][:5]
    if pool_top:
        aligned: list[str] = []
        for a in pool_top:
            tier = tier_of.get(a.agent.casefold())
            if tier:
                badge = f"{tier}-tier"
            elif a.agent.casefold() in canonical_set:
                badge = "unranked"
            else:
                badge = "new — not yet in knowledge base"
            aligned.append(
                f"{a.agent} ({badge}, {a.stats.matches}g, "
                f"{a.stats.win_rate * 100:.0f}% WR)"
            )
        lines.append(f"  • Agent pool: {', '.join(aligned)}.")

    # 2. Best & worst map by win rate (require at least 2 games to register).
    qualifying_maps = [m for m in per_map if m.stats.matches >= 2]
    if qualifying_maps:
        best_map = max(qualifying_maps, key=lambda m: m.stats.win_rate)
        worst_map = min(qualifying_maps, key=lambda m: m.stats.win_rate)
        if best_map.map_name != worst_map.map_name:
            lines.append(
                f"  • Strongest map: {best_map.map_name} "
                f"({best_map.stats.win_rate * 100:.0f}% WR, {best_map.stats.matches}g) — "
                f"keep queueing it.  Weakest: {worst_map.map_name} "
                f"({worst_map.stats.win_rate * 100:.0f}% WR, {worst_map.stats.matches}g) — "
                "either practise it deliberately or learn to dodge it."
            )

    # 3. Identify the biggest concrete weakness from the overall stats.
    weaknesses: list[str] = []
    if overall.matches >= _MIN_MATCHES_TO_TRUST:
        if overall.hs_pct < _LOW_HS_PCT:
            weaknesses.append(
                f"Headshot % is {overall.hs_pct * 100:.0f}% — well below the "
                f"{_LOW_HS_PCT * 100:.0f}% floor for climbing. Spend 15 min/day "
                "in DM with crosshair-placement focus."
            )
        elif overall.hs_pct < _GOOD_HS_PCT:
            weaknesses.append(
                f"Headshot % is {overall.hs_pct * 100:.0f}% — under the "
                f"{_GOOD_HS_PCT * 100:.0f}% benchmark. Aim trainer 10 min/day "
                "would lift this fastest."
            )
        if overall.kd < _LOW_KD:
            weaknesses.append(
                f"K/D is {overall.kd:.2f} — you're losing trades. Reduce solo "
                "peeks and stick within 5m of a teammate before opening duels."
            )

    if weaknesses:
        lines.append("  • Biggest weakness: " + weaknesses[0])
        for w in weaknesses[1:]:
            lines.append(f"  • Also work on: {w}")

    # 4. Concrete next step based on tier alignment.
    s_tier_in_pool = [a for a in pool_top if tier_of.get(a.agent.casefold()) == "S"]
    if s_tier_in_pool:
        best = max(s_tier_in_pool, key=lambda a: a.stats.win_rate)
        lines.append(
            f"  • Stick with {best.agent} (S-tier this patch, "
            f"{best.stats.win_rate * 100:.0f}% WR in your sample) as your main pick; "
            "it's the lowest-effort path to climbing."
        )

    if len(lines) == 1:
        # Only the header — no actionable data.
        return None
    return "\n".join(lines)


__all__ = [
    "format_full_meta_block",
    "format_personalised_takeaway",
    "format_player_alignment",
    "format_tier_list_panel",
]
