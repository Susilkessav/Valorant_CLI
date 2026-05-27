from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DATA_FILE = Path(__file__).parent / "data" / "meta.json"
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is None:
        with open(_DATA_FILE) as f:
            _cache = json.load(f)
        # ``meta_sync`` stamps ``sync_in_progress: true`` before writing the
        # new tier list and clears it on successful re-ingest.  If we see
        # the flag here, a previous sync crashed between the meta.json
        # write and the vector-store re-ingest — the JSON is fresh but the
        # RAG chunks are stale.  Warn once per cache-miss so the user
        # knows to re-run ``valocoach meta-refresh --force``.
        if _cache.get("sync_in_progress"):
            log.warning(
                "meta.json has sync_in_progress=True — the last "
                "meta-refresh did not complete cleanly.  RAG chunks may "
                "be stale relative to the tier list.  Re-run "
                "`valocoach meta-refresh --force`."
            )
    return _cache


def get_meta() -> dict:
    return _load()


def _lookup_key(values: dict, key: str) -> str | None:
    """Resolve metadata keys case-insensitively without normalising punctuation."""
    folded = key.casefold()
    return next((candidate for candidate in values if candidate.casefold() == folded), None)


def format_meta_context(
    agent: str | None = None,
    map_name: str | None = None,
    data_dir: Path | None = None,
) -> str:
    """Return a compact LLM-injectable meta block.

    Includes the tier list header, economy thresholds, and — when provided —
    map-specific and agent-specific meta snippets.

    Args:
        agent:    Agent name to inject per-agent meta + recent patch changes.
        map_name: Map name to inject map-specific meta + recent patch changes.
        data_dir: Root data directory.  When provided, patch diff facts from
                  ``patch_changes/{patch}.json`` (C3) are injected alongside
                  the agent / map meta so the LLM knows about buff/nerfs that
                  occurred in the current patch.
    """
    meta = _load()
    lines: list[str] = [f"META (Patch {meta['patch']}, updated {meta['updated']})"]

    # Annotate tier-list entries with each agent's role so small models can't
    # silently re-classify them (e.g. writing "Breach (Controller)" when
    # Breach is an Initiator).  Roles come from the agents.json knowledge
    # base; if lookup fails for any reason we fall back to the bare name.
    try:
        from valocoach.retrieval.agents import get_agent

        def _label(name: str) -> str:
            agent = get_agent(name)
            return f"{name} ({agent['role']})" if agent else name
    except Exception:

        def _label(name: str) -> str:
            return name

    tier = meta.get("tier_list", {})
    for rank in ("S", "A", "B", "C"):
        agents = tier.get(rank, [])
        if agents:
            lines.append(f"  {rank}-Tier: {', '.join(_label(a) for a in agents)}")

    eco = meta.get("economy", {})
    lines.append(
        f"Economy thresholds — Full buy: {eco.get('full_buy', 3900)} cr  |  "
        f"Half buy: {eco.get('half_buy', 2400)} cr  |  "
        f"Eco/save: <{eco.get('eco_save', 1600)} cr"
    )

    if map_name:
        map_meta_by_name = meta.get("map_meta", {})
        map_key = _lookup_key(map_meta_by_name, map_name)
        map_meta = map_meta_by_name.get(map_key) if map_key else None
        if map_meta:
            top = ", ".join(map_meta.get("top_agents", []))
            lines.append(f"\n{map_key} meta — Top agents: {top}")
            if map_meta.get("notes"):
                lines.append(f"  {map_meta['notes']}")

    if agent:
        agent_meta_by_name = meta.get("agent_meta", {})
        agent_key = _lookup_key(agent_meta_by_name, agent)
        agent_meta = agent_meta_by_name.get(agent_key) if agent_key else None
        if agent_meta:
            lines.append(
                f"\n{agent_key} meta — Tier: {agent_meta.get('tier', '?')}  |  "
                f"Pick rate: {agent_meta.get('pick_rate', 'N/A')}  |  "
                f"Win rate: {agent_meta.get('win_rate', 'N/A')}"
            )
            if agent_meta.get("reason"):
                lines.append(f"  {agent_meta['reason']}")

    # C4 — inject patch diff facts for the queried agent / map so the LLM
    # has grounded "what changed this patch" context rather than guessing.
    if data_dir is not None:
        patch_version = meta.get("patch", "")
        if patch_version:
            try:
                from valocoach.retrieval.patch_diff import load_patch_changes

                diff = load_patch_changes(patch_version, Path(data_dir))
                if diff:
                    diff_lines: list[str] = []

                    if agent:
                        # Case-insensitive agent lookup in diff
                        diff_agents = diff.get("agents", {})
                        diff_agent_key = next(
                            (k for k in diff_agents if k.casefold() == agent.casefold()),
                            None,
                        )
                        if diff_agent_key:
                            changes = diff_agents[diff_agent_key]
                            diff_lines.append(
                                f"\nPatch {patch_version} changes for {diff_agent_key}:"
                            )
                            for ch in changes:
                                tag = ch.get("change_type", "adjust").upper()
                                ability = f" [{ch['ability']}]" if ch.get("ability") else ""
                                diff_lines.append(f"  [{tag}]{ability} {ch['description']}")

                    if map_name:
                        diff_maps = diff.get("maps", {})
                        diff_map_key = next(
                            (k for k in diff_maps if k.casefold() == map_name.casefold()),
                            None,
                        )
                        if diff_map_key:
                            changes = diff_maps[diff_map_key]
                            diff_lines.append(
                                f"\nPatch {patch_version} changes for {diff_map_key}:"
                            )
                            for ch in changes:
                                tag = ch.get("change_type", "adjust").upper()
                                diff_lines.append(f"  [{tag}] {ch['description']}")

                    if diff_lines:
                        lines.extend(diff_lines)
            except Exception as exc:
                log.debug("C4: could not load patch diff for %s: %s", patch_version, exc)

    # F3 — provenance tag so LLM can cite the source
    lines.append(f"[SOURCE: knowledge_base/meta/{meta.get('patch', 'unknown')}]")

    return "\n".join(lines)
