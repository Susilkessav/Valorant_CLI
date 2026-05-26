"""Regression tests for the deterministic meta response builder.

These lock in the architectural decision (made in response to qwen3:8b /
qwen3:14b ability hallucinations) that ``meta`` intent uses zero LLM —
every word in the panel comes from ``agents.json`` + ``meta.json`` + the
player's stats DB.

We cover:
  * Every tier-list agent shows up in the formatted panel with their real
    role + ability names from ``agents.json``.
  * Agents present in tier_list AND ``agent_meta`` show their
    ``pick · win · reason`` subline.
  * ``format_personalised_takeaway`` gracefully returns ``None`` when no
    match data exists (fresh install).
  * Empty / None ``agent_name`` rows in the DB don't crash the takeaway
    builder — they're filtered out before string operations.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from valocoach.coach.meta_response import (
    format_full_meta_block,
    format_personalised_takeaway,
    format_player_alignment,
    format_tier_list_panel,
)
from valocoach.retrieval.agents import list_agent_names
from valocoach.retrieval.meta import get_meta


# ---------------------------------------------------------------------------
# Tier-list panel: every tier-list agent represented with real abilities
# ---------------------------------------------------------------------------


def test_tier_list_panel_contains_every_tier_list_agent() -> None:
    """No agent listed in ``meta.json::tier_list`` may go missing in the panel."""
    panel = format_tier_list_panel()
    tier_list = get_meta().get("tier_list", {})
    every_agent = [a for t in ("S", "A", "B", "C") for a in tier_list.get(t, [])]

    missing = [a for a in every_agent if a not in panel]
    assert not missing, f"Tier-list agents missing from panel: {missing}"


def test_tier_list_panel_includes_role_and_abilities_for_known_agents() -> None:
    """For agents present in agents.json, the panel must print role + 4 abilities."""
    panel = format_tier_list_panel()
    # Pick a stable representative — Omen has been S-tier since launch
    # and is guaranteed to be in agents.json.
    assert "Omen (Controller)" in panel
    # Real Omen abilities — these are the load-bearing strings the panel must emit.
    for ability in ("Shrouded Step", "Paranoia", "Dark Cover", "From the Shadows"):
        assert ability in panel, f"Omen's {ability} missing from panel"


def test_tier_list_panel_includes_pick_win_reason_for_agents_with_meta() -> None:
    """Agents with full ``agent_meta`` entries must show the pick/win/reason subline."""
    panel = format_tier_list_panel()
    # Omen is in the original agent_meta entries.
    assert "pick" in panel.lower() and "win" in panel.lower()
    # Spot-check the role-specific reasoning prose is present.
    assert "Dark Cover" in panel  # ability name from agents.json
    # The reason field for Omen mentions Dark Cover specifically.
    assert "geometry" in panel.lower() or "smoke" in panel.lower()


def test_every_tier_list_agent_now_has_agent_meta_after_h2_backfill() -> None:
    """REVIEW H2 fix: every tier-list agent has an ``agent_meta`` entry.

    Regression guard: if a future agent is added to ``tier_list`` without a
    corresponding ``agent_meta`` entry, this test fails loudly instead of
    silently rendering as bare ``Name (Role)`` lines.
    """
    meta = get_meta()
    tier_agents = {a for t in ("S", "A", "B", "C") for a in meta["tier_list"].get(t, [])}
    agent_meta_keys = set(meta.get("agent_meta", {}).keys())
    missing = tier_agents - agent_meta_keys
    assert not missing, (
        f"Agents in tier_list without agent_meta entries: {sorted(missing)}. "
        "Add them to meta.json::agent_meta or use `valocoach agents-refresh "
        "--auto-stub-meta`."
    )


# ---------------------------------------------------------------------------
# Player alignment — null/empty handling (REVIEW M1)
# ---------------------------------------------------------------------------


def test_player_alignment_skips_empty_or_none_names() -> None:
    """REVIEW M1: corrupted match data leaves NULL agent_name rows. Don't crash."""
    out = format_player_alignment(["Omen", "", None, "Jett"])  # type: ignore[list-item]
    assert "Omen" in out
    assert "Jett" in out
    # No empty bullet, no None bullet
    assert "• None" not in out
    assert "•  →" not in out  # bullet followed by space-arrow indicates empty name


def test_player_alignment_labels_new_agents_clearly() -> None:
    """An agent in ``agents.json`` but not in ``tier_list`` should label as 'unranked'."""
    # Using a name not in any tier (none currently, since H2 backfill, but a
    # synthetic name not in agents.json hits the 'new agent' branch).
    out = format_player_alignment(["NotARealAgent"])
    assert "new agent" in out.lower() or "not in bundled knowledge base" in out.lower()


def test_player_alignment_returns_none_for_empty_input() -> None:
    assert format_player_alignment([]) is None


# ---------------------------------------------------------------------------
# Personalised takeaway — gracefully handles missing match data
# ---------------------------------------------------------------------------


def test_personalised_takeaway_returns_none_when_no_match_data() -> None:
    """Fresh install: no synced matches → ``None``, not a crash."""
    settings = MagicMock()
    # load_player_data returns None when there's no data
    with patch("valocoach.data.loader.load_player_data", return_value=None):
        result = format_personalised_takeaway(settings)
    assert result is None


def test_personalised_takeaway_returns_none_when_loader_raises() -> None:
    """A DB error in the loader must surface as ``None``, not a stack trace."""
    settings = MagicMock()
    with patch(
        "valocoach.data.loader.load_player_data",
        side_effect=RuntimeError("DB closed"),
    ):
        result = format_personalised_takeaway(settings)
    assert result is None


# ---------------------------------------------------------------------------
# Composition — format_full_meta_block always includes the tier panel
# ---------------------------------------------------------------------------


def test_full_meta_block_always_includes_tier_panel() -> None:
    """The deterministic block must always lead with the tier-list panel."""
    block = format_full_meta_block(top_agents=None)
    assert "META — Patch" in block
    assert "S-Tier" in block


def test_full_meta_block_appends_alignment_when_top_agents_provided() -> None:
    block = format_full_meta_block(top_agents=["Omen", "Jett"])
    assert "YOUR AGENT ALIGNMENT" in block
    # Both supplied agents should appear in the alignment section.
    assert "Omen" in block
    assert "Jett" in block


# ---------------------------------------------------------------------------
# Coverage of every agent in agents.json
# ---------------------------------------------------------------------------


def test_every_agent_in_agents_json_has_four_abilities() -> None:
    """Any agent we ship must have C/Q/E/X abilities — incomplete kits would
    break the meta panel's ``Name (Role) — ability1, ability2, ...`` format."""
    from valocoach.retrieval.agents import get_agent

    for name in list_agent_names():
        agent = get_agent(name)
        assert agent is not None, f"get_agent({name}) returned None"
        assert set(agent["abilities"].keys()) == {"C", "Q", "E", "X"}, (
            f"{name} ability keys: {list(agent['abilities'].keys())}"
        )
