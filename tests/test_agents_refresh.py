"""Regression tests for the deterministic Liquipedia wikitext parser.

``agents-refresh --extract-kits`` parses Liquipedia's ``{{Infobox agent}}``
and ``{{AbilityCard}}`` templates with regex — no LLM, so no hallucination
risk.  These tests freeze a real Liquipedia wikitext fragment (Miks's page
as fetched 2026-05) so future template changes break this test loudly
instead of silently producing wrong kit data in ``agents.json``.

Note: the fixture is intentionally inline (not a separate file) so the
expected schema is visible alongside the assertions.
"""

from __future__ import annotations

from valocoach.cli.commands.agents_refresh import _parse_agent_wikitext

# ---------------------------------------------------------------------------
# Fixtures — frozen wikitext samples from real Liquipedia pages
# ---------------------------------------------------------------------------

# Miks's page (released 2026-03-17). Reflects the actual template layout
# used by Liquipedia as of 2026-05.
_MIKS_WIKITEXT = """\
{{Infobox agent
|name=Miks
|realname=
|country=Croatia
|image=Miks Artwork.png
|class=Controller
|ability=M-Pulse|ability2=Harmonize
|signature=Waveform
|ultimate=Bassquake
|releasedate=2026-03-17
}}

== Abilities ==

{{AbilityCard
|name=M-Pulse
|image=Miks M-Pulse Concuss.png
|hotkey=C
|ability=Basic
|cost=250
|charges=2
|description=EQUIP M-pulse. ALT-FIRE to toggle between Concuss and Healing outputs. FIRE to throw the device.
}}

{{AbilityCard
|name=Harmonize
|image=Miks_Harmonize.png
|hotkey=Q
|ability=Basic
|cost=200
|charges=1
|description=EQUIP Harmonize. Target an ally and FIRE to activate a Combat Stim on yourself and the ally.
}}

{{AbilityCard
|name=Waveform
|image=Miks_Waveform.png
|hotkey=E
|ability=Signature
|cost=100 (1 free per round)
|charges=2
|description=EQUIP a Map Targeter. FIRE to set locations. ALT-FIRE to spawn Smokes at selected locations.
}}

{{AbilityCard
|name=Bassquake
|image=Miks_Bassquake.png
|hotkey=X
|ability=Ultimate
|ultimatecost=8
|description=EQUIP Bassquake. FIRE to build up and unleash Sonic Radiance forwards, knocking back, Deafening and Slowing players.
}}
"""


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_parse_miks_returns_full_kit() -> None:
    """The Miks page must produce a complete entry: role + four abilities."""
    kit = _parse_agent_wikitext(_MIKS_WIKITEXT)

    assert kit is not None
    assert kit["name"] == "Miks"
    assert kit["role"] == "Controller"
    assert set(kit["abilities"].keys()) == {"C", "Q", "E", "X"}


def test_parse_miks_extracts_real_ability_names() -> None:
    """Real names from the AbilityCard ``name=`` field — no LLM in the loop."""
    kit = _parse_agent_wikitext(_MIKS_WIKITEXT)
    assert kit is not None
    abilities = kit["abilities"]

    assert abilities["C"]["name"] == "M-Pulse"
    assert abilities["Q"]["name"] == "Harmonize"
    assert abilities["E"]["name"] == "Waveform"
    assert abilities["X"]["name"] == "Bassquake"


def test_parse_miks_extracts_costs() -> None:
    """Costs come straight from ``cost=`` / ``ultimatecost=`` numeric values."""
    kit = _parse_agent_wikitext(_MIKS_WIKITEXT)
    assert kit is not None

    assert kit["abilities"]["C"]["cost"] == 250
    assert kit["abilities"]["Q"]["cost"] == 200
    # E has a parenthetical "(1 free per round)" — parser must extract 100, not crash.
    assert kit["abilities"]["E"]["cost"] == 100
    # Ultimate uses ult_points, not cost.
    assert kit["abilities"]["X"]["ult_points"] == 8
    assert "cost" not in kit["abilities"]["X"]


def test_parse_miks_extracts_multi_charges() -> None:
    """Abilities with charges > 1 must surface ``charges`` in the output."""
    kit = _parse_agent_wikitext(_MIKS_WIKITEXT)
    assert kit is not None

    assert kit["abilities"]["C"]["charges"] == 2
    assert kit["abilities"]["E"]["charges"] == 2
    # Q has ``charges=1`` — we omit the field since the agents.json schema
    # treats "absent" as the default 1.
    assert "charges" not in kit["abilities"]["Q"]


def test_parse_miks_strips_wikitext_markup_from_descriptions() -> None:
    """Descriptions must end up readable — no ``'''bold'''`` or ``[[link|text]]``."""
    kit = _parse_agent_wikitext(_MIKS_WIKITEXT)
    assert kit is not None

    for ability in kit["abilities"].values():
        assert "'''" not in ability["description"]
        assert "[[" not in ability["description"]
        assert "]]" not in ability["description"]


# ---------------------------------------------------------------------------
# Failure modes — the parser must return None rather than half-fill
# ---------------------------------------------------------------------------


def test_parser_returns_none_when_no_infobox() -> None:
    """Without an Infobox agent template we can't determine role — abort."""
    wikitext = "{{SomeOtherTemplate}}\n{{AbilityCard|name=Foo|hotkey=C|cost=100}}"
    assert _parse_agent_wikitext(wikitext) is None


def test_parser_returns_none_when_role_invalid() -> None:
    """If ``class=`` isn't one of the four canonical roles, refuse to write."""
    wikitext = """\
{{Infobox agent
|name=Foo
|class=Hybrid
}}
{{AbilityCard|name=A|hotkey=C|cost=100|description=test}}
{{AbilityCard|name=B|hotkey=Q|cost=100|description=test}}
{{AbilityCard|name=C|hotkey=E|cost=100|description=test}}
{{AbilityCard|name=D|hotkey=X|ultimatecost=7|description=test}}
"""
    assert _parse_agent_wikitext(wikitext) is None


def test_parser_returns_none_when_ability_missing() -> None:
    """If we don't get all four C/Q/E/X hotkeys, refuse to write a partial kit."""
    wikitext = """\
{{Infobox agent
|name=Foo
|class=Duelist
}}
{{AbilityCard|name=A|hotkey=C|cost=100|description=x}}
{{AbilityCard|name=B|hotkey=Q|cost=100|description=x}}
"""
    # Missing E and X — must NOT produce a half-filled entry.
    assert _parse_agent_wikitext(wikitext) is None


def test_parser_returns_none_on_empty_input() -> None:
    assert _parse_agent_wikitext("") is None
