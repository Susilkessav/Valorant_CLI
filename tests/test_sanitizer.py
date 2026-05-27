"""Tests for valocoach.coach.sanitizer.validate_ability_claims.

Focus on the two failure modes the sanitizer is designed to catch:
  1. Cross-attribution — a real ability assigned to the wrong agent.
  2. Pure hallucination — a phrase that doesn't match any real ability
     anywhere in agents.json.

Plus the common false-positive guards (generic descriptors, weapon names,
map callouts) that previously tripped the regex.
"""

from __future__ import annotations

from valocoach.coach.sanitizer import (
    AbilityWarning,
    validate_ability_claims,
)

# ---------------------------------------------------------------------------
# Correct attributions — must NOT trigger
# ---------------------------------------------------------------------------


def test_correct_attribution_omen_dark_cover() -> None:
    """Omen's Dark Cover is real — no warning."""
    assert validate_ability_claims("Use Omen's Dark Cover to smoke A Long.") == []


def test_correct_attribution_multi_word_ability() -> None:
    """Multi-word abilities (Tour de Force, From the Shadows) must match."""
    text = "Chamber's Tour de Force is a one-shot Op. Omen's From the Shadows is map-wide."
    assert validate_ability_claims(text) == []


def test_correct_attribution_apostrophe_in_ability() -> None:
    """Abilities containing apostrophes (Viper's Pit) match correctly."""
    assert validate_ability_claims("Viper's Viper's Pit denies the site.") == []


# ---------------------------------------------------------------------------
# Cross-attribution — real ability, wrong agent
# ---------------------------------------------------------------------------


def test_cross_attribution_fade_paranoia() -> None:
    """Paranoia belongs to Omen, not Fade — should flag."""
    warnings = validate_ability_claims("Fade's Paranoia clears B site.")
    assert len(warnings) == 1
    w = warnings[0]
    assert w.agent.lower() == "fade"
    assert w.claimed_ability.lower() == "paranoia"
    assert w.real_owner == "Omen"


def test_cross_attribution_jett_dark_cover() -> None:
    """Dark Cover is Omen's — wrong on Jett."""
    warnings = validate_ability_claims("Jett's Dark Cover blocks vision.")
    assert any(w.real_owner == "Omen" for w in warnings)


# ---------------------------------------------------------------------------
# Pure hallucination — phrase isn't any agent's ability
# ---------------------------------------------------------------------------


def test_pure_hallucination_riftwalk() -> None:
    """Riftwalk isn't a Valorant ability — should flag with real_owner=None."""
    warnings = validate_ability_claims("Use Omen's Riftwalk to push.")
    assert len(warnings) == 1
    assert warnings[0].real_owner is None
    assert warnings[0].claimed_ability.lower() == "riftwalk"


def test_pure_hallucination_crimson_hunt() -> None:
    warnings = validate_ability_claims("Omen's Crimson Hunt secures kills.")
    assert any(w.real_owner is None for w in warnings)


# ---------------------------------------------------------------------------
# False positives — must NOT trigger
# ---------------------------------------------------------------------------


def test_no_flag_for_generic_role_descriptor() -> None:
    """'Jett Duelist' / 'Omen Controller' are role labels, not abilities."""
    assert validate_ability_claims("Jett Duelist plays aggressively.") == []
    assert validate_ability_claims("Omen Controller controls map flow.") == []


def test_no_flag_for_agent_plus_weapon() -> None:
    """'Jett Operator' / 'Omen Vandal' are loadout calls, not abilities."""
    assert validate_ability_claims("Jett with Operator on Ascent.") == []
    assert validate_ability_claims("Omen Vandal full buy.") == []


def test_no_flag_for_agent_plus_map_callout() -> None:
    """'Jett Ascent' / 'Omen Split' are map names, not abilities."""
    assert validate_ability_claims("Jett dominates Ascent.") == []
    assert validate_ability_claims("Omen on Split.") == []


def test_no_flag_for_lowercase_phrase() -> None:
    """Lowercase 'mobility' isn't TitleCased so it isn't an ability candidate."""
    assert validate_ability_claims("Jett's mobility wins duels.") == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_same_warning_emitted_once() -> None:
    """If the model writes 'Omen's Riftwalk' 5 times we only warn once."""
    text = "Omen's Riftwalk. Omen's Riftwalk. Omen's Riftwalk."
    assert len(validate_ability_claims(text)) == 1


# ---------------------------------------------------------------------------
# Empty / null handling
# ---------------------------------------------------------------------------


def test_empty_text_returns_empty_list() -> None:
    assert validate_ability_claims("") == []


def test_text_with_no_agent_names_returns_empty() -> None:
    assert validate_ability_claims("Push the site with utility.") == []


# ---------------------------------------------------------------------------
# AbilityWarning.format()
# ---------------------------------------------------------------------------


def test_format_cross_attribution() -> None:
    w = AbilityWarning(
        agent="Fade",
        claimed_ability="Paranoia",
        real_owner="Omen",
        category="cross_attribution",
        snippet="",
    )
    assert "Paranoia" in w.format()
    assert "Omen" in w.format()


def test_format_hallucination() -> None:
    w = AbilityWarning(
        agent="Omen",
        claimed_ability="Riftwalk",
        real_owner=None,
        category="hallucination",
        snippet="",
    )
    assert "not a Valorant ability" in w.format()


# ---------------------------------------------------------------------------
# Real-world regression: a frozen qwen3:14b response (2026-05-17)
# ---------------------------------------------------------------------------
#
# Captured from a `valocoach coach "what's the current meta"` run during the
# session that produced the deterministic-meta refactor. This was the exact
# hallucination pattern that motivated the sanitizer: fabricated "Riftwalk"
# / "Venomous Bite" / "Warden" ultimates and weapon names treated as
# abilities (e.g. "Fade's Ghost"). The sanitizer's job is to surface enough
# of these that the user knows to verify.

_QWEN3_HALLUCINATED_PARAGRAPH = """\
Current Meta (Patch 10.08) & Your Context

S-Tier Agents:

1 Omen (Controller)
   - Key Abilities: Ghost (stealth), Smoke (area control), Flash (disables)
   - Maps: Split, Ascent, Bind.
2 Viper (Controller)
   - Key Abilities: Nerve Gas (slow + damage), Smoke (vision control).
3 Killjoy (Sentinel)
   - Key Abilities: Disruptor (area denial), Smoke (vision control).
4 Jett (Duelist)
   - Key Abilities: Dash (positioning), Blitz (burst damage), Smoke (control).

Use Fade's Ghost or Jett's Dash to reposition and avoid deaths.
"""


def test_real_qwen3_hallucination_catches_multiple_categories() -> None:
    """The sanitizer must flag a substantial portion of a real
    qwen3:14b hallucinated response across all four categories."""
    warnings = validate_ability_claims(_QWEN3_HALLUCINATED_PARAGRAPH)

    # We don't pin the exact count (set heuristics evolve) but the sanitizer
    # must catch at least one of each category that's present in the fixture.
    categories = {w.category for w in warnings}
    assert "weapon" in categories, "Ghost as ability for Omen/Fade should flag as weapon"
    assert "generic" in categories, "Smoke/Flash/Dash/Blitz should flag as generic"
    assert "hallucination" in categories, "Nerve Gas / Disruptor should flag as hallucination"

    # Specific high-stakes claims that we explicitly want surfaced:
    flagged_claims = {w.claimed_ability.lower() for w in warnings}
    assert "ghost" in flagged_claims
    assert "smoke" in flagged_claims
    assert "nerve gas" in flagged_claims
    assert "disruptor" in flagged_claims


def test_real_qwen3_hallucination_does_not_flood() -> None:
    """A response with ~10 hallucinations should produce a manageable number
    of warnings (not 50+).  Dedup must work."""
    warnings = validate_ability_claims(_QWEN3_HALLUCINATED_PARAGRAPH)
    assert 5 <= len(warnings) <= 25, (
        f"Sanitizer returned {len(warnings)} warnings for a 10-claim fixture; "
        "either dedup is broken or detection is too aggressive."
    )


# ---------------------------------------------------------------------------
# Section-scoped scan — catches "Key Abilities: <fake>, <fake>" bullets
# ---------------------------------------------------------------------------


def test_section_scan_catches_key_abilities_block() -> None:
    """The bullet-list pattern from real qwen3:8b output: agent header followed
    by 'Key Abilities:' with hallucinated names. Each fake should be flagged."""
    text = (
        "1. Omen (Controller)\n"
        "   • Role: Map control, vision denial.\n"
        "   • Key Abilities: Ghost (stealth), Smoke (area control), Flash (disables).\n"
        "   • Maps: Split, Ascent, Bind.\n"
    )
    warnings = validate_ability_claims(text)
    flagged = {w.claimed_ability.lower() for w in warnings}
    assert "ghost" in flagged  # weapon, not an ability
    assert "smoke" in flagged  # generic
    assert "flash" in flagged  # generic


def test_section_scan_flags_nerve_gas_under_viper() -> None:
    """Pure hallucination inside a Key Abilities bullet for the right agent."""
    text = (
        "2. Viper (Controller)\n"
        "   • Key Abilities: Nerve Gas (slow + damage), Smoke (vision control).\n"
    )
    warnings = validate_ability_claims(text)
    assert any(
        w.claimed_ability.lower() == "nerve gas" and w.category == "hallucination" for w in warnings
    )


def test_section_scan_flags_disruptor_under_killjoy() -> None:
    text = (
        "Killjoy (Sentinel):\n  - Key Abilities: Disruptor (area denial), Smoke (vision control).\n"
    )
    warnings = validate_ability_claims(text)
    assert any(w.claimed_ability.lower() == "disruptor" for w in warnings)


def test_section_scan_does_not_flag_real_abilities() -> None:
    """A correct Key Abilities line must produce zero warnings."""
    text = (
        "Omen (Controller):\n"
        "  - Key Abilities: Dark Cover, Paranoia, Shrouded Step, From the Shadows.\n"
    )
    assert validate_ability_claims(text) == []


def test_section_scan_does_not_flag_maps_after_agent() -> None:
    """'Maps: Split, Ascent, Bind' under an agent must not be confused with abilities."""
    text = "Jett (Duelist):\n  - Maps: Split, Ascent, Breeze.\n"
    assert validate_ability_claims(text) == []


# ---------------------------------------------------------------------------
# Weapons / generics: only flagged on possessive
# ---------------------------------------------------------------------------


def test_possessive_weapon_is_flagged() -> None:
    """'Fade's Ghost' (claiming Ghost is an ability) gets flagged."""
    warnings = validate_ability_claims("Use Fade's Ghost to reposition.")
    assert any(w.category == "weapon" and w.claimed_ability == "Ghost" for w in warnings)


def test_possessive_generic_is_flagged() -> None:
    """'Omen's Smoke' (Smoke is a generic noun, not Omen's ability) gets flagged."""
    warnings = validate_ability_claims("Prioritize Omen's Smoke for site control.")
    assert any(w.category == "generic" and w.claimed_ability == "Smoke" for w in warnings)


def test_non_possessive_loadout_is_not_flagged() -> None:
    """'Omen Vandal full buy' is loadout language, not an ability claim."""
    assert validate_ability_claims("Omen Vandal full buy.") == []
    assert validate_ability_claims("Jett with Operator on Ascent.") == []
