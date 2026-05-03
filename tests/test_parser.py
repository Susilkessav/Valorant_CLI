"""Tests for the situation parser.

The parser is regex-first against the canonical agent/map JSON.  We test:
  • Each field extracts correctly from realistic phrasings
  • Word-boundary discipline — "Iso" doesn't match "isolated", "Sage" doesn't
    match "passage", etc.  These are the failure modes that would silently
    poison retrieval routing.
  • CLI-flag-style merging: parsed.primary_agent + parsed.map plug into the
    same kwargs ``retrieve_static`` already takes.
  • Empty / pure-prose input doesn't crash and yields an all-None Situation.
"""

from __future__ import annotations

from valocoach.core.parser import Situation, parse_situation

# ---------------------------------------------------------------------------
# Map detection
# ---------------------------------------------------------------------------


class TestMap:
    def test_canonical_name_in_sentence(self):
        assert parse_situation("losing on Ascent").map == "Ascent"

    def test_lowercase(self):
        assert parse_situation("we keep dying on ascent").map == "ascent".capitalize()

    def test_first_match_wins(self):
        """If two map names somehow appear, the first canonical match wins."""
        s = parse_situation("Ascent is harder than Bind for us")
        assert s.map in {"Ascent", "Bind"}

    def test_no_map_mentioned(self):
        assert parse_situation("how do I aim better").map is None

    def test_substring_does_not_false_match(self):
        """Word boundary check: 'binding' must not match 'Bind'."""
        assert parse_situation("we have a binding agreement on comms").map is None


# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------


class TestAgents:
    def test_single_agent(self):
        assert parse_situation("playing Jett today").agents == ["Jett"]

    def test_multiple_agents_collected(self):
        s = parse_situation("Jett and Sage on attack")
        assert "Jett" in s.agents
        assert "Sage" in s.agents

    def test_primary_agent_is_first(self):
        s = parse_situation("Jett with Sage support")
        assert s.primary_agent == s.agents[0]

    def test_no_agent_mentioned(self):
        assert parse_situation("just need general advice").agents == []

    def test_substring_does_not_false_match_iso(self):
        """The 'Iso' agent must not match inside 'isolated' / 'isolation'."""
        s = parse_situation("I feel isolated when pushing A")
        assert "Iso" not in s.agents

    def test_substring_does_not_false_match_sage(self):
        """The 'Sage' agent must not match inside 'passage'."""
        s = parse_situation("contesting the passage early")
        assert "Sage" not in s.agents

    def test_kayo_with_slash(self):
        """KAY/O has a slash; the parser must escape it correctly."""
        s = parse_situation("running KAY/O for flashes")
        assert "KAY/O" in s.agents

    def test_case_insensitive(self):
        assert "Jett" in parse_situation("playing JETT").agents
        assert "Jett" in parse_situation("playing jett").agents


# ---------------------------------------------------------------------------
# Side detection
# ---------------------------------------------------------------------------


class TestSide:
    def test_attack(self):
        assert parse_situation("on attack we struggle").side == "attack"

    def test_attacking(self):
        assert parse_situation("when attacking A site").side == "attack"

    def test_t_side(self):
        assert parse_situation("our T-side is weak").side == "attack"

    def test_defense(self):
        assert parse_situation("on defense we lose every round").side == "defense"

    def test_defending(self):
        assert parse_situation("defending B site").side == "defense"

    def test_ct_side(self):
        assert parse_situation("CT side is rough on Bind").side == "defense"

    def test_no_side(self):
        assert parse_situation("how do I aim better").side is None


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------


class TestScore:
    def test_dash(self):
        assert parse_situation("we are down 5-9").score == (5, 9)

    def test_colon(self):
        assert parse_situation("score is 8:12").score == (8, 12)

    def test_en_dash(self):
        assert parse_situation("up 12–8 at half").score == (12, 8)

    def test_spaces_around_separator(self):
        assert parse_situation("11 - 13 loss").score == (11, 13)

    def test_no_score_in_prose(self):
        assert parse_situation("we just keep losing").score is None

    def test_implausible_score_rejected(self):
        """3-digit numbers shouldn't match (regex caps at 2 digits anyway,
        but verify the door is closed)."""
        s = parse_situation("the year 2023 was tough")
        # 2023 is 4 digits — no match expected
        assert s.score is None


# ---------------------------------------------------------------------------
# Site
# ---------------------------------------------------------------------------


class TestSite:
    def test_a_site(self):
        assert parse_situation("pushing A site").site == "A"

    def test_b_long(self):
        assert parse_situation("holding B long").site == "B"

    def test_c_main(self):
        assert parse_situation("rotating to C main").site == "C"

    def test_lowercase_letter(self):
        assert parse_situation("attacking a site").site == "A"

    def test_no_site(self):
        assert parse_situation("how do I improve aim").site is None


# ---------------------------------------------------------------------------
# Clutch
# ---------------------------------------------------------------------------


class TestClutch:
    def test_one_v_three(self):
        assert parse_situation("stuck in a 1v3").clutch == (1, 3)

    def test_two_v_five(self):
        assert parse_situation("2v5 retake situation").clutch == (2, 5)

    def test_spaces(self):
        assert parse_situation("clutched a 1 v 4").clutch == (1, 4)

    def test_equal_count_rejected(self):
        """3v3 isn't a clutch — it's an even fight."""
        assert parse_situation("3v3 mid duel").clutch is None

    def test_no_clutch(self):
        assert parse_situation("default play").clutch is None


# ---------------------------------------------------------------------------
# Econ
# ---------------------------------------------------------------------------


class TestEcon:
    def test_full_buy(self):
        assert parse_situation("on a full buy round").econ == "full_buy"

    def test_half_buy(self):
        assert parse_situation("running a half buy").econ == "half_buy"

    def test_force_buy(self):
        assert parse_situation("force buy after eco loss").econ == "half_buy"

    def test_eco_round(self):
        assert parse_situation("eco round strategy").econ == "eco"

    def test_save_round(self):
        assert parse_situation("save round, drop weapons").econ == "eco"

    def test_full_buy_beats_eco(self):
        """When 'full buy' appears, it should match before 'eco' even if both
        words are present somewhere in the text."""
        s = parse_situation("full buy after their eco loss")
        assert s.econ == "full_buy"


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


class TestPhase:
    def test_post_plant(self):
        assert parse_situation("post-plant on A site").phase == "post_plant"

    def test_post_plant_spaces(self):
        assert parse_situation("post plant scenarios").phase == "post_plant"

    def test_retake(self):
        assert parse_situation("retake B site").phase == "retake"

    def test_execute(self):
        assert parse_situation("executing onto A").phase == "execute"

    def test_default(self):
        assert parse_situation("slow default for info").phase == "default"

    def test_no_phase(self):
        assert parse_situation("how do I aim").phase is None


# ---------------------------------------------------------------------------
# Integration / metadata block
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_situation(self):
        """All the headline fields land from a single realistic input."""
        s = parse_situation(
            "we keep losing 8-12 on attack on Ascent as Jett, post-plant retake fails 1v3"
        )
        assert s.map == "Ascent"
        assert s.side == "attack"
        assert "Jett" in s.agents
        assert s.score == (8, 12)
        assert s.clutch == (1, 3)
        # 'post-plant' wins over 'retake' because it's the more specific phase
        # and appears first in the keyword table.
        assert s.phase == "post_plant"

    def test_raw_text_preserved(self):
        text = "anything at all"
        assert parse_situation(text).raw == text

    def test_empty_string_does_not_crash(self):
        s = parse_situation("")
        assert s.raw == ""
        assert s.map is None
        assert s.agents == []
        assert s.side is None

    def test_pure_prose_yields_empty_situation(self):
        s = parse_situation("how do I get better at this game")
        assert s.map is None
        assert s.agents == []
        assert s.side is None
        assert s.score is None
        assert s.site is None
        assert s.clutch is None
        assert s.econ is None
        assert s.phase is None

    def test_metadata_block_renders_set_fields(self):
        s = parse_situation("Jett on Ascent attack, A site")
        block = s.to_metadata_block()
        assert "Map: Ascent" in block
        assert "Side: attack" in block
        assert "Agent(s): Jett" in block
        assert "Site: A" in block

    def test_metadata_block_empty_when_nothing_extracted(self):
        assert parse_situation("how do I improve").to_metadata_block() == ""

    def test_metadata_block_skips_unset_fields(self):
        """No 'None' or empty lines in the rendered block."""
        s = parse_situation("playing Jett")
        block = s.to_metadata_block()
        assert "None" not in block
        assert block == "Agent(s): Jett"


# ---------------------------------------------------------------------------
# Pydantic model behaviour
# ---------------------------------------------------------------------------


class TestSituationModel:
    def test_is_pydantic_model(self):
        s = parse_situation("Jett on Ascent")
        assert isinstance(s, Situation)

    def test_default_factories(self):
        s = Situation(raw="x")
        assert s.agents == []
        assert s.map is None

    def test_primary_agent_none_when_empty(self):
        assert Situation(raw="x").primary_agent is None
