"""Tests for the intent classifier and templates module.

Coverage targets
----------------
intent.py   — all 9 return paths + every keyword branch
templates.py — PROMPT_TEMPLATES has all 9 keys; PANEL_TITLES has all 9 keys;
               content smoke-tests (non-empty strings, key phrases present)
"""

from __future__ import annotations

from valocoach.coach.intent import IntentType, classify_intent
from valocoach.coach.templates import PANEL_TITLES, PROMPT_TEMPLATES
from valocoach.core.parser import Situation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sit(raw: str, **kwargs) -> Situation:
    """Build a minimal Situation with only the supplied kwargs set."""
    return Situation(raw=raw, **kwargs)


ALL_INTENTS: list[IntentType] = [
    "clutch",
    "post_plant",
    "retake",
    "economy",
    "stat_analysis",
    "agent_info",
    "meta",
    "tactical",
    "general",
]


# ===========================================================================
# 1.  classify_intent — each of the 9 return paths
# ===========================================================================


class TestClutchIntent:
    """Intent 1 — highest priority."""

    def test_clutch_via_parsed_field(self):
        s = _sit("1v3 B site last alive", clutch=(1, 3))
        assert classify_intent(s, s.raw) == "clutch"

    def test_clutch_keyword_last_alive(self):
        s = _sit("I'm last alive what do I do?")
        assert classify_intent(s, s.raw) == "clutch"

    def test_clutch_keyword_1v2(self):
        s = _sit("how do I win a 1v2?")
        assert classify_intent(s, s.raw) == "clutch"

    def test_clutch_keyword_3v5(self):
        s = _sit("We're in a 3 v 5, spike just planted")
        assert classify_intent(s, s.raw) == "clutch"

    def test_clutch_beats_post_plant(self):
        """clutch has higher priority than post_plant."""
        s = _sit("1v2 post plant B site", clutch=(1, 2), phase="post_plant")
        assert classify_intent(s, s.raw) == "clutch"

    def test_clutch_beats_economy(self):
        """clutch has higher priority than economy."""
        s = _sit("clutch situation, eco round", clutch=(1, 3), econ="eco")
        assert classify_intent(s, s.raw) == "clutch"

    def test_4v5_clutch_keyword(self):
        s = _sit("4v5 push coming, what do we do?")
        assert classify_intent(s, s.raw) == "clutch"

    def test_2v3_clutch(self):
        s = _sit("2 v 3 on B, spike unplanted")
        assert classify_intent(s, s.raw) == "clutch"


class TestPostPlantIntent:
    """Intent 2."""

    def test_post_plant_phase_field(self):
        s = _sit("defending after plant", phase="post_plant")
        assert classify_intent(s, s.raw) == "post_plant"

    def test_post_plant_no_clutch(self):
        """Only fires when there's no clutch signal."""
        s = _sit("post-plant positions on B", phase="post_plant")
        assert classify_intent(s, s.raw) == "post_plant"

    def test_post_plant_beats_retake(self):
        """post_plant has higher priority than retake."""
        s = _sit("post plant retake scenario", phase="post_plant")
        assert classify_intent(s, s.raw) == "post_plant"


class TestRetakeIntent:
    """Intent 3."""

    def test_retake_via_phase_field(self):
        s = _sit("we need to retake A", phase="retake")
        assert classify_intent(s, s.raw) == "retake"

    def test_retake_keyword_only(self):
        s = _sit("best way to retake B site on Ascent")
        assert classify_intent(s, s.raw) == "retake"

    def test_retaking_keyword(self):
        s = _sit("we're retaking every round and failing")
        assert classify_intent(s, s.raw) == "retake"

    def test_retake_beats_economy(self):
        """retake has higher priority than economy."""
        s = _sit("retake with eco budget", phase="retake", econ="eco")
        assert classify_intent(s, s.raw) == "retake"

    def test_take_back_keyword(self):
        s = _sit("how do we take back B site after they plant?")
        assert classify_intent(s, s.raw) == "retake"

    def test_re_take_hyphen(self):
        s = _sit("re-take A site with Sage")
        assert classify_intent(s, s.raw) == "retake"


class TestEconomyIntent:
    """Intent 4."""

    def test_econ_via_parsed_field(self):
        s = _sit("should we save?", econ="eco")
        assert classify_intent(s, s.raw) == "economy"

    def test_full_buy_keyword(self):
        s = _sit("should we full buy this round?")
        assert classify_intent(s, s.raw) == "economy"

    def test_half_buy_keyword(self):
        s = _sit("half buy or save?")
        assert classify_intent(s, s.raw) == "economy"

    def test_credits_keyword(self):
        s = _sit("I only have 2400 credits, what do I buy?")
        assert classify_intent(s, s.raw) == "economy"

    def test_save_round_keyword(self):
        s = _sit("save round after losing pistol")
        assert classify_intent(s, s.raw) == "economy"

    def test_force_buy_keyword(self):
        s = _sit("force buy this round?")
        assert classify_intent(s, s.raw) == "economy"

    def test_pistol_round_keyword(self):
        s = _sit("pistol round tips as Reyna?")
        assert classify_intent(s, s.raw) == "economy"

    def test_eco_keyword(self):
        s = _sit("eco round tips")
        assert classify_intent(s, s.raw) == "economy"

    def test_econ_beats_stat(self):
        """economy takes priority over stat_analysis."""
        s = _sit("my stats on eco rounds", econ="eco")
        assert classify_intent(s, s.raw) == "economy"


class TestStatAnalysisIntent:
    """Intent 5."""

    def test_my_stats(self):
        s = _sit("what are my stats?")
        assert classify_intent(s, s.raw) == "stat_analysis"

    def test_my_kd(self):
        s = _sit("my kd is low how do I improve")
        assert classify_intent(s, s.raw) == "stat_analysis"

    def test_my_kda(self):
        s = _sit("analyze my kda please")
        assert classify_intent(s, s.raw) == "stat_analysis"

    def test_headshot_pct(self):
        s = _sit("my headshot % dropped this week")
        assert classify_intent(s, s.raw) == "stat_analysis"

    def test_how_am_i_doing(self):
        s = _sit("how am I doing overall?")
        assert classify_intent(s, s.raw) == "stat_analysis"

    def test_my_performance(self):
        s = _sit("analyze my performance on Ascent")
        assert classify_intent(s, s.raw) == "stat_analysis"

    def test_am_i_improving(self):
        s = _sit("am I improving over the last 10 games?")
        assert classify_intent(s, s.raw) == "stat_analysis"

    def test_my_recent_matches(self):
        s = _sit("show my recent matches breakdown")
        assert classify_intent(s, s.raw) == "stat_analysis"

    def test_stat_beats_agent_info(self):
        """stat_analysis has higher priority than agent_info."""
        s = _sit("my stats with Jett, how does she compare?")
        assert classify_intent(s, s.raw) == "stat_analysis"


class TestAgentInfoIntent:
    """Intent 6."""

    def test_how_does_agent(self):
        s = _sit("how does Killjoy work?")
        assert classify_intent(s, s.raw) == "agent_info"

    def test_what_does_agent(self):
        s = _sit("what does Sova's E do?")
        assert classify_intent(s, s.raw) == "agent_info"

    def test_explain_keyword(self):
        s = _sit("explain Astra's kit to me")
        assert classify_intent(s, s.raw) == "agent_info"

    def test_abilities_keyword(self):
        s = _sit("list Jett abilities and costs")
        assert classify_intent(s, s.raw) == "agent_info"

    def test_ult_cost(self):
        s = _sit("what is Reyna ult cost?")
        assert classify_intent(s, s.raw) == "agent_info"

    def test_how_to_play(self):
        s = _sit("how to play Neon effectively?")
        assert classify_intent(s, s.raw) == "agent_info"

    def test_playstyle_keyword(self):
        s = _sit("what is Viper playstyle in ranked?")
        assert classify_intent(s, s.raw) == "agent_info"

    def test_when_to_use(self):
        s = _sit("when to use Skye's flash vs. Breach's flash?")
        assert classify_intent(s, s.raw) == "agent_info"

    def test_agent_info_beats_meta(self):
        """agent_info has higher priority than meta."""
        s = _sit("how does the best meta agent Jett play?")
        assert classify_intent(s, s.raw) == "agent_info"


class TestMetaIntent:
    """Intent 7."""

    def test_meta_keyword(self):
        s = _sit("what's the meta right now?")
        assert classify_intent(s, s.raw) == "meta"

    def test_tier_list(self):
        s = _sit("give me a tier list for this patch")
        assert classify_intent(s, s.raw) == "meta"

    def test_best_agent(self):
        s = _sit("best agent for ranked solo queue?")
        assert classify_intent(s, s.raw) == "meta"

    def test_current_patch(self):
        s = _sit("what changed in the current patch?")
        assert classify_intent(s, s.raw) == "meta"

    def test_op_agent(self):
        s = _sit("which op agent should I play?")
        assert classify_intent(s, s.raw) == "meta"

    def test_overpowered(self):
        s = _sit("what is overpowered right now?")
        assert classify_intent(s, s.raw) == "meta"

    def test_who_is_strong(self):
        s = _sit("who is strong in this patch?")
        assert classify_intent(s, s.raw) == "meta"

    def test_what_is_good(self):
        s = _sit("what's good in the current meta?")
        assert classify_intent(s, s.raw) == "meta"

    def test_what_is_broken(self):
        s = _sit("what's broken right now?")
        assert classify_intent(s, s.raw) == "meta"

    def test_strong_agent(self):
        s = _sit("which strong agent should I learn?")
        assert classify_intent(s, s.raw) == "meta"

    def test_meta_beats_tactical(self):
        """meta has higher priority than tactical."""
        s = _sit("what's the meta on Ascent attacking side?", map="Ascent", side="attack")
        assert classify_intent(s, s.raw) == "meta"


class TestTacticalIntent:
    """Intent 8 — map AND side present, no higher-priority keyword."""

    def test_tactical_map_and_side(self):
        s = _sit("push A site on Ascent attack side", map="Ascent", side="attack")
        assert classify_intent(s, s.raw) == "tactical"

    def test_tactical_defense(self):
        s = _sit("hold B on Split defense", map="Split", side="defense")
        assert classify_intent(s, s.raw) == "tactical"

    def test_tactical_needs_both(self):
        """Map alone (no side) falls through to general."""
        s = _sit("tips for Ascent", map="Ascent")
        assert classify_intent(s, s.raw) == "general"

    def test_tactical_needs_both_side_only(self):
        """Side alone (no map) falls through to general."""
        s = _sit("how to attack?", side="attack")
        assert classify_intent(s, s.raw) == "general"

    def test_tactical_beats_general(self):
        s = _sit("generic tips on Bind defense", map="Bind", side="defense")
        assert classify_intent(s, s.raw) == "tactical"


class TestGeneralIntent:
    """Intent 9 — fallback."""

    def test_general_fallback(self):
        s = _sit("I keep losing, what should I do?")
        assert classify_intent(s, s.raw) == "general"

    def test_general_no_fields(self):
        s = _sit("tips for winning more rounds")
        assert classify_intent(s, s.raw) == "general"

    def test_general_map_no_side(self):
        s = _sit("I like Ascent but keep losing", map="Ascent")
        assert classify_intent(s, s.raw) == "general"

    def test_general_agents_no_map(self):
        s = _sit("playing Jett but not fragging", agents=["Jett"])
        assert classify_intent(s, s.raw) == "general"


# ===========================================================================
# 2.  Edge cases — priority ordering verification
# ===========================================================================


class TestPriorityOrdering:
    def test_clutch_beats_post_plant_phase(self):
        s = _sit("1v3 post plant spike", clutch=(1, 3), phase="post_plant")
        assert classify_intent(s, s.raw) == "clutch"

    def test_clutch_beats_retake(self):
        s = _sit("1v2 retake B", clutch=(1, 2), phase="retake")
        assert classify_intent(s, s.raw) == "clutch"

    def test_post_plant_beats_retake_phase(self):
        # post_plant phase field beats retake keyword in raw text
        s = _sit("post plant retake scenario", phase="post_plant")
        assert classify_intent(s, s.raw) == "post_plant"

    def test_retake_beats_economy_field(self):
        s = _sit("retake on eco round", phase="retake", econ="eco")
        assert classify_intent(s, s.raw) == "retake"

    def test_economy_beats_stat(self):
        s = _sit("my stats on eco rounds are bad", econ="eco")
        assert classify_intent(s, s.raw) == "economy"

    def test_stat_beats_agent_info_keyword(self):
        # "how am I doing" fires stat_analysis before agent_info keyword
        s = _sit("how am I doing with Jett, explain her kit")
        assert classify_intent(s, s.raw) == "stat_analysis"

    def test_agent_info_beats_meta(self):
        s = _sit("how does the meta agent Jett work?")
        # "how does" fires agent_info before "meta" keyword
        assert classify_intent(s, s.raw) == "agent_info"

    def test_meta_beats_tactical(self):
        s = _sit("what's meta on Ascent defense?", map="Ascent", side="defense")
        assert classify_intent(s, s.raw) == "meta"


# ===========================================================================
# 3.  Keyword boundary tests — regex should NOT match inside larger words
# ===========================================================================


class TestKeywordBoundaries:
    def test_no_false_positive_attackers(self):
        """'attackers' in a non-clutch context should not confuse classifier."""
        s = _sit("the attackers keep rushing B on Haven", map="Haven", side="attack")
        # Side is 'attack' extracted, map present → tactical (no clutch/econ/etc.)
        assert classify_intent(s, s.raw) == "tactical"

    def test_eco_boundary_ecology(self):
        """'ecology' should not trigger economy intent."""
        s = _sit("the ecology of Bind's layout is interesting")
        assert classify_intent(s, s.raw) == "general"

    def test_meta_boundary_metadata(self):
        """'metadata' should not trigger meta intent."""
        s = _sit("I want to understand the metadata in my match history")
        # 'my match history' → no econ/stat keyword match — might be general
        # (metadata has 'meta' as prefix but our regex uses word boundary)
        result = classify_intent(s, s.raw)
        # Either general or stat_analysis is fine; must NOT be "meta"
        assert result != "meta"

    def test_retake_partial_no_match(self):
        """'retaken' should not trigger retake intent."""
        s = _sit("the site was retaken by the enemy quickly")
        # 'retaken' — word boundary should block 'retake' match
        # (the regex pattern is 'retake' followed by non-word)
        # Actually 'retaken' ends with 'n' so (?!\w) after 'retake' would fail
        result = classify_intent(s, s.raw)
        # Should not be retake — fallback to tactical if map+side, else general
        assert result != "retake"


# ===========================================================================
# 4.  PROMPT_TEMPLATES — completeness and content smoke-tests
# ===========================================================================


class TestPromptTemplates:
    def test_all_intents_have_template(self):
        for intent in ALL_INTENTS:
            assert intent in PROMPT_TEMPLATES, f"Missing template for '{intent}'"

    def test_all_templates_are_nonempty_strings(self):
        for intent, tmpl in PROMPT_TEMPLATES.items():
            assert isinstance(tmpl, str) and len(tmpl) > 50, (
                f"Template '{intent}' is too short or not a string"
            )

    def test_no_extra_keys(self):
        assert set(PROMPT_TEMPLATES.keys()) == set(ALL_INTENTS)

    def test_clutch_template_has_decision_tree_section(self):
        assert "Primary play" in PROMPT_TEMPLATES["clutch"]

    def test_post_plant_template_mentions_spike_timer(self):
        assert "spike" in PROMPT_TEMPLATES["post_plant"].lower()

    def test_retake_template_mentions_rotation(self):
        assert "Rotate" in PROMPT_TEMPLATES["retake"] or "rotate" in PROMPT_TEMPLATES["retake"]

    def test_economy_template_has_credit_thresholds(self):
        assert "2 000" in PROMPT_TEMPLATES["economy"] or "2000" in PROMPT_TEMPLATES["economy"]

    def test_stat_analysis_template_no_forced_markdown(self):
        """stat_analysis template must NOT force the 5-section tactical layout."""
        tactical_sections = ["Pre-round", "Execute", "If it breaks"]
        for section in tactical_sections:
            assert section not in PROMPT_TEMPLATES["stat_analysis"], (
                f"stat_analysis template contains tactical section '{section}'"
            )

    def test_agent_info_template_references_grounded_context(self):
        assert "GROUNDED CONTEXT" in PROMPT_TEMPLATES["agent_info"]

    def test_meta_template_has_tier_section(self):
        assert "strong picks" in PROMPT_TEMPLATES["meta"].lower() or "tier" in PROMPT_TEMPLATES["meta"].lower()

    def test_tactical_template_has_all_five_sections(self):
        tmpl = PROMPT_TEMPLATES["tactical"]
        for section in ["Read", "Pre-round", "Execute", "If it breaks", "Key detail"]:
            assert section in tmpl, f"tactical template missing section '{section}'"

    def test_general_template_no_forced_structure(self):
        """general template should explicitly discourage rigid structure."""
        assert "five-section" in PROMPT_TEMPLATES["general"] or "template" in PROMPT_TEMPLATES["general"]

    def test_templates_contain_grounding_rules(self):
        """All templates embed the grounding rules (callouts, ability costs)."""
        for intent, tmpl in PROMPT_TEMPLATES.items():
            assert "callout" in tmpl.lower() or "GROUNDED CONTEXT" in tmpl, (
                f"Template '{intent}' appears to be missing grounding rules"
            )


# ===========================================================================
# 5.  PANEL_TITLES — completeness
# ===========================================================================


class TestPanelTitles:
    def test_all_intents_have_title(self):
        for intent in ALL_INTENTS:
            assert intent in PANEL_TITLES, f"Missing panel title for '{intent}'"

    def test_all_titles_are_nonempty_strings(self):
        for intent, title in PANEL_TITLES.items():
            assert isinstance(title, str) and len(title) >= 3, (
                f"Panel title '{intent}' is too short or not a string"
            )

    def test_no_extra_keys(self):
        assert set(PANEL_TITLES.keys()) == set(ALL_INTENTS)

    def test_titles_are_distinct(self):
        titles = list(PANEL_TITLES.values())
        assert len(titles) == len(set(titles)), "Duplicate panel titles found"

    def test_clutch_title_has_emoji(self):
        assert "⚡" in PANEL_TITLES["clutch"]

    def test_economy_title_mentions_economy(self):
        assert "Economy" in PANEL_TITLES["economy"] or "💰" in PANEL_TITLES["economy"]

    def test_stat_title_has_chart_emoji(self):
        assert "📊" in PANEL_TITLES["stat_analysis"]


# ===========================================================================
# 6.  Return type annotation — IntentType is a Literal with 9 values
# ===========================================================================


class TestIntentTypeAnnotation:
    def test_literal_values(self):
        import typing

        args = typing.get_args(IntentType)
        assert set(args) == set(ALL_INTENTS), f"IntentType args mismatch: {args}"
