from __future__ import annotations

import logging
import re

from valocoach.cli import display
from valocoach.coach import build_stats_context
from valocoach.coach.intent import classify_intent
from valocoach.coach.templates import PANEL_TITLES, PROMPT_TEMPLATES
from valocoach.core.config import load_settings
from valocoach.core.context_budget import fit_prompt
from valocoach.core.parser import Situation, parse_situation
from valocoach.llm.provider import stream_completion
from valocoach.retrieval import format_agent_context, format_map_context

log = logging.getLogger(__name__)

_META_SENSITIVE_INTENTS: frozenset[str] = frozenset({"meta", "agent_info"})
_PATCH_STALE_THRESHOLD_DAYS: int = 21


_STALE_META_WARNED: bool = False


def _maybe_warn_stale_meta(settings, *, once: bool = False) -> None:
    """Print a one-liner if the cached patch is older than the threshold.

    Used by both the LLM path (post-stream warning) and the deterministic
    meta path (printed before the early return) so the warning fires on
    every meta-sensitive answer regardless of which code path produced it.

    Args:
        once: when True, suppresses the warning after it has fired once
              in the current process.  Lets non-meta entrypoints
              (``stats``, ``profile``, ``coach`` for non-meta intents)
              show it without bombarding the user inside a single
              interactive session.
    """
    global _STALE_META_WARNED
    if once and _STALE_META_WARNED:
        return
    try:
        from valocoach.retrieval.patch_tracker import get_patch_staleness_days

        stale_days = get_patch_staleness_days(settings.data_dir)
        if stale_days is None or stale_days > _PATCH_STALE_THRESHOLD_DAYS:
            age_str = (
                "never checked" if stale_days is None else f"{stale_days:.0f}d since last check"
            )
            display.console.print(
                f"[muted]! Meta info may be outdated ({age_str}) — "
                "run [info]valocoach patch --check[/info] to refresh.[/muted]"
            )
            _STALE_META_WARNED = True
    except Exception:
        log.debug("patch staleness check failed", exc_info=True)


def warn_stale_meta_once(settings) -> None:
    """Public entry: fire the staleness warning at most once per process.

    Called by non-coach commands (``stats``, ``profile``) so the user sees
    the staleness signal even on workflows that never touch the coach.
    """
    _maybe_warn_stale_meta(settings, once=True)


# Team-roster questions ("who was in my team", "list teammates") — the schema
# stores teammate puuids but not their display names, so the LLM has no source
# of truth.  Detecting these queries lets us inject an explicit data-availability
# contract so the model stops conflating "your top-played agents" with
# "teammates in recent matches".
_TEAM_ROSTER_KW = re.compile(
    r"(?<!\w)"
    r"(?:who(?:\s+all)?\s+(?:was|were|are|is)\s+(?:in\s+)?my\s+team"
    r"|my\s+teammates?(?:'?\s+names?)?"
    r"|list\s+(?:my\s+)?teammates?"
    r"|teammate\s+names?"
    r"|players?\s+in\s+my\s+team"
    r"|team\s+roster)"
    r"(?!\w)",
    re.IGNORECASE,
)

_TEAM_ROSTER_CONTRACT = (
    "DATA AVAILABILITY — TEAM ROSTER (hard contract, overrides any inference):\n"
    "1. The match database does NOT track teammate display names — only puuids "
    "and the agent each teammate played.\n"
    "2. Your FIRST sentence MUST be exactly: \"I don't have teammate names "
    'stored — only the agents they played in each match."\n'
    "3. You MAY offer to list the agent composition of recent matches if such "
    "data is provided in the user message; otherwise say it isn't available "
    "in the prompt this turn.\n"
    "4. DO NOT list the user's own top-played agents (from PLAYER CONTEXT) "
    "as their teammates. Those are agents the USER plays.\n"
    "5. DO NOT invent enemy agents, names, or compositions."
)

# Post-game LLM occasionally echoes chat-template scaffolding ("User:",
# "Response:", "Final Answer\n") inside the panel.  These stops cut the stream
# before that garbage appears.  Only applied to post_game where the section
# template is rigid enough that no legitimate output starts with those tokens.
_POST_GAME_STOP_TOKENS: list[str] = [
    "\nUser:",
    "\n\nUser:",
    "\nResponse:",
    "\n\nResponse:",
    "\nFinal Answer",
    "\n\nFinal Answer",
]

# Per-intent output caps.  Most intents inherit ``settings.llm_max_tokens``.
# We don't cap meta/agent_info because qwen3:8b emits ~hundreds of internal
# "thinking" tokens before any visible output, so an aggressive cap silently
# eats the entire answer.  The sanitizer audits hallucinations after the fact
# regardless of length, so output-length isn't worth fighting at this layer.
_INTENT_MAX_TOKENS: dict[str, int] = {}


def _build_system_prompt(
    base_prompt: str,
    grounded_context: str | None,
    stats_context: str | None,
    notes_context: str | None = None,
) -> str:
    parts = [base_prompt]
    if grounded_context:
        parts.append(
            f"---\n\nGROUNDED CONTEXT (use these facts verbatim — do not hallucinate abilities or callouts):\n\n{grounded_context}"
        )
    if stats_context:
        parts.append(f"---\n\n{stats_context}")
    if notes_context:
        parts.append(f"---\n\n{notes_context}")
    return "\n\n".join(parts)


def _build_grounded_context(
    agent: str | None,
    map_: str | None,
    situation: str,
    side: str | None,
    data_dir,
    *,
    extra_agents: list[str] | None = None,
) -> str | None:
    """Build the GROUNDED CONTEXT block fed to the LLM.

    ``extra_agents`` are the player's most-played agents (from
    ``coach.context.get_top_played_agents``).  Their AGENT blocks are
    prepended to the vector-retrieved chunks so that even when the user
    asks an agent-less question ("how do I rank up?"), the LLM has the
    real ability lists for agents the player actually uses — preventing
    small models like qwen3:8b from hallucinating ("Fade's Smoke",
    "Fade's Paranoia", etc.).
    """
    from valocoach.retrieval.retriever import retrieve_static

    if agent and not format_agent_context(agent):
        display.warn(f"Agent '{agent}' not found in knowledge base — coach may improvise.")
    if map_ and not format_map_context(map_):
        display.warn(f"Map '{map_}' not found in knowledge base — coach may improvise.")

    result = retrieve_static(situation, data_dir, agent=agent, map_=map_, side=side)
    parts: list[str] = []

    # Prepend AGENT blocks for the player's top agents that aren't already
    # the explicit `agent`.  Skip silently when the JSON knowledge base
    # has no entry (e.g. a brand-new agent the user pulled before we
    # updated agents.json) — the grounding rule covers that case.
    if extra_agents:
        seen = {a.lower() for a in (agent,) if a}
        for extra in extra_agents:
            if not extra or extra.lower() in seen:
                continue
            block = format_agent_context(extra)
            if block:
                parts.append(block)
                seen.add(extra.lower())

    base = result.to_context_string()
    if base:
        parts.append(base)
    return "\n\n".join(parts) if parts else None


def _resolve_fields(
    situation: str,
    agent: str | None,
    map_: str | None,
    side: str | None,
) -> tuple[Situation, str | None, str | None, str | None]:
    parsed = parse_situation(situation)
    return (
        parsed,
        agent or parsed.primary_agent,
        map_ or parsed.map,
        side or parsed.side,
    )


def run_coach(
    situation: str,
    agent: str | None = None,
    map_: str | None = None,
    side: str | None = None,
    *,
    with_stats: bool = True,
    no_elicit: bool = False,
    match_context=None,  # SessionMatchContext | None — avoids circular import
    conversation_history: list[dict[str, str]] | None = None,
    force_intent: str | None = None,  # bypass classify_intent() when set
) -> str | None:
    settings = load_settings()

    from valocoach.core.preflight import check_riot_id, check_vector_store

    if with_stats:
        riot_result = check_riot_id(settings)
        if not riot_result.ok:
            display.warn(
                f"{riot_result.message} "
                "Coaching will proceed without personalised stats.\n"
                f"  {riot_result.hint}"
            )

    vs_result = check_vector_store(settings)
    if not vs_result.ok:
        display.warn(f"{vs_result.message}\n  {vs_result.hint}")

    parsed, agent, map_, side = _resolve_fields(situation, agent, map_, side)

    # Classify intent BEFORE elicitation so we only ask relevant questions.
    intent = force_intent if force_intent is not None else classify_intent(parsed, situation)

    if not no_elicit:
        from valocoach.coach.elicitation import run_elicitation, should_elicit

        if should_elicit(parsed, situation, intent):
            parsed, agent, map_, side = run_elicitation(parsed, agent, map_, side, intent=intent)
            # Persist elicited values into match_context so the REPL doesn't
            # re-ask the same fields on the next turn.
            if match_context is not None:
                if agent and not match_context.agent:
                    match_context.agent = agent
                if map_ and not match_context.map:
                    match_context.map = map_
                if side and not match_context.side:
                    match_context.side = side

    # Merge persistent match context — per-turn values take precedence
    match_context_block: str | None = None
    extra_enemies: list[str] = []
    if match_context is not None and not match_context.is_empty:
        agent, map_, side, extra_enemies = match_context.resolve_coach_kwargs(agent, map_, side)
        match_context_block = match_context.to_context_block()
    system_prompt_base = PROMPT_TEMPLATES[intent]
    panel_title = PANEL_TITLES[intent]

    # Meta intent: print BOTH the deterministic tier list AND a deterministic
    # personalised takeaway, then return without calling the LLM at all.
    # We tried prompting the LLM to only write a short takeaway, but even
    # qwen3:14b ignores the prohibitions and re-emits a full breakdown with
    # fabricated abilities.  At this model scale, the only reliable answer
    # is no model — every word in the meta panel now comes from agents.json,
    # meta.json, and the player's stats DB.
    if intent == "meta":
        from valocoach.coach.meta_response import (
            format_full_meta_block,
            format_personalised_takeaway,
        )

        top_played: list[str] = []
        if with_stats:
            try:
                from valocoach.coach import get_top_played_agents

                top_played = get_top_played_agents(settings)
            except Exception:
                log.debug("top-played agents lookup failed (meta intent)", exc_info=True)

        with display.command_frame("Meta — Current Tier List"):
            display.console.print(format_full_meta_block(top_played))

        if with_stats:
            takeaway = format_personalised_takeaway(settings)
            if takeaway:
                with display.command_frame("Personalised Takeaway"):
                    display.console.print(takeaway)
            else:
                display.info(
                    "Personalised takeaway skipped — no synced match history. "
                    "Run [info]valocoach sync[/info] first."
                )

        _maybe_warn_stale_meta(settings)
        return None  # Skip the LLM call entirely.

    # Auto-inject AGENT blocks for the player's most-played agents.  Without
    # this, agent-less questions like "how do I rank up?" produce a
    # context-free prompt and small models hallucinate abilities for
    # whichever agent name they pick from training data.  Wrapped in a
    # try because stats may be unavailable on a fresh install.
    extra_agents: list[str] = []
    if with_stats:
        try:
            from valocoach.coach import get_top_played_agents

            extra_agents = get_top_played_agents(settings)
        except Exception:
            log.debug("top-played agents lookup failed", exc_info=True)

    grounded_context = _build_grounded_context(
        agent=agent,
        map_=map_,
        situation=situation,
        side=side,
        data_dir=settings.data_dir,
        extra_agents=extra_agents,
    )

    stats_context: str | None = None
    if with_stats:
        try:
            stats_context = build_stats_context(settings)
        except Exception as e:
            display.warn(f"Couldn't load stats context (continuing without): {e}")

    if stats_context is not None:
        display.info("Personalised with your recent stats.")

    last_match_context: str | None = None
    if with_stats:
        try:
            from valocoach.coach.session_manager import (
                format_last_match_context,
                get_last_match,
            )

            lm = get_last_match(settings)
            if lm is not None:
                last_match_context = format_last_match_context(lm)
        except Exception:
            log.debug("last-match context unavailable", exc_info=True)

    notes_context: str | None = None
    if with_stats:
        try:
            from valocoach.coach.session_manager import (
                format_open_notes_context,
                get_player_puuid,
                list_open_notes,
            )

            puuid = get_player_puuid(settings)
            if puuid:
                open_notes = list_open_notes(settings, puuid, limit=5)
                notes_context = format_open_notes_context(open_notes)
        except Exception:
            log.debug("open-notes context unavailable", exc_info=True)

    resolved_agents: list[str] = parsed.agents
    if agent and agent not in parsed.agents:
        resolved_agents = [agent, *parsed.agents]
    # Merge extra enemies from match context (not already in parsed enemies)
    resolved_enemies = list(parsed.enemy_agents)
    for e in extra_enemies:
        if e not in resolved_enemies:
            resolved_enemies.append(e)
    resolved_display = parsed.model_copy(
        update={
            "agents": resolved_agents,
            "enemy_agents": resolved_enemies,
            "map": map_,
            "side": side,
        }
    )
    user_msg_parts: list[str] = []
    # Persistent match context header comes first so the LLM sees it before the
    # per-turn metadata block — it acts as the "ground truth" for the session.
    if match_context_block:
        user_msg_parts.append(match_context_block)
    metadata_block = resolved_display.to_metadata_block()
    if metadata_block:
        user_msg_parts.append(metadata_block)
    if last_match_context:
        user_msg_parts.append(last_match_context)
    user_msg_parts.append(f"Situation: {situation}")
    user_msg = "\n\n".join(user_msg_parts)

    grounded_context, stats_context = fit_prompt(
        system_base=system_prompt_base,
        grounded_context=grounded_context,
        stats_context=stats_context,
        user_msg=user_msg,
    )

    system_prompt = _build_system_prompt(
        system_prompt_base, grounded_context, stats_context, notes_context
    )

    if _TEAM_ROSTER_KW.search(situation):
        system_prompt = f"{system_prompt}\n\n---\n\n{_TEAM_ROSTER_CONTRACT}"

    display.info(
        f"Using model: [heading]{settings.ollama_model}[/heading] [muted][{intent}][/muted]"
    )

    stop_tokens = _POST_GAME_STOP_TOKENS if intent == "post_game" else None
    max_tokens_override = _INTENT_MAX_TOKENS.get(intent)

    try:
        token_stream = stream_completion(
            settings=settings,
            system_prompt=system_prompt,
            user_message=user_msg,
            conversation_history=conversation_history,
            stop=stop_tokens,
            max_tokens=max_tokens_override,
        )
        response_text = display.stream_to_panel(
            token_stream,
            title=panel_title,
            subtitle=settings.ollama_model,
        )
    except Exception as e:
        display.error_with_hint(
            f"LLM call failed: {e}",
            "Check Ollama is running: ollama list",
        )
        raise

    if response_text:
        try:
            from valocoach.coach.sanitizer import validate_ability_claims

            ability_warnings = validate_ability_claims(response_text)
            if ability_warnings:
                # Bucket by category so the user sees the failure mode at a glance.
                by_cat: dict[str, list] = {}
                for w in ability_warnings:
                    by_cat.setdefault(w.category, []).append(w)

                category_labels = {
                    "hallucination": "fabricated abilities (don't exist in Valorant)",
                    "cross_attribution": "wrong-agent attributions",
                    "weapon": "weapons mis-cast as abilities",
                    "generic": "generic descriptors used as ability names",
                }

                lines = [
                    "",
                    f"[warning]! Ability fact-check — {len(ability_warnings)} "
                    "claim(s) don't match the canonical agent kit:[/warning]",
                ]
                for cat in ("hallucination", "cross_attribution", "weapon", "generic"):
                    items = by_cat.get(cat, [])
                    if not items:
                        continue
                    label = category_labels.get(cat, cat)
                    lines.append(f"  [heading]{label}:[/heading]")
                    for w in items[:8]:
                        lines.append(f"    • {w.format()}")
                    if len(items) > 8:
                        lines.append(f"    …and {len(items) - 8} more.")
                lines.append(
                    "  [muted]This is a model limitation, not a bug — "
                    "always verify ability names / costs against in-game tooltips.[/muted]"
                )
                display.console.print("\n".join(lines))
        except Exception:
            log.debug("ability claim sanitizer failed", exc_info=True)

        # Numeric stats sanitizer — symmetric to the ability one.  Pull
        # ``stats_context`` (if present) and check the LLM didn't misquote
        # the player's real K/D / ACS / ADR / HS% / Win-rate.
        try:
            from valocoach.coach.stats_sanitizer import validate_stat_claims

            stat_warnings = validate_stat_claims(response_text, stats_context or "")
            if stat_warnings:
                stat_lines = [
                    "",
                    f"[warning]! Stat fact-check — {len(stat_warnings)} "
                    "numeric claim(s) don't match your real PLAYER CONTEXT:[/warning]",
                ]
                for w in stat_warnings[:8]:
                    stat_lines.append(f"  • {w.format()}")
                if len(stat_warnings) > 8:
                    stat_lines.append(f"  …and {len(stat_warnings) - 8} more.")
                stat_lines.append(
                    "  [muted]The model occasionally misquotes the stats we "
                    "give it.  Trust the numbers in `valocoach stats`.[/muted]"
                )
                display.console.print("\n".join(stat_lines))
        except Exception:
            log.debug("stat claim sanitizer failed", exc_info=True)

    if intent in _META_SENSITIVE_INTENTS:
        _maybe_warn_stale_meta(settings)

    return response_text or None
