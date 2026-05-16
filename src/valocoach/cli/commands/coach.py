from __future__ import annotations

import logging

from valocoach.cli import display

log = logging.getLogger(__name__)
from valocoach.coach import build_stats_context
from valocoach.coach.intent import classify_intent
from valocoach.coach.templates import PANEL_TITLES, PROMPT_TEMPLATES
from valocoach.core.config import load_settings
from valocoach.core.context_budget import fit_prompt
from valocoach.core.parser import Situation, parse_situation
from valocoach.llm.provider import stream_completion
from valocoach.retrieval import format_agent_context, format_map_context

_META_SENSITIVE_INTENTS: frozenset[str] = frozenset({"meta", "agent_info"})
_PATCH_STALE_THRESHOLD_DAYS: int = 21


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
) -> str | None:
    from valocoach.retrieval.retriever import retrieve_static

    if agent and not format_agent_context(agent):
        display.warn(f"Agent '{agent}' not found in knowledge base — coach may improvise.")
    if map_ and not format_map_context(map_):
        display.warn(f"Map '{map_}' not found in knowledge base — coach may improvise.")

    result = retrieve_static(situation, data_dir, agent=agent, map_=map_, side=side)
    return result.to_context_string()


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

    if not no_elicit:
        from valocoach.coach.elicitation import run_elicitation, should_elicit

        if should_elicit(parsed, situation):
            parsed, agent, map_, side = run_elicitation(parsed, agent, map_, side)

    # Merge persistent match context — per-turn values take precedence
    match_context_block: str | None = None
    extra_enemies: list[str] = []
    if match_context is not None and not match_context.is_empty:
        agent, map_, side, extra_enemies = match_context.resolve_coach_kwargs(agent, map_, side)
        match_context_block = match_context.to_context_block()

    intent = force_intent if force_intent is not None else classify_intent(parsed, situation)
    system_prompt_base = PROMPT_TEMPLATES[intent]
    panel_title = PANEL_TITLES[intent]

    grounded_context = _build_grounded_context(
        agent=agent,
        map_=map_,
        situation=situation,
        side=side,
        data_dir=settings.data_dir,
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

    display.info(f"Using model: [heading]{settings.ollama_model}[/heading] [muted][{intent}][/muted]")

    try:
        token_stream = stream_completion(
            settings=settings,
            system_prompt=system_prompt,
            user_message=user_msg,
            conversation_history=conversation_history,
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

    if intent in _META_SENSITIVE_INTENTS:
        try:
            from valocoach.retrieval.patch_tracker import get_patch_staleness_days

            stale_days = get_patch_staleness_days(settings.data_dir)
            if stale_days is None or stale_days > _PATCH_STALE_THRESHOLD_DAYS:
                age_str = (
                    "never checked"
                    if stale_days is None
                    else f"{stale_days:.0f}d since last check"
                )
                display.console.print(
                    f"[muted]⚠ Meta info may be outdated ({age_str}) — "
                    "run [info]valocoach patch --check[/info] to refresh.[/muted]"
                )
        except Exception:
            log.debug("patch staleness check failed", exc_info=True)

    return response_text or None
