from __future__ import annotations

from valocoach.cli import display
from valocoach.coach import build_stats_context
from valocoach.coach.intent import classify_intent
from valocoach.coach.templates import PANEL_TITLES, PROMPT_TEMPLATES
from valocoach.core.config import load_settings
from valocoach.core.context_budget import fit_prompt
from valocoach.core.parser import Situation, parse_situation
from valocoach.llm.provider import stream_completion
from valocoach.retrieval import format_agent_context, format_map_context

# Intents that rely on current patch meta data — show a staleness warning
# when these are used and the last patch check is older than the threshold.
_META_SENSITIVE_INTENTS: frozenset[str] = frozenset({"meta", "agent_info"})

# Days after which patch meta data is considered potentially stale.
_PATCH_STALE_THRESHOLD_DAYS: int = 21


def _build_system_prompt(
    base_prompt: str,
    grounded_context: str | None,
    stats_context: str | None,
) -> str:
    parts = [base_prompt]
    if grounded_context:
        parts.append(
            f"---\n\nGROUNDED CONTEXT (use these facts verbatim — do not hallucinate abilities or callouts):\n\n{grounded_context}"
        )
    if stats_context:
        parts.append(f"---\n\n{stats_context}")
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
    """Merge CLI flags with parsed-from-text fields.

    CLI flags take precedence — if the user passed ``--agent Jett`` we use
    that even when the situation text mentions someone else.  The parser
    fills in the gaps so e.g. ``"push A on Ascent attack"`` no longer
    requires ``--map`` and ``--side`` flags to route retrieval correctly.
    """
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
    conversation_history: list[dict[str, str]] | None = None,
) -> str | None:
    """Run a single coaching turn.

    Args:
        situation:             Free-text description of the match situation.
        agent:                 Optional agent override (beats parser output).
        map_:                  Optional map override.
        side:                  Optional side override ("attack" / "defense").
        with_stats:            If True, fetch and inject player recent-form stats.
        conversation_history:  Prior turns from ``ConversationMemory.messages``.
                               When provided, the LLM sees the full multi-turn
                               context before the current user message.

    Returns:
        The full assistant response text on success, or None when streaming
        produced no output.  The interactive REPL uses this return value to
        store the assistant turn in conversation memory.
    """
    settings = load_settings()

    from valocoach.core.preflight import check_riot_id, check_vector_store

    # Non-fatal: warn when stats are requested but Riot ID is not configured.
    # Coaching still works; the user just loses personalised stats injection.
    if with_stats:
        riot_result = check_riot_id(settings)
        if not riot_result.ok:
            display.warn(
                f"{riot_result.message} "
                "Coaching will proceed without personalised stats.\n"
                f"  {riot_result.hint}"
            )

    # Non-fatal: warn when the vector store is empty.  The LLM falls back to
    # its training knowledge but cannot cite exact ability costs or callouts.
    vs_result = check_vector_store(settings)
    if not vs_result.ok:
        display.warn(f"{vs_result.message}\n  {vs_result.hint}")

    # Parse the situation up front so retrieval gets the same agent/map/side
    # the LLM will see in the user message — keeps the two paths consistent.
    parsed, agent, map_, side = _resolve_fields(situation, agent, map_, side)

    # Classify intent — determines which prompt template and panel title to use.
    intent = classify_intent(parsed, situation)
    system_prompt_base = PROMPT_TEMPLATES[intent]
    panel_title = PANEL_TITLES[intent]

    # Retrieve grounded context (abilities, callouts, meta) — always included.
    grounded_context = _build_grounded_context(
        agent=agent,
        map_=map_,
        situation=situation,
        side=side,
        data_dir=settings.data_dir,
    )

    # Fetch player context — non-fatal.
    stats_context: str | None = None
    if with_stats:
        try:
            stats_context = build_stats_context(settings)
        except Exception as e:
            display.warn(f"Couldn't load stats context (continuing without): {e}")

    if stats_context is not None:
        display.info("Personalised with your recent stats.")

    # Fetch last-match context — a compact one-liner injected into the user
    # message so the LLM can reference the player's most recent game without
    # needing to ask.  Non-fatal: any error silently skips this block.
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
            pass  # last-match context is advisory — never crash the coaching turn

    # User message: structured metadata block (when any field was extracted),
    # followed by last-match context (when available), then the verbatim situation.
    #
    # Build the metadata block from resolved fields — CLI flags (agent, map_,
    # side) take precedence over the parser, so if the user passed --agent Jett
    # but didn't write "Jett" in the situation text the block still shows the
    # agent so the LLM is keyed into the right frame.
    resolved_agents: list[str] = parsed.agents
    if agent and agent not in parsed.agents:
        resolved_agents = [agent, *parsed.agents]
    resolved_display = parsed.model_copy(
        update={
            "agents": resolved_agents,
            "map": map_,
            "side": side,
        }
    )
    user_msg_parts: list[str] = []
    metadata_block = resolved_display.to_metadata_block()
    if metadata_block:
        user_msg_parts.append(metadata_block)
    if last_match_context:
        user_msg_parts.append(last_match_context)
    user_msg_parts.append(f"Situation: {situation}")
    user_msg = "\n\n".join(user_msg_parts)

    # Adaptive context trimming — keep the prompt within the hard token limit.
    # Trims lowest-priority blocks first (vector hits, then stats) so the
    # load-bearing JSON facts and the verbatim user message are never cut.
    grounded_context, stats_context = fit_prompt(
        system_base=system_prompt_base,
        grounded_context=grounded_context,
        stats_context=stats_context,
        user_msg=user_msg,
    )

    system_prompt = _build_system_prompt(system_prompt_base, grounded_context, stats_context)

    display.info(f"Using model: {settings.ollama_model} [{intent}]")

    try:
        token_stream = stream_completion(
            settings=settings,
            system_prompt=system_prompt,
            user_message=user_msg,
            conversation_history=conversation_history,
        )
        response_text = display.stream_to_panel(token_stream, title=panel_title)
    except Exception as e:
        display.error(f"LLM call failed: {e}")
        display.warn("Check Ollama is running: `ollama list`")
        raise

    # ── Patch-staleness warning ─────────────────────────────────────────────
    # Only shown for intents that specifically draw on current patch meta data.
    # Non-fatal: any error inside get_patch_staleness_days returns None, which
    # is treated as "never checked" → warning fires to encourage a refresh.
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
                    f"[dim]⚠ Meta info may be outdated ({age_str}) — "
                    "run [cyan]valocoach patch --check[/cyan] to refresh.[/dim]"
                )
        except Exception:
            pass  # staleness check is advisory — never crash the coaching turn

    return response_text or None
