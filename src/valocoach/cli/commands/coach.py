from __future__ import annotations

from valocoach.cli import display
from valocoach.coach import build_stats_context
from valocoach.core.config import load_settings
from valocoach.core.context_budget import fit_prompt
from valocoach.core.parser import Situation, parse_situation
from valocoach.llm.provider import stream_completion
from valocoach.retrieval import format_agent_context, format_map_context

SYSTEM_PROMPT_STUB = """You are ValorantCoach, an Immortal-ranked Valorant tactical coach with 5000+ hours.
You give advice like a real coach reviewing a VOD — specific, actionable, and grounded in actual game mechanics.

Rules:
- Reference SPECIFIC map callouts (e.g. "A Long", "CT spawn", "Elbow", "Cubby") not vague locations.
- Name EXACT abilities with costs using the GROUNDED CONTEXT provided below — never invent ability names or costs.
- Give CONCRETE timing (e.g. "smoke A Long at 1:25 before pushing", "wait 2 seconds after flash").
- If the player names an agent, tailor the plan to that agent's full kit as defined in GROUNDED CONTEXT.
- If economy is mentioned, use the economy thresholds from GROUNDED CONTEXT.
- Never give generic advice like "communicate with team" — assume comms are fine.
- When PLAYER CONTEXT is provided below, use it to tailor advice to this player's actual recent form — reference their agents, maps, and tendencies where relevant. Do NOT dump the stats back at them.
- In multi-turn coaching sessions, track context across turns and build on earlier advice rather than repeating it.

Respond in markdown with these sections:

🎯 **Read** — What's actually going wrong and why (root cause, not symptoms).

🛠️ **Pre-round** — Exact utility usage, buy decisions, and positioning before the barrier drops.

⚔️ **Execute** — Numbered step-by-step with specific callouts, timings, and ability usage.
  Each step should say WHO does WHAT, WHERE, and WHEN.

🔄 **If it breaks** — Concrete re-route or retake plan, not "regroup and try again".

💡 **Key detail** — One non-obvious tip that separates ranked players from pros in this scenario.

Keep response under 350 words. Prioritize specificity over completeness.
""".strip()


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

    # User message: structured metadata block (when any field was extracted)
    # followed by the player's verbatim situation text.  The LLM gets both
    # the parser's signal and the original wording.
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
    user_msg_parts.append(f"Situation: {situation}")
    user_msg = "\n\n".join(user_msg_parts)

    # Adaptive context trimming — keep the prompt within the hard token limit.
    # Trims lowest-priority blocks first (vector hits, then stats) so the
    # load-bearing JSON facts and the verbatim user message are never cut.
    grounded_context, stats_context = fit_prompt(
        system_base=SYSTEM_PROMPT_STUB,
        grounded_context=grounded_context,
        stats_context=stats_context,
        user_msg=user_msg,
    )

    system_prompt = _build_system_prompt(SYSTEM_PROMPT_STUB, grounded_context, stats_context)

    display.info(f"Using model: {settings.ollama_model}")

    try:
        token_stream = stream_completion(
            settings=settings,
            system_prompt=system_prompt,
            user_message=user_msg,
            conversation_history=conversation_history,
        )
        response_text = display.stream_to_panel(token_stream)
    except Exception as e:
        display.error(f"LLM call failed: {e}")
        display.warn("Check Ollama is running: `ollama list`")
        raise

    return response_text or None
