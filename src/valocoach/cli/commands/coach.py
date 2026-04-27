from __future__ import annotations

from valocoach.cli import display
from valocoach.coach import build_stats_context
from valocoach.core.config import load_settings
from valocoach.llm.provider import stream_completion
from valocoach.retrieval import format_agent_context, format_map_context, format_meta_context

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
        parts.append(f"---\n\nGROUNDED CONTEXT (use these facts verbatim — do not hallucinate abilities or callouts):\n\n{grounded_context}")
    if stats_context:
        parts.append(f"---\n\n{stats_context}")
    return "\n\n".join(parts)


def _build_grounded_context(
    agent: str | None,
    map_: str | None,
    situation: str,
    data_dir,
) -> str | None:
    parts: list[str] = []

    # Static JSON — precise, structured, always reliable.
    if agent:
        agent_ctx = format_agent_context(agent)
        if agent_ctx:
            parts.append(agent_ctx)
        else:
            display.warn(f"Agent '{agent}' not found in knowledge base — coach may improvise.")

    if map_:
        map_ctx = format_map_context(map_)
        if map_ctx:
            parts.append(map_ctx)
        else:
            display.warn(f"Map '{map_}' not found in knowledge base — coach may improvise.")

    parts.append(format_meta_context(agent=agent, map_name=map_))

    # Vector search — supplemental patch notes, YouTube insights, etc.
    # Non-fatal: if Ollama isn't running or the store is empty, skip silently.
    try:
        from valocoach.retrieval.searcher import search

        query = " ".join(filter(None, [situation, agent, map_]))
        hits = search(
            query,
            data_dir,
            n_results=3,
            doc_types=["patch_note", "youtube", "web"],
            max_distance=0.45,
        )
        for hit in hits:
            name = hit["metadata"].get("name", "supplemental")
            parts.append(f"[{hit['metadata']['type'].upper()}: {name}]\n{hit['text']}")
    except Exception:
        pass  # Vector store is optional — static JSON is the baseline.

    return "\n\n".join(parts) if parts else None


def run_coach(
    situation: str,
    agent: str | None = None,
    map_: str | None = None,
    side: str | None = None,
    *,
    with_stats: bool = True,
) -> None:
    settings = load_settings()

    # Retrieve grounded context (abilities, callouts, meta) — always included.
    grounded_context = _build_grounded_context(
        agent=agent,
        map_=map_,
        situation=situation,
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

    system_prompt = _build_system_prompt(SYSTEM_PROMPT_STUB, grounded_context, stats_context)

    user_msg_parts = [f"Situation: {situation}"]
    if agent:
        user_msg_parts.append(f"Agent: {agent}")
    if map_:
        user_msg_parts.append(f"Map: {map_}")
    if side:
        user_msg_parts.append(f"Side: {side}")
    user_msg = "\n".join(user_msg_parts)

    display.info(f"Using model: {settings.ollama_model}")

    try:
        token_stream = stream_completion(
            settings=settings,
            system_prompt=system_prompt,
            user_message=user_msg,
        )
        display.stream_to_panel(token_stream)
    except Exception as e:
        display.error(f"LLM call failed: {e}")
        display.warn("Check Ollama is running: `ollama list`")
        raise
