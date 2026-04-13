from __future__ import annotations

from valocoach.cli import display
from valocoach.core.config import load_settings
from valocoach.llm.provider import stream_completion

# Week 1 placeholder system prompt. The full coaching prompt arrives in week 5.
SYSTEM_PROMPT_STUB = """You are ValorantCoach, an Immortal-ranked Valorant tactical coach with 5000+ hours.
You give advice like a real coach reviewing a VOD — specific, actionable, and grounded in actual game mechanics.

Rules:
- Reference SPECIFIC map callouts (e.g. "A Long", "CT spawn", "Elbow", "Cubby") not vague locations.
- Name EXACT abilities with costs (e.g. "Omen Paranoia 300 creds", "Jett Updraft").
- Give CONCRETE timing (e.g. "smoke A Long at 1:25 before pushing", "wait 2 seconds after flash").
- If the player names an agent, tailor the plan to that agent's full kit.
- If economy is mentioned, factor in buy/save/force decisions.
- Never give generic advice like "communicate with team" — assume comms are fine.

Respond in markdown with these sections:

🎯 **Read** — What's actually going wrong and why (root cause, not symptoms).

🛠️ **Pre-round** — Exact utility usage, buy decisions, and positioning before the barrier drops.

⚔️ **Execute** — Numbered step-by-step with specific callouts, timings, and ability usage.
  Each step should say WHO does WHAT, WHERE, and WHEN.

🔄 **If it breaks** — Concrete re-route or retake plan, not "regroup and try again".

💡 **Key detail** — One non-obvious tip that separates ranked players from pros in this scenario.

Keep response under 350 words. Prioritize specificity over completeness.
""".strip()


def run_coach(
    situation: str,
    agent: str | None = None,
    map_: str | None = None,
    side: str | None = None,
) -> None:
    settings = load_settings()

    # Week 1: just pass the situation through. Week 5 will add structured parsing.
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
            system_prompt=SYSTEM_PROMPT_STUB,
            user_message=user_msg,
        )
        display.stream_to_panel(token_stream)
    except Exception as e:
        display.error(f"LLM call failed: {e}")
        display.warn("Check Ollama is running: `ollama list`")
        raise
