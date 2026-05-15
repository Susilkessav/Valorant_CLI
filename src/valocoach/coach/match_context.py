"""SessionMatchContext — persistent match state for the interactive REPL.

The REPL is multi-turn but each turn previously had no memory of match
specifics (map, side, agent, score) set in earlier turns.  A user who typed
"Ascent attack as Jett" in turn 1 had to repeat that context in turn 3 if
the conversation moved on.

``SessionMatchContext`` is the single mutable bag that holds all such fields
for the lifetime of one REPL session.  Slash commands write into it; every
``run_coach`` call reads from it as fallback values for agent/map/side and
injects a compact ``MATCH CONTEXT`` header into the user message so the LLM
always sees the current match state.

Public API
----------
    SessionMatchContext          — dataclass, one instance per REPL session
    SessionMatchContext.to_context_block()  -> str | None
    SessionMatchContext.merge_into_situation(parsed, agent, map_, side) -> ...
    SessionMatchContext.reset()  -> None
    SessionMatchContext.summary_line() -> str   (for /context display)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class SessionMatchContext:
    """Mutable match state carried across all turns of one REPL session.

    All fields are ``None`` / empty until explicitly set by a slash command.
    The ``is_empty`` property returns ``True`` when nothing has been set so
    callers can skip injection when there is no useful context.
    """

    map: str | None = None
    side: str | None = None          # "attack" | "defense"
    agent: str | None = None
    enemy_agents: list[str] = field(default_factory=list)
    score: tuple[int, int] | None = None   # (own, opp) from player's POV
    result: str | None = None        # "won" | "lost"
    econ: str | None = None          # "eco" | "half_buy" | "full_buy"
    phase: str | None = None         # "post_plant" | "retake" | "execute" | "default"

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def is_empty(self) -> bool:
        """True when no fields have been set yet."""
        return (
            self.map is None
            and self.side is None
            and self.agent is None
            and not self.enemy_agents
            and self.score is None
            and self.result is None
            and self.econ is None
            and self.phase is None
        )

    # ------------------------------------------------------------------ #
    # Mutation helpers
    # ------------------------------------------------------------------ #

    def set_side(self, value: str) -> None:
        """Normalise and store side."""
        _SIDE = {"attack": "attack", "atk": "attack", "offense": "attack",
                 "defense": "defense", "def": "defense", "defend": "defense"}
        self.side = _SIDE.get(value.lower(), value.lower())

    def flip_side(self) -> None:
        """Toggle attack ↔ defense (used at half-time with /half)."""
        if self.side == "attack":
            self.side = "defense"
        elif self.side == "defense":
            self.side = "attack"
        # If unknown, leave unchanged

    def add_enemy(self, name: str) -> bool:
        """Add *name* to enemy_agents if not already present.  Returns True on add."""
        if name not in self.enemy_agents:
            self.enemy_agents.append(name)
            return True
        return False

    def reset(self) -> None:
        """Clear all fields back to defaults."""
        self.map = None
        self.side = None
        self.agent = None
        self.enemy_agents = []
        self.score = None
        self.result = None
        self.econ = None
        self.phase = None

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def to_context_block(self) -> str | None:
        """Return a compact ``MATCH CONTEXT`` block for injection into the user message.

        Returns ``None`` when nothing has been set so callers can skip the
        section without an empty header.

        Example output::

            MATCH CONTEXT (persistent across this session):
            Map: Ascent  ·  Side: attack  ·  Agent: Jett  ·  Enemy: Cypher, Killjoy
            Score: 9-11  ·  Result: won  ·  Economy: eco
        """
        if self.is_empty:
            return None

        parts_line1: list[str] = []
        if self.map:
            parts_line1.append(f"Map: {self.map}")
        if self.side:
            parts_line1.append(f"Side: {self.side}")
        if self.agent:
            parts_line1.append(f"Agent: {self.agent}")
        if self.enemy_agents:
            parts_line1.append(f"Enemy: {', '.join(self.enemy_agents)}")

        parts_line2: list[str] = []
        if self.score:
            parts_line2.append(f"Score: {self.score[0]}-{self.score[1]}")
        if self.result:
            parts_line2.append(f"Result: {self.result}")
        if self.econ:
            parts_line2.append(f"Economy: {self.econ.replace('_', ' ')}")
        if self.phase:
            parts_line2.append(f"Phase: {self.phase.replace('_', ' ')}")

        lines = ["MATCH CONTEXT (persistent across this session):"]
        if parts_line1:
            lines.append("  " + "  ·  ".join(parts_line1))
        if parts_line2:
            lines.append("  " + "  ·  ".join(parts_line2))
        return "\n".join(lines)

    def summary_line(self) -> str:
        """One-line human-readable summary for /context display."""
        if self.is_empty:
            return "No match context set.  Use /agent, /map, /side, /score to set up."
        parts: list[str] = []
        if self.agent:
            parts.append(self.agent)
        if self.map:
            parts.append(self.map)
        if self.side:
            parts.append(self.side)
        if self.score:
            parts.append(f"{self.score[0]}-{self.score[1]}")
        if self.result:
            parts.append(self.result)
        if self.econ:
            parts.append(self.econ.replace("_", " "))
        if self.enemy_agents:
            parts.append(f"vs {', '.join(self.enemy_agents)}")
        return " · ".join(parts) if parts else "No context."

    # ------------------------------------------------------------------ #
    # Merging into run_coach kwargs
    # ------------------------------------------------------------------ #

    def resolve_coach_kwargs(
        self,
        agent: str | None,
        map_: str | None,
        side: str | None,
    ) -> tuple[str | None, str | None, str | None, list[str]]:
        """Merge context fields with per-turn parsed values.

        Per-turn values (from the situation parser or CLI flags) take
        precedence over stored context — the user can always override
        temporarily by mentioning "Bind" in one turn without resetting the
        stored map.

        Returns ``(agent, map_, side, extra_enemies)`` where ``extra_enemies``
        are context enemy agents not already in the parsed Situation.
        """
        resolved_agent = agent or self.agent
        resolved_map = map_ or self.map
        resolved_side = side or self.side
        return resolved_agent, resolved_map, resolved_side, list(self.enemy_agents)

    # ------------------------------------------------------------------ #
    # JSON serialisation (for DB persistence)
    # ------------------------------------------------------------------ #

    def to_json(self) -> str:
        return json.dumps({
            "map": self.map,
            "side": self.side,
            "agent": self.agent,
            "enemy_agents": self.enemy_agents,
            "score": list(self.score) if self.score else None,
            "result": self.result,
            "econ": self.econ,
            "phase": self.phase,
        })

    @classmethod
    def from_json(cls, data: str) -> "SessionMatchContext":
        d = json.loads(data)
        score = d.get("score")
        return cls(
            map=d.get("map"),
            side=d.get("side"),
            agent=d.get("agent"),
            enemy_agents=d.get("enemy_agents") or [],
            score=tuple(score) if score else None,  # type: ignore[arg-type]
            result=d.get("result"),
            econ=d.get("econ"),
            phase=d.get("phase"),
        )


__all__ = ["SessionMatchContext"]
