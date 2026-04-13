from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CoachRequest(BaseModel):
    """Minimal coaching request passed from CLI to the orchestrator."""

    message: str = Field(min_length=1)
    agent: str | None = None
    map_name: str | None = None
    side: Literal["attack", "defense"] | None = None

    def render_user_prompt(self) -> str:
        """Expand structured CLI flags into a single user message."""
        parts = [f"Situation: {self.message.strip()}"]
        if self.agent:
            parts.append(f"Agent: {self.agent}")
        if self.map_name:
            parts.append(f"Map: {self.map_name}")
        if self.side:
            parts.append(f"Side: {self.side}")
        return "\n".join(parts)
