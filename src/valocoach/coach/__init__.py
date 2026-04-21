"""valocoach.coach — coaching-side logic beyond the CLI shell.

Currently exposes the player-context builder that personalises the coach
system prompt. Will grow to hold the full coaching system prompt, response
parsing, and follow-up-suggestion logic once those move out of
``valocoach.cli.commands.coach``.
"""

from __future__ import annotations

from valocoach.coach.context import build_stats_context

__all__ = ["build_stats_context"]
