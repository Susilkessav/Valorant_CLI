from __future__ import annotations


class ValoCoachError(Exception):
    """Base project exception."""


class ConfigurationError(ValoCoachError):
    """Raised when configuration is missing or invalid."""


class ProviderError(ValoCoachError):
    """Raised when the active LLM provider fails."""


class CommandNotReadyError(ValoCoachError):
    """Raised by CLI commands that are intentionally stubbed."""
