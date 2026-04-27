from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScrapedContent:
    url: str
    title: str
    text: str
    fetched_at: str
    source: str  # "web", "patch_note", "youtube", etc.
