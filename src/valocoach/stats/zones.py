"""Callout zone classifier for Valorant maps (Phase E6).

Maps raw (x, y) coordinates from the HenrikDev API to named callout zones
(e.g. "A Showers", "Mid Catwalk", "B Main").

Coordinate system
-----------------
Valorant's internal coordinate system is centimetres in Unreal Engine space,
with origin varying per map.  The HenrikDev v4 API exposes raw (x, y) as
integers after scaling; exact ranges differ by map.

Calibration status (2026-05)
----------------------------
Bounding boxes below are ESTIMATED from publicly-available Valorant mapping
resources and community tooling.  They should be considered v1 approximations.
Once real coordinate data has been collected (2-3 weeks of synced matches with
Phase A columns populated), run ``scripts/calibrate_zones.py`` to refine them.

Adding new maps
---------------
1. Add an entry to ``_ZONE_BOXES`` with the map name (case-insensitive match
   against ``Match.map_name``).
2. Each zone is a 4-tuple ``(x_min, y_min, x_max, y_max)``.
3. Zones are checked in order; the first match is returned.  Put smaller
   zones before larger fallback zones.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounding box data
# (map_name → list of (zone_name, x_min, y_min, x_max, y_max))
# ---------------------------------------------------------------------------
# Coordinate estimates based on Valorant map dimensions (centimetre scale).
# These are APPROXIMATE — see module docstring for calibration notes.

_ZONE_BOXES: dict[str, list[tuple[str, int, int, int, int]]] = {
    "Ascent": [
        # A Site area
        ("A Heaven",        5200, 1500, 7200, 3500),
        ("A Rafters",       5200, 3500, 7200, 5000),
        ("A Default",       6500, 4500, 9000, 7000),
        ("A Main",          9000, 4000, 11500, 7000),
        ("A Lobby",         9500, 1500, 12500, 4500),
        ("A Catwalk",       5000, 4800, 7500, 6500),
        ("A Market",        3500, 5000, 5500, 7500),
        # B Site area
        ("B Heaven",        5000, 8000, 7500, 10000),
        ("B Main",          9000, 8000, 12000, 11000),
        ("B Site",          6000, 7000, 9500, 10000),
        ("B Market",        3500, 7500, 6000, 10000),
        ("B Lobby",         10000, 10000, 13000, 13000),
        # Mid
        ("Mid Market",      4500, 6000, 7000, 8000),
        ("Mid Top",         7000, 6000, 9500, 8000),
        ("Mid Pizza",       7000, 6500, 9000, 8500),
        ("Mid Catwalk",     5500, 5500, 7500, 7500),
        # Fallback
        ("A Site",          5000, 3000, 9500, 8000),
        ("B Site",          4500, 7500, 10000, 12000),
    ],
    "Bind": [
        # A Site
        ("A Showers",       3500, 2000, 6000, 5000),
        ("A Heaven",        1500, 2000, 4000, 5000),
        ("A Bath",          4000, 5000, 6500, 7500),
        ("A Short",         6500, 4500, 9000, 7000),
        ("A Long",          7000, 7000, 10000, 10000),
        ("A Lamps",         3000, 5500, 5500, 8000),
        # B Site
        ("B Window",        3000, 8000, 5500, 10500),
        ("B Garden",        5000, 8000, 7500, 11000),
        ("B Elbow",         6500, 9500, 9000, 12000),
        ("B Short",         7000, 10500, 10500, 13000),
        ("B Long",          9000, 8500, 12500, 11000),
        # Teleporters / Mid
        ("Teleporter A",    5000, 6500, 7500, 9000),
        ("Teleporter B",    7500, 6500, 10000, 9000),
        # Fallback
        ("A Site",          1500, 1500, 7500, 8000),
        ("B Site",          3000, 8000, 12500, 13500),
    ],
    "Haven": [
        # A Site
        ("A Long",          1500, 1500, 5000, 5000),
        ("A Short",         4000, 3000, 7000, 6000),
        ("A Heaven",        3500, 5500, 6500, 8500),
        ("A Sewer",         5500, 4500, 8000, 7500),
        ("A Site",          5000, 6000, 8500, 9500),
        # B Site
        ("B Garage",        6500, 1500, 9500, 5000),
        ("B Window",        8000, 4000, 11000, 7000),
        ("B Site",          8000, 6500, 12000, 10000),
        # C Site
        ("C Long",          9500, 1500, 13000, 5500),
        ("C Sewer",         10000, 5000, 13000, 7500),
        ("C Catwalk",       10500, 7000, 13000, 9500),
        ("C Site",          10000, 8500, 13500, 12000),
        # Mid
        ("Mid Doors",       6500, 5500, 9500, 8500),
        ("Mid Top",         5500, 5000, 8000, 8000),
        ("Courtyard",       7000, 4500, 10000, 7000),
    ],
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_location(map_name: str, x: int | None, y: int | None) -> str | None:
    """Return the callout zone name for a coordinate pair on the given map.

    Args:
        map_name: Map name as stored in ``Match.map_name`` (case-insensitive).
        x, y:     Coordinates from the HenrikDev API (``Kill.killer_x``,
                  ``Kill.victim_x``, ``Round.plant_x``, etc.).

    Returns:
        The zone name string (e.g. ``"A Showers"``), or ``None`` when:
        - the map is not in the classifier database,
        - the coordinates are NULL (pre-migration match), or
        - the point does not fall inside any defined bounding box.
    """
    if x is None or y is None:
        return None

    # Case-insensitive map lookup
    boxes = next(
        (v for k, v in _ZONE_BOXES.items() if k.casefold() == map_name.casefold()),
        None,
    )
    if boxes is None:
        return None

    for zone_name, x_min, y_min, x_max, y_max in boxes:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return zone_name

    return None


def supported_maps() -> list[str]:
    """Return the list of map names supported by the zone classifier."""
    return list(_ZONE_BOXES.keys())


__all__ = ["classify_location", "supported_maps"]
