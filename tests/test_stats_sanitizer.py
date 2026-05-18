"""Tests for valocoach.coach.stats_sanitizer.validate_stat_claims.

Two failure modes the sanitizer is designed to catch:
  1. Numeric mismatch — the model writes a stat that contradicts PLAYER CONTEXT.
  2. Wrong unit / scale — e.g. claiming K/D 12 when actual is K/D 1.2.

Plus the false-positive guards:
  - Close-enough rounding (27% vs 27.4%) shouldn't fire.
  - Missing PLAYER CONTEXT returns empty.
  - Numbers without an explicit stat label aren't flagged.
"""

from __future__ import annotations

from valocoach.coach.stats_sanitizer import StatWarning, validate_stat_claims

# Canonical PLAYER CONTEXT block shape — built by build_stats_context.
_CTX = """\
PLAYER CONTEXT — Yoursaviour02#SBSK · Gold 3 · NA
Recent form (18 competitive match(es)):
- Record: 11-7 (61% WR) · ACS 188 · K/D 0.93 · KDA 1.31 · HS 27% · ADR 124
- Entry: FB 31 / FD 28 (diff +3)
Top agents:
- Omen (6g): 67% WR · ACS 152 · K/D 0.81
"""


# ---------------------------------------------------------------------------
# Mismatches that MUST flag
# ---------------------------------------------------------------------------


def test_flags_wrong_kd() -> None:
    text = "Your K/D is 1.2, keep it up!"
    warnings = validate_stat_claims(text, _CTX)
    assert any(w.stat == "K/D" and w.claimed == "1.2" for w in warnings)


def test_flags_wrong_hs_percent() -> None:
    text = "Your HS% is 45% — strong fundamentals."
    warnings = validate_stat_claims(text, _CTX)
    assert any(w.stat == "HS%" and "45" in w.claimed for w in warnings)


def test_flags_wrong_acs() -> None:
    text = "Your ACS of 280 is above the Gold average."
    warnings = validate_stat_claims(text, _CTX)
    assert any(w.stat == "ACS" and w.claimed == "280" for w in warnings)


def test_flags_wrong_win_rate() -> None:
    text = "Your WR is 80% — that's excellent."
    warnings = validate_stat_claims(text, _CTX)
    assert any(w.stat == "Win rate" and "80" in w.claimed for w in warnings)


# ---------------------------------------------------------------------------
# Close-enough rounding MUST NOT flag
# ---------------------------------------------------------------------------


def test_does_not_flag_close_kd() -> None:
    # K/D 0.95 vs actual 0.93 — within 0.05 tolerance.
    text = "Your K/D is 0.95 right now."
    assert validate_stat_claims(text, _CTX) == []


def test_does_not_flag_close_hs() -> None:
    # HS 27% rounded vs actual 27.4% — within 2pp tolerance.
    text = "Your HS rate is 28%."
    assert validate_stat_claims(text, _CTX) == []


def test_does_not_flag_close_acs() -> None:
    # ACS 190 vs actual 188 — within 5 points.
    text = "ACS: 190."
    assert validate_stat_claims(text, _CTX) == []


# ---------------------------------------------------------------------------
# Edge cases — empty input, missing context
# ---------------------------------------------------------------------------


def test_empty_text_returns_empty() -> None:
    assert validate_stat_claims("", _CTX) == []


def test_empty_context_returns_empty() -> None:
    """Without PLAYER CONTEXT we have no ground truth — don't flag anything."""
    assert validate_stat_claims("Your K/D is 99.9!", "") == []


def test_context_without_stats_returns_empty() -> None:
    """A PLAYER CONTEXT block with no numeric stats produces no warnings."""
    ctx = "PLAYER CONTEXT — Nobody#NA\nNothing here."
    assert validate_stat_claims("Your K/D is 1.5.", ctx) == []


def test_bare_numbers_are_not_flagged() -> None:
    """Numbers without an explicit stat label are not the sanitizer's concern."""
    text = "Round 5 had 1.2 trades and 0.8 multikills."
    assert validate_stat_claims(text, _CTX) == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_same_claim_emitted_once() -> None:
    text = "Your K/D is 1.5. Your K/D is 1.5. Your K/D is 1.5."
    warnings = validate_stat_claims(text, _CTX)
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------


def test_format_includes_both_values() -> None:
    w = StatWarning(
        stat="K/D",
        claimed="1.5",
        actual="0.93",
        snippet="Your K/D is 1.5.",
    )
    formatted = w.format()
    assert "K/D" in formatted
    assert "1.5" in formatted
    assert "0.93" in formatted
