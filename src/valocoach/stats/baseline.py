"""Rolling baseline and anomaly detection over a match window.

Rolling baseline:
    The "baseline" is the player's aggregate stats over the full row set
    supplied (e.g., last 20 competitive matches). The "form" is their stats
    over the N most-recent rows within that same set (default: last 5).

    Because form is a strict subset of baseline, the comparison directly
    answers "has the player's output shifted recently?" -- exactly the signal
    a coach needs. No external data store is required; everything is derived
    from the already-fetched rows.

Z-score approach:
    Each match is treated as one independent observation. For every tracked
    metric we compute a per-match list (ACS = score/rounds for that match,
    K/D = kills/deaths for that match, etc.) and derive the population mean
    and standard deviation across all baseline matches.

    The form window's per-match mean is then compared to the baseline mean
    via a z-score: z = (form_mean - baseline_mean) / baseline_stddev.

    Why z-scores over fixed thresholds?
        A 200-ACS player at 170 is a small blip on a wild day but a
        meaningful drop for someone who scores 200 every single game.
        Fixed thresholds can't tell the difference; z-scores adapt to
        each player's own variance automatically.

Stddev floor:
    Highly consistent players can have stddevs close to zero, which would
    inflate z-scores for trivial fluctuations. We clamp the effective stddev
    to a floor of max(2% of baseline mean, 1e-6) so that near-constant
    players still require a meaningful absolute shift before an anomaly fires.

Severity tiers (z-score thresholds):
    notable      |z| >= 1.5  (~86th percentile)
    significant  |z| >= 2.0  (~97.5th percentile; the classic "2-sigma rule")

Anomaly dataclass:
    Carries both the raw values (for further computation) and pre-formatted
    display strings (so presentation layers need no formatting knowledge).
    Includes z_score and baseline_stddev for rich coach context output.

Output:
    compare_baseline() returns a BaselineComparison dataclass with aggregate
    PlayerStats for both windows plus a sorted list of Anomaly objects.
    Callers (coach context, stats CLI) own rendering -- this module is pure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

from valocoach.data.orm_models import MatchPlayer
from valocoach.stats.calculator import PlayerStats, compute_player_stats
from valocoach.stats.filters import recent_form

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORM_WINDOW: Final[int] = 5
"""Default recent-form window -- last N matches compared against the baseline."""

MIN_BASELINE_MATCHES: Final[int] = 5
"""Skip comparison when the baseline has fewer matches than this."""

MIN_FORM_MATCHES: Final[int] = 3
"""Skip anomaly detection when the form window has fewer matches than this."""

# Z-score anomaly thresholds.
# 1.5-sigma: unusual, worth mentioning. 2.0-sigma: statistically significant.
Z_NOTABLE: Final[float] = 1.5
Z_SIGNIFICANT: Final[float] = 2.0

# Stddev floor: clamp effective stddev to at least this fraction of the
# baseline mean. Prevents trivial fluctuations from scoring huge z-scores
# on highly consistent players.
_STDDEV_FLOOR_FRAC: Final[float] = 0.02

# Absolute stddev floor: prevents division-by-zero when baseline_mean ~ 0
# (e.g. win_rate on a perfect-loss or perfect-win window).
_STDDEV_ABS_FLOOR: Final[float] = 1e-6

# ---------------------------------------------------------------------------
# Per-metric config
# ---------------------------------------------------------------------------
# Stripped down from the threshold-based version: mode/notable/severe are
# gone (replaced by unified z-score tiers). Only label, positive, and fmt
# remain -- the pure presentation metadata.
#
#   label     Human-readable metric name.
#   positive  True when "higher is better". False = rising value is a decline.
#   fmt       ".0f" | ".2f" | "pct" (stored 0.0-1.0, rendered as %).

_METRIC_CONFIG: Final[dict[str, dict[str, object]]] = {
    "acs": {
        "label": "ACS",
        "positive": True,
        "fmt": ".0f",
    },
    "kd": {
        "label": "K/D",
        "positive": True,
        "fmt": ".2f",
    },
    "adr": {
        "label": "ADR",
        "positive": True,
        "fmt": ".0f",
    },
    "hs_pct": {
        "label": "HS%",
        "positive": True,
        "fmt": "pct",
    },
    "win_rate": {
        "label": "Win rate",
        "positive": True,
        "fmt": "pct",
    },
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Anomaly:
    """One detected deviation between baseline and form.

    baseline_val / form_val are per-match means (not round-weighted aggregates)
    so that they match the scale on which z_score was computed. The
    BaselineComparison.baseline / .form PlayerStats are the round-weighted
    aggregates used for display elsewhere -- they will differ slightly.
    """

    metric: str                          # internal key: "acs", "kd", ...
    label: str                           # display label: "ACS", "K/D", ...
    baseline_val: float                  # per-match mean over baseline window
    form_val: float                      # per-match mean over form window
    delta: float                         # form_val - baseline_val
    pct_delta: float | None              # delta / baseline_val (None if baseline = 0)
    direction: Literal["up", "down"]     # "up" = form > baseline
    severity: Literal["notable", "significant"]
    is_improvement: bool                 # True = player got better
    fmt: str                             # ".0f" | ".2f" | "pct"
    z_score: float                       # (form_mean - baseline_mean) / eff_stddev
    baseline_stddev: float               # raw population stddev of baseline

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _render(self, v: float) -> str:
        if self.fmt == "pct":
            return f"{v * 100:.0f}%"
        return f"{v:{self.fmt}}"

    @property
    def arrow(self) -> str:
        return "\u2191" if self.direction == "up" else "\u2193"

    @property
    def baseline_display(self) -> str:
        return self._render(self.baseline_val)

    @property
    def form_display(self) -> str:
        return self._render(self.form_val)

    @property
    def delta_display(self) -> str:
        """Compact signed delta: '+6pp', '-22%', '+0.31'."""
        if self.fmt == "pct":
            pp = self.delta * 100
            return f"{pp:+.0f}pp"
        if self.pct_delta is not None:
            return f"{self.pct_delta * 100:+.0f}%"
        return f"{self.delta:+.2f}"

    @property
    def trend_word(self) -> str:
        return "improvement" if self.is_improvement else "slump"

    def one_liner(self) -> str:
        """Compact summary for coach context / CLI.

        Includes z-score so the LLM (or a human skimming the stats panel)
        can immediately see how statistically unusual the shift is.
        """
        sev = "significant" if self.severity == "significant" else "notable"
        return (
            f"{self.label} {self.arrow} {self.form_display} vs {self.baseline_display}"
            f" ({self.delta_display}, {self.z_score:+.1f}\u03c3) -- {sev} {self.trend_word}"
        )


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """Baseline vs. recent form with detected anomalies.

    .baseline and .form are round-weighted PlayerStats for display in tables
    (baseline uses ALL rows; form uses the recent subset).
    Anomaly.baseline_val is the per-match mean of the HISTORICAL rows only
    (baseline_rows minus form_rows), so the z-score is an uncontaminated
    comparison of "recent form" vs "what came before".
    """

    baseline: PlayerStats
    form: PlayerStats
    baseline_matches: int
    form_matches: int
    anomalies: list[Anomaly]

    @property
    def has_anomalies(self) -> bool:
        return bool(self.anomalies)

    @property
    def declines(self) -> list[Anomaly]:
        """Anomalies where performance worsened -- coaching priorities."""
        return [a for a in self.anomalies if not a.is_improvement]

    @property
    def improvements(self) -> list[Anomaly]:
        return [a for a in self.anomalies if a.is_improvement]


# ---------------------------------------------------------------------------
# Per-match decomposition  (the key contribution from the z-score approach)
# ---------------------------------------------------------------------------


def _per_match_metrics(rows: list[MatchPlayer]) -> dict[str, list[float]]:
    """Compute per-match values for every tracked metric.

    Each match contributes exactly one observation regardless of round count.
    This is the correct population for statistical norm estimation -- a
    30-round match and a 13-round stomp are both single data points.

    hs_pct and win_rate are stored as 0.0-1.0 to match PlayerStats convention.
    """
    buckets: dict[str, list[float]] = {k: [] for k in _METRIC_CONFIG}
    for mp in rows:
        rds = max(mp.rounds_played, 1)
        deaths = max(mp.deaths, 1)
        shots = max(mp.headshots + mp.bodyshots + mp.legshots, 1)
        buckets["acs"].append(mp.score / rds)
        buckets["kd"].append(mp.kills / deaths)
        buckets["adr"].append(mp.damage_dealt / rds)
        buckets["hs_pct"].append(mp.headshots / shots)       # 0.0-1.0
        buckets["win_rate"].append(1.0 if mp.won else 0.0)
    return buckets


def _mean_stddev(values: list[float]) -> tuple[float, float]:
    """Population mean and standard deviation over a list of floats."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(variance)


# ---------------------------------------------------------------------------
# Detection logic  (pure -- no I/O)
# ---------------------------------------------------------------------------


def _check_one(
    metric: str,
    form_vals: list[float],
    baseline_mean: float,
    baseline_stddev: float,
) -> Anomaly | None:
    """Evaluate one metric via z-score. Returns Anomaly or None.

    Args:
        metric:           Key into _METRIC_CONFIG.
        form_vals:        Per-match values for the form window.
        baseline_mean:    Per-match mean of the baseline window.
        baseline_stddev:  Population stddev of the baseline window.

    Returns None when:
      - Metric not in config.
      - form_vals is empty.
      - |z| < Z_NOTABLE (shift is within normal variance).
    """
    cfg = _METRIC_CONFIG.get(metric)
    if cfg is None or not form_vals:
        return None

    form_mean = sum(form_vals) / len(form_vals)
    delta = form_mean - baseline_mean
    if delta == 0.0:
        return None

    # Clamp effective stddev: prevents near-zero stddev from inflating z-scores
    # for highly consistent players, while still requiring a meaningful absolute
    # shift to trigger an anomaly.
    effective_stddev = max(
        baseline_stddev,
        abs(baseline_mean) * _STDDEV_FLOOR_FRAC,
        _STDDEV_ABS_FLOOR,
    )

    z = delta / effective_stddev
    if abs(z) < Z_NOTABLE:
        return None

    severity: Literal["notable", "significant"] = (
        "significant" if abs(z) >= Z_SIGNIFICANT else "notable"
    )
    direction: Literal["up", "down"] = "up" if delta > 0 else "down"
    positive: bool = cfg["positive"]  # type: ignore[assignment]
    is_improvement = (direction == "up") == positive
    pct_delta: float | None = (delta / baseline_mean) if baseline_mean != 0.0 else None

    return Anomaly(
        metric=metric,
        label=cfg["label"],  # type: ignore[arg-type]
        baseline_val=baseline_mean,
        form_val=form_mean,
        delta=delta,
        pct_delta=pct_delta,
        direction=direction,
        severity=severity,
        is_improvement=is_improvement,
        fmt=cfg["fmt"],  # type: ignore[arg-type]
        z_score=z,
        baseline_stddev=baseline_stddev,
    )


def detect_anomalies(
    baseline_rows: list[MatchPlayer],
    form_rows: list[MatchPlayer],
) -> list[Anomaly]:
    """Detect performance anomalies by comparing form against baseline.

    Each match is one observation. The z-score measures how many standard
    deviations the form window's per-match mean is from the historical baseline
    per-match mean. This adapts to each player's own variance, unlike fixed
    percentage thresholds.

    Args:
        baseline_rows:  Full match window (e.g., last 20 matches). The
                        historical baseline is derived from these rows with
                        form_rows *excluded* -- so the z-score compares
                        "recent form" against "everything before the form
                        window", not against the contaminated combined pool.
        form_rows:      Recent subset (e.g., last 5 matches).

    Returns a sorted list -- significant declines first (most actionable),
    then notable declines, then improvements.
    """
    # Exclude form rows from the historical baseline so we're comparing
    # "recent window" against "what came before", not against a mixed pool
    # that includes the very window we're testing.
    form_ids = {mp.match_id for mp in form_rows}
    historical_rows = [mp for mp in baseline_rows if mp.match_id not in form_ids]
    if not historical_rows:
        return []

    baseline_metrics = _per_match_metrics(historical_rows)
    form_metrics = _per_match_metrics(form_rows)

    results: list[Anomaly] = []
    for metric in _METRIC_CONFIG:
        b_vals = baseline_metrics.get(metric, [])
        f_vals = form_metrics.get(metric, [])
        if len(b_vals) < MIN_BASELINE_MATCHES or not f_vals:
            continue
        mean, stddev = _mean_stddev(b_vals)
        anomaly = _check_one(metric, f_vals, mean, stddev)
        if anomaly is not None:
            results.append(anomaly)

    results.sort(
        key=lambda a: (
            0 if a.severity == "significant" else 1,
            1 if a.is_improvement else 0,
        )
    )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compare_baseline(
    rows: list[MatchPlayer],
    form_n: int = FORM_WINDOW,
) -> BaselineComparison | None:
    """Compare the player's full-window baseline to their recent form.

    Args:
        rows:   All match rows to use. The baseline is derived from the full
                set; the form window is the ``form_n`` most-recent by
                started_at. Period/agent/map filtering is the caller's job.
        form_n: Number of recent matches in the form window.

    Returns:
        BaselineComparison when the comparison is meaningful, None when:
        - The baseline is too thin (< MIN_BASELINE_MATCHES).
        - The form window covers the entire baseline (no older data to
          compare against -- form == baseline, delta is always 0).

        When form_rows < MIN_FORM_MATCHES, a comparison is returned but
        anomalies is empty -- callers can still display baseline/form stats
        without flagging thin-window false positives.
    """
    if len(rows) < MIN_BASELINE_MATCHES:
        return None

    form_rows = recent_form(rows, form_n)

    if len(form_rows) >= len(rows):
        return None

    baseline_stats = compute_player_stats(rows)
    form_stats = compute_player_stats(form_rows)

    anomalies: list[Anomaly] = []
    if len(form_rows) >= MIN_FORM_MATCHES:
        anomalies = detect_anomalies(rows, form_rows)

    return BaselineComparison(
        baseline=baseline_stats,
        form=form_stats,
        baseline_matches=len(rows),
        form_matches=len(form_rows),
        anomalies=anomalies,
    )
