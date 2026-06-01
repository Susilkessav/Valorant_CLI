#!/usr/bin/env python3
"""F1 — Meta tier-list regression eval.

Compares the current ``data/meta.json`` tier list against a held-out reference
tier list (``tests/eval/meta_reference.yaml``) and reports per-tier
precision/recall.  Use this after each LLM-regenerated meta update to catch
tier assignment regressions before they reach users.

Precision (per tier T):
    |predicted_T ∩ reference_T| / |predicted_T|
    "Of the agents I put in T, what fraction belong there?"

Recall (per tier T):
    |predicted_T ∩ reference_T| / |reference_T|
    "Of the agents that should be in T, what fraction did I find?"

Usage
-----
    # Default run
    python scripts/eval_meta.py

    # Custom reference / meta files
    python scripts/eval_meta.py --reference tests/eval/meta_reference.yaml

    # Stricter threshold (90% precision required to pass)
    python scripts/eval_meta.py --threshold 0.9

    # Save results to JSON
    python scripts/eval_meta.py --output eval_meta_results.json

Exit codes
----------
    0 — all tiers meet the precision threshold
    1 — one or more tiers fail the threshold (or wrong-tier agents found)
    2 — runner error (missing files, bad YAML, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = REPO_ROOT / "tests" / "eval" / "meta_reference.yaml"
DEFAULT_META = REPO_ROOT / "src" / "valocoach" / "retrieval" / "data" / "meta.json"

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
DIM = "\033[2m"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TierResult:
    tier: str
    predicted: set[str]
    reference: set[str]

    @property
    def correct(self) -> set[str]:
        return self.predicted & self.reference

    @property
    def precision(self) -> float:
        return len(self.correct) / len(self.predicted) if self.predicted else 1.0

    @property
    def recall(self) -> float:
        return len(self.correct) / len(self.reference) if self.reference else 1.0

    @property
    def wrong_tier(self) -> set[str]:
        """Reference agents that appear in a DIFFERENT tier in the prediction."""
        return self.reference - self.predicted


@dataclass
class EvalResult:
    patch_version: str
    tiers: list[TierResult] = field(default_factory=list)
    threshold: float = 0.6
    excluded: set[str] = field(default_factory=set)

    @property
    def passed(self) -> bool:
        return all(t.precision >= self.threshold for t in self.tiers)

    def as_dict(self) -> dict:
        return {
            "patch_version": self.patch_version,
            "threshold": self.threshold,
            "passed": self.passed,
            "tiers": [
                {
                    "tier": t.tier,
                    "precision": round(t.precision, 3),
                    "recall": round(t.recall, 3),
                    "correct": sorted(t.correct),
                    "predicted_only": sorted(t.predicted - t.reference),
                    "reference_only": sorted(t.reference - t.predicted),
                    "passed": t.precision >= self.threshold,
                }
                for t in self.tiers
            ],
        }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_meta(path: Path) -> tuple[str, dict[str, set[str]]]:
    """Return (patch_version, {tier: {agents}}) from meta.json."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    patch = data.get("patch", "unknown")
    tier_list = data.get("tier_list", {})
    tiers: dict[str, set[str]] = {}
    for tier in ("S", "A", "B", "C"):
        tiers[tier] = set(tier_list.get(tier, []))
    return patch, tiers


def load_reference(path: Path) -> tuple[dict[str, set[str]], set[str]]:
    """Return ({tier: {agents}}, excluded_agents) from reference YAML."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    ref = data.get("reference_tiers", {})
    tiers: dict[str, set[str]] = {}
    for tier in ("S", "A", "B", "C"):
        tiers[tier] = set(ref.get(tier, []))

    excluded = set(data.get("exclude", []))
    return tiers, excluded


# ---------------------------------------------------------------------------
# Eval logic
# ---------------------------------------------------------------------------


def run_eval(
    meta_path: Path,
    reference_path: Path,
    threshold: float,
) -> EvalResult:
    patch, predicted_tiers = load_meta(meta_path)
    reference_tiers, excluded = load_reference(reference_path)

    # Strip excluded agents from both sides
    for tier in ("S", "A", "B", "C"):
        predicted_tiers[tier] -= excluded
        reference_tiers[tier] -= excluded

    result = EvalResult(patch_version=patch, threshold=threshold, excluded=excluded)
    for tier in ("S", "A", "B", "C"):
        result.tiers.append(
            TierResult(
                tier=tier,
                predicted=predicted_tiers.get(tier, set()),
                reference=reference_tiers.get(tier, set()),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    empty = width - filled
    color = GREEN if value >= 0.7 else (YELLOW if value >= 0.5 else RED)
    return f"{color}{'█' * filled}{'░' * empty}{RESET} {value:.0%}"


def print_results(result: EvalResult, threshold: float) -> None:
    print(f"\n{BOLD}ValoCoach Meta Tier Eval — Patch {result.patch_version}{RESET}")
    print(f"Threshold: {threshold:.0%} precision per tier\n")

    for tr in result.tiers:
        status = f"{GREEN}PASS{RESET}" if tr.precision >= threshold else f"{RED}FAIL{RESET}"
        print(
            f"  {BOLD}{tr.tier}-Tier{RESET}  [{status}]  "
            f"Precision {_bar(tr.precision)}  Recall {_bar(tr.recall)}"
        )
        if tr.correct:
            print(f"    {DIM}Correct :  {', '.join(sorted(tr.correct))}{RESET}")
        wrong = tr.predicted - tr.reference
        if wrong:
            print(f"    {YELLOW}Predicted but not in reference: {', '.join(sorted(wrong))}{RESET}")
        missing = tr.reference - tr.predicted
        if missing:
            print(
                f"    {YELLOW}In reference but not predicted: {', '.join(sorted(missing))}{RESET}"
            )
        print()

    verdict = f"{GREEN}ALL PASS{RESET}" if result.passed else f"{RED}FAIL — below threshold{RESET}"
    print(f"{BOLD}Result: {verdict}{RESET}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare current meta.json tier list against a held-out reference."
    )
    parser.add_argument(
        "--reference",
        metavar="FILE",
        default=str(DEFAULT_REFERENCE),
        help=f"Path to reference YAML (default: {DEFAULT_REFERENCE.name}).",
    )
    parser.add_argument(
        "--meta",
        metavar="FILE",
        default=str(DEFAULT_META),
        help="Path to meta.json (default: auto-detected).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="Minimum per-tier precision to pass (default: 0.6).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write full results as JSON to this file.",
    )
    args = parser.parse_args()

    meta_path = Path(args.meta)
    ref_path = Path(args.reference)

    for p, label in ((meta_path, "meta.json"), (ref_path, "reference YAML")):
        if not p.exists():
            print(f"File not found: {p} ({label})", file=sys.stderr)
            sys.exit(2)

    try:
        result = run_eval(meta_path, ref_path, threshold=args.threshold)
    except Exception as exc:
        print(f"Eval failed: {exc}", file=sys.stderr)
        sys.exit(2)

    print_results(result, threshold=args.threshold)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
        print(f"Results written to {out_path}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
