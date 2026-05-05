#!/usr/bin/env python
"""ValoCoach RAGAS evaluation runner.

Evaluates RAG quality using RAGAS metrics (faithfulness, answer relevancy,
context precision, context recall) against the ground-truth dataset in
``tests/eval/ragas_samples.yaml``.

Usage
-----
    # Full eval — runs all samples against the live pipeline
    python scripts/run_ragas_eval.py

    # Single sample (for debugging)
    python scripts/run_ragas_eval.py --sample jett_dash_ascent_a

    # Skip pipeline calls and score a pre-collected results file
    python scripts/run_ragas_eval.py --from-file results.json

    # Choose metrics (default: all four)
    python scripts/run_ragas_eval.py --metrics faithfulness,answer_relevancy

    # Write scores to JSON
    python scripts/run_ragas_eval.py --output ragas_scores.json

Requirements
------------
    pip install ragas               # not installed by default (heavy)
    pip install datasets            # comes with ragas

The pipeline calls require Ollama to be running.  Use ``--from-file`` to
score a pre-collected dataset without a live LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_FILE = REPO_ROOT / "tests" / "eval" / "ragas_samples.yaml"

# RAGAS metric names → import path
METRIC_NAMES = {
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
}

DEFAULT_METRICS = list(METRIC_NAMES)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_ragas_samples(path: Path = SAMPLES_FILE) -> list[dict[str, Any]]:
    """Load RAGAS evaluation samples from *path*.

    Returns a list of sample dicts with keys:
        id, question, ground_truth, metadata (optional)
    """
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    samples = data.get("samples", [])
    _validate_samples(samples)
    return samples


def _validate_samples(samples: list[dict]) -> None:
    """Raise ``ValueError`` if any sample is missing required fields."""
    required = {"id", "question", "ground_truth"}
    for i, s in enumerate(samples):
        missing = required - s.keys()
        if missing:
            raise ValueError(
                f"Sample at index {i} (id={s.get('id', '?')!r}) "
                f"is missing required fields: {missing}"
            )


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------


def collect_pipeline_outputs(
    samples: list[dict[str, Any]],
    *,
    settings=None,
) -> list[dict[str, Any]]:
    """Run the ValoCoach RAG pipeline for each sample.

    Returns a list of result dicts with keys added to each sample:
        response          — LLM answer string
        retrieved_contexts — list of retrieved passage strings
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from valocoach.cli.commands.coach import run_coach
    from valocoach.core.config import load_settings
    from valocoach.retrieval.retriever import retrieve

    if settings is None:
        settings = load_settings()

    results = []
    for sample in samples:
        question = sample["question"]
        agent = sample.get("metadata", {}).get("agent")
        map_name = sample.get("metadata", {}).get("map")

        # Retrieve context passages
        try:
            docs = retrieve(
                query=question,
                agent=agent,
                map_name=map_name,
                settings=settings,
            )
            contexts = [d.get("document", "") for d in docs]
        except Exception as exc:
            print(f"  [warn] retrieval failed for {sample['id']!r}: {exc}", file=sys.stderr)
            contexts = []

        # Generate answer
        try:
            response = run_coach(
                situation=question,
                agent=agent,
                map_name=map_name,
            )
        except Exception as exc:
            print(f"  [warn] coaching failed for {sample['id']!r}: {exc}", file=sys.stderr)
            response = ""

        results.append(
            {
                **sample,
                "response": response or "",
                "retrieved_contexts": contexts,
            }
        )
    return results


# ---------------------------------------------------------------------------
# RAGAS dataset builder
# ---------------------------------------------------------------------------


def build_ragas_dataset(results: list[dict[str, Any]]):
    """Convert collected results into a RAGAS-compatible ``Dataset``.

    Requires ``datasets`` and ``ragas`` to be installed.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("The 'datasets' package is required: pip install datasets") from exc

    rows = []
    for r in results:
        rows.append(
            {
                "user_input": r["question"],
                "response": r.get("response", ""),
                "retrieved_contexts": r.get("retrieved_contexts", []),
                "reference": r["ground_truth"],
            }
        )
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# RAGAS metric instantiation
# ---------------------------------------------------------------------------


def _get_metrics(names: list[str]):
    """Return RAGAS metric objects for the given *names*.

    Raises ``ImportError`` if ragas is not installed.
    """
    try:
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise ImportError("RAGAS is not installed.  Install it with:  pip install ragas") from exc

    mapping = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
    }
    unknown = set(names) - mapping.keys()
    if unknown:
        raise ValueError(f"Unknown RAGAS metrics: {unknown}.  Choose from {set(mapping.keys())}")
    return [mapping[n] for n in names]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def run_ragas_eval(
    dataset,
    metric_names: list[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Score *dataset* with RAGAS.

    Returns a dict mapping metric name -> mean score (0-1).
    """
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise ImportError("RAGAS is not installed.  Install it with:  pip install ragas") from exc

    metrics = _get_metrics(metric_names)
    result = evaluate(dataset=dataset, metrics=metrics)
    return {k: float(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_scores(scores: dict[str, float]) -> None:
    """Print a pretty score table to stdout."""
    width = max((len(k) for k in scores), default=12) + 2
    print()
    print("RAGAS Evaluation Results")
    print("=" * (width + 10))
    for name, score in sorted(scores.items()):
        bar = "█" * int(score * 20)
        print(f"  {name:<{width}} {score:.3f}  {bar}")
    print()
    mean = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'Overall mean':<{width}} {mean:.3f}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RAGAS evaluation for ValoCoach.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sample", metavar="ID", help="Evaluate only this sample ID.")
    p.add_argument(
        "--from-file",
        metavar="FILE",
        help="Skip pipeline calls; load pre-collected results from FILE (JSON).",
    )
    p.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help=f"Comma-separated list of metrics (default: all). Choices: {', '.join(sorted(METRIC_NAMES))}",
    )
    p.add_argument("--output", metavar="FILE", help="Write scores to FILE as JSON.")
    p.add_argument(
        "--samples-file",
        metavar="FILE",
        default=str(SAMPLES_FILE),
        help=f"YAML samples file (default: {SAMPLES_FILE})",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # 1. Load samples
    samples_path = Path(args.samples_file)
    try:
        samples = load_ragas_samples(samples_path)
    except Exception as exc:
        print(f"ERROR: could not load samples from {samples_path}: {exc}", file=sys.stderr)
        return 2

    if args.sample:
        samples = [s for s in samples if s["id"] == args.sample]
        if not samples:
            print(f"ERROR: no sample found with id={args.sample!r}", file=sys.stderr)
            return 2

    print(f"Loaded {len(samples)} sample(s) from {samples_path.name}")

    # 2. Collect pipeline outputs (or load from file)
    if args.from_file:
        try:
            with open(args.from_file, encoding="utf-8") as fh:
                results = json.load(fh)
        except Exception as exc:
            print(f"ERROR: could not load results file: {exc}", file=sys.stderr)
            return 2
    else:
        print(f"Running pipeline for {len(samples)} sample(s)…")
        try:
            results = collect_pipeline_outputs(samples)
        except Exception as exc:
            print(f"ERROR: pipeline collection failed: {exc}", file=sys.stderr)
            return 2

    # 3. Build RAGAS dataset
    try:
        dataset = build_ragas_dataset(results)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # 4. Run RAGAS evaluation
    print(f"Scoring with metrics: {', '.join(metric_names)}")
    try:
        scores = run_ragas_eval(dataset, metric_names=metric_names)
    except (ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # 5. Display
    _print_scores(scores)

    # 6. Optionally write JSON output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"Scores written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
