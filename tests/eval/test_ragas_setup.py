"""Offline tests for the RAGAS evaluation infrastructure.

Tests the dataset loading, sample validation, dataset building, and metric
configuration logic WITHOUT calling RAGAS, an LLM, or a vector store.

``ragas`` and ``datasets`` are optional heavy dependencies — tests that
need them are individually skipped via ``importlib.util.find_spec``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Ensure scripts/ is importable without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SAMPLES_FILE = REPO_ROOT / "tests" / "eval" / "ragas_samples.yaml"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_ragas_eval import (  # noqa: E402
    DEFAULT_METRICS,
    METRIC_NAMES,
    _print_scores,
    _validate_samples,
    build_ragas_dataset,
    load_ragas_samples,
    run_ragas_eval,
)

_HAS_DATASETS = importlib.util.find_spec("datasets") is not None
_HAS_RAGAS = importlib.util.find_spec("ragas") is not None

skip_no_datasets = pytest.mark.skipif(not _HAS_DATASETS, reason="datasets not installed")
skip_no_ragas = pytest.mark.skipif(not _HAS_RAGAS, reason="ragas not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_samples() -> list[dict]:
    return [
        {
            "id": "test_jett",
            "question": "How do I use Jett on Ascent?",
            "ground_truth": "Use Tailwind to dash through A Long.",
            "metadata": {"agent": "Jett", "map": "Ascent"},
        },
        {
            "id": "test_omen",
            "question": "How do I smoke with Omen on Split?",
            "ground_truth": "Throw Dark Cover on B Heaven.",
            "metadata": {"agent": "Omen", "map": "Split"},
        },
    ]


@pytest.fixture()
def minimal_results(minimal_samples) -> list[dict]:
    """Samples with response + retrieved_contexts added (as if pipeline ran)."""
    return [
        {
            **s,
            "response": f"Generated answer for: {s['question']}",
            "retrieved_contexts": [f"Context chunk for {s['id']}", "Another relevant chunk"],
        }
        for s in minimal_samples
    ]


# ---------------------------------------------------------------------------
# TestLoadRagasSamples
# ---------------------------------------------------------------------------


class TestLoadRagasSamples:
    def test_loads_real_samples_file(self):
        samples = load_ragas_samples(SAMPLES_FILE)
        assert len(samples) > 0

    def test_returns_list_of_dicts(self):
        samples = load_ragas_samples(SAMPLES_FILE)
        assert all(isinstance(s, dict) for s in samples)

    def test_every_sample_has_id(self):
        samples = load_ragas_samples(SAMPLES_FILE)
        for s in samples:
            assert "id" in s, f"Sample missing 'id': {s}"

    def test_every_sample_has_question(self):
        samples = load_ragas_samples(SAMPLES_FILE)
        for s in samples:
            assert "question" in s

    def test_every_sample_has_ground_truth(self):
        samples = load_ragas_samples(SAMPLES_FILE)
        for s in samples:
            assert "ground_truth" in s

    def test_sample_ids_are_unique(self):
        samples = load_ragas_samples(SAMPLES_FILE)
        ids = [s["id"] for s in samples]
        assert len(ids) == len(set(ids)), "Duplicate sample IDs found"

    def test_loads_from_tmp_file(self, tmp_path: Path):
        data = {
            "samples": [
                {
                    "id": "a",
                    "question": "q",
                    "ground_truth": "gt",
                }
            ]
        }
        f = tmp_path / "s.yaml"
        f.write_text(yaml.dump(data))
        samples = load_ragas_samples(f)
        assert len(samples) == 1
        assert samples[0]["id"] == "a"

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_ragas_samples(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# TestValidateSamples
# ---------------------------------------------------------------------------


class TestValidateSamples:
    def test_valid_samples_do_not_raise(self, minimal_samples):
        _validate_samples(minimal_samples)  # should not raise

    def test_raises_on_missing_id(self):
        bad = [{"question": "q", "ground_truth": "gt"}]
        with pytest.raises(ValueError, match="missing required fields"):
            _validate_samples(bad)

    def test_raises_on_missing_question(self):
        bad = [{"id": "x", "ground_truth": "gt"}]
        with pytest.raises(ValueError, match="missing required fields"):
            _validate_samples(bad)

    def test_raises_on_missing_ground_truth(self):
        bad = [{"id": "x", "question": "q"}]
        with pytest.raises(ValueError, match="missing required fields"):
            _validate_samples(bad)

    def test_empty_list_is_valid(self):
        _validate_samples([])  # no samples is OK

    def test_extra_fields_are_allowed(self):
        samples = [
            {
                "id": "x",
                "question": "q",
                "ground_truth": "gt",
                "metadata": {"agent": "Jett"},
                "extra": "value",
            }
        ]
        _validate_samples(samples)  # should not raise


# ---------------------------------------------------------------------------
# TestBuildRagasDataset
# ---------------------------------------------------------------------------


class TestBuildRagasDataset:
    @skip_no_datasets
    def test_builds_dataset_with_correct_columns(self, minimal_results):
        ds = build_ragas_dataset(minimal_results)
        assert "user_input" in ds.column_names
        assert "response" in ds.column_names
        assert "retrieved_contexts" in ds.column_names
        assert "reference" in ds.column_names

    @skip_no_datasets
    def test_dataset_has_correct_row_count(self, minimal_results):
        ds = build_ragas_dataset(minimal_results)
        assert len(ds) == len(minimal_results)

    @skip_no_datasets
    def test_question_maps_to_user_input(self, minimal_results):
        ds = build_ragas_dataset(minimal_results)
        assert ds[0]["user_input"] == minimal_results[0]["question"]

    @skip_no_datasets
    def test_ground_truth_maps_to_reference(self, minimal_results):
        ds = build_ragas_dataset(minimal_results)
        assert ds[0]["reference"] == minimal_results[0]["ground_truth"]

    def test_raises_import_error_without_datasets(self, minimal_results):
        with patch.dict("sys.modules", {"datasets": None}), pytest.raises(ImportError, match="datasets"):
            build_ragas_dataset(minimal_results)


# ---------------------------------------------------------------------------
# TestMetricNames
# ---------------------------------------------------------------------------


class TestMetricNames:
    def test_default_metrics_is_nonempty(self):
        assert len(DEFAULT_METRICS) > 0

    def test_all_metric_names_known(self):
        assert set(DEFAULT_METRICS).issubset(METRIC_NAMES)

    def test_faithfulness_in_metric_names(self):
        assert "faithfulness" in METRIC_NAMES

    def test_answer_relevancy_in_metric_names(self):
        assert "answer_relevancy" in METRIC_NAMES

    def test_context_precision_in_metric_names(self):
        assert "context_precision" in METRIC_NAMES

    def test_context_recall_in_metric_names(self):
        assert "context_recall" in METRIC_NAMES


# ---------------------------------------------------------------------------
# TestRunRagasEval (mocked ragas — always runs, no ragas install needed)
# ---------------------------------------------------------------------------


class TestRunRagasEval:
    def _ragas_mock_modules(self, scores: dict) -> dict:
        """Return a sys.modules overlay that fakes ragas + ragas.metrics."""
        ragas_mod = MagicMock()
        ragas_mod.evaluate.return_value = dict(scores.items())

        metrics_mod = MagicMock()
        metrics_mod.Faithfulness.return_value = MagicMock()
        metrics_mod.AnswerRelevancy.return_value = MagicMock()
        metrics_mod.ContextPrecision.return_value = MagicMock()
        metrics_mod.ContextRecall.return_value = MagicMock()

        return {
            "ragas": ragas_mod,
            "ragas.metrics": metrics_mod,
        }

    def test_returns_dict_of_float_scores(self):
        fake_scores = {"faithfulness": 0.85, "answer_relevancy": 0.9}
        mocks = self._ragas_mock_modules(fake_scores)
        with patch.dict("sys.modules", mocks):
            import importlib

            import run_ragas_eval as rre
            importlib.reload(rre)
            scores = rre.run_ragas_eval(MagicMock(), metric_names=["faithfulness", "answer_relevancy"])
        assert isinstance(scores, dict)
        # After mock module teardown, just verify the function returned something dict-like.
        assert all(isinstance(v, float) for v in scores.values())

    def test_raises_import_error_without_ragas(self):
        # Temporarily remove ragas from sys.modules
        saved = sys.modules.pop("ragas", None)
        try:
            with patch.dict("sys.modules", {"ragas": None}), pytest.raises(ImportError, match="RAGAS"):
                run_ragas_eval(MagicMock())
        finally:
            if saved is not None:
                sys.modules["ragas"] = saved


# ---------------------------------------------------------------------------
# TestPrintScores
# ---------------------------------------------------------------------------


class TestPrintScores:
    def test_prints_metric_names(self, capsys):
        _print_scores({"faithfulness": 0.85, "answer_relevancy": 0.7})
        out = capsys.readouterr().out
        assert "faithfulness" in out
        assert "answer_relevancy" in out

    def test_prints_numeric_scores(self, capsys):
        _print_scores({"faithfulness": 0.85})
        out = capsys.readouterr().out
        assert "0.850" in out

    def test_prints_overall_mean(self, capsys):
        _print_scores({"faithfulness": 0.8, "answer_relevancy": 0.6})
        out = capsys.readouterr().out
        assert "Overall mean" in out or "mean" in out.lower()

    def test_handles_empty_scores(self, capsys):
        _print_scores({})
        out = capsys.readouterr().out
        assert "0.000" in out or "mean" in out.lower()
