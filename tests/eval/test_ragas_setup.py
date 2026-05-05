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
    _get_metrics,
    _print_scores,
    _validate_samples,
    build_ragas_dataset,
    collect_pipeline_outputs,
    load_ragas_samples,
    main,
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
        with (
            patch.dict("sys.modules", {"datasets": None}),
            pytest.raises(ImportError, match="datasets"),
        ):
            build_ragas_dataset(minimal_results)

    def test_builds_dataset_via_mock(self, minimal_results):
        """build_ragas_dataset happy path with mocked datasets module."""
        fake_dataset = MagicMock(name="Dataset")
        datasets_mod = MagicMock()
        datasets_mod.Dataset.from_list.return_value = fake_dataset

        with patch.dict("sys.modules", {"datasets": datasets_mod}):
            import importlib

            import run_ragas_eval as rre

            importlib.reload(rre)
            result = rre.build_ragas_dataset(minimal_results)

        assert result is fake_dataset
        # Verify column mapping
        call_rows = datasets_mod.Dataset.from_list.call_args[0][0]
        assert call_rows[0]["user_input"] == minimal_results[0]["question"]
        assert call_rows[0]["reference"] == minimal_results[0]["ground_truth"]


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
            scores = rre.run_ragas_eval(
                MagicMock(), metric_names=["faithfulness", "answer_relevancy"]
            )
        assert isinstance(scores, dict)
        # After mock module teardown, just verify the function returned something dict-like.
        assert all(isinstance(v, float) for v in scores.values())

    def test_raises_import_error_without_ragas(self):
        # Temporarily remove ragas from sys.modules
        saved = sys.modules.pop("ragas", None)
        try:
            with (
                patch.dict("sys.modules", {"ragas": None}),
                pytest.raises(ImportError, match="RAGAS"),
            ):
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


# ---------------------------------------------------------------------------
# TestLoadRagasSamplesCount — 20-sample assertion
# ---------------------------------------------------------------------------


class TestRagasSamplesCount:
    def test_samples_file_has_20_entries(self):
        """ragas_samples.yaml must have exactly 20 samples."""
        samples = load_ragas_samples(SAMPLES_FILE)
        assert len(samples) == 20, f"Expected 20 samples, got {len(samples)}"

    def test_all_samples_have_metadata(self):
        """Every sample should have at least an 'id', 'question', and 'ground_truth'."""
        samples = load_ragas_samples(SAMPLES_FILE)
        for s in samples:
            assert s.get("question"), f"Sample {s['id']} has empty question"
            gt = s.get("ground_truth", "")
            assert gt and len(str(gt).strip()) > 20, f"Sample {s['id']} ground_truth too short"


# ---------------------------------------------------------------------------
# TestCollectPipelineOutputs — collect_pipeline_outputs() (LLM/retriever mocked)
# ---------------------------------------------------------------------------


class TestCollectPipelineOutputs:
    """collect_pipeline_outputs() is tested here with mocked valocoach modules."""

    def _samples(self) -> list[dict]:
        return [
            {
                "id": "s1",
                "question": "How should I use Jett on Ascent?",
                "ground_truth": "Use Tailwind.",
                "metadata": {"agent": "Jett", "map": "Ascent"},
            },
            {
                "id": "s2",
                "question": "Eco decision on pistol round?",
                "ground_truth": "Save credits.",
                "metadata": {},
            },
        ]

    def _mock_modules(self, coach_response="Tailwind dash.", contexts=None):
        if contexts is None:
            contexts = [{"document": "Jett uses Tailwind for entry."}]
        return {
            "valocoach.cli.commands.coach": MagicMock(
                run_coach=MagicMock(return_value=coach_response)
            ),
            "valocoach.core.config": MagicMock(load_settings=MagicMock(return_value=MagicMock())),
            "valocoach.retrieval.retriever": MagicMock(
                retrieve=MagicMock(return_value=contexts)
            ),
        }

    def test_returns_one_result_per_sample(self):
        samples = self._samples()
        with patch.dict("sys.modules", self._mock_modules()):
            results = collect_pipeline_outputs(samples)
        assert len(results) == len(samples)

    def test_response_populated_from_run_coach(self):
        samples = self._samples()[:1]
        with patch.dict("sys.modules", self._mock_modules(coach_response="Tailwind entry.")):
            results = collect_pipeline_outputs(samples)
        assert results[0]["response"] == "Tailwind entry."

    def test_retrieved_contexts_populated_from_retriever(self):
        samples = self._samples()[:1]
        contexts = [{"document": "chunk 1"}, {"document": "chunk 2"}]
        with patch.dict("sys.modules", self._mock_modules(contexts=contexts)):
            results = collect_pipeline_outputs(samples)
        assert results[0]["retrieved_contexts"] == ["chunk 1", "chunk 2"]

    def test_retrieval_failure_results_in_empty_contexts(self):
        """A retrieval error should not abort the run — contexts become []."""
        samples = self._samples()[:1]
        mocks = self._mock_modules()
        mocks["valocoach.retrieval.retriever"].retrieve.side_effect = RuntimeError("chroma down")
        with patch.dict("sys.modules", mocks):
            results = collect_pipeline_outputs(samples)
        assert results[0]["retrieved_contexts"] == []

    def test_coach_failure_results_in_empty_response(self):
        """A coaching error should not abort the run — response becomes ''."""
        samples = self._samples()[:1]
        mocks = self._mock_modules()
        mocks["valocoach.cli.commands.coach"].run_coach.side_effect = RuntimeError("llm down")
        with patch.dict("sys.modules", mocks):
            results = collect_pipeline_outputs(samples)
        assert results[0]["response"] == ""

    def test_original_sample_fields_preserved(self):
        """Result dicts contain all original sample fields plus response + retrieved_contexts."""
        samples = self._samples()[:1]
        with patch.dict("sys.modules", self._mock_modules()):
            results = collect_pipeline_outputs(samples)
        r = results[0]
        assert r["id"] == "s1"
        assert r["ground_truth"] == "Use Tailwind."
        assert "response" in r
        assert "retrieved_contexts" in r

    def test_settings_none_loads_from_config(self):
        """When settings=None, load_settings() is called to obtain settings."""
        samples = self._samples()[:1]
        mocks = self._mock_modules()
        mock_load = mocks["valocoach.core.config"].load_settings
        with patch.dict("sys.modules", mocks):
            collect_pipeline_outputs(samples, settings=None)
        mock_load.assert_called_once()

    def test_settings_provided_skips_load_settings(self):
        """When settings is already supplied, load_settings() is NOT called."""
        samples = self._samples()[:1]
        mocks = self._mock_modules()
        mock_load = mocks["valocoach.core.config"].load_settings
        fake_settings = MagicMock()
        with patch.dict("sys.modules", mocks):
            collect_pipeline_outputs(samples, settings=fake_settings)
        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# TestGetMetrics — _get_metrics() success path (ragas mocked)
# ---------------------------------------------------------------------------


class TestGetMetrics:
    def _ragas_metrics_mock(self):
        m = MagicMock()
        m.Faithfulness.return_value = MagicMock(name="Faithfulness")
        m.AnswerRelevancy.return_value = MagicMock(name="AnswerRelevancy")
        m.ContextPrecision.return_value = MagicMock(name="ContextPrecision")
        m.ContextRecall.return_value = MagicMock(name="ContextRecall")
        return m

    def test_returns_list_of_metric_objects(self):
        metrics_mod = self._ragas_metrics_mock()
        with patch.dict("sys.modules", {"ragas.metrics": metrics_mod}):
            result = _get_metrics(["faithfulness", "answer_relevancy"])
        assert len(result) == 2

    def test_unknown_metric_raises_value_error(self):
        metrics_mod = self._ragas_metrics_mock()
        with (
            patch.dict("sys.modules", {"ragas.metrics": metrics_mod}),
            pytest.raises(ValueError, match="Unknown RAGAS metrics"),
        ):
            _get_metrics(["not_a_metric"])

    def test_all_four_default_metrics_instantiated(self):
        metrics_mod = self._ragas_metrics_mock()
        with patch.dict("sys.modules", {"ragas.metrics": metrics_mod}):
            result = _get_metrics(list(METRIC_NAMES))
        assert len(result) == len(METRIC_NAMES)

    def test_raises_import_error_when_ragas_missing(self):
        with (
            patch.dict("sys.modules", {"ragas.metrics": None}),
            pytest.raises((ImportError, AttributeError)),
        ):
            _get_metrics(["faithfulness"])


# ---------------------------------------------------------------------------
# TestMain (ragas_eval) — smoke tests for main() with all I/O mocked
# ---------------------------------------------------------------------------


class TestMainRagasEval:
    def _fake_samples(self) -> list[dict]:
        return [
            {
                "id": "t1",
                "question": "q1",
                "ground_truth": "gt1",
                "response": "r1",
                "retrieved_contexts": ["c1"],
            }
        ]

    def test_main_returns_0_on_success(self, tmp_path):
        """main() returns 0 when the pipeline succeeds end-to-end (all mocked)."""
        fake_samples = self._fake_samples()
        fake_ds = MagicMock()

        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=fake_samples),
            patch("run_ragas_eval.collect_pipeline_outputs", return_value=fake_samples),
            patch("run_ragas_eval.build_ragas_dataset", return_value=fake_ds),
            patch("run_ragas_eval.run_ragas_eval", return_value={"faithfulness": 0.9}),
            patch("sys.argv", ["run_ragas_eval.py"]),
        ):
            exit_code = main()

        assert exit_code == 0

    def test_main_returns_2_on_missing_samples_file(self, tmp_path):
        """main() returns 2 when samples file does not exist."""
        with (
            patch("sys.argv", ["run_ragas_eval.py", "--samples-file", "/no/such/file.yaml"]),
        ):
            exit_code = main()

        assert exit_code == 2

    def test_main_returns_2_on_unknown_sample_id(self):
        """--sample with an unknown ID returns exit code 2."""
        fake_samples = self._fake_samples()
        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=fake_samples),
            patch("sys.argv", ["run_ragas_eval.py", "--sample", "does_not_exist"]),
        ):
            exit_code = main()

        assert exit_code == 2

    def test_main_writes_output_file(self, tmp_path):
        """--output FILE writes a JSON scores file."""
        import json

        fake_samples = self._fake_samples()
        out_file = tmp_path / "scores.json"
        fake_ds = MagicMock()

        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=fake_samples),
            patch("run_ragas_eval.collect_pipeline_outputs", return_value=fake_samples),
            patch("run_ragas_eval.build_ragas_dataset", return_value=fake_ds),
            patch("run_ragas_eval.run_ragas_eval", return_value={"faithfulness": 0.85}),
            patch("sys.argv", ["run_ragas_eval.py", "--output", str(out_file)]),
        ):
            main()

        data = json.loads(out_file.read_text())
        assert data["faithfulness"] == pytest.approx(0.85)

    def test_main_from_file_skips_pipeline(self, tmp_path):
        """--from-file loads pre-collected results instead of calling the pipeline."""
        import json

        pre_results = self._fake_samples()
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(pre_results))

        fake_ds = MagicMock()

        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=pre_results),
            patch("run_ragas_eval.collect_pipeline_outputs") as mock_collect,
            patch("run_ragas_eval.build_ragas_dataset", return_value=fake_ds),
            patch("run_ragas_eval.run_ragas_eval", return_value={"faithfulness": 0.8}),
            patch("sys.argv", ["run_ragas_eval.py", "--from-file", str(results_file)]),
        ):
            exit_code = main()

        # collect_pipeline_outputs must NOT have been called
        mock_collect.assert_not_called()
        assert exit_code == 0

    def test_main_returns_2_on_build_dataset_import_error(self):
        """ImportError from build_ragas_dataset (datasets not installed) → exit 2."""
        fake_samples = self._fake_samples()

        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=fake_samples),
            patch("run_ragas_eval.collect_pipeline_outputs", return_value=fake_samples),
            patch("run_ragas_eval.build_ragas_dataset", side_effect=ImportError("no datasets")),
            patch("sys.argv", ["run_ragas_eval.py"]),
        ):
            exit_code = main()

        assert exit_code == 2

    def test_main_single_sample_filter(self):
        """--sample filters to exactly one sample."""
        fake_samples = [
            {"id": "s1", "question": "q1", "ground_truth": "g1", "response": "r1", "retrieved_contexts": []},
            {"id": "s2", "question": "q2", "ground_truth": "g2", "response": "r2", "retrieved_contexts": []},
        ]
        fake_ds = MagicMock()
        mock_collect = MagicMock(return_value=fake_samples[:1])

        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=fake_samples),
            patch("run_ragas_eval.collect_pipeline_outputs", mock_collect),
            patch("run_ragas_eval.build_ragas_dataset", return_value=fake_ds),
            patch("run_ragas_eval.run_ragas_eval", return_value={"faithfulness": 0.9}),
            patch("sys.argv", ["run_ragas_eval.py", "--sample", "s1"]),
        ):
            exit_code = main()

        # Only sample s1 passed to collect
        passed_samples = mock_collect.call_args[0][0]
        assert len(passed_samples) == 1
        assert passed_samples[0]["id"] == "s1"
        assert exit_code == 0

    def test_main_corrupt_from_file_returns_2(self, tmp_path):
        """--from-file pointing at a non-JSON file → return 2."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{{")

        fake_samples = self._fake_samples()
        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=fake_samples),
            patch("sys.argv", ["run_ragas_eval.py", "--from-file", str(bad_file)]),
        ):
            exit_code = main()

        assert exit_code == 2

    def test_main_collect_pipeline_raises_returns_2(self):
        """collect_pipeline_outputs raising → return 2."""
        fake_samples = self._fake_samples()

        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=fake_samples),
            patch(
                "run_ragas_eval.collect_pipeline_outputs",
                side_effect=RuntimeError("no ollama"),
            ),
            patch("sys.argv", ["run_ragas_eval.py"]),
        ):
            exit_code = main()

        assert exit_code == 2

    def test_main_run_ragas_eval_import_error_returns_2(self):
        """run_ragas_eval raising ImportError (ragas not installed) → return 2."""
        fake_samples = self._fake_samples()
        fake_ds = MagicMock()

        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=fake_samples),
            patch("run_ragas_eval.collect_pipeline_outputs", return_value=fake_samples),
            patch("run_ragas_eval.build_ragas_dataset", return_value=fake_ds),
            patch(
                "run_ragas_eval.run_ragas_eval",
                side_effect=ImportError("RAGAS not installed"),
            ),
            patch("sys.argv", ["run_ragas_eval.py"]),
        ):
            exit_code = main()

        assert exit_code == 2

    def test_main_run_ragas_eval_value_error_returns_2(self):
        """run_ragas_eval raising ValueError (unknown metric) → return 2."""
        fake_samples = self._fake_samples()
        fake_ds = MagicMock()

        with (
            patch("run_ragas_eval.load_ragas_samples", return_value=fake_samples),
            patch("run_ragas_eval.collect_pipeline_outputs", return_value=fake_samples),
            patch("run_ragas_eval.build_ragas_dataset", return_value=fake_ds),
            patch(
                "run_ragas_eval.run_ragas_eval",
                side_effect=ValueError("Unknown RAGAS metrics"),
            ),
            patch("sys.argv", ["run_ragas_eval.py"]),
        ):
            exit_code = main()

        assert exit_code == 2
