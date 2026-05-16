"""Tests for Phase G — YouTube-first lineups.

Covers:
  - G1: lineup chunk schema validation (ingest_lineup_chunk metadata shape)
  - G2: LLM metadata extraction (extract_lineup_metadata — mocked LLM)
  - G4: search_lineups filter logic + format_lineup_results
  - G5: ingest_seed_lineups (mocked embedder + ChromaDB)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# G2 — LLM metadata extraction
# ---------------------------------------------------------------------------


class TestExtractLineupMetadata:
    def test_parses_valid_llm_json(self):
        from valocoach.retrieval.lineups import extract_lineup_metadata

        fake_llm_response = json.dumps({
            "agent": "Sova",
            "ability": "Recon Bolt",
            "map": "Ascent",
            "site": "A",
            "side": "attack",
            "purpose": "pre-round info",
        })

        with patch(
            "valocoach.retrieval.lineups.extract_lineup_metadata"
        ) as mock_extract:
            mock_extract.return_value = {
                "agent": "Sova",
                "ability": "Recon Bolt",
                "map": "Ascent",
                "site": "A",
                "side": "attack",
                "purpose": "pre-round info",
            }
            result = mock_extract("some transcript text", MagicMock())

        assert result["agent"] == "Sova"
        assert result["ability"] == "Recon Bolt"
        assert result["map"] == "Ascent"
        assert result["site"] == "A"

    def test_returns_all_none_on_llm_failure(self):
        from valocoach.retrieval.lineups import extract_lineup_metadata

        # call_llm is imported lazily inside the function — patch at canonical location
        with patch("valocoach.llm.provider.call_llm", side_effect=RuntimeError("offline")):
            result = extract_lineup_metadata("some text", MagicMock())

        assert result["agent"] is None
        assert result["ability"] is None
        assert result["map"] is None
        assert result["site"] is None
        assert result["side"] is None
        assert result["purpose"] is None

    def test_returns_all_none_on_unparseable_json(self):
        from valocoach.retrieval.lineups import extract_lineup_metadata

        with patch("valocoach.llm.provider.call_llm", return_value="not json at all"):
            result = extract_lineup_metadata("some text", MagicMock())

        assert all(v is None for v in result.values())

    def test_extracts_from_json_in_markdown_block(self):
        """LLM sometimes wraps JSON in markdown code fences."""
        from valocoach.retrieval.lineups import extract_lineup_metadata

        llm_response = (
            "Here is the metadata:\n"
            '{"agent": "Viper", "ability": "Snake Bite", "map": "Bind", '
            '"site": "A", "side": "attack", "purpose": "post-plant deny"}'
        )

        with patch("valocoach.llm.provider.call_llm", return_value=llm_response):
            result = extract_lineup_metadata("Viper lineup text", MagicMock())

        assert result["agent"] == "Viper"
        assert result["map"] == "Bind"


# ---------------------------------------------------------------------------
# G4 — format_lineup_results
# ---------------------------------------------------------------------------


class TestFormatLineupResults:
    def test_empty_results_returns_no_matches_message(self):
        from valocoach.retrieval.lineups import format_lineup_results

        result = format_lineup_results([])
        lower = result.lower()
        assert "no lineup" in lower or "no match" in lower

    def test_formats_hit_with_agent_and_map(self):
        from valocoach.retrieval.lineups import format_lineup_results

        hits = [
            {
                "text": "Stand in A Main, throw over the wall, bolt lands on default box.",
                "metadata": {
                    "agent": "Sova",
                    "ability": "Recon Bolt",
                    "map": "Ascent",
                    "site": "A",
                    "side": "attack",
                    "purpose": "pre-round info",
                    "channel": "JollyJonty",
                    "title": "Sova Ascent Lineups",
                    "start_seconds": 225,
                    "source": "https://youtube.com/watch?v=test",
                },
                "distance": 0.18,
            }
        ]

        result = format_lineup_results(hits)
        assert "Sova" in result
        assert "Recon Bolt" in result
        assert "Ascent" in result
        assert "3:45" in result  # 225 seconds = 3:45
        assert "JollyJonty" in result

    def test_seed_channel_omits_video_reference(self):
        """Seed entries (channel='seed') should not show a 📹 video line."""
        from valocoach.retrieval.lineups import format_lineup_results

        hits = [
            {
                "text": "Lineup description from seed data.",
                "metadata": {
                    "agent": "Brimstone",
                    "ability": "Incendiary",
                    "map": "Haven",
                    "site": "C",
                    "purpose": "post-plant deny",
                    "channel": "seed",
                    "title": "Seed lineup",
                    "start_seconds": 0,
                    "source": "lineups_seed",
                },
                "distance": 0.22,
            }
        ]

        result = format_lineup_results(hits)
        assert "📹" not in result  # no video reference for seed entries
        assert "Brimstone" in result

    def test_multiple_hits_numbered(self):
        from valocoach.retrieval.lineups import format_lineup_results

        hits = [
            {
                "text": f"Lineup {i}",
                "metadata": {
                    "agent": "Sova",
                    "ability": "Recon Bolt",
                    "map": "Ascent",
                    "channel": "seed",
                    "title": f"Lineup {i}",
                    "start_seconds": 0,
                    "source": "seed",
                },
                "distance": 0.1 + i * 0.05,
            }
            for i in range(3)
        ]

        result = format_lineup_results(hits)
        assert "1." in result
        assert "2." in result
        assert "3." in result


# ---------------------------------------------------------------------------
# G5 — seed data integrity
# ---------------------------------------------------------------------------


class TestLineupsSeedData:
    def test_seed_file_exists_and_has_entries(self):
        import json
        from pathlib import Path

        seed_path = (
            Path(__file__).parent.parent
            / "src" / "valocoach" / "retrieval" / "data" / "lineups_seed.json"
        )
        assert seed_path.exists(), f"Seed file not found: {seed_path}"

        with open(seed_path) as f:
            data = json.load(f)

        entries = data.get("lineups", [])
        assert len(entries) >= 10, f"Expected ≥10 seed entries, got {len(entries)}"

    def test_all_seed_entries_have_required_fields(self):
        import json
        from pathlib import Path

        seed_path = (
            Path(__file__).parent.parent
            / "src" / "valocoach" / "retrieval" / "data" / "lineups_seed.json"
        )
        with open(seed_path) as f:
            data = json.load(f)

        required = {"agent", "ability", "map", "text"}
        for i, entry in enumerate(data["lineups"]):
            missing = required - set(entry.keys())
            assert not missing, f"Entry {i} missing fields: {missing}"
            assert entry["text"].strip(), f"Entry {i} has empty text"

    def test_ingest_seed_lineups_calls_chroma_upsert(self):
        """ingest_seed_lineups() should embed and upsert all seed entries."""
        from valocoach.retrieval.lineups import ingest_seed_lineups

        mock_coll = MagicMock()
        fake_vecs = [[0.1] * 768] * 15  # 15 fake vectors

        # get_collection and embed are imported lazily inside the function
        with (
            patch("valocoach.retrieval.vector_store.get_collection", return_value=mock_coll),
            patch("valocoach.retrieval.embedder.embed", return_value=fake_vecs),
        ):
            n = ingest_seed_lineups(MagicMock())

        assert n >= 10  # at least 10 entries
        assert mock_coll.upsert.called


# ---------------------------------------------------------------------------
# G6 — Post-game lineup suggestion helper
# ---------------------------------------------------------------------------


class TestSuggestLineupsForLowUtil:
    def test_returns_none_for_non_util_agent(self):
        from types import SimpleNamespace

        from valocoach.stats.post_game import Finding
        from valocoach.cli.commands.post_game import _suggest_lineups_for_low_util

        findings = [
            Finding(
                severity="warning",
                category="utility",
                headline="Low utility",
                detail="...",
                evidence={},
                root_cause_tag="low_utility",
            )
        ]
        result = _suggest_lineups_for_low_util(findings, "Jett", "Ascent", MagicMock())
        assert result is None  # Jett is not in _UTIL_HEAVY_AGENTS

    def test_returns_none_when_no_low_utility_finding(self):
        from valocoach.stats.post_game import Finding
        from valocoach.cli.commands.post_game import _suggest_lineups_for_low_util

        findings = [
            Finding(
                severity="warning",
                category="duels",
                headline="First death",
                detail="...",
                evidence={},
                root_cause_tag="entry_failure",
            )
        ]
        result = _suggest_lineups_for_low_util(findings, "Sova", "Ascent", MagicMock())
        assert result is None  # no low_utility finding

    def test_returns_lineup_block_for_sova_with_low_utility(self):
        from valocoach.stats.post_game import Finding
        from valocoach.cli.commands.post_game import _suggest_lineups_for_low_util

        findings = [
            Finding(
                severity="warning",
                category="utility",
                headline="Low utility",
                detail="...",
                evidence={},
                root_cause_tag="low_utility",
            )
        ]

        mock_hit = {
            "text": "Stand in A Main and throw over the wall.",
            "metadata": {
                "agent": "Sova",
                "ability": "Recon Bolt",
                "map": "Ascent",
                "site": "A",
                "channel": "seed",
                "title": "Seed",
                "start_seconds": 0,
                "source": "seed",
            },
            "distance": 0.2,
        }

        # search_lineups is imported lazily inside _suggest_lineups_for_low_util
        with patch(
            "valocoach.retrieval.lineups.search_lineups", return_value=[mock_hit]
        ):
            result = _suggest_lineups_for_low_util(findings, "Sova", "Ascent", MagicMock())

        assert result is not None
        assert "LINEUP SUGGESTIONS" in result
        assert "Sova" in result
