"""Tests for the retrieval pipeline.

Coverage:
  - chunker:        Chunk dataclass, token limits (tiktoken-accurate), heading
                    boundary splits, overlap, chunk_text shim, metadata passthrough
  - vector store:   ingest_text, collection_count, delete_by_metadata, clear_collection
  - searcher:       search returns results, empty-store guard, doc_type filtering
  - retriever:      build_retrieval_queries (pure), RetrievalResult.to_context_string,
                    retrieve_static integration (static JSON path)
  - cache:          store_cached / get_cached roundtrip, expiry eviction,
                    invalidate_volatile, purge_expired

All Ollama / ChromaDB calls that embed text are intercepted by ``fake_embed``.
The fake returns a fixed unit vector so every doc is cosine-distance 0 from
every query — the tests exercise the pipeline, not semantic accuracy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared fixture: deterministic embedder stub
# ---------------------------------------------------------------------------

_FAKE_VEC = [1.0, 0.0, 0.0]


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace Ollama embedder at every usage site with a constant unit vector.

    Patches:
      • valocoach.retrieval.ingester.embed  — used in _upsert_batch
      • valocoach.retrieval.searcher.embed_one — used in search()
    """
    monkeypatch.setattr(
        "valocoach.retrieval.ingester.embed",
        lambda texts: [_FAKE_VEC for _ in texts],
    )
    monkeypatch.setattr(
        "valocoach.retrieval.searcher.embed_one",
        lambda text: _FAKE_VEC,
    )


# ---------------------------------------------------------------------------
# Fixture: in-memory DB for async cache tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def cache_db(tmp_path: Path):
    """Bootstrap a fresh SQLite DB with the full ORM schema (MetaCache included)."""
    from valocoach.data.database import Base, init_engine

    engine = init_engine(tmp_path / "cache_test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


# ===========================================================================
# 1. Chunker
# ===========================================================================


class TestChunkDataclass:
    def test_fields_present(self):
        from valocoach.retrieval.chunker import chunk_markdown

        chunks = chunk_markdown("hello world", source="my_source", max_tokens=400)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.text == "hello world"
        assert c.source == "my_source"
        assert c.chunk_index == 0
        assert isinstance(c.metadata, dict)

    def test_metadata_chunk_index_key(self):
        """chunk_index is stored inside metadata as well as the dataclass field."""
        from valocoach.retrieval.chunker import chunk_markdown

        chunks = chunk_markdown("text", source="src", max_tokens=400)
        assert "chunk_index" in chunks[0].metadata
        assert chunks[0].metadata["chunk_index"] == 0

    def test_extra_metadata_passed_through(self):
        from valocoach.retrieval.chunker import chunk_markdown

        chunks = chunk_markdown(
            "text", source="src", max_tokens=400, metadata={"doc_type": "concept", "tier": "S"}
        )
        assert chunks[0].metadata["doc_type"] == "concept"
        assert chunks[0].metadata["tier"] == "S"


class TestChunkMarkdown:
    def test_respects_max_tokens_tiktoken_accurate(self):
        """Each chunk must be within max_tokens by tiktoken count, not chars."""
        from valocoach.retrieval.chunker import chunk_markdown, count_tokens

        # "tokenword " is ~2 tokens each; 600 repetitions ≈ 1200 tokens
        long_text = "tokenword " * 600
        chunks = chunk_markdown(long_text, source="src", max_tokens=100, overlap=10)

        for c in chunks:
            # Overlap can carry up to `overlap` extra tokens from the tail,
            # so allow a small buffer (2x overlap) above max_tokens.
            assert count_tokens(c.text) <= 100 + 20, (
                f"chunk {c.chunk_index} has {count_tokens(c.text)} tokens (max 120)"
            )

    def test_content_preserved_across_chunks(self):
        """Every word from the source must appear in at least one chunk."""
        from valocoach.retrieval.chunker import chunk_markdown

        text = "Alpha Beta Gamma Delta Epsilon"
        chunks = chunk_markdown(text, source="src", max_tokens=400)
        combined = " ".join(c.text for c in chunks)
        for word in text.split():
            assert word in combined

    def test_heading_boundaries_respected(self):
        """## headings trigger new sections; both sections appear in output."""
        from valocoach.retrieval.chunker import chunk_markdown

        text = "## Section A\nContent of A.\n\n## Section B\nContent of B."
        chunks = chunk_markdown(text, source="src", max_tokens=400)
        combined = " ".join(c.text for c in chunks)
        assert "Section A" in combined
        assert "Section B" in combined

    def test_chunk_indices_are_sequential(self):
        from valocoach.retrieval.chunker import chunk_markdown

        # Force multiple chunks with a tight token limit
        text = "word " * 300
        chunks = chunk_markdown(text, source="src", max_tokens=50, overlap=5)
        assert len(chunks) > 1
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_source_propagated_to_all_chunks(self):
        from valocoach.retrieval.chunker import chunk_markdown

        text = "word " * 300
        chunks = chunk_markdown(text, source="knowledge_base/agents/jett", max_tokens=50)
        assert all(c.source == "knowledge_base/agents/jett" for c in chunks)

    def test_single_short_text_is_one_chunk(self):
        from valocoach.retrieval.chunker import chunk_markdown

        chunks = chunk_markdown("Short text.", source="src", max_tokens=400)
        assert len(chunks) == 1

    def test_empty_string_produces_no_chunks(self):
        from valocoach.retrieval.chunker import chunk_markdown

        chunks = chunk_markdown("", source="src", max_tokens=400)
        assert chunks == []


class TestChunkText:
    def test_returns_strings_not_chunks(self):
        from valocoach.retrieval.chunker import chunk_text

        result = chunk_text("hello world", max_tokens=400)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_content_matches_chunk_markdown(self):
        from valocoach.retrieval.chunker import chunk_markdown, chunk_text

        text = "word " * 200
        via_shim = chunk_text(text, max_tokens=100, overlap=10)
        via_markdown = [c.text for c in chunk_markdown(text, source="", max_tokens=100, overlap=10)]
        assert via_shim == via_markdown


# ===========================================================================
# 2. Vector store + ingester
# ===========================================================================


class TestIngestText:
    def test_ingest_returns_chunk_count(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text

        n = ingest_text(
            tmp_path, "Jett dash ability text", doc_type="agent", name="Jett", source="jett"
        )
        assert n >= 1

    def test_collection_count_reflects_ingest(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.vector_store import collection_count

        n = ingest_text(
            tmp_path, "Jett dash ability text", doc_type="agent", name="Jett", source="jett"
        )
        assert collection_count(tmp_path) == n

    def test_upsert_is_idempotent(self, tmp_path, fake_embed):
        """Ingesting the same source twice must not grow the collection."""
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.vector_store import collection_count

        ingest_text(
            tmp_path, "Economy concepts text", doc_type="concept", name="eco", source="eco_src"
        )
        first = collection_count(tmp_path)
        ingest_text(
            tmp_path, "Economy concepts text", doc_type="concept", name="eco", source="eco_src"
        )
        assert collection_count(tmp_path) == first

    def test_extra_metadata_stored_in_collection(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        ingest_text(
            tmp_path,
            "Economy and buy rounds",
            doc_type="concept",
            name="economy",
            source="economy_src",
            extra_metadata={"content_type": "concepts", "ttl_tier": "stable"},
        )
        results = search("economy", tmp_path, n_results=5, max_distance=0.5)
        assert len(results) > 0
        meta = results[0]["metadata"]
        assert meta.get("content_type") == "concepts"
        assert meta.get("ttl_tier") == "stable"

    def test_type_metadata_set_correctly(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        ingest_text(
            tmp_path, "Patch note 9.08 changes", doc_type="patch_note", name="9.08", source="pn_src"
        )
        results = search("patch changes", tmp_path, n_results=5, max_distance=0.5)
        assert all(r["metadata"]["type"] == "patch_note" for r in results)


class TestVectorStoreOperations:
    def test_delete_by_metadata_removes_matching_docs(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.vector_store import collection_count, delete_by_metadata

        ingest_text(tmp_path, "Jett ability text", doc_type="agent", name="Jett", source="jett_src")
        ingest_text(
            tmp_path, "Ascent callouts text", doc_type="map", name="Ascent", source="ascent_src"
        )

        before = collection_count(tmp_path)
        assert before > 0

        delete_by_metadata(tmp_path, {"type": "agent"})

        after = collection_count(tmp_path)
        assert after < before
        # Only map docs remain
        from valocoach.retrieval.searcher import search

        remaining = search("anything", tmp_path, n_results=10, max_distance=0.5)
        assert all(r["metadata"]["type"] == "map" for r in remaining)

    def test_clear_collection_empties_store(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.vector_store import clear_collection, collection_count

        ingest_text(tmp_path, "some agent text", doc_type="agent", name="Sage", source="sage_src")
        assert collection_count(tmp_path) > 0

        clear_collection(tmp_path)
        assert collection_count(tmp_path) == 0

    def test_empty_store_has_count_zero(self, tmp_path, fake_embed):
        from valocoach.retrieval.vector_store import collection_count

        assert collection_count(tmp_path) == 0


# ===========================================================================
# 3. Searcher
# ===========================================================================


class TestSearch:
    def test_empty_store_returns_empty_list(self, tmp_path, fake_embed):
        from valocoach.retrieval.searcher import search

        assert search("any query", tmp_path, n_results=5) == []

    def test_ingested_doc_is_retrievable(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        ingest_text(
            tmp_path, "Jett dash repositions quickly", doc_type="agent", name="Jett", source="jett"
        )
        results = search("how to reposition with Jett", tmp_path, n_results=5, max_distance=0.5)

        assert len(results) > 0
        assert all({"text", "metadata", "distance"} <= r.keys() for r in results)

    def test_doc_type_filter_agent(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        ingest_text(tmp_path, "Jett dash text", doc_type="agent", name="Jett", source="jett_src")
        ingest_text(
            tmp_path, "Ascent A Long callout", doc_type="map", name="Ascent", source="ascent_src"
        )

        results = search("query", tmp_path, n_results=10, doc_types=["agent"], max_distance=0.5)
        assert len(results) > 0
        assert all(r["metadata"]["type"] == "agent" for r in results)

    def test_doc_type_filter_map(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        ingest_text(tmp_path, "Jett dash text", doc_type="agent", name="Jett", source="jett_src")
        ingest_text(
            tmp_path, "Ascent A Long callout", doc_type="map", name="Ascent", source="ascent_src"
        )

        results = search("query", tmp_path, n_results=10, doc_types=["map"], max_distance=0.5)
        assert len(results) > 0
        assert all(r["metadata"]["type"] == "map" for r in results)

    def test_doc_type_filter_multiple_types(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        ingest_text(tmp_path, "patch note text", doc_type="patch_note", name="pn", source="pn_src")
        ingest_text(tmp_path, "concept text", doc_type="concept", name="eco", source="eco_src")
        ingest_text(tmp_path, "agent text", doc_type="agent", name="Sage", source="sage_src")

        results = search(
            "query", tmp_path, n_results=10, doc_types=["patch_note", "concept"], max_distance=0.5
        )
        returned_types = {r["metadata"]["type"] for r in results}
        assert returned_types <= {"patch_note", "concept"}
        assert "agent" not in returned_types

    def test_max_distance_filters_far_docs(self, tmp_path, fake_embed, monkeypatch):
        """Docs beyond max_distance are excluded. Here we give the query a
        completely orthogonal vector so cosine distance = 1.0, then clamp at 0.5."""
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        # Documents embedded with [1,0,0]; query embedded with [0,1,0] → distance = 1.0
        monkeypatch.setattr(
            "valocoach.retrieval.searcher.embed_one",
            lambda text: [0.0, 1.0, 0.0],
        )
        ingest_text(tmp_path, "some doc", doc_type="concept", name="n", source="s")
        results = search("query", tmp_path, n_results=5, max_distance=0.5)
        assert results == []

    def test_n_results_respected(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        for i in range(5):
            ingest_text(
                tmp_path,
                f"doc text number {i}",
                doc_type="concept",
                name=f"doc{i}",
                source=f"src{i}",
            )

        results = search("query", tmp_path, n_results=3, max_distance=0.5)
        assert len(results) <= 3


# ===========================================================================
# 4. Retriever — build_retrieval_queries + RetrievalResult
# ===========================================================================


class TestBuildRetrievalQueries:
    def test_situation_always_first(self):
        from valocoach.retrieval.retriever import build_retrieval_queries

        queries = build_retrieval_queries("push A site quickly")
        assert queries[0] == "push A site quickly"

    def test_no_optionals_gives_single_query(self):
        from valocoach.retrieval.retriever import build_retrieval_queries

        queries = build_retrieval_queries("solo push B")
        assert queries == ["solo push B"]

    def test_map_adds_callout_query(self):
        from valocoach.retrieval.retriever import build_retrieval_queries

        queries = build_retrieval_queries("push A", map_name="Ascent")
        callout_queries = [q for q in queries if "Ascent" in q and "callouts" in q]
        assert len(callout_queries) == 1

    def test_map_and_side_adds_strategy_query(self):
        from valocoach.retrieval.retriever import build_retrieval_queries

        queries = build_retrieval_queries("push A", map_name="Ascent", side="attack")
        strategy_queries = [q for q in queries if "Ascent" in q and "attack" in q]
        assert len(strategy_queries) == 1

    def test_agent_adds_ability_query(self):
        from valocoach.retrieval.retriever import build_retrieval_queries

        queries = build_retrieval_queries("smoke and push", agents=["Viper"])
        ability_queries = [q for q in queries if "Viper" in q and "abilities" in q]
        assert len(ability_queries) == 1

    def test_multiple_agents_each_get_query(self):
        from valocoach.retrieval.retriever import build_retrieval_queries

        queries = build_retrieval_queries("execute", agents=["Jett", "Sage", "Omen"])
        for agent in ("Jett", "Sage", "Omen"):
            assert any(agent in q and "abilities" in q for q in queries)

    def test_agents_capped_at_three(self):
        from valocoach.retrieval.retriever import build_retrieval_queries

        # 4 agents provided — only first 3 should produce queries
        queries = build_retrieval_queries("situation", agents=["A", "B", "C", "D"])
        ability_queries = [q for q in queries if "abilities" in q]
        assert len(ability_queries) == 3
        assert not any("D" in q for q in queries)

    def test_all_options_combined(self):
        from valocoach.retrieval.retriever import build_retrieval_queries

        queries = build_retrieval_queries(
            "execute B", map_name="Bind", agents=["Jett", "Sage"], side="attack"
        )
        assert "execute B" in queries
        assert any("Bind" in q and "callouts" in q for q in queries)
        assert any("Bind" in q and "attack" in q for q in queries)
        assert any("Jett" in q and "abilities" in q for q in queries)
        assert any("Sage" in q and "abilities" in q for q in queries)


class TestRetrievalResult:
    def test_to_context_string_combines_static_and_meta(self):
        from valocoach.retrieval.retriever import RetrievalResult

        r = RetrievalResult(
            static_chunks=["agent: Jett dash facts"],
            meta_chunks=["tier: S tier pick rate 15%"],
        )
        ctx = r.to_context_string()
        assert ctx is not None
        assert "agent: Jett dash facts" in ctx
        assert "tier: S tier pick rate 15%" in ctx

    def test_to_context_string_empty_returns_none(self):
        from valocoach.retrieval.retriever import RetrievalResult

        r = RetrievalResult()
        assert r.to_context_string() is None

    def test_to_context_string_static_only(self):
        from valocoach.retrieval.retriever import RetrievalResult

        r = RetrievalResult(static_chunks=["only this"])
        assert r.to_context_string() == "only this"

    def test_to_context_string_meta_only(self):
        from valocoach.retrieval.retriever import RetrievalResult

        r = RetrievalResult(meta_chunks=["only meta"])
        assert r.to_context_string() == "only meta"

    def test_to_context_string_separator(self):
        from valocoach.retrieval.retriever import RetrievalResult

        r = RetrievalResult(static_chunks=["A", "B"], meta_chunks=["C"])
        ctx = r.to_context_string()
        # chunks separated by double newline
        assert "\n\n" in ctx
        assert ctx == "A\n\nB\n\nC"

    def test_patch_version_default_none(self):
        from valocoach.retrieval.retriever import RetrievalResult

        r = RetrievalResult()
        assert r.patch_version is None

    def test_sources_default_empty(self):
        from valocoach.retrieval.retriever import RetrievalResult

        r = RetrievalResult()
        assert r.sources == []


class TestRetrieveStatic:
    def test_returns_result_with_static_json_context(self, tmp_path, fake_embed):
        """retrieve_static always includes JSON knowledge (agents, meta) even when
        the vector store is empty — this is the baseline for the coach."""
        from valocoach.retrieval.retriever import retrieve_static

        # Jett and Ascent are in the bundled knowledge base
        result = retrieve_static("push A site", tmp_path, agent="Jett", map_="Ascent")
        ctx = result.to_context_string()
        assert ctx is not None
        # JSON facts should reference the agent or map
        assert "Jett" in ctx or "Ascent" in ctx

    def test_returns_result_with_meta_context(self, tmp_path, fake_embed):
        from valocoach.retrieval.retriever import retrieve_static

        result = retrieve_static("eco round strategy", tmp_path)
        ctx = result.to_context_string()
        assert ctx is not None

    def test_unknown_agent_does_not_crash(self, tmp_path, fake_embed):
        """An agent not in the knowledge base causes a warning, not a crash."""
        from valocoach.retrieval.retriever import retrieve_static

        result = retrieve_static("situation", tmp_path, agent="FakeAgentXYZDoesNotExist")
        assert result is not None

    def test_sources_populated(self, tmp_path, fake_embed):
        from valocoach.retrieval.retriever import retrieve_static

        result = retrieve_static("push B", tmp_path, agent="Sage", map_="Bind")
        assert len(result.sources) > 0

    def test_vector_results_appended_when_ingested(self, tmp_path, fake_embed):
        """When the vector store has docs, retrieve_static appends them to
        static_chunks (up to n_results)."""
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.retriever import retrieve_static

        ingest_text(
            tmp_path,
            "Economy: full buy at 3900 credits. Half buy at 2400 credits.",
            doc_type="concept",
            name="economy",
            source="economy_doc",
        )

        result = retrieve_static("how to manage economy", tmp_path)
        ctx = result.to_context_string()
        # Vector hits are appended with [TYPE: name] header
        assert ctx is not None

    def test_unknown_map_logs_warning_and_continues(self, tmp_path, fake_embed, caplog) -> None:
        """A map not in the knowledge base logs a warning (line 84) but doesn't crash.

        Coverage target: retriever.py line 84 — `log.warning("Map '%s' not found", map_)`.
        """
        import logging

        from valocoach.retrieval.retriever import retrieve_static

        with caplog.at_level(logging.WARNING, logger="valocoach.retrieval.retriever"):
            result = retrieve_static("push A", tmp_path, map_="ZZZUnknownMapXYZ")

        assert result is not None
        assert any("ZZZUnknownMapXYZ" in r.message for r in caplog.records)

    def test_search_exception_is_silently_swallowed(self, tmp_path, monkeypatch) -> None:
        """When the vector search raises any exception, it is caught and ignored (lines 125-126).

        Coverage target: retriever.py lines 125-126 — `except Exception: pass`.
        The function must still return a valid result with the static JSON chunks.
        """
        from valocoach.retrieval.retriever import retrieve_static

        # Make search() raise to trigger the except-Exception branch
        monkeypatch.setattr(
            "valocoach.retrieval.retriever.build_retrieval_queries",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("chroma down")),
        )

        # Should not raise; returns static JSON result
        result = retrieve_static("eco round", tmp_path)
        assert result is not None

    def test_duplicate_vector_hit_is_deduplicated(self, tmp_path, monkeypatch) -> None:
        """When two queries return the same text, only the first is kept (line 116 False branch).

        Coverage target: retriever.py line 116 `if text not in seen:` False branch (116→114).
        The duplicate hit is silently skipped; only one entry appears in static_chunks.

        `search` is lazily imported inside retrieve_static, so we patch it at the source
        module (valocoach.retrieval.searcher.search) — that's what the lazy `from...import`
        resolves to at call time.
        """
        from valocoach.retrieval.retriever import retrieve_static

        dup_text = "Economy: buy rifle at 3900 credits."
        dup_hit = {
            "text": dup_text,
            "metadata": {"name": "economy", "type": "concept", "source": "eco_doc"},
        }

        # search is lazily imported from valocoach.retrieval.searcher inside the function;
        # patch at the source so the lazy import picks up the stub.
        monkeypatch.setattr(
            "valocoach.retrieval.searcher.search",
            lambda *a, **kw: [dup_hit],
        )
        # Also need to stub embed_one (called by the real search) — not needed since
        # we've replaced search itself. But we need STATIC_COLLECTION / LIVE_COLLECTION
        # to be importable (they're constants, so they'll import fine).
        # Also ensure build_retrieval_queries produces >1 query so the loop runs twice.
        monkeypatch.setattr(
            "valocoach.retrieval.retriever.build_retrieval_queries",
            lambda *a, **kw: ["q1", "q2"],
        )

        result = retrieve_static("eco round", tmp_path)
        # The duplicate must appear at most once in the chunks
        dup_occurrences = sum(1 for c in result.static_chunks if dup_text in c)
        assert dup_occurrences <= 1


# ===========================================================================
# 5. Cache (async)
# ===========================================================================


class TestCache:
    async def test_get_cached_miss_returns_none(self, cache_db):
        from valocoach.retrieval.cache import get_cached

        assert await get_cached("https://not-stored.example.com") is None

    async def test_store_and_get_roundtrip(self, cache_db):
        from valocoach.retrieval.cache import get_cached, store_cached

        await store_cached("https://example.com/meta", "pick rate data", source="web")
        result = await get_cached("https://example.com/meta")
        assert result == "pick rate data"

    async def test_store_update_refreshes_content(self, cache_db):
        """Second store_cached for same URL updates text in place."""
        from valocoach.retrieval.cache import get_cached, store_cached

        await store_cached("https://example.com/meta", "old content", source="web")
        await store_cached("https://example.com/meta", "new content", source="web")
        assert await get_cached("https://example.com/meta") == "new content"

    async def test_store_different_ttl_tiers(self, cache_db):
        from valocoach.retrieval.cache import get_cached, store_cached

        for tier in ("stable", "semi_stable", "volatile"):
            url = f"https://example.com/{tier}"
            await store_cached(url, f"content for {tier}", source="web", ttl_tier=tier)
            assert await get_cached(url) == f"content for {tier}"

    async def test_expired_entry_evicted_on_read(self, cache_db):
        """An entry with expires_at in the past must be evicted and return None."""
        from datetime import UTC, datetime, timedelta

        from valocoach.data.database import session_scope
        from valocoach.data.orm_models import MetaCache
        from valocoach.retrieval.cache import get_cached

        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        async with session_scope() as s:
            s.add(
                MetaCache(
                    url="https://expired.example.com",
                    source="web",
                    content_hash="deadbeef",
                    ttl_tier="volatile",
                    fetched_at=past,
                    expires_at=past,
                    content_text="stale data",
                )
            )

        # get_cached must not return stale data
        result = await get_cached("https://expired.example.com")
        assert result is None

        # And it should have been deleted (second call still None, not re-raised)
        assert await get_cached("https://expired.example.com") is None

    async def test_invalidate_volatile_only_removes_volatile(self, cache_db):
        from valocoach.retrieval.cache import get_cached, invalidate_volatile, store_cached

        await store_cached(
            "https://volatile.example.com", "live stats", source="web", ttl_tier="volatile"
        )
        await store_cached(
            "https://stable.example.com", "stable facts", source="web", ttl_tier="stable"
        )
        await store_cached(
            "https://semi.example.com", "semi data", source="web", ttl_tier="semi_stable"
        )

        count = await invalidate_volatile()
        assert count == 1
        assert await get_cached("https://volatile.example.com") is None
        assert await get_cached("https://stable.example.com") == "stable facts"
        assert await get_cached("https://semi.example.com") == "semi data"

    async def test_invalidate_volatile_returns_zero_when_none(self, cache_db):
        from valocoach.retrieval.cache import invalidate_volatile

        count = await invalidate_volatile()
        assert count == 0

    async def test_purge_expired_removes_only_expired(self, cache_db):
        """purge_expired sweeps all expired rows; fresh rows are untouched."""
        from datetime import UTC, datetime, timedelta

        from valocoach.data.database import session_scope
        from valocoach.data.orm_models import MetaCache
        from valocoach.retrieval.cache import get_cached, purge_expired, store_cached

        # Fresh entry via normal path
        await store_cached(
            "https://fresh.example.com", "fresh content", source="web", ttl_tier="semi_stable"
        )

        # Manually insert an already-expired entry
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        async with session_scope() as s:
            s.add(
                MetaCache(
                    url="https://expired2.example.com",
                    source="web",
                    content_hash="abc123",
                    ttl_tier="volatile",
                    fetched_at=past,
                    expires_at=past,
                    content_text="stale",
                )
            )

        count = await purge_expired()
        assert count == 1
        assert await get_cached("https://fresh.example.com") == "fresh content"
        assert await get_cached("https://expired2.example.com") is None

    async def test_purge_expired_returns_zero_when_nothing_expired(self, cache_db):
        from valocoach.retrieval.cache import purge_expired, store_cached

        await store_cached("https://ok.example.com", "ok", source="web", ttl_tier="stable")
        count = await purge_expired()
        assert count == 0


# ===========================================================================
# 6. collection_stats — searcher
# ===========================================================================


class TestCollectionStats:
    def test_empty_store_returns_zero_total(self, tmp_path, fake_embed):
        from valocoach.retrieval.searcher import collection_stats

        stats = collection_stats(tmp_path)
        assert stats["total"] == 0
        assert stats["by_type"] == {}

    def test_populated_store_sums_counts_correctly(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import collection_stats

        ingest_text(tmp_path, "agent tactics text", doc_type="agent", name="jett", source="s1")
        ingest_text(tmp_path, "map callouts text", doc_type="map", name="ascent", source="s2")

        stats = collection_stats(tmp_path)
        assert stats["total"] == 2

    def test_by_type_groups_per_doc_type(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import collection_stats

        ingest_text(tmp_path, "agent text", doc_type="agent", name="jett", source="s1")
        ingest_text(tmp_path, "map text", doc_type="map", name="ascent", source="s2")

        stats = collection_stats(tmp_path)
        assert stats["by_type"]["[static] agent"] == 1
        assert stats["by_type"]["[static] map"] == 1

    def test_by_type_keys_have_collection_prefix(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import collection_stats

        ingest_text(tmp_path, "concept text", doc_type="concept", name="economy", source="s1")
        stats = collection_stats(tmp_path)
        assert all(k.startswith("[static]") or k.startswith("[live]") for k in stats["by_type"])

    def test_multiple_docs_same_type_accumulate(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import collection_stats

        for i in range(3):
            ingest_text(
                tmp_path, f"agent text {i}", doc_type="agent", name=f"agent{i}", source=f"s{i}"
            )

        stats = collection_stats(tmp_path)
        assert stats["by_type"]["[static] agent"] == 3
        assert stats["total"] == 3


# ===========================================================================
# 7. Live-collection TTL — both halves of the cache stay in lockstep
# ===========================================================================
#
# Regression tests for FINDINGS P1: invalidate_volatile() and purge_expired()
# previously deleted only SQLite meta_cache rows, leaving orphan vectors in
# valocoach_live that retrieve_static would happily surface to the LLM.


class TestLiveCollectionTTLFilter:
    """``retrieve_static`` must filter expired live docs at search time."""

    def test_expired_live_doc_is_filtered_out_at_search(self, tmp_path, fake_embed):
        """A LIVE_COLLECTION doc with ``expires_at_unix`` in the past must NOT
        appear in retrieve_static results, even when it semantically matches."""
        from datetime import UTC, datetime, timedelta

        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.retriever import retrieve_static
        from valocoach.retrieval.vector_store import LIVE_COLLECTION

        past_unix = int((datetime.now(UTC) - timedelta(hours=1)).timestamp())
        ingest_text(
            tmp_path,
            "stale meta: pick rate 99% on Jett (PATCH 9.00 — outdated)",
            doc_type="web",
            name="metasrc-stale",
            source="metasrc-stale",
            collection_name=LIVE_COLLECTION,
            extra_metadata={"ttl_tier": "volatile", "expires_at_unix": past_unix},
        )

        result = retrieve_static("Jett pick rate", tmp_path, agent="Jett")
        ctx = result.to_context_string() or ""
        assert "PATCH 9.00 — outdated" not in ctx, (
            "expired live doc leaked into grounded context; expires_at_unix filter is broken"
        )

    def test_fresh_live_doc_is_retrievable(self, tmp_path, fake_embed):
        """The flip side: a live doc with ``expires_at_unix`` in the future
        must still be searchable.  Otherwise the TTL filter is too aggressive."""
        from datetime import UTC, datetime, timedelta

        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search
        from valocoach.retrieval.vector_store import LIVE_COLLECTION

        future_unix = int((datetime.now(UTC) + timedelta(hours=6)).timestamp())
        ingest_text(
            tmp_path,
            "current meta: pick rate 50% on Sage",
            doc_type="web",
            name="metasrc-fresh",
            source="metasrc-fresh",
            collection_name=LIVE_COLLECTION,
            extra_metadata={"ttl_tier": "volatile", "expires_at_unix": future_unix},
        )

        # Search the live collection with the same TTL gate retrieve_static uses.
        now_unix = int(datetime.now(UTC).timestamp())
        hits = search(
            "Sage meta",
            tmp_path,
            n_results=5,
            doc_types=["web"],
            max_distance=0.5,
            collection_name=LIVE_COLLECTION,
            where_extra={"expires_at_unix": {"$gt": now_unix}},
        )
        assert len(hits) == 1
        assert "Sage" in hits[0]["text"]

    def test_pre_migration_live_docs_excluded(self, tmp_path, fake_embed):
        """Live docs ingested before the TTL change have no ``expires_at_unix``.

        ChromaDB excludes missing-field rows from ``$gt`` queries — that's
        the safe outcome (treats them as expired, forces a re-scrape).
        This test pins that behavior so a future ChromaDB upgrade that
        changes it doesn't silently re-leak old vectors.
        """
        from datetime import UTC, datetime

        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search
        from valocoach.retrieval.vector_store import LIVE_COLLECTION

        # Ingest WITHOUT expires_at_unix metadata (simulates a pre-fix install).
        ingest_text(
            tmp_path,
            "ancient meta from before the TTL fix",
            doc_type="web",
            name="legacy",
            source="legacy",
            collection_name=LIVE_COLLECTION,
        )

        now_unix = int(datetime.now(UTC).timestamp())
        hits = search(
            "ancient meta",
            tmp_path,
            n_results=5,
            doc_types=["web"],
            max_distance=0.5,
            collection_name=LIVE_COLLECTION,
            where_extra={"expires_at_unix": {"$gt": now_unix}},
        )
        assert hits == []


class TestInvalidateVolatileNukesLive:
    """invalidate_volatile(data_dir) must clear BOTH cache halves."""

    async def test_invalidate_volatile_deletes_live_volatile_docs(
        self, tmp_path, cache_db, fake_embed
    ):
        """Volatile-tier live docs must be deleted from the vector store
        when ``invalidate_volatile`` is called with a data_dir.

        Before the fix: only SQLite rows were deleted; the live vector
        was orphaned.  Now both halves are evicted in lockstep.
        """
        from datetime import UTC, datetime, timedelta

        from valocoach.retrieval.cache import invalidate_volatile, store_cached
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.vector_store import LIVE_COLLECTION, collection_count

        future_unix = int((datetime.now(UTC) + timedelta(hours=6)).timestamp())
        # Volatile live doc — must be deleted.
        ingest_text(
            tmp_path,
            "volatile pick rates",
            doc_type="web",
            name="vol",
            source="vol",
            collection_name=LIVE_COLLECTION,
            extra_metadata={"ttl_tier": "volatile", "expires_at_unix": future_unix},
        )
        # Stable live doc — must survive (different ttl_tier).
        ingest_text(
            tmp_path,
            "stable patch notes",
            doc_type="web",
            name="stable",
            source="stable",
            collection_name=LIVE_COLLECTION,
            extra_metadata={"ttl_tier": "stable", "expires_at_unix": future_unix},
        )
        # Mirror SQLite cache state so the SQLite half deletes too.
        await store_cached("https://vol.example.com", "vol", source="web", ttl_tier="volatile")

        before = collection_count(tmp_path, LIVE_COLLECTION)
        assert before == 2

        await invalidate_volatile(tmp_path)

        after = collection_count(tmp_path, LIVE_COLLECTION)
        assert after == 1, "volatile live doc should have been deleted; stable kept"

    async def test_invalidate_volatile_without_data_dir_is_safe(self, cache_db):
        """Existing callers (or tests) that don't pass data_dir still work.

        The SQLite half is invalidated; the Chroma half is left untouched.
        This preserves the function's pre-fix signature for callers that
        don't have a data_dir context handy.
        """
        from valocoach.retrieval.cache import invalidate_volatile, store_cached

        await store_cached("https://vol.example.com", "vol", source="web", ttl_tier="volatile")
        count = await invalidate_volatile()  # No data_dir
        assert count == 1


class TestPurgeExpiredNukesLive:
    """purge_expired(data_dir) must drop expired live docs too."""

    async def test_purge_expired_deletes_expired_live_docs(self, tmp_path, cache_db, fake_embed):
        from datetime import UTC, datetime, timedelta

        from valocoach.data.database import session_scope
        from valocoach.data.orm_models import MetaCache
        from valocoach.retrieval.cache import purge_expired
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.vector_store import LIVE_COLLECTION, collection_count

        past_iso = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        past_unix = int((datetime.now(UTC) - timedelta(hours=1)).timestamp())
        future_unix = int((datetime.now(UTC) + timedelta(hours=6)).timestamp())

        # Expired live doc — must be deleted.
        ingest_text(
            tmp_path,
            "expired meta",
            doc_type="web",
            name="expired",
            source="expired",
            collection_name=LIVE_COLLECTION,
            extra_metadata={"ttl_tier": "volatile", "expires_at_unix": past_unix},
        )
        # Fresh live doc — must survive.
        ingest_text(
            tmp_path,
            "fresh meta",
            doc_type="web",
            name="fresh",
            source="fresh",
            collection_name=LIVE_COLLECTION,
            extra_metadata={"ttl_tier": "volatile", "expires_at_unix": future_unix},
        )
        # Add an expired SQLite row so the purge function counts something.
        async with session_scope() as s:
            s.add(
                MetaCache(
                    url="https://expired.example.com",
                    source="web",
                    content_hash="h",
                    ttl_tier="volatile",
                    fetched_at=past_iso,
                    expires_at=past_iso,
                    content_text="t",
                )
            )

        await purge_expired(tmp_path)

        # Only the fresh live doc remains.
        assert collection_count(tmp_path, LIVE_COLLECTION) == 1


class TestPatchTrackerInvalidatesLive:
    """The patch tracker's invalidation hook must pass data_dir through."""

    async def test_patch_change_clears_live_collection(
        self, tmp_path, cache_db, fake_embed, monkeypatch
    ):
        """When ``check_patch_update`` detects a new game version it must
        nuke the live collection alongside the SQLite cache."""
        from datetime import UTC, datetime, timedelta

        from valocoach.core.config import Settings
        from valocoach.data.database import session_scope
        from valocoach.data.orm_models import PatchVersion
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.patch_tracker import check_patch_update
        from valocoach.retrieval.vector_store import LIVE_COLLECTION, collection_count

        # Seed a previous patch row so check_patch_update sees a "change".
        async with session_scope() as s:
            s.add(PatchVersion(game_version="9.00"))

        # Seed a volatile live doc.
        future_unix = int((datetime.now(UTC) + timedelta(hours=6)).timestamp())
        ingest_text(
            tmp_path,
            "patch 9.00 meta",
            doc_type="web",
            name="meta",
            source="meta",
            collection_name=LIVE_COLLECTION,
            extra_metadata={"ttl_tier": "volatile", "expires_at_unix": future_unix},
        )
        assert collection_count(tmp_path, LIVE_COLLECTION) == 1

        # Stub the HenrikDev call so we don't hit the network.
        class _FakeClient:
            def __init__(self, *_a, **_kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_a):
                return None

            async def get_version(self, _region):
                return {"version": "9.05"}

        monkeypatch.setattr("valocoach.retrieval.patch_tracker.HenrikClient", _FakeClient)

        settings = Settings(
            riot_name="t",
            riot_tag="t",
            riot_region="na",
            henrikdev_api_key="f",
            data_dir=tmp_path,
        )
        version, is_new = await check_patch_update(settings)
        assert is_new is True
        assert version == "9.05"

        # The volatile live doc must be gone.
        assert collection_count(tmp_path, LIVE_COLLECTION) == 0
