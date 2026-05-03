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
            # so allow a small buffer (2×overlap) above max_tokens.
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

        n = ingest_text(tmp_path, "Jett dash ability text", doc_type="agent", name="Jett", source="jett")
        assert n >= 1

    def test_collection_count_reflects_ingest(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.vector_store import collection_count

        n = ingest_text(tmp_path, "Jett dash ability text", doc_type="agent", name="Jett", source="jett")
        assert collection_count(tmp_path) == n

    def test_upsert_is_idempotent(self, tmp_path, fake_embed):
        """Ingesting the same source twice must not grow the collection."""
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.vector_store import collection_count

        ingest_text(tmp_path, "Economy concepts text", doc_type="concept", name="eco", source="eco_src")
        first = collection_count(tmp_path)
        ingest_text(tmp_path, "Economy concepts text", doc_type="concept", name="eco", source="eco_src")
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

        ingest_text(tmp_path, "Patch note 9.08 changes", doc_type="patch_note", name="9.08", source="pn_src")
        results = search("patch changes", tmp_path, n_results=5, max_distance=0.5)
        assert all(r["metadata"]["type"] == "patch_note" for r in results)


class TestVectorStoreOperations:
    def test_delete_by_metadata_removes_matching_docs(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.vector_store import collection_count, delete_by_metadata

        ingest_text(tmp_path, "Jett ability text", doc_type="agent", name="Jett", source="jett_src")
        ingest_text(tmp_path, "Ascent callouts text", doc_type="map", name="Ascent", source="ascent_src")

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

        ingest_text(tmp_path, "Jett dash repositions quickly", doc_type="agent", name="Jett", source="jett")
        results = search("how to reposition with Jett", tmp_path, n_results=5, max_distance=0.5)

        assert len(results) > 0
        assert all({"text", "metadata", "distance"} <= r.keys() for r in results)

    def test_doc_type_filter_agent(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        ingest_text(tmp_path, "Jett dash text", doc_type="agent", name="Jett", source="jett_src")
        ingest_text(tmp_path, "Ascent A Long callout", doc_type="map", name="Ascent", source="ascent_src")

        results = search("query", tmp_path, n_results=10, doc_types=["agent"], max_distance=0.5)
        assert len(results) > 0
        assert all(r["metadata"]["type"] == "agent" for r in results)

    def test_doc_type_filter_map(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        ingest_text(tmp_path, "Jett dash text", doc_type="agent", name="Jett", source="jett_src")
        ingest_text(tmp_path, "Ascent A Long callout", doc_type="map", name="Ascent", source="ascent_src")

        results = search("query", tmp_path, n_results=10, doc_types=["map"], max_distance=0.5)
        assert len(results) > 0
        assert all(r["metadata"]["type"] == "map" for r in results)

    def test_doc_type_filter_multiple_types(self, tmp_path, fake_embed):
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.searcher import search

        ingest_text(tmp_path, "patch note text", doc_type="patch_note", name="pn", source="pn_src")
        ingest_text(tmp_path, "concept text", doc_type="concept", name="eco", source="eco_src")
        ingest_text(tmp_path, "agent text", doc_type="agent", name="Sage", source="sage_src")

        results = search("query", tmp_path, n_results=10, doc_types=["patch_note", "concept"], max_distance=0.5)
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
                tmp_path, f"doc text number {i}",
                doc_type="concept", name=f"doc{i}", source=f"src{i}"
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
            s.add(MetaCache(
                url="https://expired.example.com",
                source="web",
                content_hash="deadbeef",
                ttl_tier="volatile",
                fetched_at=past,
                expires_at=past,
                content_text="stale data",
            ))

        # get_cached must not return stale data
        result = await get_cached("https://expired.example.com")
        assert result is None

        # And it should have been deleted (second call still None, not re-raised)
        assert await get_cached("https://expired.example.com") is None

    async def test_invalidate_volatile_only_removes_volatile(self, cache_db):
        from valocoach.retrieval.cache import get_cached, invalidate_volatile, store_cached

        await store_cached("https://volatile.example.com", "live stats", source="web", ttl_tier="volatile")
        await store_cached("https://stable.example.com", "stable facts", source="web", ttl_tier="stable")
        await store_cached("https://semi.example.com", "semi data", source="web", ttl_tier="semi_stable")

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
        await store_cached("https://fresh.example.com", "fresh content", source="web", ttl_tier="semi_stable")

        # Manually insert an already-expired entry
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        async with session_scope() as s:
            s.add(MetaCache(
                url="https://expired2.example.com",
                source="web",
                content_hash="abc123",
                ttl_tier="volatile",
                fetched_at=past,
                expires_at=past,
                content_text="stale",
            ))

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
            ingest_text(tmp_path, f"agent text {i}", doc_type="agent", name=f"agent{i}", source=f"s{i}")

        stats = collection_stats(tmp_path)
        assert stats["by_type"]["[static] agent"] == 3
        assert stats["total"] == 3
