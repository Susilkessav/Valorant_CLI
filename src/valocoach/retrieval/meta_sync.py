"""Meta auto-update pipeline orchestrator.

Full flow triggered on every new patch (or manually via ``--force``):

  1. Patch detection   — HenrikDev API → record in DB, evict volatile cache
  2. Patch notes scrape — playvalorant.com (URL auto-constructed from version)
  3. Stats scrape       — tracker.gg (Diamond+ ranked) + vlr.gg (pro/VCT)
  4. YouTube ingest     — optional list of video IDs / URLs
  5. LLM tier regen     — full tier list regenerated from scraped data
  6. Write meta.json    — validated JSON written to disk
  7. Re-ingest KB       — meta + agents + maps re-embedded into ChromaDB

Each step is attempted independently — a failure in scraping or LLM
generation is recorded in :attr:`SyncResult.errors` but never stops
subsequent steps unless the data they depend on is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from valocoach.core.config import Settings

log = logging.getLogger(__name__)

_META_FILE = Path(__file__).parent / "data" / "meta.json"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Outcome of a single :func:`run_meta_sync` call."""

    patch_version: str
    is_new_patch: bool

    patch_notes_scraped: bool = False
    ranked_stats_scraped: bool = False
    pro_stats_scraped: bool = False
    youtube_chunks_ingested: int = 0
    meta_regenerated: bool = False
    meta_written: bool = False
    meta_ingested: bool = False

    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"Patch:          {self.patch_version}"
            + (" (new)" if self.is_new_patch else " (unchanged)"),
            f"Patch notes:    {'✓' if self.patch_notes_scraped else '✗'}",
            f"Ranked stats:   {'✓' if self.ranked_stats_scraped else '✗'}",
            f"Pro stats:      {'✓' if self.pro_stats_scraped else '✗'}",
            f"YouTube chunks: {self.youtube_chunks_ingested}",
            f"Meta regen:     {'✓' if self.meta_regenerated else '✗'}",
            f"meta.json:      {'written' if self.meta_written else 'not written'}",
            f"Re-ingested:    {'✓' if self.meta_ingested else '✗'}",
        ]
        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  • {e}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_meta_sync(
    settings: Settings,
    *,
    force: bool = False,
    dry_run: bool = False,
    youtube_videos: list[str] | None = None,
    on_step: object = None,
) -> SyncResult:
    """Run the full meta sync pipeline.

    Args:
        settings:       Application settings.
        force:          Run all steps even when no new patch is detected.
        dry_run:        Execute every step but skip writing ``meta.json``
                        and re-ingesting — useful for previewing changes.
        youtube_videos: List of YouTube video IDs or full URLs to ingest
                        as supplemental meta context.
        on_step:        Optional ``callable(step: str, status: str)``
                        invoked before and after each step for progress
                        reporting.  The CLI passes a Rich status updater.

    Returns:
        :class:`SyncResult` describing what happened in each step.
    """

    def _step(name: str, status: str = "start") -> None:
        log.info("[meta_sync] step=%s status=%s", name, status)
        if callable(on_step):
            on_step(name, status)

    # ── 1. Patch detection ─────────────────────────────────────────────────
    _step("patch_check")
    from valocoach.retrieval.patch_tracker import check_patch_update

    try:
        patch_version, is_new_patch = await check_patch_update(settings)
    except Exception as exc:
        log.error("Patch check failed: %s", exc)
        return SyncResult(
            patch_version="unknown",
            is_new_patch=False,
            errors=[f"Patch check failed: {exc}"],
        )

    result = SyncResult(patch_version=patch_version, is_new_patch=is_new_patch)

    if not is_new_patch and not force:
        _step("patch_check", "no_new_patch — exiting early")
        return result

    _step(
        "patch_check",
        f"new_patch={patch_version}" if is_new_patch else f"forced (patch={patch_version})",
    )

    # ── 2. Patch notes scrape ──────────────────────────────────────────────
    _step("patch_notes")
    patch_notes_text = ""
    try:
        from valocoach.retrieval.scrapers.patch_notes import fetch_patch_notes

        content = fetch_patch_notes(patch_version)
        if content:
            patch_notes_text = content.text
            result.patch_notes_scraped = True
            _step("patch_notes", f"ok ({len(patch_notes_text):,} chars)")
        else:
            _step("patch_notes", "no content returned")
            result.errors.append(
                f"Could not scrape patch notes for {patch_version} — "
                "the page may not be live yet."
            )
    except Exception as exc:
        result.errors.append(f"Patch notes scrape error: {exc}")
        _step("patch_notes", f"error: {exc}")

    # ── 3. Stats scrape ────────────────────────────────────────────────────
    _step("stats_scrape")
    stats_text = ""
    try:
        from valocoach.retrieval.scrapers.meta_stats import fetch_all_stats

        stats = fetch_all_stats()
        stats_text = stats.combined
        result.ranked_stats_scraped = bool(stats.ranked_text)
        result.pro_stats_scraped = bool(stats.pro_text)
        _step(
            "stats_scrape",
            f"ranked={'ok' if result.ranked_stats_scraped else 'fail'}, "
            f"pro={'ok' if result.pro_stats_scraped else 'fail'}",
        )

        # Cache and ingest the combined stats so the coach can also
        # reference live pick-rate data in coaching responses.
        if stats_text and not dry_run:
            from valocoach.retrieval.cache import store_cached
            from valocoach.retrieval.ingester import ingest_text
            from valocoach.retrieval.vector_store import LIVE_COLLECTION

            await store_cached(
                "meta_stats:combined",
                stats_text,
                source="meta_stats",
                ttl_tier="volatile",
            )
            ingest_text(
                settings.data_dir,
                stats_text,
                doc_type="web",
                name="Agent Stats (Ranked + Pro)",
                source="meta_stats:combined",
                collection_name=LIVE_COLLECTION,
            )
    except Exception as exc:
        result.errors.append(f"Stats scrape error: {exc}")
        _step("stats_scrape", f"error: {exc}")

    # ── 4. YouTube transcripts ─────────────────────────────────────────────
    if youtube_videos:
        _step("youtube_ingest")
        from valocoach.retrieval.ingester import ingest_text
        from valocoach.retrieval.scrapers.youtube import fetch_transcript
        from valocoach.retrieval.vector_store import LIVE_COLLECTION

        total_chunks = 0
        for video in youtube_videos:
            try:
                transcript = fetch_transcript(video)
                if transcript and not dry_run:
                    n = ingest_text(
                        settings.data_dir,
                        transcript.text,
                        doc_type="youtube",
                        name=transcript.title,
                        source=transcript.url,
                        collection_name=LIVE_COLLECTION,
                    )
                    total_chunks += n
                    log.info("Ingested %d chunk(s) from %s", n, video)
            except Exception as exc:
                result.errors.append(f"YouTube ingest error ({video}): {exc}")

        result.youtube_chunks_ingested = total_chunks
        _step("youtube_ingest", f"ok ({total_chunks} chunk(s))")

    # ── 5. LLM tier regeneration ───────────────────────────────────────────
    _step("meta_generate")

    if not patch_notes_text and not stats_text:
        result.errors.append(
            "No source data for LLM — skipping meta regeneration. "
            "Both patch notes and stats scrapes failed."
        )
        _step("meta_generate", "skipped — no source data")
        return result

    try:
        with open(_META_FILE) as f:
            existing_meta = json.load(f)

        from valocoach.retrieval.meta_generator import generate_meta_update

        # Extract a clean "X.YY" version string for the patch field.
        m = re.search(r"(\d+\.\d+)", patch_version)
        clean_patch = m.group(1) if m else patch_version

        new_meta_data = generate_meta_update(
            settings=settings,
            patch_version=clean_patch,
            patch_notes_text=patch_notes_text,
            stats_text=stats_text,
            existing_meta=existing_meta,
        )

        if new_meta_data is None:
            result.errors.append("LLM returned no valid JSON for meta update.")
            _step("meta_generate", "failed — LLM returned no JSON")
            return result

        # Stamp patch version and update timestamp.
        new_meta_data["patch"] = clean_patch
        new_meta_data["updated"] = datetime.now(UTC).strftime("%Y-%m")
        new_meta_data.setdefault(
            "notes",
            "Tier list reflects Diamond+ ranked play and pro/VCT data. "
            "Auto-generated by valocoach meta-refresh.",
        )
        # Preserve economy thresholds (LLM doesn't update these).
        new_meta_data.setdefault("economy", existing_meta.get("economy", {}))

        result.meta_regenerated = True
        _step("meta_generate", f"ok (patch={clean_patch})")

        # ── 6. Write meta.json ─────────────────────────────────────────────
        _step("meta_write")
        if not dry_run:
            with open(_META_FILE, "w") as f:
                json.dump(new_meta_data, f, indent=2)
                f.write("\n")

            # Invalidate the module-level cache so the next call to
            # get_meta() / format_meta_context() reads the fresh file.
            import valocoach.retrieval.meta as _meta_mod
            _meta_mod._cache = None

            result.meta_written = True
            _step("meta_write", "ok")
        else:
            _step("meta_write", "skipped (dry_run)")

    except Exception as exc:
        result.errors.append(f"Meta generation/write error: {exc}")
        _step("meta_generate", f"error: {exc}")
        return result

    # ── 7. Re-ingest knowledge base ────────────────────────────────────────
    if result.meta_written:
        _step("re_ingest")
        try:
            from valocoach.retrieval.ingester import ingest_knowledge_base

            counts = ingest_knowledge_base(settings.data_dir)
            result.meta_ingested = True
            _step("re_ingest", f"ok ({counts['total']} docs)")
        except Exception as exc:
            result.errors.append(f"Re-ingest error: {exc}")
            _step("re_ingest", f"error: {exc}")

    return result
