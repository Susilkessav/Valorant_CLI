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
    patch_diff_extracted: bool = False   # C3 — set when patch changes are extracted

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

        content = fetch_patch_notes(patch_version, settings)
        if content:
            patch_notes_text = content.text
            result.patch_notes_scraped = True
            _step("patch_notes", f"ok ({len(patch_notes_text):,} chars)")
        else:
            _step("patch_notes", "no content returned")
            result.errors.append(
                f"Could not scrape patch notes for {patch_version} — the page may not be live yet."
            )
    except Exception as exc:
        result.errors.append(f"Patch notes scrape error: {exc}")
        _step("patch_notes", f"error: {exc}")

    # ── 2b. C3 — Extract patch changes into {data_dir}/patch_changes/ ────────
    if patch_notes_text and not dry_run:
        try:
            from valocoach.retrieval.patch_diff import extract_patch_changes

            m_clean = re.search(r"(\d+\.\d+)", patch_version)
            clean_for_diff = m_clean.group(1) if m_clean else patch_version
            extract_patch_changes(
                patch_notes_text=patch_notes_text,
                patch_version=clean_for_diff,
                data_dir=settings.data_dir,
                settings=settings,
            )
            result.patch_diff_extracted = True
            _step("patch_diff", "ok")
        except Exception as exc:
            result.errors.append(f"Patch diff extraction error: {exc}")
            log.warning("C3: patch diff extraction failed: %s", exc)

    # ── 3. Stats scrape ────────────────────────────────────────────────────
    _step("stats_scrape")
    stats_text = ""
    try:
        from valocoach.retrieval.scrapers.meta_stats import fetch_all_stats

        stats = fetch_all_stats(settings)
        stats_text = stats.combined
        result.ranked_stats_scraped = bool(stats.ranked_text)
        result.pro_stats_scraped = bool(stats.pro_text)
        _step(
            "stats_scrape",
            f"ranked={stats.ranked_source if result.ranked_stats_scraped else 'fail'}, "
            f"pro={stats.pro_source if result.pro_stats_scraped else 'fail'}",
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

    # ── 4. YouTube transcripts (Phase D pipeline) ─────────────────────────
    if youtube_videos:
        _step("youtube_ingest")
        from valocoach.retrieval.youtube_ingest import ingest_youtube_video

        total_chunks = 0
        for video in youtube_videos:
            try:
                if not dry_run:
                    n = ingest_youtube_video(
                        settings.data_dir,
                        video,
                        settings,
                        force=force,
                        summarize=False,
                    )
                    total_chunks += n
                    log.info("Ingested %d chunk(s) from %s", n, video)
                else:
                    log.info("dry_run: skipping YouTube ingest for %s", video)
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
            # Stamp ``sync_in_progress: true`` BEFORE writing the new tier
            # list so downstream consumers (``get_meta``, the deterministic
            # meta panel) can detect the half-baked state if step 7 fails.
            # The flag is cleared on successful re-ingest at the bottom of
            # this function.  Without this, a re_ingest failure would leave
            # the new meta.json live alongside a stale vector index — the
            # coach prompt would read fresh tier data but pull old RAG
            # chunks for the agent blocks.
            new_meta_data["sync_in_progress"] = True
            with open(_META_FILE, "w") as f:
                json.dump(new_meta_data, f, indent=2)
                f.write("\n")

            # Invalidate the module-level cache so the next call to
            # get_meta() / format_meta_context() reads the fresh file.
            import valocoach.retrieval.meta as _meta_mod

            _meta_mod._cache = None

            result.meta_written = True
            _step("meta_write", "ok")

            # ── C5: update last_verified_patch on every map ────────────────
            try:
                _update_maps_verified_patch(clean_patch)
                _step("maps_verify", f"ok (patch={clean_patch})")
            except Exception as exc:
                log.warning("C5: could not update last_verified_patch: %s", exc)
        else:
            _step("meta_write", "skipped (dry_run)")

    except Exception as exc:
        result.errors.append(f"Meta generation/write error: {exc}")
        _step("meta_generate", f"error: {exc}")
        return result

    # ── 7. Re-ingest knowledge base (only fires when meta.json was written) ──
    if result.meta_written:
        _step("re_ingest")
        try:
            from valocoach.retrieval.ingester import ingest_knowledge_base

            counts = ingest_knowledge_base(settings.data_dir)
            result.meta_ingested = True
            _step("re_ingest", f"ok ({counts['total']} docs)")

            # Re-ingest succeeded — clear the in-progress flag.  We re-read
            # the file we just wrote (rather than trusting in-memory state)
            # so the flag is removed even if some other writer touched the
            # file between our write and this clear.
            try:
                with open(_META_FILE) as f:
                    final_meta = json.load(f)
                final_meta.pop("sync_in_progress", None)
                with open(_META_FILE, "w") as f:
                    json.dump(final_meta, f, indent=2)
                    f.write("\n")
                import valocoach.retrieval.meta as _meta_mod

                _meta_mod._cache = None
            except Exception as exc:
                log.warning("Could not clear sync_in_progress flag: %s", exc)
        except Exception as exc:
            result.errors.append(f"Re-ingest error: {exc}")
            _step("re_ingest", f"error: {exc}")

    return result


# ---------------------------------------------------------------------------
# C5 helper — stamp last_verified_patch on every map entry
# ---------------------------------------------------------------------------

_MAPS_FILE = Path(__file__).parent / "data" / "maps.json"


def _update_maps_verified_patch(patch_version: str) -> None:
    """Write *patch_version* as ``last_verified_patch`` on every map in maps.json.

    Called after a successful ``meta.json`` write so map data is considered
    verified against the current patch.
    """
    with open(_MAPS_FILE) as f:
        data = json.load(f)

    for map_entry in data.get("maps", []):
        map_entry["last_verified_patch"] = patch_version

    with open(_MAPS_FILE, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    log.info(
        "C5: set last_verified_patch=%s on %d map(s)",
        patch_version,
        len(data.get("maps", [])),
    )
