from __future__ import annotations

from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from valocoach.cli import display
from valocoach.core.config import load_settings


def run_ingest(
    url: str | None,
    youtube: str | None,
    corpus: bool,
    seed: bool,
    clear: bool,
    show_stats: bool,
    force: bool = False,
    preview: bool = False,
) -> None:
    settings = load_settings()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if show_stats:
        from valocoach.retrieval.searcher import collection_stats

        s = collection_stats(data_dir)
        with display.command_frame("Knowledge Base"):
            display.console.print(f"Vector store: [stat.value]{s['total']}[/stat.value] document(s)")
            for doc_type, count in sorted(s["by_type"].items()):
                display.console.print(f"  [stat.label]{doc_type}:[/stat.label] {count}")
        return

    if clear:
        from valocoach.retrieval.vector_store import (
            LIVE_COLLECTION,
            STATIC_COLLECTION,
            clear_collection,
        )

        clear_collection(data_dir, STATIC_COLLECTION)
        clear_collection(data_dir, LIVE_COLLECTION)
        display.success("Vector store cleared (static + live collections).")
        return

    nothing_specified = not corpus and url is None and youtube is None

    with display.command_frame("Knowledge Base"):
        if seed or nothing_specified:
            _do_seed(data_dir)

        if corpus:
            _do_corpus(data_dir)

        if url:
            _do_url(data_dir, url)

        if youtube:
            _do_youtube(data_dir, youtube, settings=settings, force=force, preview=preview)


def _do_seed(data_dir: Path) -> None:
    from valocoach.retrieval.ingester import ingest_knowledge_base

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[muted]{task.completed}/{task.total} docs[/muted]"),
        TimeElapsedColumn(),
        console=display.console,
        transient=False,
    ) as progress:
        task = progress.add_task("Embedding knowledge base…", total=None)

        def _on_progress(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        try:
            counts = ingest_knowledge_base(data_dir, on_progress=_on_progress)
        except Exception as e:
            display.error(f"Seed failed: {e}")
            raise typer.Exit(1) from e

    display.success(
        f"Seeded {counts['total']} docs — "
        f"{counts['agents']} agents · {counts['maps']} maps · {counts['meta']} meta"
    )

    # G5 — also seed the built-in lineup entries
    try:
        from valocoach.retrieval.lineups import ingest_seed_lineups

        n_lineups = ingest_seed_lineups(data_dir)
        if n_lineups:
            display.success(f"Seeded {n_lineups} lineup entries (valocoach lineup ready)")
    except Exception as exc:
        display.console.print(f"[muted]Lineup seed skipped: {exc}[/muted]")


def _do_url(data_dir: Path, url: str) -> None:
    from valocoach.retrieval.ingester import ingest_text
    from valocoach.retrieval.scrapers.web import scrape_url

    with display.console.status(f"[info]Scraping {url} …[/info]"):
        content = scrape_url(url, source="patch_note")

    if content is None:
        display.error(f"Could not extract content from {url}")
        raise typer.Exit(1)

    with display.console.status("[info]Embedding and indexing…[/info]"):
        n = ingest_text(
            data_dir,
            content.text,
            doc_type="patch_note",
            name=content.title,
            source=content.url,
            extra_metadata={"fetched_at": content.fetched_at, "ttl_tier": "live"},
        )
    display.success(f"Ingested {n} chunk(s) from URL.")


def _do_corpus(data_dir: Path, corpus_root: Path | None = None) -> None:
    from valocoach.retrieval.ingester import ingest_text

    if corpus_root is None:
        corpus_root = Path(__file__).resolve().parents[4] / "corpus"
    if not corpus_root.exists():
        display.error_with_hint(
            f"Corpus directory not found: {corpus_root}",
            "Run: python scripts/build_corpus.py",
        )
        raise typer.Exit(1)

    md_files = list(corpus_root.rglob("*.md"))
    if not md_files:
        display.warn("No .md files found in corpus/. Run scripts/build_corpus.py first.")
        return

    display.info(f"Ingesting {len(md_files)} corpus file(s) from {corpus_root} …")
    total = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=display.console,
        transient=False,
    ) as progress:
        task = progress.add_task("Ingesting corpus…", total=len(md_files))
        for path in sorted(md_files):
            text = path.read_text(encoding="utf-8")
            folder = path.parent.name.rstrip("s")
            doc_type = folder if folder in ("agent", "map", "meta", "concept") else "web"
            n = ingest_text(
                data_dir,
                text,
                doc_type=doc_type,
                name=path.stem,
                source=str(path),
                extra_metadata={"content_type": path.parent.name, "ttl_tier": "stable"},
            )
            total += n
            progress.advance(task)

    display.success(f"Ingested {total} chunk(s) from {len(md_files)} corpus file(s).")


def _do_youtube(
    data_dir: Path,
    youtube: str,
    *,
    settings: object,
    force: bool = False,
    preview: bool = False,
) -> None:
    """Analyse a YouTube video, show a preview panel, prompt for confirmation, then ingest."""
    from valocoach.retrieval.youtube_ingest import analyze_youtube_video, apply_youtube_ingest_plan

    with display.console.status("[info]Fetching + classifying YouTube transcript…[/info]"):
        try:
            plan = analyze_youtube_video(data_dir, youtube, settings, force=force)
        except Exception as e:
            display.error(f"YouTube analysis failed: {e}")
            raise typer.Exit(1) from e

    _print_youtube_plan(plan)

    if plan.skipped_reason:
        return

    if plan.kept_count == 0:
        return

    if preview:
        display.console.print("[muted]Preview only — nothing written.[/muted]")
        return

    confirmed = typer.confirm(
        f"\nIngest these {plan.kept_count} chunk(s)?",
        default=False,
    )
    if not confirmed:
        display.console.print("[muted]Aborted — nothing written.[/muted]")
        return

    with display.console.status("[info]Embedding and indexing…[/info]"):
        try:
            n = apply_youtube_ingest_plan(plan, data_dir, settings)
        except Exception as e:
            display.error(f"YouTube ingest failed: {e}")
            raise typer.Exit(1) from e

    display.success(
        f"Ingested {n} chunk(s) from [bold]{plan.title}[/bold]"
        f" — {plan.lineup_count} lineup · {plan.youtube_count} regular"
    )


def _print_youtube_plan(plan: object) -> None:
    """Render the YouTube ingest plan as a Rich-formatted summary."""
    from valocoach.retrieval.youtube_ingest import YouTubeIngestPlan

    assert isinstance(plan, YouTubeIngestPlan)
    c = display.console

    if plan.skipped_reason == "invalid_url":
        display.error("Could not extract a valid YouTube video ID from the URL.")
        return

    if plan.skipped_reason == "already_ingested":
        c.print(
            f"[warning]Video already in the knowledge base — use [bold]--force[/bold] to re-ingest.[/warning]"
        )
        return

    if plan.skipped_reason == "ip_blocked":
        c.print("[warning]YouTube is rate-limiting this IP — wait a few minutes and try again.[/warning]")
        c.print("[muted]Tip: avoid running ingest --youtube rapidly in succession.[/muted]")
        return

    if plan.skipped_reason == "no_captions":
        c.print("[warning]This video has captions disabled — no transcript available.[/warning]")
        return

    if plan.skipped_reason == "no_language":
        c.print("[warning]No English transcript found for this video.[/warning]")
        c.print("[muted]Try a video with English (auto-generated or manual) captions.[/muted]")
        return

    if plan.skipped_reason == "no_transcript":
        c.print("[warning]No transcript available for this video.[/warning]")
        c.print("[muted]Possible causes: captions disabled, transcript unavailable in English.[/muted]")
        return

    # ── Video header ──────────────────────────────────────────────────────
    c.print(f"[bold]{plan.title}[/bold]")
    c.print(f"[stat.label]Channel:[/stat.label]  {plan.channel}")
    c.print()

    # ── Chunk counts ─────────────────────────────────────────────────────
    c.print(f"[stat.label]Chunks fetched:[/stat.label]   [stat.value]{plan.fetched_count}[/stat.value]")

    kept_detail = ""
    if plan.lineup_count or plan.youtube_count:
        parts = []
        if plan.lineup_count:
            parts.append(f"lineup: {plan.lineup_count}")
        if plan.youtube_count:
            parts.append(f"regular: {plan.youtube_count}")
        kept_detail = f"  [muted]({' · '.join(parts)})[/muted]"
    c.print(f"[stat.label]Chunks kept:[/stat.label]      [stat.value]{plan.kept_count}[/stat.value]{kept_detail}")

    if plan.dropped_counts:
        total_dropped = sum(plan.dropped_counts.values())
        drop_parts = []
        if plan.dropped_counts.get("off_topic"):
            drop_parts.append(f"off-topic: {plan.dropped_counts['off_topic']}")
        if plan.dropped_counts.get("low_score"):
            drop_parts.append(f"low-relevance: {plan.dropped_counts['low_score']}")
        if plan.dropped_counts.get("unknown"):
            drop_parts.append(f"no-embedding: {plan.dropped_counts['unknown']}")
        c.print(
            f"[stat.label]Dropped:[/stat.label]          [muted]{total_dropped}  ({' · '.join(drop_parts)})[/muted]"
        )

    if plan.kept_count == 0:
        c.print()
        reason_parts = []
        if plan.dropped_counts.get("unknown"):
            reason_parts.append(
                f"{plan.dropped_counts['unknown']} chunk(s) could not be classified — "
                "is Ollama running with nomic-embed-text pulled?"
            )
        if plan.dropped_counts.get("off_topic"):
            reason_parts.append(f"{plan.dropped_counts['off_topic']} chunk(s) were off-topic (intro/outro/sponsor)")
        if plan.dropped_counts.get("low_score"):
            reason_parts.append(f"{plan.dropped_counts['low_score']} chunk(s) scored below the relevance threshold")
        for msg in reason_parts:
            c.print(f"[warning]»[/warning]  {msg}")
        c.print("[muted]Nothing to ingest.[/muted]")
        return

    # ── Lineup candidates ─────────────────────────────────────────────────
    lineup_chunks = [chunk for chunk in plan.candidates if chunk.category == "lineups" and chunk.drop_reason is None]
    if lineup_chunks:
        c.print()
        c.print(f"[val.red]Lineup candidates[/val.red] — {len(lineup_chunks)}")
        c.print("[muted]" + "─" * 44 + "[/muted]")
        for i, chunk in enumerate(lineup_chunks, 1):
            mins, secs = divmod(chunk.start_seconds, 60)
            meta = chunk.lineup_metadata or {}

            id_parts = []
            if meta.get("agent"):
                id_parts.append(meta["agent"])
            if meta.get("ability"):
                id_parts.append(meta["ability"])
            if meta.get("map"):
                id_parts.append(meta["map"])
            if meta.get("site"):
                id_parts.append(f"{meta['site']} site")
            label = " · ".join(id_parts) if id_parts else "[muted]metadata extraction incomplete[/muted]"
            purpose = meta.get("purpose", "")
            purpose_str = f"  [muted][{purpose}][/muted]" if purpose else ""

            snippet = chunk.text[:110].replace("\n", " ").strip()
            if len(chunk.text) > 110:
                snippet += "…"

            c.print(f"  [stat.value]{i}.[/stat.value]  [muted]{mins}:{secs:02d}[/muted]  {label}{purpose_str}")
            c.print(f'       [muted]"{snippet}"[/muted]')
            c.print()
