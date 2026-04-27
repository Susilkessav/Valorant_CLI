from __future__ import annotations

from pathlib import Path

import typer

from valocoach.cli import display
from valocoach.core.config import load_settings


def run_ingest(
    url: str | None,
    youtube: str | None,
    corpus: bool,
    seed: bool,
    clear: bool,
    show_stats: bool,
) -> None:
    settings = load_settings()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    if show_stats:
        from valocoach.retrieval.searcher import collection_stats

        s = collection_stats(data_dir)
        display.console.print(f"Vector store: [bold]{s['total']}[/bold] document(s)")
        for doc_type, count in sorted(s["by_type"].items()):
            display.console.print(f"  {doc_type}: {count}")
        return

    if clear:
        from valocoach.retrieval.vector_store import clear_collection

        clear_collection(data_dir)
        display.success("Vector store cleared.")
        return

    nothing_specified = not corpus and url is None and youtube is None

    # Default: seed from JSON when nothing else is specified.
    if seed or nothing_specified:
        _do_seed(data_dir)

    if corpus:
        _do_corpus(data_dir)

    if url:
        _do_url(data_dir, url)

    if youtube:
        _do_youtube(data_dir, youtube)


def _do_seed(data_dir) -> None:
    from valocoach.retrieval.ingester import ingest_knowledge_base

    display.info("Seeding vector store from JSON knowledge base…")
    try:
        counts = ingest_knowledge_base(data_dir)
        display.success(
            f"Seeded {counts['total']} docs — "
            f"{counts['agents']} agents · {counts['maps']} maps · {counts['meta']} meta"
        )
    except Exception as e:
        display.error(f"Seed failed: {e}")
        raise typer.Exit(1) from e


def _do_url(data_dir, url: str) -> None:
    from valocoach.retrieval.ingester import ingest_text
    from valocoach.retrieval.scrapers.web import scrape_url

    display.info(f"Scraping {url} …")
    content = scrape_url(url, source="patch_note")
    if content is None:
        display.error(f"Could not extract content from {url}")
        raise typer.Exit(1)
    n = ingest_text(
        data_dir, content.text, doc_type="patch_note",
        name=content.title, source=content.url,
        extra_metadata={"fetched_at": content.fetched_at, "ttl_tier": "live"},
    )
    display.success(f"Ingested {n} chunk(s) from URL.")


def _do_corpus(data_dir) -> None:
    from valocoach.retrieval.ingester import ingest_text

    corpus_root = Path(__file__).resolve().parents[4] / "corpus"
    if not corpus_root.exists():
        display.error(
            f"Corpus directory not found: {corpus_root}\n"
            "  Run  python scripts/build_corpus.py  first."
        )
        raise typer.Exit(1)

    md_files = list(corpus_root.rglob("*.md"))
    if not md_files:
        display.warn("No .md files found in corpus/. Run scripts/build_corpus.py first.")
        return

    display.info(f"Ingesting {len(md_files)} corpus file(s) from {corpus_root} …")
    total = 0
    for path in sorted(md_files):
        text = path.read_text(encoding="utf-8")
        # Derive doc_type from parent folder name (agents → agent, maps → map, etc.)
        folder = path.parent.name.rstrip("s")  # agents→agent, maps→map, concepts→concept
        doc_type = folder if folder in ("agent", "map", "meta", "concept") else "web"
        n = ingest_text(
            data_dir, text, doc_type=doc_type, name=path.stem, source=str(path),
            extra_metadata={"content_type": path.parent.name, "ttl_tier": "stable"},
        )
        total += n

    display.success(f"Ingested {total} chunk(s) from {len(md_files)} corpus file(s).")


def _do_youtube(data_dir, youtube: str) -> None:
    from valocoach.retrieval.ingester import ingest_text
    from valocoach.retrieval.scrapers.youtube import fetch_transcript

    display.info(f"Fetching YouTube transcript: {youtube} …")
    content = fetch_transcript(youtube)
    if content is None:
        display.error(f"Could not fetch transcript for {youtube}")
        raise typer.Exit(1)
    try:
        n = ingest_text(
            data_dir, content.text, doc_type="youtube",
            name=content.title, source=content.url,
            extra_metadata={"fetched_at": content.fetched_at, "ttl_tier": "live"},
        )
        display.success(f"Ingested {n} chunk(s) from YouTube video.")
    except Exception as e:
        display.error(f"YouTube ingest failed: {e}")
        raise typer.Exit(1) from e
