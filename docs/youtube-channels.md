# YouTube Channel Tier List

Curated channels for `valocoach ingest --youtube`. Ordered by information
density and tactical specificity. Ingest Tier-1 first; apply `--summarize`
on Tier-2 and Tier-3 to clean up noisier transcripts.

---

## Tier 1 — High density, structured teaching

Content is segment-structured with clear topic transitions. The 2-minute
time-window chunker captures clean, self-contained lessons. Anchor filter
passes 60–70% of chunks.

| Channel | Focus | Notes |
|---------|-------|-------|
| **Woohoojin** | High-ELO game sense, reads, rotations | Best-in-class structure; every segment teaches a principle |
| **Thinking Man's Valorant** | Tactical theory, team-play, executes | Dense tactical breakdowns; minimal filler |
| **Sero** | Agent-specific guides, ability usage | Per-agent deep dives; high lineup content |

**Recommended ingest:**
```bash
# Woohoojin: ranked coaching, game-sense series
uv run valocoach meta-refresh --youtube dQw4w9WgXcQ   # replace with real IDs

# Thinking Man's Valorant: map-specific breakdowns
# Sero: agent kit deep-dives
```

---

## Tier 2 — Good insights, less structured

Strong tactical content but looser structure — more mid-segment topic shifts,
more stream-style delivery. Anchor filter works harder; expect 40–55% pass rate.
`--summarize` recommended to produce cleaner embeddings.

| Channel | Focus | Notes |
|---------|-------|-------|
| **Sliggy** | Pro team analysis, VCT breakdowns | High intel value; heavy on team-level reads |
| **Curry (Nerd Street)** | Agent tier lists, patch analysis | Good meta coverage; some editorial padding |
| **fl0m** | Ranked coaching, aim mechanics | Practical ranked tips; lower production structure |
| **ProGuides Valorant** | Beginner-to-intermediate guides | Broad coverage; quality varies by presenter |

**Recommended ingest with summarization:**
```bash
uv run valocoach meta-refresh --youtube <video_id> --summarize
```

---

## Tier 3 — Useful but noisy

Useful data buried in gameplay commentary or highlight narration. Expect
50–60% drop rate from the anchor filter. Only ingest specific episodes
(patch reviews, VOD reviews) rather than full channels.

| Channel | Focus | Notes |
|---------|-------|-------|
| Pro POV channels | Kill breakdowns, mechanical highlights | Specific rounds have high value; most is reaction content |
| Ranked gameplay commentary | Situational decision-making | Audio quality varies; transcripts messy |
| Patch reaction streams | Tier-list changes, initial reactions | High relevance in first 48h of a patch; drops quickly |

---

## Skip entirely

Don't ingest these — anchor filter would drop nearly everything and waste
embed credits:

- **Highlights / funny moments** — no tactical content
- **Tier list debates without gameplay** — opinion, no mechanics
- **Watch parties** — audio often unclear; content is reaction not instruction
- **Unboxing / inventory showcases** — zero tactical value
- **Esports event streams** — commentary not coaching; patch context degrades quickly

---

## Lineup-focused channels (Phase G)

These channels are reserved for Phase G's structured lineup extraction.
The YouTube pipeline runs an extra LLM metadata pass on chunks classified
as `lineups` to extract agent/ability/map/site structured data.

| Channel | Agent focus | Coverage |
|---------|-------------|---------|
| **JollyJonty** | Sova | Ascent, Haven, Bind, Split — deep coverage |
| **Unidaro** | Viper | All maps; one of the most complete Viper lineup libraries |
| **Snapiex** | Multi-agent | Broad coverage across controllers and initiators |
| **BananaGaming** | Brimstone, Omen | Smoke lineups and molly setups |
| **Average Jonas** | KAY/O, Breach | Flash and suppression lineups |

---

## Ingestion frequency

| Tier | When to re-ingest |
|------|------------------|
| Tier 1 | After each major patch (when `meta-refresh --force` runs) |
| Tier 2 | Every 2–3 patches, or when a specific meta topic is queried without good hits |
| Tier 3 | Only specific videos, on-demand |
| Lineup channels | Phase G pipeline handles these automatically |
