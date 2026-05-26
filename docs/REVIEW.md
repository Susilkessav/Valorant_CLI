# Valorant CLI — Full-Project Code Review

**Date:** 2026-05-17
**Scope:** Full source tree (77 files, ~19,900 LOC) + tests (58 files)
**Methodology:** Five parallel module-level reviews + spot-verification of every HIGH claim by reading the cited code directly. Findings the verification pass disproved are marked *(disproved)* below so they don't bleed into future audits.

---

## TL;DR

The project is in good shape architecturally. The data → retrieval → coach → CLI separation is clean, async lifetimes are mostly correct, the test suite is large (1,798 passing) and exercises real failure paths.

**Two verified HIGH-severity issues:**

1. **Unreachable meta-intent code in `coach.py`** (lines 283–307). When we refactored meta intent to skip the LLM entirely, the older "inject tier-list AGENT blocks + ability roster" branch was orphaned. It runs only on the now-impossible path where intent is `meta` *and* the early `return None` didn't fire. ~50 lines of dead code to delete.
2. **`meta.json::agent_meta` is incomplete** — 19/29 agents have pick/win/reason entries; 10 are missing (Astra, Clove, Deadlock, Gekko, Harbor, Neon, Phoenix, Skye, Vyse, Yoru). The deterministic meta panel renders those agents as bare `name (role)` lines instead of the rich `pick · win · reason` format. Easy fix: backfill the 10 entries; the structural plumbing already handles it.

Everything else is medium-or-low severity. The biggest narrative theme is that **prompt-engineering can't beat training-data priors at qwen3:8b/14b scale** — and our response (deterministic meta panel + post-hoc sanitizer) is the right structural move, but there are still places where we ask the LLM to write content it can't reliably produce.

---

## What's Working Well

* **The deterministic meta panel** (`coach/meta_response.py` + the early-return in `run_coach` for meta intent) is the highest-impact architectural decision in the codebase. Every word printed in `Meta — Current Tier List` comes from JSON files, not an LLM. That's the only reason `meta` answers are usable on a local 14B model.
* **The ability sanitizer** (`coach/sanitizer.py`) is a defence-in-depth complement: even when other intents call the LLM, post-stream validation catches cross-attributions, weapon-as-ability claims, and hallucinated ability names. Two-pass (section-scoped + direct-attribution) coverage is solid.
* **Data-layer async safety.** Event-loop crashes are eliminated by `NullPool` on the aiosqlite engine. `expire_on_commit=False` is correctly set so detached rows survive across async boundaries — `loader.load_player_data` callers can hold rows.
* **HenrikDev mapper** (`data/mapper.py`) has correct null-defaulting for optional v3/v4 fields. The Red/Blue team and attack/defense side constraints are enforced at the ORM level via `CheckConstraint`, not just code.
* **Round-side inference.** `round_analyzer.get_side(round_number, attacker_team)` correctly handles half-time swap and OT alternation. The XOR-style formula at `round_analyzer.py:292` is correct (truth-table verified) — one reviewer flagged it as buggy; that claim is disproved.
* **Test suite breadth.** 1,798 passing tests cover intent classification, side-split correctness on attacker-defender start configs, sanitizer false-positive guards (loadout vs. ability claim), context-budget trimming order, meta-sync partial-failure recording, and post-game analyzer threshold logic. The new sanitizer-tests cover both cross-attribution and pure hallucination cases.
* **`agents-refresh` deterministic kit import.** Building on Liquipedia's structured `{{Infobox agent}}` + `{{AbilityCard}}` wikitext templates and regex-parsing them avoids the LLM entirely — verified to parse correctly for Miks, Tejo, Veto, Waylay. This is the right model: when a structured source exists, use it.

---

## HIGH — Fix soon

### H1. Dead code: `coach.py:283–307` (meta-intent ability-roster injection)

**File:** [src/valocoach/cli/commands/coach.py](../src/valocoach/cli/commands/coach.py)

After the meta-intent early return at line ~263, the block at lines 283–307 still says `if intent == "meta":` and injects tier-list agents into `extra_agents` + builds `ability_roster_prefix`. None of it runs anymore for meta intent (which returns earlier) and the variables it populates are only consumed by the LLM-call path that meta intent skips.

**Fix.** Delete lines 283–307 and the `ability_roster_prefix` variable. The `format_ability_roster` import in `retrieval/__init__.py` becomes unused by `coach.py` but is still imported elsewhere — leave the export.

### H2. `meta.json::agent_meta` covers only 19 of 29 agents

**File:** [src/valocoach/retrieval/data/meta.json](../src/valocoach/retrieval/data/meta.json)

`tier_list` lists all 29 agents (after the auto-stub work), but `agent_meta` only has detailed entries (`pick_rate`, `win_rate`, `reason`) for 19 of them. The 10 missing: Astra, Clove, Deadlock, Gekko, Harbor, Neon, Phoenix, Skye, Vyse, Yoru.

**Impact.** In the deterministic meta panel, agents without an `agent_meta` entry render as just `Name (Role) — ability list`, missing the pick%/win%/reasoning subline that the panel format expects.

**Fix.** Either (a) hand-fill the 10 entries with current Diamond+/VCT data, or (b) make `agent_meta` fully derived from `tier_list` with placeholder values on the missing rows (similar to how `agents-refresh --auto-stub-meta` handles tier-list gaps).

---

## MEDIUM — Worth fixing this sprint

### M1. `format_personalised_takeaway` can crash on Unicode-name agents

**File:** `src/valocoach/coach/meta_response.py`

`_load_agents` is called inside `_canonical_agents`. Match data with NULL or empty `agent_name` rows will pass the existing `list_agent_names()` filter but would crash on `name.casefold()` in adjacent callers — guard with an early-skip for empty strings.

### M2. Lineup metadata canonicalisation asymmetric on the write path

**File:** `src/valocoach/retrieval/lineups.py`

`ingest_lineup_chunk` runs LLM-based metadata extraction (line ~213) but doesn't apply `_canon_agent` / `_canon_map` / `_canon_site` to the extracted fields before upsert. Read path canonicalises in `search_lineups`. If the LLM emits `"sova"` (lowercase) on write, the ChromaDB `$eq` filter on `"Sova"` at read time will miss it.

**Fix.** Apply `_canon_*` to extracted fields immediately after the LLM call, before passing to the upsert.

### M3. Cache: orphan documents without `expires_at_unix`

**File:** `src/valocoach/retrieval/cache.py` + `retrieval/lineups.py` + `retrieval/youtube_ingest.py`

Today's chunks are stamped with `expires_at_unix` on write. Chunks ingested before the TTL change have no field and will never expire via `purge_expired`. Low-frequency drift but worth either (a) one-time cleanup script, or (b) lazy-stamping on read so they roll forward.

### M4. Patch tracker — regional version variants

**File:** `src/valocoach/retrieval/patch_tracker.py`

`_extract_patch_number` regex matches `\d+\.\d{2}` against multiple HenrikDev version fields. Korean/JP/regional variants like `10.08-kr` will match the leading numeric and lose the suffix — fine for our use (we only need patch number), but worth documenting so future maintainers don't add a regional-aware path thinking it's missing.

### M5. Meta sync atomicity

**File:** `src/valocoach/retrieval/meta_sync.py`

If `meta.json` write succeeds (step 6) but `ingest_knowledge_base` re-ingest fails (step 7), the new tier list is live in JSON but the vector store still has the old one. Downstream RAG retrieves stale context. Either wrap 6+7 in a rollback or stamp `meta.json` with a `complete: false` flag that downstream readers gate on.

### M6. Anchor classifier falls open on Ollama failure

**File:** `src/valocoach/retrieval/youtube_ingest.py`

If `get_classifier()` returns degraded `("unknown", 0.0)` results (e.g., embedding service down), the relevance filter accepts all chunks because the score isn't below threshold. Result: noisy LIVE collection on partial outages. Tighten by treating `unknown` as a separate skip path, not a pass-through.

### M7. `repository.upsert_player` race risk

**File:** `src/valocoach/data/repository.py:80–101`

`session.merge(player)` is correct (and `await` IS valid on `AsyncSession.merge` — one reviewer's claim it was sync is wrong, *disproved*). But concurrent sync calls for the same puuid can both read NULL → both INSERT → one IntegrityError on commit. `upsert_match_details` (line ~310) uses a SAVEPOINT pattern that handles this; apply the same pattern here.

### M8. `analyze_first_contact` divides by `rounds_counted` without explicit guard

**File:** `src/valocoach/stats/post_game.py:219`

Line 206 returns `[]` when there are no kills, but in a match where every round has kills yet none qualify as "first contact" (edge case), `rounds_counted` can be 0. Add `if rounds_counted == 0: return []` before the division.

### M9. CLI subcommand: `meta-refresh --youtube` empty-list coercion

**File:** `src/valocoach/cli/app.py` (the `meta_refresh` command def, around line 389)

`list(youtube) or None` is redundant — an empty list is already falsy so coerces to `None`, but the intent is unclear. Pass `youtube or None` directly.

### M10. `_detect_tilt` rolling window may misbehave across UTC midnight on local timezones

**File:** `src/valocoach/coach/context.py:47–99`

`started_at` is stored as ISO UTC. The 8-hour cutoff is computed against `datetime.now(UTC)`. For players in non-UTC regions, "today's tilted session" may not align with their actual play day. Acceptable for now (we mostly care about contiguous runs, not calendar days) but worth a docstring note.

---

## LOW — Polish

### L1. `intent.py:47` adds `post_game` to `IntentType` literal but the docstring's "priority order" list (lines 14–23) only mentions 9 intents.

**Fix.** Update the docstring to mention `post_game` is exclusively set via `force_intent` from the post-game command and isn't part of the classifier's priority chain.

### L2. `formatter.py:230–231` defines an inner `_delta_style` whose two branches are symmetric — both call sites produce the same Rich style. Either remove the function or differentiate the branches.

### L3. `memory.py:113–114` (`ConversationMemory._evict`) recomputes `token_count` on every loop iteration. O(n²) on long conversations. Track a running tally as turns pop.

### L4. `session_store.py:103` validates `"role" in t and "content" in t` but doesn't check the role value. Malformed turns with `role="system"` or empty strings load silently.

### L5. `constants.py:40–41` comment says "first half is rounds 0–11 = 12 rounds". Correct, but rephrase to remove the off-by-one risk for future readers.

### L6. Sanitizer's `_FALSE_POSITIVE_PHRASES` and `_DESCRIPTORS` sets have grown organically — some overlap (e.g. `"smoke"` in both). Consolidate to one source-of-truth set and document the categories.

### L7. `elicitation.py` keeps a hardcoded agent-name tuple at ~line 54. Replace with `list_agent_names()` so new agents added via `agents-refresh --extract-kits` are immediately accepted in elicitation prompts.

---

## Disproved claims (kept here so the next audit doesn't re-raise them)

* **`sanitizer.py:367` — "missing Unicode apostrophe `’`".** Line 367 already reads `"'" in between or "’" in between`. Both ASCII U+0027 and the curly U+2019 are handled. **Not a bug.**
* **`repository.py:93` — "`await session.merge(player)` is syntactically wrong because `merge()` is sync".** In SQLAlchemy 2.0's `AsyncSession`, `merge` IS an async method and IS awaited. The race-condition concern in M7 is real, but the `await` is correct.
* **`round_analyzer.py:292` — "side-assignment formula inverts on odd-OT".** The formula `on_attack = (side == "attack") == (player_team == attacker_team)` is a XOR identity that's correct in all four combinations of (player on attacker team y/n) × (attacker on attack this round y/n). OT correctness is handled inside `get_side`, not here. **Not a bug.**

---

## Test coverage gaps to consider

* No test for the new `Personalised Takeaway` frame in `coach/meta_response.py::format_personalised_takeaway` — would catch regressions when adding new agents or changing stats schema.
* No integration test for `agents-refresh --extract-kits` parsing a real Liquipedia wikitext fixture. Capture one page once and freeze it as a test fixture so future template changes are detected.
* Sanitizer tests cover cross-attribution and pure hallucination but no fixture from a real qwen3 response. Adding a regression fixture from today's outputs would lock in the section-aware coverage.
* No test that the deterministic meta block actually short-circuits the LLM (`stream_completion` should not be called when intent == `"meta"`). Easy to add: mock `stream_completion`, call `run_coach("what's the meta")`, assert `mock_stream.call_count == 0`.

---

## Recommended next steps, ordered

1. **Delete dead block at `coach.py:283–307`** (H1) — 5 min, removes the largest "wait, this still does something?" footgun.
2. **Backfill the 10 missing `agent_meta` entries** in `meta.json` (H2) — 20 min, makes the deterministic meta panel uniform across all agents.
3. **Add the four regression tests** above — 30 min, locks in today's architectural wins.
4. **M2 (lineup canonicalisation on write)** + **M7 (upsert_player race)** — both small, high-value.
5. Everything else is opportunistic.

---

## Architectural observations not tied to specific files

* **Hallucination defence in depth.** The triple "deterministic answer for risky intents + grounding context for the rest + post-hoc sanitizer" is the right design. Each layer compensates for the next's limits.
* **Wikitext-as-source is underused.** Liquipedia's structured templates work well for agent kits. The same approach could replace LLM-based metadata extraction on lineup chunks if a structured source exists (it doesn't yet for lineups, but if Riot exposes lineup data, switch.)
* **`call_llm` vs `stream_completion` split.** Two LLM entry points with similar but not identical kwargs. Both accept `stop` now, but only `stream_completion` has `max_tokens` override surfaced. Worth unifying into a single options dataclass when next touched.
* **The "scrape → LLM-extract → write JSON" pipeline pattern** appears in three places (`meta_sync` for tier list, `lineups` for lineup metadata, `patch_diff` for change extraction). Each handles failures slightly differently. Worth a shared `extract_or_fallback(scraper, validator, fallback)` helper.
