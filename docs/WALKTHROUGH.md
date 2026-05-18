# ValoCoach — End-to-End Walkthrough & UX Audit

**Date:** 2026-05-18
**Approach:** Ran every user-facing command in a realistic workflow order, captured what each produces, then synthesized usability findings independent of the static code review. Where this overlaps with [REVIEW.md](REVIEW.md) (architectural / correctness audit), it complements rather than repeats.

---

## TL;DR

The CLI is **functionally excellent** — the deterministic Meta panel, the ability sanitizer, and the agents-refresh wikitext parser all work cleanly. **Stats panels are rich and reliable. Profile, lineup, agents-refresh, patch — all snappy.** The places to spend the next chunk of polish effort are:

1. **One real bug**: `meta` command and `meta-refresh` command disagree about whether a meta refresh is needed. The user gets told "run meta-refresh" but `meta-refresh` then refuses without `--force`.
2. **First-run defaults make the dashboard look empty.** `valocoach stats` defaults to `--period 30d`, which returns nothing for any casual player. The empty-state hint doesn't suggest the obvious fix.
3. **Several small inconsistencies** between commands that should share state (profile vs stats top-agent count, patch-staleness banner appearing on `meta` but not `coach` or `stats`).
4. **No machine-readable output.** Every command renders Rich tables. Useful for humans, but means you can't pipe `stats` into anything.

Everything else is polish — better empty states, better latency feedback during LLM streams, a quickstart command.

---

## Workflow walkthrough

### 1. Discovery surface (`valocoach`, `--help`)

✅ **Branded banner on bare invocation** is the right call — doesn't fall through to Typer's auto-help dump.
✅ **`--help` groups commands** into Coaching / Performance / Data / Game Info. Discoverable.

**Issues**:
* The "Quick start" snippet on bare invocation lists 6 commands but skips `meta-refresh`, `agents-refresh`, and `interactive` — newer users won't find them without `--help`.
* `config` shows up as a stand-alone bucket above the rich panels (Typer quirk when a subcommand isn't grouped). Cosmetic.
* No global `--verbose` / `--debug` / `--no-color` flags. Hard to capture clean output for bug reports.

### 2. `config show`

```
{
    'riot_name': 'Yoursaviour02',
    'riot_tag': 'SBSK',
    ...
    'data_dir': PosixPath('/Users/susil/.valocoach/data')
}
```

❌ **This is `print(dict)`, not a configured renderer.** `PosixPath(...)` literal is leaky — that's an internal Python type, not user-facing. Compare to every other panel in the app, which uses Rich tables.

**Fix sketch**: format as a 2-column table (Key | Value), cast Path values to strings, sort keys, and put it inside the same Rule frame the other commands use.

### 3. `patch` / `patch --check`

✅ Both work. `--check` is fast (one HenrikDev call).
✅ Output is consistent with other commands' framing.

**Note**: HenrikDev returned `12.09` here, but `meta.json` is still pinned to `10.08`. That mismatch is a recurring theme — see finding #1 below.

### 4. `stats` (default → empty)

```
$ valocoach stats
⚠  No matches after filters (period=30d).
```

❌ **Default is `--period 30d`**, which is too narrow. A casual player who comes back monthly sees an empty dashboard on first run.

**Fix sketch**:
* Change default to `90d` or `all` (the data is local — no API cost to scanning everything).
* When the result is empty, **suggest the wider window** in the warning: "Try `--period 365d` or `--period all`."

`valocoach stats --period 365d` produced exactly the panel I'd want — Overall card, Win/Loss split table, Round Mastery, Agent Breakdown, Map Breakdown. Small-sample warnings consistently flagged. **All good once you find the right flag.**

### 5. `profile`

The full profile renders well:

* **Player identity** with rank + ELO
* **Last 18 matches summary**
* **Round Mastery table**
* **Top Agents (3 rows)**
* **Recent Coaching Sessions (5 rows)**

❌ **"Rank Progression" frame is empty** — the heading bar renders but the body is blank. Either the data isn't being loaded or the rendering condition is wrong. Cleaner: hide the section entirely when empty.

❌ **Top Agents truncates to 3 here** but `stats` shows 5. Inconsistent. One source of truth (`compute_per_agent`) used by both; the truncation lives in the formatter. Pick a number (`5` seems right since stats already uses it) and unify.

### 6. `meta` family

`valocoach meta` produces:

```
⚠  New patch detected: 12.09 (meta data still reflects 10.08 — run
   `valocoach meta-refresh` to update)

━━━ Current Meta ━━━
Patch 10.08  ·  updated 2025-04  ...

╭─ Agent Tier List ─╮
│ Tier   Agents     │
│ S      Omen, ...  │
│ A      Cypher, ...│
│ B      Sage, ...  │
│ C      Harbor, ...│
╰───────────────────╯

Full buy: 3900 cr  Half buy: 2400 cr  Eco/save: <1600 cr
```

✅ **Patch-staleness warning is excellent UX** — the user knows the data is old AND what to run.
✅ **Tier list is dense and readable.**

❌ **The patch-staleness warning only fires on `meta`** — but users will more frequently call `coach`, `stats`, `profile`. The same warning should appear wherever the meta data influences output.

❌ **No role labels on the tier list rows** (`Omen, Viper, Killjoy` instead of `Omen (Controller), Viper (Controller), Killjoy (Sentinel)`). The `format_meta_context` helper now does emit roles for LLM prompts (added during the session) — surface that to the user-facing `meta` command too.

`valocoach meta --agent Jett` / `--map Ascent` — both work great. Show full kit + tier + reasoning + map callouts. Detailed and accurate.

### 7. `coach "what's the current meta"` (deterministic path)

✅ Zero LLM. Prints the deterministic tier-list panel (29 agents, all real abilities + role labels + real costs from `agents.json`) and then a `Personalised Takeaway` derived from local stats.
✅ Five agent pool entries match what `stats` showed.
✅ "Strongest map: Split — keep queueing it. Weakest: Pearl — practise or dodge." Actionable.

**Polish**: the takeaway shows `pick — · win —` for placeholder-tier agents (Miks/Raze/Tejo/Waylay/Veto). Visually noisy — could collapse those to a footer "Tier data pending for: Miks, Raze, Tejo, Waylay, Veto (run meta-refresh --force)."

### 8. `coach "<tactical question>"` (LLM-backed)

Asked: *"I'm Sage defending Haven A site, getting hit through A Sewer every round, give me one concrete adjustment"*

✅ **Output quality is genuinely good.** Real Sage abilities (Barrier Orb, Slow Orb, Healing Orb), real Haven callouts (A Sewer, A Long, Mid Doors, A Box), real costs (400 cr for Barrier Orb).
✅ **No fact-check warning panel** appeared — the sanitizer found no fabricated claims. Working as intended.

❌ **Latency: 1:09 on qwen3:14b.** That's the cost of running a 14B model locally. The user sees nothing for ~10 seconds before the stream starts — qwen3 emits internal "thinking" tokens that the panel suppresses, so there's no visible progress indicator until real output begins.

**Fix sketch**:
* Show a `Pulling context… → Streaming response…` two-stage indicator with elapsed seconds.
* If the model supports it, expose `/no_think` directive as a flag so users can opt out of the thinking phase.

❌ **The LLM invented round-timer values** (`0:00–0:05`, `0:10–0:20`). Valorant rounds have a 30s buy phase before any action — these timings don't make sense. The sanitizer doesn't catch this because it only validates ability names. Adding a `_validate_timings` pass with a hard "round actions start at 0:00 = buy-phase-end" rule could surface it.

### 9. `lineup Sova --map Ascent --site A`

✅ Returns 2 hand-curated Sova lineups for Ascent A site. Real callouts (A Main, A Heaven, A Default, A Lobby). Fast (<2s) — RAG retrieval over ChromaDB seed data.

**Polish**: results lack inline images or video timestamps. The seed data has these fields (we saw them during the lineups module review). If the underlying lineup chunk has a `[SOURCE: youtube/...]` tag, surface it as a clickable hyperlink so the user can jump to the moment.

### 10. `notes list` / `sessions list`

✅ Empty-state for notes is good: "No open coaching notes. Add one with: `valocoach notes add <text>`".
✅ Sessions list shows all 18 saved sessions.

❌ **All sessions show `Title: 2026-05-11 / 2026-05-17`** — the auto-generated title is the start date. Two months of sessions all named the same way is hard to navigate. The first user message would make a better default title.

❌ **2 open sessions** even though every test session ended with `/quit`. Either `/quit` isn't closing the session row, or there's an older never-closed session pair. Worth a `sessions close --all` command for cleanup.

### 11. `agents-refresh`

```
»  Fetching roster from Liquipedia (Category:Agents API)
»  Found 29 candidate names on Liquipedia.

✔  Knowledge base is up to date — 29 agents, all present in the meta tier list.
```

✅ **Snappy** (~2s total, one HTTP request). Output is correct and concise.
✅ Cleanly says "already up to date" when nothing's missing.

### 12. `meta-refresh --dry-run`

```
⚠  Dry-run mode: no files will be written.

»  No new patch detected (12.09). Meta is already up to date.  Use --force
   to refresh anyway.
```

❌ **THIS IS THE ONE REAL BUG.** `meta` says "new patch detected, run meta-refresh". `meta-refresh` says "no new patch detected, use --force". They disagree because they compare different pairs:

| Command | Compares | Verdict |
|---|---|---|
| `meta` (display) | `meta.json::patch` ("10.08") vs DB `patch_versions.game_version` ("12.09") | "New patch! Run meta-refresh." |
| `meta-refresh` | Fresh HenrikDev call ("12.09") vs DB `patch_versions.game_version` ("12.09") | "No new patch." |

`meta-refresh` is checking "has Riot bumped the version since we last checked?" when it should be checking "is `meta.json` aligned with the current patch?". The user is correct to be confused.

**Fix sketch**: `meta-refresh` should ALSO compare against `meta.json::patch` and run when those disagree, even if the HenrikDev version matches what we last stored. Or fold both checks into one: "refresh if the meta data is older than the current patch, regardless of where the staleness signal came from."

---

## Cross-cutting observations

### A. Patch-staleness signal is unevenly surfaced

Only `meta` warns about stale meta data. The same condition affects:
* `coach` (LLM responses will be grounded against an old tier list)
* `profile`'s Top Agents tier alignment in mental terms
* The deterministic Personalised Takeaway

Suggest a small `_maybe_warn_stale_meta` (already exists for meta-sensitive intents inside `coach.py`!) wired into the entry of `stats`, `profile`, and `lineup` so the user is told *once* per session that their meta data is N days stale.

### B. No machine-readable output

Everything renders Rich tables. No `--json`, no `--csv`. Means:
* You can't easily script "what's my K/D over the last 100 games?"
* You can't graph rank progression externally.
* CI/automation can't reason about the output.

Adding `--json` to `stats`, `profile`, `meta` would be ~50 LOC each — emit the underlying `PlayerStats` / `dict` as JSON when the flag is set, skip the Rich rendering.

### C. LLM latency masking

Today's user-facing flow when running a tactical `coach` call:
1. Hit enter → silence for ~10s (Ollama loads model, qwen3 thinking phase)
2. Panel header appears, blank body
3. Tokens stream in over ~50s
4. Sanitizer fact-check pass (instant)

A simple cosmetic fix: when the panel is empty but the LLM call is alive, show a spinner or "qwen3:14b — thinking…" hint inside the panel. Today the user can't distinguish "model is working" from "I should Ctrl-C and try again."

### D. State commands could share more

* `profile` shows `Top Agents (3)`; `stats` shows `By Agent (5)`.
* `profile` shows "Recent Coaching Sessions (5)"; `sessions list` shows all 18.
* `profile` summary K/D and `stats` K/D use the same `PlayerStats.kd` but pass through different formatters (`0.93` vs `⚠ 0.93`).

A small `formatter.py` constants block (`TOP_AGENT_LIMIT = 5`, `RECENT_SESSIONS_LIMIT = 5`, `WARN_PREFIX = "⚠ "`) referenced by both commands would eliminate this drift.

---

## Performance / latency budget

Measured on this machine (M-series Mac, qwen3:14b loaded):

| Command | Duration | Notes |
|---|---|---|
| `valocoach` (banner) | <100ms | Pure Python startup |
| `valocoach --help` | <100ms | Same |
| `valocoach config show` | ~200ms | Loads pydantic settings |
| `valocoach patch` | ~300ms | Local DB read |
| `valocoach patch --check` | ~800ms | One HenrikDev call |
| `valocoach stats --period 365d` | ~500ms | Pure local DB + compute |
| `valocoach profile` | ~500ms | Same path |
| `valocoach meta` | ~250ms | JSON load |
| `valocoach meta --agent Jett` | ~250ms | JSON load |
| `valocoach coach "what's the meta"` (deterministic) | ~700ms | JSON + stats compute |
| `valocoach coach "<tactical>"` (qwen3:14b) | **~70s** | LLM + sanitizer |
| `valocoach lineup Sova --map Ascent --site A` | ~1.5s | ChromaDB query + ranking |
| `valocoach agents-refresh` | ~2s | One Liquipedia API call |
| `valocoach meta-refresh --dry-run` | ~3s | Patch check + ahem, no real work |

Everything non-LLM is fast. The LLM path is the only place a user waits.

---

## Recommended action plan, prioritised

### Now (high-value, low-effort)

1. **Fix the `meta-refresh` no-new-patch bug** — compare against `meta.json::patch` not just the stored DB version. ~5 LOC.
2. **Change `stats` default period to `90d`** and improve the empty-state hint. ~3 LOC.
3. **Fix `config show`** to render via Rich table instead of `print(dict)`. ~20 LOC.
4. **Hide empty `Rank Progression` section in `profile`** when there's no data. ~5 LOC.

### Next sprint (medium-effort, polish)

5. **Unify the `profile` ↔ `stats` Top Agents row count** + warn-prefix formatting via shared constants.
6. **Add a session-aware staleness warning** — fire `_maybe_warn_stale_meta` once per CLI invocation, not per command.
7. **Better session titles** — use the first user message as the default title for `sessions list`.
8. **Latency indicator on LLM streams** — Rich spinner / elapsed-time line inside the empty panel.

### Bigger lifts (worth doing once the above land)

9. **`--json` output mode** for `stats`, `profile`, `meta`. Unlocks scripting.
10. **`valocoach quickstart`** — interactive first-run flow that walks through config init → sync → first stats look.
11. **Numeric stats sanitizer** — same pattern as the ability sanitizer, but validates K/D / HS% / ACS / WR claims in LLM output against PLAYER CONTEXT. Catches the rare LLM misquote of a real stat. (This was a recommendation in REVIEW.md too.)
12. **Round-timer sanitizer** — flag LLM-invented timings that contradict Valorant's 30s buy + 1:40 action structure.

---

## What was rock-solid (worth not breaking)

* **The deterministic Meta panel.** Replacing the LLM here remains the single best architectural call in the project.
* **The agents-refresh wikitext parser.** Liquipedia template scraping with no LLM = correct kit data, every time.
* **The ability sanitizer.** Quiet when the LLM behaves, loud when it hallucinates. Real value.
* **Small-sample warnings** in every stats panel. The user is constantly reminded which numbers to trust.
* **Empty-state hints** on `notes list` and `sessions list` — they point at the exact next command.
* **Test coverage.** 1,822 passing tests. Today's regression tests around the deterministic-meta architecture, the wikitext parser, and the sanitizer make the load-bearing decisions impossible to silently undo.
