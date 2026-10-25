# Known issues

## HA retention: declared 10 days, effective boundary unverified

`apps/home-assistant/config/configuration.yaml` sets only `recorder.db_url` —
there is no `purge_keep_days`, no `commit_interval`, and no `exclude`/`include`
override, so Home Assistant's documented default of 10 days applies. The
*effective* coverage boundary is unverified and may be shorter; this package
does not probe retention or write synthetic rows to test availability (doing
either would change live Home Assistant configuration, which ADR-0003 forbids
in this phase).

A 2026-10-23 run with `--lookback-days 9` declared 10 chunks, wrote 9 parts
(39 rows), and observed one empty chunk (2026-10-14 → 2026-10-15). With 10-day
retention reaching back to roughly 2026-10-13, that empty chunk is **not**
explained by retention — with only two appliances producing 39 rows in nine
days, a day with no state transition is the likelier reading. An earliest
*observed row* is not a retention boundary; the entities may simply not have
changed state.

The operational conclusion stands regardless of the exact number: enum history
must be extracted recurringly well inside the declared retention because **HA
recorder history is not re-extractable**, unlike the Prometheus archive. The
recurring production schedule is **still not implemented**; today the cadence is
manual, on windows that deliberately overlap. That is only safe because
re-extraction is deduplicated at row level — see
[Overlapping re-extraction is deduplicated at row
level](#overlapping-re-extraction-is-deduplicated-at-row-level).

## Per-chunk emptiness is now captured in `quality_flags`

`extract_ha.py` materialises every declared chunk into a `ChunkOutcome`
*before* fetching, so a zero-row chunk produces flags rather than vanishing
from the manifest. A declared/written mismatch raises
`chunk_accounting_mismatch` with the empty windows named; per-chunk ledger
entries live in `params.chunk_outcomes[]` and summary counts in
`params.chunk_accounting`. Two flag prefixes distinguish the cause:
`empty_chunk:` (payload carried nothing — causes indistinguishable) and
`empty_chunk_after_window_filter:` (payload carried rows, none inside the
half-open window). `coverage_notes` spells out the disagreement in words.

The mismatch is reconciled against `parts_written + fully_suppressed_chunks`,
so a chunk that found nothing new because everything was already stored is not
counted as a gap — only a chunk that yielded no new observation at all is.

## Overlapping re-extraction is deduplicated at row level

Manual extraction on a short recurring cycle overlaps windows by design (10-day
retention leaves no room for gap-free scheduling), and ADR-0003 asserts that
overlapping pulls are "cheap and idempotent to re-pull". That claim was false
until 2026-10-23: two real runs (9 parts / 39 rows, then 3 parts / 12 rows)
stored 51 rows for 39 distinct observations — 12 duplicates, 0 genuinely new
rows — while the second manifest recorded `quality_flags: []`, so it looked like
a clean, fully productive run even though the store had grown by ~31 %.

`extract_ha.py` now compares every candidate row against what is **already
stored** before writing:

- **Identity** is `(entity, last_changed, last_updated, state, attributes_json)`
  — the five `RAW_ENUM_STATE` columns that describe what Home Assistant recorded.
  `extraction_id`, `source_window_start` and `carry_in` are excluded: they describe
  the pull, not the observation. `extraction_id` is a fresh UUID per run, which is
  precisely why the content-addressed no-clobber store could never recognise two
  overlapping parts as duplicates — some of those part files were byte-identical
  except for that one column. Timestamps are compared as whole microseconds since
  the epoch, so a one-microsecond difference is never collapsed.
- **Prior coverage comes from the part files on disk**, not from manifests and not
  from window arithmetic: a row physically present is a row that must not be
  written twice, and a run that died after writing some parts left those parts
  behind. Manifests are scanned only to report provenance — which parts a successful
  run claimed, which are orphans.
- **The requested window is never narrowed.** Chunk boundaries stay a deterministic
  function of `[start, end)` so carry-in anchoring, part naming and part content all
  remain reproducible; dedup happens per row on top of that.
- **Nothing already stored is touched.** Dedup means not writing a duplicate —
  never rewriting, appending to, or deleting an existing part.

Accounting is recorded per chunk in `params.chunk_outcomes[]` and in totals in
`params.row_accounting`: `raw_rows` (what the response carried), `rows_seen`
(candidates surviving carry-in detection, the half-open filter and intra-chunk
dedup), `duplicate_rows_suppressed` (already present, not re-written), and
`new_rows` / `rows_written` (identical, because a new row is always written — a
failing write aborts the run before any manifest exists). `params.prior_coverage`
records what was scanned: parts, rows, distinct keys, pre-existing duplicates,
orphan parts and unreadable parts. `params.row_identity` records the key itself.

When any row is suppressed the manifest raises
`duplicate_rows_suppressed:<window>: seen=… already_present=… new=… rows_written=…`
for that chunk plus a `duplicate_rows_suppressed:total:` entry. **This is an
accounting disclosure, not an alarm:** on an intentional overlap suppression is the
expected outcome, each observation stays stored exactly once, and nothing is lost.
It is deliberately distinct in prefix and wording from the genuine-problem flags
(`empty_chunk:`, `empty_chunk_after_window_filter:`, `chunk_accounting_mismatch`,
`no_rows_for_entity:`, `no_rows_extracted:all`). A chunk whose rows were all already
present is marked `fully_suppressed`: no part file, but not a coverage gap, so it
does not raise `chunk_accounting_mismatch` either — and `no_rows_extracted:all` is
keyed on rows *seen*, so a productive overlap never claims that coverage is missing.

Three further disclosures describe real problems rather than intent:

- `carry_in_classification_disagreement:<window>` — a suppressed duplicate would be
  classified as carry-in differently from the copy already stored, which means the
  two runs' chunk boundaries did not line up. The stored row is kept as-is; check
  the chunk width and window start.
- `duplicate_rows_already_stored:<n>` — the prior parts themselves hold a logical key
  more than once, the residue of pre-dedup overlaps. This run reports the count and
  repairs nothing.
- `prior_part_unreadable:<part name>: <error type>` — a damaged part contributed no
  keys, so rows it holds may be written again. Duplication is the acceptable failure
  mode; losing not-re-extractable history is not. Only the part *name* is recorded,
  never its path.

**Idempotence holds:** a re-run of a window that is already fully stored writes zero
new parts and zero rows, discloses the suppression counts, and raises neither
`no_rows_extracted:all` nor `chunk_accounting_mismatch`. Covered by
`tests/test_extract_ha.py::TestCrossRunDedup`, including the mis-aligned-chunk case
where the tail chunk's window differs from the run that stored its rows and still
duplicates nothing. Idempotence is exact when the chunk width stays the same across
runs: carry-in rows are anchored to the chunk start (unchanged, deliberate behaviour),
so a run with a different `chunk_seconds` also synthesises carry-in rows at new anchors
— genuinely different rows, which are written, not deduplicated. Keep the chunk width
constant and overlapping pulls cost nothing.

**The 12 rows already duplicated in the private data dir** (manifest `1ec3085f…`,
overlapping `750bce0e…`) were not — and will not be — removed by this package. They
were inert because `build.py` has no HA reader at all, and they stay inert for a
consumer that does exist: the read-side `ha-states` reader (see
[§`ha-states`](cli.md#ha-states)) **collapses them at read time** on exactly the identity
described here, rather than rewriting immutable history. Two things follow. First,
the read-side rule was mandatory for any HA reader regardless of when the store
is cleaned, and it is now implemented: dedup on the same row identity, exactly as the Prometheus path collapses
`(series_key, ts, value)`. Second, if a physically clean store is ever wanted, either
revert the whole contributing run (its manifest together with the parts that *only*
that manifest claims, after verifying every one of their keys also exists in another
part, and after a snapshot) or rebuild into a fresh data directory. Never rewrite a
part in place, and never delete a part while its manifest still claims it — `validate`
checks manifest part hashes against the files on disk, so a half-revert breaks
validation instead of fixing it.

## `source_window_start` / `carry_in` flags

Both fields are preserved in the raw Parquet for every HA row. `carry_in=True`
marks rows whose `last_updated` was at or before the chunk start (not a real
transition). Consumers must filter on `carry_in=False` before treating
`last_changed` as a real event time.

Both are provenance about the *query*, not about the observation, which is why
neither is part of the dedup identity above: `source_window_start` is anchored to
the chunk start, and `carry_in` is derived from it. That anchoring is what makes
carry-in rows collapse correctly on re-extraction — the same chunk start produces
the same synthetic row, so an overlapping re-pull suppresses it like any other
duplicate rather than inventing an appliance event per run.
