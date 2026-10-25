"""Which manifest quality flags mean the extracted data has a coverage gap.

``extract`` writes its manifest first and *then* judges it, so a run with gaps keeps its data and
stays inspectable while a scheduler (a Kubernetes CronJob) still sees a non-zero exit.

Only flags that mean **data is missing or does not reconcile** are gaps.  Accounting disclosures
(suppressed duplicates, dropped non-finite samples, conflicting samples, carry-in disagreements,
unreadable prior parts, dead-duplicate entities) describe how the run went without saying that
data was lost, and stay informational.  The prefixes are matched at a ``:`` boundary, so
``empty_chunk`` never swallows ``empty_chunk_after_window_filter``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from battery_optimizer.extract_omie import OMIE_GAP_FLAG_PREFIXES
from battery_optimizer.extract_prom import DEAD_DUPLICATE_ENTITY_IDS

#: Process exit code of ``extract`` when a run recorded coverage gaps.  ``1`` is a failed run
#: (no manifest), ``2`` is argparse's usage error, so a coverage gap gets its own value.
EXIT_COVERAGE_GAPS = 3

#: Flags (prefix before the first ``:``) meaning declared coverage is missing or inconsistent.
COVERAGE_GAP_PREFIXES: frozenset[str] = frozenset(
    {
        # HA history (not re-extractable once the recorder purges it)
        "empty_chunk",
        "empty_chunk_after_window_filter",
        "chunk_accounting_mismatch",
        "no_rows_for_entity",
        "no_rows_extracted",
        # Prometheus
        "no_samples_extracted",
        "no_series_chunks",
        "series_without_samples_chunks",
        # OMIE: a day that is missing after its expected publication, a file that does not
        # parse, a download that kept failing, a stored part that cannot be read
        *OMIE_GAP_FLAG_PREFIXES,
    }
)

#: Prefixes whose flag is ``<prefix>:<entity>...``; a known-dead entity is expected to be empty.
_ENTITY_SCOPED = frozenset(
    {
        "no_rows_for_entity",
        "no_samples_extracted",
        "no_series_chunks",
        "series_without_samples_chunks",
    }
)


def coverage_gap_flags(flags: Iterable[str]) -> list[str]:
    """Return the flags that are coverage gaps, in input order."""
    gaps: list[str] = []
    for flag in flags:
        prefix, _, rest = flag.partition(":")
        if prefix not in COVERAGE_GAP_PREFIXES:
            continue
        if prefix in _ENTITY_SCOPED and rest.split(":", 1)[0] in DEAD_DUPLICATE_ENTITY_IDS:
            continue
        gaps.append(flag)
    return gaps


def manifest_coverage_gaps(manifest_path: str | Path) -> list[str]:
    """Coverage-gap flags recorded in the manifest file at *manifest_path*."""
    data = json.loads(Path(manifest_path).read_text())
    return coverage_gap_flags(data.get("quality_flags", []))


def format_gap_summary(source: str, manifest_path: str | Path, gaps: list[str]) -> str:
    """Multi-line stderr summary naming each gap flag (chunk windows included)."""
    lines = [
        f"extract: {source} recorded {len(gaps)} coverage gap flag(s); data and manifest are "
        f"kept ({manifest_path})"
    ]
    lines.extend(f"  - {_one_line(flag)}" for flag in gaps)
    return "\n".join(lines)


def _one_line(flag: str) -> str:
    """Flags can carry a wrapped explanation; collapse whitespace, keep every window named."""
    return " ".join(flag.split())
