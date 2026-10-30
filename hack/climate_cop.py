#!/usr/bin/env python3
"""Fit and validate the heat-pump electricity (COP) model on real data (PR 2 of the C2 design).

Read-only on the climate feature set and the dataset prices; nothing is written into the
repository, a feature set or a dataset.  Both folds of ``config/climate.yaml`` are run: fit on the
training seasons, validate on the held-out one given the measured heat, against the COP table.

Usage:
    uv run python hack/climate_cop.py \\
        --feature-set <climate feature set id or prefix> \\
        --dataset ~/data/<store>/analysis/v1/<dataset_id> \\
        [--config config/climate.yaml] [--data-dir ~/data/<store>] [--name <suffix>]

The result is written to ``<data_dir>/analysis/v1/climate-cop/<feature set>-<config>[-name]/`` as
``cop.json`` (the versioned parameter records, the validation and the sanity checks) and
``cop.md``.  It holds private figures: keep it in the data dir.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from battery_optimizer.climate.config import load_climate_config  # noqa: E402
from battery_optimizer.climate.cop import fit_and_validate, render_markdown  # noqa: E402
from battery_optimizer.climate.data import load_climate_data  # noqa: E402
from battery_optimizer.features import json_bytes  # noqa: E402
from battery_optimizer.paths import resolve_data_dir, write_part_atomic  # noqa: E402


def _report_name(value: str) -> str:
    """A directory suffix: letters, digits, dot, underscore and dash, never ``..``."""
    if value and (not re.fullmatch(r"[A-Za-z0-9._-]+", value) or ".." in value):
        raise argparse.ArgumentTypeError(f"{value!r}: use only [A-Za-z0-9._-] and no '..'")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--feature-set", required=True, help="climate feature set id (or prefix)")
    parser.add_argument("--dataset", required=True, help="canonical dataset directory (prices)")
    parser.add_argument("--config", default="config/climate.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--name", default="", type=_report_name, help="suffix of the directory")
    args = parser.parse_args(argv)

    root = resolve_data_dir(args.data_dir)
    cfg = load_climate_config(args.config)
    data = load_climate_data(
        cfg, feature_set=args.feature_set, dataset=args.dataset, data_dir=str(root)
    )
    report = fit_and_validate(data, cfg)
    name = f"{data.feature_set_id[:12]}-{cfg.sha256[:12]}" + (f"-{args.name}" if args.name else "")
    out_dir = root / "analysis" / "v1" / "climate-cop" / name
    write_part_atomic(out_dir / "cop.json", json_bytes(report))
    write_part_atomic(out_dir / "cop.md", render_markdown(report).encode("utf-8"))
    print(f"climate_cop: result written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
