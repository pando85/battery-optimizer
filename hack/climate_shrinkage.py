#!/usr/bin/env python3
"""Choose the shrinkage of the weekly thermal refits on S2 only (the C2 model iteration).

Read-only on the climate feature set and the dataset prices; nothing is written into the
repository, a feature set or a dataset.  The prior is fitted on the S2 fit-month days up to
``model.adaptive.selection.prior_until``, the weekly refits run over the rest of the S2 gate months
for every grid value, and each is scored by its Q-driven forecast RMSE
(:func:`battery_optimizer.climate.fit.select_shrinkage`).  No S3 day is fitted or scored.  The
chosen values go into ``model.adaptive.shrinkage`` of ``config/climate.yaml`` in their own commit,
before ``climate-fit`` runs on S3.

Usage:
    uv run python hack/climate_shrinkage.py \\
        --feature-set <climate feature set id or prefix> \\
        --dataset ~/data/<store>/analysis/v1/<dataset_id> \\
        [--config config/climate.yaml] [--data-dir ~/data/<store>]

The result is written to ``<data_dir>/analysis/v1/climate-shrinkage/<inputs>-<config>/`` as
``selection.json`` and ``selection.md``.  It holds private figures: keep it in the data dir.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from battery_optimizer.climate.config import load_climate_config  # noqa: E402
from battery_optimizer.climate.data import load_climate_data  # noqa: E402
from battery_optimizer.climate.fit import render_selection, select_shrinkage  # noqa: E402
from battery_optimizer.features import json_bytes  # noqa: E402
from battery_optimizer.manifest import code_version  # noqa: E402
from battery_optimizer.paths import resolve_data_dir, write_part_atomic  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--feature-set", required=True, help="climate feature set id (or prefix)")
    parser.add_argument("--dataset", required=True, help="canonical dataset directory (prices)")
    parser.add_argument("--config", default="config/climate.yaml")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = resolve_data_dir(args.data_dir)
    cfg = load_climate_config(args.config)
    data = load_climate_data(
        cfg, feature_set=args.feature_set, dataset=args.dataset, data_dir=str(root)
    )
    report = select_shrinkage(data, cfg)
    report["lineage"] = {
        "input_sha256": data.input_sha256,
        "config_sha256": cfg.sha256,
        "code_version": code_version(),
    }
    name = f"{data.input_sha256[:12]}-{cfg.sha256[:12]}"
    out_dir = root / "analysis" / "v1" / "climate-shrinkage" / name
    write_part_atomic(out_dir / "selection.json", json_bytes(report))
    write_part_atomic(out_dir / "selection.md", render_selection(report).encode("utf-8"))
    chosen = {k: m["chosen"] for k, m in report["models"].items()}
    print(f"climate_shrinkage: chosen {chosen}; result written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
