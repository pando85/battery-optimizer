#!/usr/bin/env python3
"""Fit the prior of the live climate shadow (track B, step B2). Read-only on its inputs.

The shadow's model is B0's ``b2d`` (B1 replay's model): every parameter, the heat law, the valve
cap, the observer gain and the COP model come from one complete heating season, and the service
re-estimates ``ua``/``q0`` weekly on live data. This script fits that season prior exactly as the
B1 replay does (``mpc_replay.fit_prior`` and ``fit_cop``), on the **causal** outdoor series (the
last valid reading held, as the live observer reads it), adds every thermostat room's median
offset from the comfort-room mean (the preregistered room-mean includes office A) and writes it,
write-once, to ``<data_dir>/analysis/v1/climate-shadow/priors/<prior_id>/prior.json``.

Usage:
    uv run python hack/climate_shadow_prior.py \\
        --feature-set <climate feature set id or prefix> --dataset <dataset dir or id> \\
        [--season S3] [--data-dir <store>] [--config config/climate.yaml] \\
        [--mpc-config config/climate-mpc.yaml]

The dataset supplies the PVPC prices of the legacy permission replay, which selects the slots the
heat law is fitted on (as in the B1 replay).

Then set ``climate_shadow.prior: <prior_id>`` in the service config and copy the prior directory
and ``climate/site.yaml`` into the service's data dir. The file holds model parameters only (no
time series); it stays in the private data dir all the same.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from battery_optimizer.climate import mpc_replay, shadow  # noqa: E402
from battery_optimizer.climate.config import load_climate_config  # noqa: E402
from battery_optimizer.climate.data import SITE_FILE, load_climate_data, load_site  # noqa: E402
from battery_optimizer.climate.mpc_config import load_mpc_config  # noqa: E402
from battery_optimizer.manifest import code_version  # noqa: E402
from battery_optimizer.paths import resolve_data_dir  # noqa: E402

log = logging.getLogger("climate_shadow_prior")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--feature-set", required=True, help="climate feature set id (or prefix)")
    parser.add_argument("--dataset", required=True, help="canonical dataset directory or id")
    parser.add_argument("--season", default="S3", choices=("S2", "S3"), help="prior season")
    parser.add_argument("--config", default=str(_PROJECT / "config" / "climate.yaml"))
    parser.add_argument("--mpc-config", default=str(_PROJECT / "config" / "climate-mpc.yaml"))
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ccfg = load_climate_config(args.config)
    cfg = load_mpc_config(args.mpc_config)
    mpc_replay.check_configs(ccfg, cfg)
    if cfg.model.kind != "b2d":
        raise SystemExit("the live shadow runs model.kind b2d only")
    root = resolve_data_dir(args.data_dir)
    site = load_site(root / SITE_FILE)
    log.info("loading climate data")
    d = load_climate_data(
        ccfg, feature_set=args.feature_set, dataset=args.dataset, data_dir=str(root), site=site
    )
    phi = site.phi if site.phi is not None else float(ccfg.model.response.phi_default)
    log.info("fitting the %s prior", args.season)
    prior = shadow.fit_shadow_prior(
        d,
        ccfg,
        cfg,
        args.season,
        phi=phi,
        lineage={"code_version": code_version(), "dataset_id": d.dataset_id},
    )
    path = shadow.save_prior(root, prior)
    print(prior.prior_id)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
