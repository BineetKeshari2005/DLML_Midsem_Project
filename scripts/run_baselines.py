"""Verify the existing M0/M1 baselines in a fresh, traceable run directory.

Defaults to validation only. Test scores, if explicitly requested, are exploratory:
the original test results have already been inspected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT, load_config, load_series
from _provenance import create_manifest

from trafficml.data.splits import make_windows
from trafficml.eval.predictions import save_predictions, save_targets
from trafficml.models.baselines import HistoricalAverage, persistence


def fit_training_profile(values, index, splits, horizon):
    """Fit only through the last retained training target, never future validation targets."""
    train_end = int(splits.train[-1] + horizon + 1)
    if train_end - 1 > splits.val_tune[0]:
        raise ValueError("training fit interval crosses the first validation anchor")
    return HistoricalAverage().fit(values[:train_end], index[:train_end])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data.yaml")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/runs/correction-v1")
    parser.add_argument(
        "--splits", nargs="+", choices=["val_tune", "val_stack", "test"], default=["val_tune"]
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    speed, values, ids, splits = load_series(cfg)
    history, horizon = cfg["windows"]["history"], cfg["windows"]["horizon"]
    ha = fit_training_profile(values, speed.index, splits, horizon)
    manifest = create_manifest(args.run_dir, cfg, args.config, speed, splits, args.splits)
    pred_dir = args.run_dir / "preds"
    for split, anchors in splits.items():
        if split not in args.splits:
            continue
        x, y = make_windows(values, anchors, history, horizon)
        save_targets(split, anchors, ids, y, pred_dir, context=manifest)
        ha_pred = ha.predict(anchors, speed.index, horizon)
        save_predictions("M1_ha", split, anchors, ha_pred, pred_dir, sensor_ids=ids)
        save_predictions(
            "M0_persistence",
            split,
            anchors,
            persistence(x, horizon, ha_pred),
            pred_dir,
            sensor_ids=ids,
        )
        print(f"{split}: {len(anchors)} samples saved to {pred_dir}")
    if "test" in args.splits:
        print("CAUTION: test verification is exploratory, not an untouched final holdout.")


if __name__ == "__main__":
    main()
