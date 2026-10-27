"""Generate training-focused EDA and reproducible forecast/error plots from a verified run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _common import ROOT, load_config, load_series
from _provenance import validate_manifest

from trafficml.data.load import load_locations
from trafficml.eval.predictions import available_models, load_predictions, load_targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    cfg = load_config()
    manifest = validate_manifest(args.run_dir, cfg, ROOT / "configs/data.yaml")
    speed, values, ids, splits = load_series(cfg)
    out = args.run_dir / "analysis"
    out.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )
    end = int(splits.train[-1] + cfg["windows"]["horizon"] + 1)
    train = values[:end]

    def save(name):
        plt.tight_layout()
        plt.savefig(out / (name + ".png"))
        plt.close()

    audit = {
        "shape": list(values.shape),
        "start": str(speed.index[0]),
        "end": str(speed.index[-1]),
        "timezone": str(speed.index.tz),
        "sampling_minutes": 5,
        "missing_fraction": float(np.isnan(values).mean()),
        "train_missing_fraction": float(np.isnan(train).mean()),
        "observed_min": float(np.nanmin(values)),
        "observed_max": float(np.nanmax(values)),
        "source_unit": "mph as described by benchmark source; not encoded by this loader",
        "input_channels": ["speed"],
        "training_fit_end": str(speed.index[end - 1]),
        "licensing": "Code license: MIT. Raw-data redistribution terms remain unverified.",
        "test_exposure": "Already inspected; descriptive/exploratory results only.",
    }
    missing = np.isnan(train)
    runs = []
    for i in range(train.shape[1]):
        padded = np.r_[False, missing[:, i], False].astype(int)
        starts = np.flatnonzero(np.diff(padded) == 1)
        stops = np.flatnonzero(np.diff(padded) == -1)
        runs.extend((stops - starts).tolist())
    audit["outage_runs"] = len(runs)
    audit["longest_outage_minutes"] = max(runs, default=0) * 5
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    pd.DataFrame(
        {
            "sensor_id": ids,
            "train_missing_fraction": missing.mean(axis=0),
            "train_mean_speed": np.nanmean(train, axis=0),
        }
    ).to_csv(out / "sensor_audit.csv", index=False)
    plt.figure(figsize=(8, 3))
    plt.hist(train[np.isfinite(train)], bins=60, color="#3069a7")
    plt.xlabel("Speed (mph, benchmark units)")
    plt.ylabel("Training readings")
    save("speed_distribution")
    daily = pd.DataFrame(missing, index=speed.index[:end]).resample("D").mean()
    plt.figure(figsize=(10, 3))
    plt.imshow(daily.T, aspect="auto", cmap="magma", vmin=0, vmax=1)
    plt.colorbar(label="Missing fraction")
    plt.xlabel("Training day index")
    plt.ylabel("Sensor index")
    save("missingness")
    plt.figure(figsize=(8, 3))
    plt.hist(np.array(runs) * 5, bins=np.geomspace(5, max(10, max(runs) * 5), 40), color="#3069a7")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Missing run duration (minutes)")
    plt.ylabel("Number of runs")
    save("outage_runs")
    counts = np.isfinite(train).sum(axis=1)
    aggregate = np.divide(
        np.nansum(train, axis=1), counts, out=np.full(len(train), np.nan), where=counts > 0
    )
    audit["training_all_sensor_missing_steps"] = int((counts == 0).sum())
    (out / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    mean = pd.Series(aggregate, index=speed.index[:end])
    plt.figure(figsize=(8, 3))
    for weekend, label in [(False, "Weekday"), (True, "Weekend")]:
        group = mean[(mean.index.dayofweek >= 5) == weekend]
        profile = group.groupby(group.index.hour * 60 + group.index.minute).mean()
        plt.plot(profile.index / 60, profile.values, label=label)
    plt.xlabel("Recorded clock hour (timezone unresolved)")
    plt.ylabel("Mean speed (mph)")
    plt.legend()
    save("daily_profile")
    plt.figure(figsize=(10, 3))
    day = train[:288].T
    plt.imshow(
        day,
        aspect="auto",
        cmap="viridis",
        vmin=np.nanpercentile(train, 1),
        vmax=np.nanpercentile(train, 99),
    )
    plt.colorbar(label="Speed (mph)")
    plt.xlabel("5-minute step in first training day")
    plt.ylabel("Sensor index")
    save("speed_heatmap")
    locations = load_locations(cfg["paths"]["locations"]).set_index("sensor_id").reindex(ids)
    plt.figure(figsize=(7, 4))
    plt.scatter(
        locations.longitude, locations.latitude, c=np.nanmean(train, axis=0), s=15, cmap="viridis"
    )
    plt.colorbar(label="Training mean speed (mph)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    save("sensor_locations")
    frames = []
    for split in ["val_tune", "val_stack", "test"]:
        metric_file = args.run_dir / "tables" / f"metrics_{split}.csv"
        if metric_file.exists():
            frame = pd.read_csv(metric_file)
            frame["split"] = split
            frames.append(frame)
    metrics = pd.concat(frames)
    metrics.to_csv(out / "all_metrics.csv", index=False)
    for split in metrics.split.unique():
        frame = metrics[(metrics.split == split) & (metrics.slice == "all")]
        plt.figure(figsize=(8, 3))
        for model, part in frame.groupby("model"):
            plt.plot(part.minutes, part.MAE, marker="o", markersize=3, label=model)
        plt.xlabel("Forecast horizon (minutes)")
        plt.ylabel("MAE (mph)")
        plt.legend(ncol=2)
        save("mae_horizons_" + split)
        peak = metrics[
            (metrics.split == split)
            & (metrics.h == 12)
            & metrics["slice"].isin(["peak", "off_peak"])
        ]
        pivot = peak.pivot(index="model", columns="slice", values="MAE")
        ax = pivot.plot.bar(figsize=(8, 3), rot=0)
        ax.set_ylabel("60-minute MAE (mph)")
        ax.set_xlabel("Model")
        save("peak_comparison_" + split)
    # Pick an illustrative sensor by fixed index, never by which model looks best.
    split = "test"
    targets = load_targets(split, args.run_dir / "preds")
    coverage = manifest["coverage_sensor_ids"]
    i = len(coverage) // 2
    preds = {
        m: load_predictions(m, split, args.run_dir / "preds")
        for m in available_models(split, args.run_dir / "preds")
    }
    anchors = targets["anchors"]
    h = 11
    day0 = speed.index[anchors[0] + h + 1].normalize() + pd.Timedelta(days=1)
    times = speed.index[anchors + h + 1]
    selected = (times >= day0) & (times < day0 + pd.Timedelta(days=1))
    plt.figure(figsize=(11, 4))
    plt.plot(
        times[selected],
        targets["y_true"][selected, h, i],
        color="black",
        linewidth=1.5,
        label="Actual",
    )
    for model, pred in preds.items():
        plt.plot(times[selected], pred[selected, h, i], label=model, alpha=0.8)
    plt.ylabel("Speed (mph)")
    plt.xlabel("Target timestamp; 60-minute forecasts")
    plt.legend(ncol=3)
    save("forecast_example")
    errors = []
    for model, pred in preds.items():
        diff = np.abs(pred[:, h, :] - targets["y_true"][:, h, :])
        for j, sensor in enumerate(coverage):
            errors.append(
                {"model": model, "sensor_id": sensor, "mae_60": float(np.nanmean(diff[:, j]))}
            )
    pd.DataFrame(errors).to_csv(out / "per_sensor_mae_60.csv", index=False)
    records = json.loads((args.run_dir / "training_records.json").read_text())
    importance = []
    for kind in ["rf", "xgb"]:
        gains = [r[kind]["feature_importance"] for r in records if r[kind]["feature_importance"]]
        for name, value in zip(manifest["feature_names"], np.mean(gains, axis=0), strict=True):
            importance.append({"model": kind, "feature": name, "importance": float(value)})
    pd.DataFrame(importance).to_csv(out / "feature_importance.csv", index=False)
    print(f"Analysis complete: {out}", flush=True)


if __name__ == "__main__":
    main()
