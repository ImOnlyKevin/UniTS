#!/usr/bin/env python3
"""
export_telemetry.py — Export satellite telemetry + anomaly predictions to CSV

Output format (wide): one row per timestamp, one column per channel, plus
anomaly_score, is_anomaly_predicted, is_anomaly_ground_truth columns.

    timestamp            | channel_1 | channel_2 | ... | anomaly_score | is_anomaly_predicted | is_anomaly_ground_truth
    2002-10-01 00:00:00  | 0.7412    | 0.0000    | ... | 0.0523        | 0                    | 0

Usage:
    python scripts/export_telemetry.py --mission ESA-Mission2

    # Limit to a time window
    python scripts/export_telemetry.py --mission ESA-Mission2 \\
        --start 2003-01-01 --end 2003-02-01

    # Only export anomalous windows (predicted=1) plus N minutes of context
    python scripts/export_telemetry.py --mission ESA-Mission2 \\
        --anomalies_only --context_min 30

    # Specific channels only
    python scripts/export_telemetry.py --mission ESA-Mission2 \\
        --channels channel_1 channel_47 channel_100
"""

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mission",        default="ESA-Mission2")
    p.add_argument("--start",          default=None,
                   help="Start of time window e.g. 2003-01-01")
    p.add_argument("--end",            default=None,
                   help="End of time window e.g. 2003-02-01")
    p.add_argument("--anomalies_only", action="store_true",
                   help="Only export rows within predicted anomaly windows + context")
    p.add_argument("--context_min",    type=float, default=30,
                   help="Minutes of context before/after each anomaly window (default: 30)")
    p.add_argument("--channels",       nargs="*", default=None,
                   help="Channel names to include e.g. channel_1 channel_47. Default: all.")
    p.add_argument("--points",         default=None,
                   help="Path to points CSV. Auto-detected if not specified.")
    p.add_argument("--dataset",        default=None,
                   help="Path to dataset dir. Default: dataset/<mission>/")
    p.add_argument("--out",            default=None,
                   help="Output CSV path. Default: results/<mission>/<mission>_telemetry.csv")
    p.add_argument("--downsample",     type=int, default=1,
                   help="Keep every Nth row (default: 1 = no downsampling). "
                        "Use 2 for 1-minute resolution, 10 for 5-minute resolution.")
    return p.parse_args()


def find_points_csv(mission: str) -> Path:
    patterns = [
        f"checkpoints/ALL_task_esa_{mission.lower().replace('esa-', 'esa_')}_UniTS_*/anomaly_results/{mission}_points.csv",
        f"checkpoints/*{mission}*/anomaly_results/{mission}_points.csv",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[-1])
    raise FileNotFoundError(
        f"Could not find points CSV for {mission}. Use --points to specify path.")


def main():
    args = parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────
    points_path = Path(args.points) if args.points else find_points_csv(args.mission)
    dataset_dir = Path(args.dataset) if args.dataset else Path(f"dataset/{args.mission}")

    out_path = Path(args.out) if args.out else \
               Path(f"results/{args.mission}/{args.mission}_telemetry.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"Loading points CSV : {points_path}")
    points = pd.read_csv(points_path, parse_dates=["timestamp"])
    points = points.loc[:, ~points.columns.str.match(r"^Unnamed")]
    points = points.sort_values("timestamp").reset_index(drop=True)

    print(f"Loading test array : {dataset_dir}/{args.mission}_test.npy")
    test_arr = np.load(dataset_dir / f"{args.mission}_test.npy")

    ch_txt = dataset_dir / f"{args.mission}_channels.txt"
    all_channel_names = ch_txt.read_text().splitlines() if ch_txt.exists() else \
                        [f"channel_{i}" for i in range(test_arr.shape[1])]

    print(f"  {len(points):,} timesteps  |  {test_arr.shape[1]} channels")

    # ── Channel selection ─────────────────────────────────────────────────
    if args.channels:
        missing = set(args.channels) - set(all_channel_names)
        if missing:
            print(f"WARNING: unknown channel names: {missing}")
        col_indices = [all_channel_names.index(c) for c in args.channels
                       if c in all_channel_names]
        selected_names = [all_channel_names[i] for i in col_indices]
    else:
        col_indices    = list(range(len(all_channel_names)))
        selected_names = all_channel_names

    print(f"  Exporting {len(selected_names)} channels")

    # ── Time window filter ────────────────────────────────────────────────
    mask = pd.Series([True] * len(points))
    if args.start:
        mask &= points["timestamp"] >= pd.Timestamp(args.start)
    if args.end:
        mask &= points["timestamp"] <= pd.Timestamp(args.end)

    # Anomaly-only mode: expand mask to include context around anomaly windows
    if args.anomalies_only:
        context = pd.Timedelta(minutes=args.context_min)
        anom_times = points.loc[points["is_anomaly_predicted"] == 1, "timestamp"]
        anom_mask  = pd.Series([False] * len(points))
        for t in anom_times:
            anom_mask |= (points["timestamp"] >= t - context) & \
                         (points["timestamp"] <= t + context)
        mask &= anom_mask
        print(f"  Anomaly-only mode: {mask.sum():,} rows within {args.context_min}min "
              f"of predicted anomalies")

    orig_idx = mask[mask].index
    if len(orig_idx) == 0:
        print("No rows matched filters. Exiting.")
        return

    points_sub   = points.loc[orig_idx].reset_index(drop=True)
    test_sub     = test_arr[orig_idx.min():orig_idx.max()+1][::args.downsample,  :]
    points_sub   = points_sub.iloc[::args.downsample].reset_index(drop=True)

    # ── Build output DataFrame ────────────────────────────────────────────
    print(f"\nBuilding output table: {len(points_sub):,} rows × "
          f"{len(selected_names) + 3} columns ...")

    channel_df = pd.DataFrame(
        test_sub[:, col_indices],
        columns=selected_names,
    )

    out_df = pd.concat([
        points_sub[["timestamp"]].reset_index(drop=True),
        channel_df,
        points_sub[["anomaly_score", "is_anomaly_predicted",
                    "is_anomaly_ground_truth"]].reset_index(drop=True),
    ], axis=1)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"Saving to {out_path} ...")
    out_df.to_csv(out_path, index=False)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nDone.")
    print(f"  Rows    : {len(out_df):,}")
    print(f"  Columns : {len(out_df.columns):,}  "
          f"(timestamp + {len(selected_names)} channels + 3 anomaly cols)")
    print(f"  Size    : {size_mb:.1f} MB")
    print(f"  Output  : {out_path}")


if __name__ == "__main__":
    main()