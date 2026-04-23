#!/usr/bin/env python3
"""
attribute_anomalies.py — Per-channel attribution for predicted anomaly windows.

For each contiguous predicted-anomaly window, scores every channel by how far
it deviated from its normal baseline (median ± IQR). Outputs a ranked
attribution parquet with one row per window × channel.

Output columns:
    window_id           integer window index
    start               window start timestamp
    end                 window end timestamp
    duration_min        window duration in minutes
    channel             channel name
    deviation_score     abs(value - median) / (IQR + epsilon) — higher = more anomalous
    rank                rank within window (1 = most anomalous channel)
    mean_during         mean channel value during the window
    median_normal       median channel value during normal periods
    iqr_normal          IQR of channel during normal periods

Usage:
    python scripts/attribute_anomalies.py --mission STPSat7-EPS

    # Specify paths manually
    python scripts/attribute_anomalies.py \
        --mission STPSat7-EPS \
        --points  checkpoints/.../STPSat7-EPS_points.csv \
        --out     results/STPSat7-EPS/telemetry/STPSat7-EPS_attribution.parquet
"""

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mission",  required=True)
    p.add_argument("--points",   default=None,
                   help="Path to points CSV. Auto-detected if not specified.")
    p.add_argument("--dataset",  default=None,
                   help="Path to dataset dir. Default: dataset/<mission>/")
    p.add_argument("--out",      default=None,
                   help="Output parquet path. Default: results/<mission>/telemetry/<mission>_attribution.parquet")
    p.add_argument("--top_n",    type=int, default=None,
                   help="Only keep top N channels per window (default: all)")
    p.add_argument("--min_duration", type=int, default=1,
                   help="Minimum window duration in rows to include (default: 1)")
    return p.parse_args()


def find_points_csv(mission: str) -> Path:
    patterns = [
        f"checkpoints/ALL_task_{mission.lower()}_UniTS_*/anomaly_results/{mission}_points.csv",
        f"checkpoints/ALL_task_esa_{mission.lower()}_UniTS_*/anomaly_results/{mission}_points.csv",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[-1])
    raise FileNotFoundError(
        f"Could not find points CSV for {mission}. Use --points to specify path.")


def extract_windows(df: pd.DataFrame) -> list:
    """Extract contiguous anomaly windows from is_anomaly_predicted column."""
    preds = df["is_anomaly_predicted"].values
    changes = np.diff(np.concatenate([[0], preds, [0]]))
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    windows = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        windows.append({
            "window_id": i + 1,
            "start_idx": s,
            "end_idx":   e,
            "start":     df["timestamp"].iloc[s],
            "end":       df["timestamp"].iloc[e - 1],
            "n_rows":    e - s,
        })
    return windows


def compute_attribution(test_arr: np.ndarray, channel_names: list,
                         windows: list, normal_mask: np.ndarray) -> pd.DataFrame:
    """
    For each window, score each channel by deviation from normal baseline.
    Uses robust median/IQR scoring.
    """
    eps = 1e-8
    n_channels = test_arr.shape[1]

    # Compute per-channel baseline stats from normal periods
    normal_data = test_arr[normal_mask]
    if len(normal_data) == 0:
        # Fallback: use full array if no normal periods
        normal_data = test_arr

    medians = np.median(normal_data, axis=0)
    q75     = np.percentile(normal_data, 75, axis=0)
    q25     = np.percentile(normal_data, 25, axis=0)
    iqrs    = q75 - q25

    rows = []
    for w in windows:
        s, e = w["start_idx"], w["end_idx"]
        window_data = test_arr[s:e]  # shape: (n_rows, n_channels)

        if len(window_data) == 0:
            continue

        window_means = window_data.mean(axis=0)

        # Deviation score: how many IQRs away from normal median
        deviation_scores = np.abs(window_means - medians) / (iqrs + eps)

        for ch_idx in range(n_channels):
            rows.append({
                "window_id":      w["window_id"],
                "start":          w["start"],
                "end":            w["end"],
                "duration_min":   round(w["n_rows"] / 60, 2),  # assumes 60s resolution; adjust if needed
                "channel":        channel_names[ch_idx],
                "deviation_score": float(deviation_scores[ch_idx]),
                "mean_during":    float(window_means[ch_idx]),
                "median_normal":  float(medians[ch_idx]),
                "iqr_normal":     float(iqrs[ch_idx]),
            })

    if not rows:
        return pd.DataFrame()

    df_attr = pd.DataFrame(rows)

    # Add rank within each window (1 = most anomalous)
    df_attr["rank"] = (
        df_attr.groupby("window_id")["deviation_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    return df_attr.sort_values(["window_id", "rank"]).reset_index(drop=True)


def main():
    args = parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────────
    points_path = Path(args.points) if args.points else find_points_csv(args.mission)
    dataset_dir = Path(args.dataset) if args.dataset else Path(f"dataset/{args.mission}")
    out_path    = Path(args.out) if args.out else \
                  Path(f"results/{args.mission}/telemetry/{args.mission}_attribution.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading {points_path} ...")
    df = pd.read_csv(points_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Loading test array ...")
    test_arr = np.load(dataset_dir / f"{args.mission}_test.npy")

    ch_txt = dataset_dir / f"{args.mission}_channels.txt"
    channel_names = ch_txt.read_text().splitlines() if ch_txt.exists() else \
                    [f"channel_{i}" for i in range(test_arr.shape[1])]

    n_predicted = int(df["is_anomaly_predicted"].sum())
    print(f"  {len(df):,} timesteps  |  {test_arr.shape[1]} channels  |  "
          f"{n_predicted:,} predicted anomaly points")

    if n_predicted == 0:
        print("\nNo predicted anomalies — attribution parquet will be empty.")
        print("This is expected when the threshold is NaN (insufficient training data).")
        pd.DataFrame().to_parquet(out_path, index=False, engine="pyarrow")
        print(f"Saved empty parquet to {out_path}")
        return

    # ── Extract windows ───────────────────────────────────────────────────────
    windows = extract_windows(df)
    if args.min_duration > 1:
        windows = [w for w in windows if w["n_rows"] >= args.min_duration]
    print(f"  {len(windows)} anomaly windows (min_duration={args.min_duration} rows)")

    # ── Align lengths (points CSV and test array may differ by a few rows) ──────
    n_common = min(len(df), test_arr.shape[0])
    if len(df) != test_arr.shape[0]:
        print(f"  NOTE: points CSV has {len(df):,} rows, test array has "
              f"{test_arr.shape[0]:,} rows — aligning to {n_common:,}")
        df       = df.iloc[:n_common].reset_index(drop=True)
        test_arr = test_arr[:n_common]

    # ── Normal mask (rows not flagged as anomaly) ─────────────────────────────
    normal_mask = df["is_anomaly_predicted"].values == 0

    # ── Attribution ───────────────────────────────────────────────────────────
    print("Computing per-channel attribution scores ...")
    df_attr = compute_attribution(test_arr, channel_names, windows, normal_mask)

    if df_attr.empty:
        print("No attribution rows generated.")
        pd.DataFrame().to_parquet(out_path, index=False, engine="pyarrow")
        return

    # Optionally filter to top N channels per window
    if args.top_n:
        df_attr = df_attr[df_attr["rank"] <= args.top_n].reset_index(drop=True)
        print(f"  Filtered to top {args.top_n} channels per window")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving to {out_path} ...")
    df_attr.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nDone.")
    print(f"  Windows   : {df_attr['window_id'].nunique():,}")
    print(f"  Rows      : {len(df_attr):,}  (windows × channels)")
    print(f"  Size      : {size_mb:.2f} MB")
    print(f"  Output    : {out_path}")

    # Print top culprits summary
    print(f"\nTop 10 most anomalous channels across all windows:")
    top = (df_attr.groupby("channel")["deviation_score"]
           .mean()
           .sort_values(ascending=False)
           .head(10))
    for ch, score in top.items():
        print(f"  {score:6.2f}  {ch}")


if __name__ == "__main__":
    main()