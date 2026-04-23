#!/usr/bin/env python3
"""
prepare_stpsat4_data.py — Prepare STPSat-4 health & status data for UniTS anomaly detection

Each subsystem (ADCS, HRR, MRR, PCE1, PCE2, TCS) is prepared as a separate UniTS mission:
    STPSat4-ADCS, STPSat4-HRR, STPSat4-MRR, STPSat4-PCE1, STPSat4-PCE2, STPSat4-TCS

Pipeline per subsystem:
  1. Find all CSVs for this subsystem across all date folders
  2. Concatenate chronologically
  3. Parse UTC_Time_String timestamp
  4. Resample to 60-second grid (forward-fill)
  5. Ordinal-encode any categorical (string) columns
  6. Fill NaNs with column means (nan_to_num safe)
  7. 70/30 train/test split by date
  8. Save: <mission>_train.npy, <mission>_test.npy,
           <mission>_test_label.npy (all zeros — no ground truth),
           <mission>_test_timestamps.npy, <mission>_channels.txt

Usage:
    python scripts/prepare_stpsat4_data.py

    # Single subsystem only
    python scripts/prepare_stpsat4_data.py --subsystems ADCS TCS

    # Custom resample interval
    python scripts/prepare_stpsat4_data.py --resample_sec 30

    # Custom train/test split date
    python scripts/prepare_stpsat4_data.py --split_date 2020-03-19
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

ALL_SUBSYSTEMS = ["ADCS", "HRR", "MRR", "PCE1", "PCE2", "TCS"]

TIMESTAMP_COL  = "UTC_Time_String"
RAW_DATA_DIR   = "data/STPSat-4-raw/Sat-4_HS_data"
DATASET_BASE   = "dataset"
MISSION_PREFIX = "STPSat4"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subsystems", nargs="*", default=ALL_SUBSYSTEMS,
                   help="Which subsystems to process (default: all 6)")
    p.add_argument("--raw_dir",    default=RAW_DATA_DIR)
    p.add_argument("--out_dir",    default=DATASET_BASE)
    p.add_argument("--resample_sec", type=int, default=60,
                   help="Resample interval in seconds (default: 60)")
    p.add_argument("--split_date", default="2020-03-19",
                   help="Train/test split date. Train = before, Test = on or after. "
                        "Default: 2020-03-19 (~70/30)")
    p.add_argument("--min_train_rows", type=int, default=1000,
                   help="Skip subsystem if train set has fewer rows than this")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_csvs(raw_dir: str, subsystem: str) -> list:
    """Find all CSVs for a subsystem across all date folders, sorted chronologically."""
    pattern = os.path.join(raw_dir, f"*__HS_Data*", f"{subsystem}_HS_Data_*.csv")
    files   = sorted(glob.glob(pattern))
    return files


def load_and_concat(files: list, subsystem: str) -> pd.DataFrame:
    """Load all CSVs for a subsystem and concatenate."""
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if TIMESTAMP_COL not in df.columns:
                print(f"  WARNING: No '{TIMESTAMP_COL}' column in {f}, skipping")
                continue
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not read {f}: {e}")
    if not dfs:
        raise ValueError(f"No valid CSVs found for subsystem {subsystem}")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(files)} files → {len(combined):,} rows before dedup")
    return combined


def clean_and_resample(df: pd.DataFrame, resample_sec: int) -> pd.DataFrame:
    """Parse timestamps, drop duplicates, resample to uniform grid."""
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    n_bad = df[TIMESTAMP_COL].isna().sum()
    if n_bad > 0:
        print(f"  WARNING: {n_bad} rows with unparseable timestamps dropped")
    df = df.dropna(subset=[TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL)
    df = df.drop_duplicates(subset=[TIMESTAMP_COL], keep="last")
    df = df.set_index(TIMESTAMP_COL)

    # Resample to uniform grid
    rule = f"{resample_sec}s"
    df   = df.resample(rule).last()   # forward-fill via last observation
    df   = df.ffill()                 # fill any remaining gaps
    print(f"  After resample ({resample_sec}s): {len(df):,} rows  "
          f"[{df.index[0]} → {df.index[-1]}]")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple:
    """
    Ordinal-encode any object/string columns.
    Returns (encoded_df, list_of_encoded_col_names).
    """
    encoded = []
    for col in df.columns:
        if df[col].dtype == object:
            categories = sorted(df[col].dropna().unique())
            mapping    = {v: i for i, v in enumerate(categories)}
            df[col]    = df[col].map(mapping).astype(float)
            encoded.append(col)
    if encoded:
        print(f"  Ordinal-encoded {len(encoded)} categorical columns: {encoded[:5]}"
              f"{'...' if len(encoded) > 5 else ''}")
    return df, encoded


def to_float_array(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to float32 numpy array, coercing non-numeric to NaN."""
    arr = df.apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    return arr


def fill_nans(arr: np.ndarray) -> np.ndarray:
    """Fill NaN values with column means (nan_to_num safe for all-NaN columns)."""
    col_means = np.nan_to_num(np.nanmean(arr, axis=0, keepdims=True))
    nan_mask  = np.isnan(arr)
    arr       = np.where(nan_mask, col_means, arr)
    n_nan     = nan_mask.sum()
    if n_nan > 0:
        print(f"  Filled {n_nan:,} NaN values with column means")
    return arr


def save_dataset(out_dir: Path, mission: str,
                 train_arr: np.ndarray, test_arr: np.ndarray,
                 test_timestamps: pd.DatetimeIndex,
                 channel_names: list):
    """Save all output files for one mission."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Arrays
    np.save(out_dir / f"{mission}_train.npy",           train_arr)
    np.save(out_dir / f"{mission}_test.npy",            test_arr)
    np.save(out_dir / f"{mission}_test_label.npy",
            np.zeros(len(test_arr), dtype=np.int32))    # no ground truth
    np.save(out_dir / f"{mission}_test_timestamps.npy",
            test_timestamps.astype(np.int64))

    # Channel names
    (out_dir / f"{mission}_channels.txt").write_text("\n".join(channel_names))

    # Size summary
    train_mb = train_arr.nbytes / 1024 / 1024
    test_mb  = test_arr.nbytes  / 1024 / 1024
    print(f"  Saved → {out_dir}")
    print(f"    train : {train_arr.shape}  ({train_mb:.1f} MB)")
    print(f"    test  : {test_arr.shape}   ({test_mb:.1f} MB)")
    print(f"    labels: all zeros (unsupervised — no ground truth)")


def update_yaml(yaml_path: str, missions: list, enc_in_map: dict):
    """Append new STPSat4 mission entries to the anomaly detection YAML."""
    import yaml

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    changed = False
    for mission, enc_in in enc_in_map.items():
        if mission not in config:
            config[mission] = {
                "data":       "anomaly_detection_ESA",
                "root_path":  f"./dataset/{mission}/",
                "data_path":  mission,
                "enc_in":     enc_in,
                "dec_in":     enc_in,
                "c_out":      enc_in,
            }
            print(f"  Added YAML entry: {mission} (enc_in={enc_in})")
            changed = True
        else:
            print(f"  YAML entry already exists: {mission}, skipping")

    if changed:
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  Saved {yaml_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def process_subsystem(subsystem: str, args) -> dict:
    """Full pipeline for one subsystem. Returns {mission: enc_in} or empty dict."""
    mission  = f"{MISSION_PREFIX}-{subsystem}"
    out_dir  = Path(args.out_dir) / mission
    split_ts = pd.Timestamp(args.split_date)

    print(f"\n{'='*70}")
    print(f"Processing: {mission}")
    print(f"{'='*70}")

    # ── Find and load ─────────────────────────────────────────────────────
    files = find_csvs(args.raw_dir, subsystem)
    if not files:
        print(f"  No CSV files found for {subsystem} in {args.raw_dir}")
        return {}
    print(f"  Found {len(files)} CSV files")

    df = load_and_concat(files, subsystem)

    # ── Clean and resample ────────────────────────────────────────────────
    df = clean_and_resample(df, args.resample_sec)

    # ── Encode categoricals ───────────────────────────────────────────────
    df, _ = encode_categoricals(df)

    # ── Channel names (all columns except timestamp, which is now index) ──
    channel_names = list(df.columns)
    print(f"  Channels: {len(channel_names)}")

    # ── Convert to float array ────────────────────────────────────────────
    arr = to_float_array(df)
    arr = fill_nans(arr)

    # ── Train/test split ──────────────────────────────────────────────────
    split_idx  = df.index.searchsorted(split_ts)
    train_arr  = arr[:split_idx]
    test_arr   = arr[split_idx:]
    test_times = df.index[split_idx:]

    if len(train_arr) < args.min_train_rows:
        print(f"  SKIP: train set only {len(train_arr)} rows "
              f"(< min_train_rows={args.min_train_rows})")
        return {}

    train_pct = 100 * len(train_arr) / len(arr)
    print(f"  Split at {split_ts.date()}: "
          f"train={len(train_arr):,} ({train_pct:.0f}%)  "
          f"test={len(test_arr):,} ({100-train_pct:.0f}%)")

    # ── Save ──────────────────────────────────────────────────────────────
    save_dataset(out_dir, mission, train_arr, test_arr, test_times, channel_names)

    return {mission: len(channel_names)}


def main():
    args = parse_args()

    print(f"STPSat-4 Data Preparation")
    print(f"  Raw data  : {args.raw_dir}")
    print(f"  Output    : {args.out_dir}")
    print(f"  Resample  : {args.resample_sec}s")
    print(f"  Split date: {args.split_date}")
    print(f"  Subsystems: {args.subsystems}")

    enc_in_map = {}
    for subsystem in args.subsystems:
        result = process_subsystem(subsystem, args)
        enc_in_map.update(result)

    # ── Update YAML ───────────────────────────────────────────────────────
    yaml_path = "data_provider/anomaly_detection_esa.yaml"
    if enc_in_map and os.path.exists(yaml_path):
        print(f"\n{'='*70}")
        print(f"Updating YAML: {yaml_path}")
        update_yaml(yaml_path, list(enc_in_map.keys()), enc_in_map)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PREP COMPLETE — {len(enc_in_map)}/{len(args.subsystems)} subsystems")
    print(f"{'='*70}")
    for mission, enc_in in enc_in_map.items():
        print(f"  {mission:<25} enc_in={enc_in}")

    if enc_in_map:
        missions_str = " ".join(enc_in_map.keys())
        print(f"\nTo run anomaly detection:")
        print(f"  MISSIONS='{missions_str}' sbatch slurm/02_run_anomaly.sh")


if __name__ == "__main__":
    main()