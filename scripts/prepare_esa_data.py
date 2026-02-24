#!/usr/bin/env python3
"""
prepare_esa_data.py
───────────────────
Reads the raw ESA-ADB Zenodo download and produces the three .npy files
that UniTS needs for anomaly detection.  No ESA-ADB codebase required.

Expected raw layout (exactly what you get from the Zenodo download):

    UniTS/data/ESA-ADB-raw/
    ├── ESA-Mission1/
    │   ├── channels/
    │   │   ├── channel_1.zip      # pd.read_pickle → DataFrame(datetime_index, col='channel_N')
    │   │   └── ...
    │   ├── labels.csv
    │   ├── anomaly_types.csv      # ID, Category  ("Anomaly" | "Rare Event" | "Gap")
    │   └── telecommands.csv
    └── ESA-Mission2/
        └── ...                    # same structure

Output (placed inside UniTS dataset folder, ready to use):

    UniTS/dataset/ESA-Mission1/
    ├── ESA-Mission1_train.npy          float32  [T_train × C]
    ├── ESA-Mission1_test.npy           float32  [T_test  × C]
    ├── ESA-Mission1_test_label.npy     int32    [T_test]       1 = anomaly
    ├── ESA-Mission1_test_timestamps.npy str     [T_test]       ISO timestamps
    └── ESA-Mission1_channels.txt                channel names in column order

Usage:
    python scripts/prepare_esa_data.py

    # Use only the 6-channel subset from the ESA-ADB paper:
    python scripts/prepare_esa_data.py --channels channel_41 channel_42 channel_43 channel_44 channel_45 channel_46

    # Change train/test split date:
    python scripts/prepare_esa_data.py --train_end 2003-07-01 --test_start 2007-01-01
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Label encoding (mirrors ESA-ADB AnnotationLabel enum) ─────────────────
LABEL_NOMINAL    = 0
LABEL_ANOMALY    = 1
LABEL_RARE_EVENT = 2
LABEL_GAP        = 3

# Treat both Anomaly and Rare Event as positive class for UniTS
ANOMALOUS_LABELS = {LABEL_ANOMALY, LABEL_RARE_EVENT}

# ── Default paths (relative to this script's location = UniTS/scripts/) ──
SCRIPT_DIR   = Path(__file__).resolve().parent
UNITS_ROOT   = SCRIPT_DIR.parent          # UniTS/
RAW_DIR      = UNITS_ROOT / "data" / "ESA-ADB-raw" / "ESA-Mission1"
OUT_DIR      = UNITS_ROOT / "dataset" / "ESA-Mission1"
DATASET_NAME = "ESA-Mission1"

# Mission 1 date splits (from the ESA-ADB paper)
# Training uses data UP TO train_end; test uses data AFTER test_start.
# The gap between them is a validation period (not used here).
DEFAULT_TRAIN_END  = "2006-10-01"   # ~84 months of training
DEFAULT_TEST_START = "2007-01-01"
RESAMPLE_FREQ      = "30s"          # 30-second zero-order-hold resampling


def parse_args():
    p = argparse.ArgumentParser(description="Convert raw ESA-ADB → UniTS .npy")
    p.add_argument("--raw_dir",    default=str(RAW_DIR),
                   help=f"Path to one mission's raw folder, e.g. data/ESA-ADB-raw/ESA-Mission1 (default: {RAW_DIR})")
    p.add_argument("--out_dir",    default=str(OUT_DIR),
                   help=f"Output directory (default: {OUT_DIR})")
    p.add_argument("--name",       default=DATASET_NAME)
    p.add_argument("--train_end",  default=DEFAULT_TRAIN_END,
                   help="Last timestamp included in training data (exclusive)")
    p.add_argument("--test_start", default=DEFAULT_TEST_START,
                   help="First timestamp included in test data (inclusive)")
    p.add_argument("--channels",   nargs="*", default=None,
                   help="Subset of channel names to use, e.g. channel_41 channel_42. "
                        "Default: all channels.")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────

def load_labels(raw_dir: Path):
    labels_df       = pd.read_csv(raw_dir / "labels.csv",
                                  parse_dates=["StartTime", "EndTime"])
    anomaly_types   = pd.read_csv(raw_dir / "anomaly_types.csv")
    # Merge so each label row carries its Category
    labels_df = labels_df.merge(anomaly_types[["ID", "Category"]], on="ID", how="left")
    # Strip timezone info if present — channel DataFrames are tz-naive
    for col in ["StartTime", "EndTime"]:
        if hasattr(labels_df[col].dt, "tz") and labels_df[col].dt.tz is not None:
            labels_df[col] = labels_df[col].dt.tz_localize(None)
    return labels_df


def category_to_label(category: str) -> int:
    if category == "Anomaly":
        return LABEL_ANOMALY
    elif category == "Rare Event":
        return LABEL_RARE_EVENT
    else:
        return LABEL_GAP


def load_channel(raw_dir: Path, channel_name: str,
                 labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load one channel zip file, apply derivative for monotonic channels (4-11),
    assign per-sample labels, and return a DataFrame with columns
    ['value', 'label'] on a DatetimeIndex.
    """
    zip_path = raw_dir / "channels" / f"{channel_name}.zip"
    df = pd.read_pickle(zip_path)               # col = channel_name, datetime index

    # Ensure tz-naive index so loc slicing against labels works
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.rename(columns={channel_name: "value"})
    df["label"] = LABEL_NOMINAL

    # Derivative for monotonic counter channels (channels 4–11)
    try:
        chan_num = int(channel_name.split("_")[1])
        if 4 <= chan_num <= 11:
            df["value"] = np.diff(df["value"].values, append=df["value"].values[-1])
    except (IndexError, ValueError):
        pass

    # Annotate labels from labels.csv
    chan_labels = labels_df[labels_df["Channel"] == channel_name]
    for _, row in chan_labels.iterrows():
        label_val = category_to_label(row["Category"])
        df.loc[row["StartTime"]:row["EndTime"], "label"] = label_val

    return df


def resample_zoh(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample to a uniform grid using zero-order hold (forward-fill).
    """
    first = pd.Timestamp(df.index[0]).floor(freq=freq)
    last  = pd.Timestamp(df.index[-1]).ceil(freq=freq)
    grid  = pd.date_range(first, last, freq=freq)
    resampled = df.reindex(grid, method="ffill")
    resampled.iloc[0] = df.iloc[0]   # preserve first sample
    return resampled


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    raw_dir  = Path(args.raw_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: raw_dir does not exist: {raw_dir}")
        print("       Expected layout: data/ESA-ADB-raw/ESA-Mission1/channels/, labels.csv, etc.")
        sys.exit(1)

    print(f"Raw data : {raw_dir}")
    print(f"Output   : {out_dir}")

    # ── Discover channels ────────────────────────────────────────────────
    all_zips = sorted(glob(str(raw_dir / "channels" / "*.zip")))
    all_channels = [Path(z).stem for z in all_zips]

    if args.channels:
        missing = set(args.channels) - set(all_channels)
        if missing:
            print(f"ERROR: requested channels not found in raw data: {missing}")
            sys.exit(1)
        channels = args.channels
    else:
        channels = all_channels

    print(f"Channels : {len(channels)}  ({channels[0]} … {channels[-1]})")

    train_end  = pd.Timestamp(args.train_end)
    test_start = pd.Timestamp(args.test_start)

    # ── Load & process all channels ──────────────────────────────────────
    print(f"\nLoading and resampling {len(channels)} channels to {RESAMPLE_FREQ} ...")
    labels_df = load_labels(raw_dir)

    # Determine full time range first (from the first channel)
    ch0 = load_channel(raw_dir, channels[0], labels_df)
    ch0_r = resample_zoh(ch0, RESAMPLE_FREQ)

    # Build a full time index spanning train + test
    full_train_df = ch0_r[ch0_r.index <= train_end]
    full_test_df  = ch0_r[ch0_r.index >= test_start]

    train_index = full_train_df.index
    test_index  = full_test_df.index

    # Pre-allocate arrays
    n_train, n_test = len(train_index), len(test_index)
    n_chan           = len(channels)
    train_arr   = np.zeros((n_train, n_chan), dtype=np.float32)
    test_arr    = np.zeros((n_test,  n_chan), dtype=np.float32)
    test_labels = np.zeros(n_test,           dtype=np.int32)

    for i, ch_name in enumerate(tqdm(channels, desc="Channels")):
        df = load_channel(raw_dir, ch_name, labels_df)
        df = resample_zoh(df, RESAMPLE_FREQ)

        # Train slice
        t = df[df.index <= train_end].reindex(train_index)
        t["value"] = t["value"].ffill().bfill()
        train_arr[:, i] = t["value"].values.astype(np.float32)

        # Test slice — values
        s = df[df.index >= test_start].reindex(test_index)
        s["value"] = s["value"].ffill().bfill()
        test_arr[:, i] = s["value"].values.astype(np.float32)

        # Test labels — OR across channels: anomaly if ANY channel is annotated
        s_label = s["label"].fillna(LABEL_NOMINAL).astype(int).values
        # Mark as anomaly if this channel has anomaly/rare_event label
        test_labels |= np.isin(s_label, list(ANOMALOUS_LABELS)).astype(np.int32)

    # ── NaN safety ───────────────────────────────────────────────────────
    if np.isnan(train_arr).any():
        print("  WARNING: NaNs in train — filling with column means.")
        col_means = np.nanmean(train_arr, axis=0, keepdims=True)
        mask = np.isnan(train_arr)
        train_arr = np.where(mask, np.broadcast_to(col_means, train_arr.shape), train_arr)

    if np.isnan(test_arr).any():
        print("  WARNING: NaNs in test — filling with column means.")
        col_means = np.nanmean(test_arr, axis=0, keepdims=True)
        mask = np.isnan(test_arr)
        test_arr = np.where(mask, np.broadcast_to(col_means, test_arr.shape), test_arr)

    # ── Report ───────────────────────────────────────────────────────────
    n_anomaly = test_labels.sum()
    print(f"\nTrain : {train_arr.shape}  ({train_index[0]} → {train_index[-1]})")
    print(f"Test  : {test_arr.shape}  ({test_index[0]} → {test_index[-1]})")
    print(f"Anomaly timesteps : {n_anomaly} / {n_test}  ({100*n_anomaly/n_test:.2f}%)")

    # ── Save ─────────────────────────────────────────────────────────────
    np.save(out_dir / f"{args.name}_train.npy",           train_arr)
    np.save(out_dir / f"{args.name}_test.npy",            test_arr)
    np.save(out_dir / f"{args.name}_test_label.npy",      test_labels)
    np.save(out_dir / f"{args.name}_test_timestamps.npy",
            test_index.strftime("%Y-%m-%d %H:%M:%S").values)

    (out_dir / f"{args.name}_channels.txt").write_text("\n".join(channels))

    print(f"\n✓ Saved to {out_dir}/")
    print(f"  {args.name}_train.npy")
    print(f"  {args.name}_test.npy")
    print(f"  {args.name}_test_label.npy")
    print(f"  {args.name}_test_timestamps.npy")
    print(f"\nSet enc_in / dec_in / c_out = {n_chan} in dataset/ESA-Mission1/ YAML config")


if __name__ == "__main__":
    main()