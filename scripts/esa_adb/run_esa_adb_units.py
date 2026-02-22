#!/usr/bin/env python3
"""Prepare ESA-ADB mission data and run UniTS anomaly inference.

Assumes raw ESA-ADB data exists at:
  data/ESA-ADB-raw/ESA-Mission1
  data/ESA-ADB-raw/ESA-Mission2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.tools import adjustment


DEFAULT_TEST_SPLITS = {
    "ESA-Mission1": "2007-01-01",
    "ESA-Mission2": "2001-10-01",
}


@dataclass
class MissionPreparedData:
    mission: str
    train: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    test_timestamps: pd.DatetimeIndex


class SlidingWindowDataset(Dataset):
    def __init__(self, values: np.ndarray, labels: np.ndarray, win_size: int) -> None:
        self.values = values
        self.labels = labels
        self.win_size = win_size

    def __len__(self) -> int:
        return max(self.values.shape[0] - self.win_size + 1, 0)

    def __getitem__(self, idx: int):
        x = self.values[idx : idx + self.win_size]
        y = self.labels[idx : idx + self.win_size]
        return np.float32(x), np.float32(y)


def _read_channel_df(channel_path: Path) -> pd.DataFrame:
    channel_df = pd.read_pickle(channel_path)
    channel_df.index = pd.to_datetime(channel_df.index)
    channel_df = channel_df[~channel_df.index.duplicated(keep="last")]
    return channel_df.sort_index()


def prepare_mission(
    mission_path: Path,
    mission_name: str,
    split_at: str,
    max_channels: int | None,
) -> MissionPreparedData:
    channels_dir = mission_path / "channels"
    labels_path = mission_path / "labels.csv"
    channels_csv_path = mission_path / "channels.csv"

    if not channels_dir.exists() or not labels_path.exists() or not channels_csv_path.exists():
        raise FileNotFoundError(f"Missing required ESA files under {mission_path}")

    channels_meta = pd.read_csv(channels_csv_path)
    channel_names = channels_meta["Channel"].astype(str).tolist()
    if max_channels is not None:
        channel_names = channel_names[:max_channels]

    series_parts: list[pd.Series] = []
    for channel_name in channel_names:
        channel_file = channels_dir / f"{channel_name}.zip"
        if not channel_file.exists():
            continue
        channel_df = _read_channel_df(channel_file)

        if channel_name in channel_df.columns:
            values = channel_df[channel_name]
        else:
            numeric_cols = channel_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                continue
            values = channel_df[numeric_cols[0]]
        series_parts.append(values.rename(channel_name))

    if not series_parts:
        raise RuntimeError(f"No channel data loaded for mission {mission_name}")

    full_df = pd.concat(series_parts, axis=1).sort_index()
    full_df = full_df.ffill().bfill()

    labels_df = pd.read_csv(labels_path, parse_dates=["StartTime", "EndTime"])
    label_series = pd.Series(np.zeros(len(full_df), dtype=np.int64), index=full_df.index)
    available_channels = set(full_df.columns)
    for _, row in labels_df.iterrows():
        row_channel = str(row.get("Channel", ""))
        if row_channel and row_channel not in available_channels:
            continue
        mask = (full_df.index >= row["StartTime"]) & (full_df.index <= row["EndTime"])
        label_series.loc[mask] = 1

    split_ts = pd.Timestamp(split_at)
    train_df = full_df[full_df.index <= split_ts]
    test_df = full_df[full_df.index > split_ts]
    test_labels = label_series.loc[test_df.index]

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError(
            f"Invalid split for {mission_name}: train={len(train_df)} test={len(test_df)}"
        )

    return MissionPreparedData(
        mission=mission_name,
        train=train_df.to_numpy(dtype=np.float32),
        test=test_df.to_numpy(dtype=np.float32),
        test_labels=test_labels.to_numpy(dtype=np.int64),
        test_timestamps=test_df.index,
    )


def build_units_model(enc_in: int, checkpoint: Path, device: torch.device, args):
    from models.UniTS import Model as UniTSModel
    model_args = SimpleNamespace(
        prompt_num=args.prompt_num,
        d_model=args.d_model,
        patch_len=args.patch_len,
        stride=args.stride,
        dropout=args.dropout,
        e_layers=args.e_layers,
        n_heads=args.n_heads,
    )
    task_cfg = [["ESA", {"dataset": "ESA", "enc_in": enc_in, "task_name": "anomaly_detection"}]]
    model = UniTSModel(model_args, task_cfg).to(device)

    ckpt = torch.load(checkpoint, map_location="cpu")
    if "student" in ckpt:
        ckpt = ckpt["student"]
    cleaned = {}
    for k, v in ckpt.items():
        name = k[7:] if k.startswith("module.") else k
        if name in model.state_dict() and model.state_dict()[name].shape == v.shape:
            cleaned[name] = v
    missing = model.load_state_dict(cleaned, strict=False)
    print(f"Loaded checkpoint keys={len(cleaned)}; missing={len(missing.missing_keys)}")
    model.eval()
    return model


def aggregate_window_scores(window_scores: np.ndarray, length: int, win_size: int) -> np.ndarray:
    sums = np.zeros(length, dtype=np.float64)
    counts = np.zeros(length, dtype=np.float64)
    for i in range(window_scores.shape[0]):
        sums[i : i + win_size] += window_scores[i]
        counts[i : i + win_size] += 1.0
    counts[counts == 0] = 1.0
    return sums / counts


def infer_energy(
    model,
    values: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = SlidingWindowDataset(values, labels, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    all_scores = []
    mse = torch.nn.MSELoss(reduction="none")
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            out = model(batch_x, None, None, None, task_id=0, task_name="anomaly_detection")
            score = torch.mean(mse(batch_x, out), dim=-1).cpu().numpy()
            all_scores.append(score)

    if not all_scores:
        raise RuntimeError("No windows produced. Lower --seq-len or provide longer time series.")
    win_scores = np.concatenate(all_scores, axis=0)
    return aggregate_window_scores(win_scores, len(values), seq_len)


def contiguous_anomaly_ranges(pred: np.ndarray, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    start = None
    for i, p in enumerate(pred):
        if p and start is None:
            start = i
        if (not p) and start is not None:
            rows.append((timestamps[start], timestamps[i - 1]))
            start = None
    if start is not None:
        rows.append((timestamps[start], timestamps[len(pred) - 1]))
    return pd.DataFrame(rows, columns=["start_timestamp", "end_timestamp"])


def run_mission(mission_data: MissionPreparedData, args, output_dir: Path, device: torch.device) -> None:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(mission_data.train)
    test_scaled = scaler.transform(mission_data.test)

    model = build_units_model(enc_in=train_scaled.shape[1], checkpoint=args.checkpoint, device=device, args=args)

    train_labels_stub = np.zeros(len(train_scaled), dtype=np.int64)
    train_energy = infer_energy(model, train_scaled, train_labels_stub, args.seq_len, args.batch_size, device)
    test_energy = infer_energy(model, test_scaled, mission_data.test_labels, args.seq_len, args.batch_size, device)

    combined = np.concatenate([train_energy, test_energy])
    threshold = np.percentile(combined, 100 - args.anomaly_ratio)

    pred = (test_energy > threshold).astype(int)
    gt = mission_data.test_labels.astype(int)
    gt_adj, pred_adj = adjustment(gt.copy(), pred.copy())

    mission_out = output_dir / mission_data.mission
    mission_out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "timestamp": mission_data.test_timestamps,
            "energy": test_energy,
            "gt": gt_adj,
            "pred": pred_adj,
        }
    ).to_csv(mission_out / "predicted_anomaly_points.csv", index=False)

    contiguous_anomaly_ranges(pred_adj, mission_data.test_timestamps).to_csv(
        mission_out / "predicted_anomaly_ranges.csv", index=False
    )

    summary = {
        "mission": mission_data.mission,
        "train_samples": int(len(train_scaled)),
        "test_samples": int(len(test_scaled)),
        "features": int(train_scaled.shape[1]),
        "threshold": float(threshold),
        "predicted_anomalies": int(np.sum(pred_adj)),
    }
    pd.Series(summary).to_json(mission_out / "run_summary.json", indent=2)
    print(f"[{mission_data.mission}] wrote outputs to {mission_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ESA-ADB and run UniTS anomaly detection.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/ESA-ADB-raw"))
    parser.add_argument("--missions", nargs="+", default=["ESA-Mission1", "ESA-Mission2"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/esa_adb_units"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--anomaly-ratio", type=float, default=1.0)
    parser.add_argument("--max-channels", type=int, default=None)
    parser.add_argument("--split-override", type=str, default=None, help="Override test split date for all missions.")
    parser.add_argument("--prompt-num", type=int, default=10)
    parser.add_argument("--patch-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for mission in args.missions:
        mission_path = args.raw_root / mission
        split_at = args.split_override or DEFAULT_TEST_SPLITS.get(mission)
        if split_at is None:
            raise ValueError(f"No default split is defined for mission {mission}; use --split-override")
        prepared = prepare_mission(
            mission_path=mission_path,
            mission_name=mission,
            split_at=split_at,
            max_channels=args.max_channels,
        )
        run_mission(prepared, args=args, output_dir=args.output_dir, device=device)


if __name__ == "__main__":
    main()
