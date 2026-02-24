#!/usr/bin/env python3
"""
infer_anomalies.py
──────────────────
Loads a UniTS checkpoint (or uses Isolation Forest as a fallback),
runs it over ESA-ADB test data, and writes two CSVs:

  results/ESA-Mission1_points.csv   — anomaly score for every timestep
  results/ESA-Mission1_windows.csv  — contiguous anomaly windows with timestamps

Run from UniTS root:

    # With pretrained UniTS checkpoint:
    python scripts/infer_anomalies.py \
        --ckpt newcheckpoints/units_x32_pretrain_checkpoint.pth

    # Fast CPU baseline (no checkpoint needed):
    python scripts/infer_anomalies.py --iforest

    # Different dataset name or anomaly sensitivity:
    python scripts/infer_anomalies.py --ckpt <path> --name ESA-Mission1 --anomaly_ratio 0.5
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

SCRIPT_DIR  = Path(__file__).resolve().parent
UNITS_ROOT  = SCRIPT_DIR.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name",          default="ESA-Mission1")
    p.add_argument("--data_dir",      default=None,
                   help="Path to .npy files. Default: dataset/{name}/")
    p.add_argument("--out_dir",       default=None,
                   help="Output directory. Default: results/{name}/")
    p.add_argument("--ckpt",          default=None,
                   help="UniTS checkpoint .pth file")
    p.add_argument("--iforest",       action="store_true",
                   help="Use Isolation Forest baseline (CPU, no checkpoint)")
    p.add_argument("--anomaly_ratio", type=float, default=1.0,
                   help="Flag top N%% of scores as anomalies (default: 1.0)")
    p.add_argument("--seq_len",       type=int,   default=96)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--device",        default="cuda")
    return p.parse_args()


# ── Data loading ───────────────────────────────────────────────────────────

def load_data(data_dir: Path, name: str):
    train  = np.load(data_dir / f"{name}_train.npy").astype(np.float32)
    test   = np.load(data_dir / f"{name}_test.npy").astype(np.float32)
    labels = np.load(data_dir / f"{name}_test_label.npy").astype(np.int32)

    ts_path = data_dir / f"{name}_test_timestamps.npy"
    timestamps = (np.load(ts_path, allow_pickle=True)
                  if ts_path.exists() else np.arange(len(test)).astype(str))

    scaler = StandardScaler()
    scaler.fit(train)
    return scaler.transform(train), scaler.transform(test), labels, timestamps


# ── Standard TSAD detection adjustment ────────────────────────────────────

def adjustment(gt, pred):
    """Credit entire anomaly segment if any point inside is detected."""
    gt, pred = list(gt), list(pred)
    in_anom = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not in_anom:
            in_anom = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1
        elif gt[i] == 0:
            in_anom = False
        if in_anom:
            pred[i] = 1
    return np.array(gt), np.array(pred)


# ── Output helpers ─────────────────────────────────────────────────────────

def report_metrics(gt, pred, label=""):
    acc = accuracy_score(gt, pred)
    p, r, f, _ = precision_recall_fscore_support(gt, pred, average="binary", zero_division=0)
    print(f"\n{'['+label+'] ' if label else ''}"
          f"Accuracy={acc:.4f}  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")
    return dict(accuracy=acc, precision=p, recall=r, f1=f)


def save_results(out_dir: Path, name: str,
                 timestamps, scores, pred, gt, metrics: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(timestamps), len(pred))

    # Per-point CSV
    df = pd.DataFrame({
        "timestamp":               timestamps[:n],
        "anomaly_score":           scores[:n],
        "is_anomaly_predicted":    pred[:n].astype(int),
        "is_anomaly_ground_truth": gt[:n].astype(int),
    })
    df.to_csv(out_dir / f"{name}_points.csv", index=False)

    # Contiguous windows
    mask        = df["is_anomaly_predicted"] == 1
    df["_blk"]  = (mask != mask.shift()).cumsum()
    wins = df[mask].groupby("_blk").agg(
        start=("timestamp",     "first"),
        end=("timestamp",       "last"),
        peak_score=("anomaly_score", "max"),
        n_points=("timestamp",  "count"),
    ).reset_index(drop=True)
    wins.to_csv(out_dir / f"{name}_windows.csv", index=False)

    # Metrics
    pd.DataFrame([metrics]).to_csv(out_dir / f"{name}_metrics.csv", index=False)

    print(f"\n✓ Saved to {out_dir}/")
    print(f"  {name}_points.csv   — {n:,} timestep anomaly scores")
    print(f"  {name}_windows.csv  — {len(wins)} anomaly windows")
    print(f"  {name}_metrics.csv")

    if len(wins):
        print("\nDetected anomaly windows:")
        print(wins.to_string(index=False))
    else:
        print("\nNo anomaly windows detected.")
        print(f"Try --anomaly_ratio higher than {args.anomaly_ratio} if ground truth has anomalies.")


# ── Isolation Forest baseline ──────────────────────────────────────────────

def run_iforest(train, test, labels, timestamps, args, out_dir, name):
    from sklearn.ensemble import IsolationForest
    print("\nRunning Isolation Forest baseline (CPU) ...")

    win = args.seq_len

    def windows(data):
        return np.array([data[i:i+win].flatten()
                         for i in range(0, len(data)-win+1, win)])

    clf = IsolationForest(n_estimators=200,
                          contamination=max(0.001, args.anomaly_ratio/100),
                          random_state=42, n_jobs=-1)
    clf.fit(windows(train))
    raw = -clf.decision_function(windows(test))

    # Expand back to point level (non-overlapping windows)
    scores = np.zeros(len(test))
    for i, start in enumerate(range(0, len(test)-win+1, win)):
        scores[start:start+win] = raw[i]

    thr  = np.percentile(scores, 100 - args.anomaly_ratio)
    pred = (scores > thr).astype(int)
    gt, pred = adjustment(labels[:len(pred)].copy(), pred.copy())

    metrics = report_metrics(gt, pred, "IForest")
    save_results(out_dir, name, timestamps, scores, pred, gt, metrics)


# ── UniTS inference ────────────────────────────────────────────────────────

def run_units(train, test, labels, timestamps, args, out_dir, name):
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(UNITS_ROOT))
    from models.UniTS import Model as UniTSModel

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nRunning UniTS on {device} ...")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)

    # Build a minimal args-like namespace from the checkpoint metadata
    import types
    margs = types.SimpleNamespace(
        seq_len    = args.seq_len,
        pred_len   = 0,
        label_len  = 0,
        enc_in     = train.shape[1],
        dec_in     = train.shape[1],
        c_out      = train.shape[1],
        d_model    = ckpt.get("d_model",    64),
        e_layers   = ckpt.get("e_layers",    3),
        patch_len  = ckpt.get("patch_len",  16),
        stride     = ckpt.get("stride",     16),
        dropout    = 0.0,
        prompt_num = ckpt.get("prompt_num", 10),
        num_task   = 1,
        task_names = ["anomaly_detection"],
    )

    model = UniTSModel(margs).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    criterion = nn.MSELoss(reduction="none")

    def score(data):
        results = []
        for start in range(0, len(data) - args.seq_len + 1, 1):
            batch = []
            for b in range(start, min(start + args.batch_size, len(data) - args.seq_len + 1)):
                batch.append(data[b:b+args.seq_len])
            x = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)
            with torch.no_grad():
                out  = model(x, None, None, None,
                             task_id=0, task_name="anomaly_detection")
                err  = torch.mean(criterion(x, out), dim=-1)  # [B, T]
                results.append(err.cpu().numpy())
            start += len(batch) - 1
        return np.concatenate(results).reshape(-1)

    print("  Scoring train set ...")
    e_train = score(train)
    print("  Scoring test set ...")
    e_test  = score(test)

    thr  = np.percentile(np.concatenate([e_train, e_test]), 100 - args.anomaly_ratio)
    print(f"  Threshold (top {args.anomaly_ratio}%) = {thr:.6f}")

    pred = (e_test > thr).astype(int)
    gt, pred = adjustment(labels[:len(pred)].copy(), pred.copy())

    metrics = report_metrics(gt, pred, "UniTS")
    save_results(out_dir, name, timestamps, e_test, pred, gt, metrics)


# ── Entry ──────────────────────────────────────────────────────────────────

def main():
    global args
    args     = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else UNITS_ROOT / "dataset" / args.name
    out_dir  = Path(args.out_dir)  if args.out_dir  else UNITS_ROOT / "results"  / args.name

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        print("       Run scripts/prepare_esa_data.py first.")
        sys.exit(1)

    print(f"Loading data from {data_dir} ...")
    train, test, labels, timestamps = load_data(data_dir, args.name)
    print(f"  Train : {train.shape}")
    print(f"  Test  : {test.shape}")
    print(f"  Labels: anomaly rate = {labels.mean()*100:.2f}%")

    if args.iforest:
        run_iforest(train, test, labels, timestamps, args, out_dir, args.name)
    else:
        if not args.ckpt:
            print("ERROR: provide --ckpt <path> or use --iforest for a baseline.")
            sys.exit(1)
        run_units(train, test, labels, timestamps, args, out_dir, args.name)


if __name__ == "__main__":
    main()