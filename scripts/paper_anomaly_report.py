#!/usr/bin/env python3
"""
paper_anomaly_report.py
-----------------------
Build a paper-friendly anomaly-detection study bundle from a TSV manifest of
UniTS runs. The script consolidates per-run point outputs into:

  - figures/*.png          Cross-mission and sweep comparison figures
  - tables/*.csv           Run summaries, mission profiles, sweep tables
  - README.md              Browseable study index with short descriptions
  - report/paper_report.pdf Central PDF summary

The manifest is expected to be a tab-separated file with at least:
  run_id, mission, mode, subsample_pct, prompt_tune_epoch, train_epochs,
  base_anomaly_ratio, dataset_dir, points_csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


COLOR_NAVY = "#16324f"
COLOR_TEAL = "#1f7a8c"
COLOR_GOLD = "#c58f00"
COLOR_RED = "#c0392b"
COLOR_GREEN = "#2e8b57"
COLOR_GRAY = "#6c757d"
COLOR_LIGHT = "#eef3f8"
COLOR_DARK = "#243b53"
PALETTE = [
    "#1f7a8c",
    "#2e8b57",
    "#4c6ef5",
    "#d97706",
    "#c0392b",
    "#7b2cbf",
    "#0f766e",
    "#475569",
]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#d0d7de",
        "axes.labelcolor": COLOR_DARK,
        "text.color": COLOR_DARK,
        "xtick.color": COLOR_DARK,
        "ytick.color": COLOR_DARK,
        "grid.color": "#dfe7ef",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.45,
        "legend.frameon": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-style anomaly report bundle from a study manifest."
    )
    parser.add_argument("--manifest", required=True, help="TSV manifest produced by the Slurm sweep")
    parser.add_argument("--study-dir", required=True, help="Output study directory")
    parser.add_argument("--title", default="UniTS Anomaly Detection Study")
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.1, 0.25, 0.5, 1.0, 2.0],
        help="Anomaly-ratio values to evaluate from each saved points CSV",
    )
    parser.add_argument(
        "--progress-bins",
        type=int,
        default=20,
        help="Number of normalized test-progress bins for the heatmap",
    )
    parser.add_argument(
        "--score-sample-size",
        type=int,
        default=5000,
        help="Per-run sample size for score-distribution figure",
    )
    parser.add_argument(
        "--top-channels",
        type=int,
        default=8,
        help="Top attributed channels to show for each reference run",
    )
    return parser.parse_args()


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    adjusted = y_pred.astype(int).copy()
    padded = np.concatenate([[0], y_true.astype(int), [0]])
    starts = np.where(np.diff(padded) == 1)[0]
    ends = np.where(np.diff(padded) == -1)[0]
    for start, end in zip(starts, ends):
        if adjusted[start:end].any():
            adjusted[start:end] = 1
    return adjusted


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep="\t")
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {path}")
    required = {"run_id", "mission", "mode", "points_csv", "dataset_dir"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    manifest = manifest.fillna("")
    return manifest


def load_points(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    required = {"timestamp", "anomaly_score", "is_anomaly_predicted", "is_anomaly_ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required point columns: {sorted(missing)}")
    return df


def contiguous_windows(mask: np.ndarray, timestamps: pd.Series, scores: np.ndarray) -> pd.DataFrame:
    mask = np.asarray(mask).astype(int)
    changes = np.diff(np.concatenate([[0], mask, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    rows = []
    for idx, (start, end) in enumerate(zip(starts, ends), start=1):
        rows.append(
            {
                "window_id": idx,
                "start_idx": int(start),
                "end_idx": int(end),
                "start": timestamps.iloc[start],
                "end": timestamps.iloc[end - 1],
                "n_points": int(end - start),
                "duration_hours": float(end - start) / 60.0,
                "peak_score": float(np.max(scores[start:end])) if end > start else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def overlap_fraction(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def compute_window_overlap_metrics(
    pred_windows: pd.DataFrame, gt_windows: pd.DataFrame
) -> tuple[float, float]:
    if gt_windows.empty:
        return float("nan"), float("nan")
    if pred_windows.empty:
        return 0.0, 0.0

    pred_hits = 0
    for _, pred_row in pred_windows.iterrows():
        hit = any(
            overlap_fraction(
                int(pred_row["start_idx"]),
                int(pred_row["end_idx"]),
                int(gt_row["start_idx"]),
                int(gt_row["end_idx"]),
            )
            > 0
            for _, gt_row in gt_windows.iterrows()
        )
        pred_hits += int(hit)

    gt_hits = 0
    for _, gt_row in gt_windows.iterrows():
        hit = any(
            overlap_fraction(
                int(pred_row["start_idx"]),
                int(pred_row["end_idx"]),
                int(gt_row["start_idx"]),
                int(gt_row["end_idx"]),
            )
            > 0
            for _, pred_row in pred_windows.iterrows()
        )
        gt_hits += int(hit)

    window_precision = pred_hits / max(len(pred_windows), 1)
    window_recall = gt_hits / max(len(gt_windows), 1)
    return window_precision, window_recall


def apply_ratio_threshold(
    scores: np.ndarray,
    baseline_pred: np.ndarray,
    ratio: float,
    baseline_ratio: float = 1.0,
) -> tuple[np.ndarray, float]:
    scores = np.asarray(scores, dtype=float)
    baseline_pred = np.asarray(baseline_pred).astype(int)
    baseline_flagged = int(baseline_pred.sum())

    if baseline_flagged > 0:
        target_n = int(round(baseline_flagged * (ratio / max(baseline_ratio, 1e-9))))
    else:
        target_n = int(round(len(scores) * (ratio / 100.0)))

    target_n = max(1, min(target_n, len(scores)))
    threshold = float(np.partition(scores, -target_n)[-target_n])
    pred = (scores >= threshold).astype(int)
    return pred, threshold


def compute_operational_heuristic(metrics: dict) -> float:
    target_flag_rate = 0.5
    separation = metrics.get("score_separation", 0.0)
    flag_penalty = abs(metrics.get("flagged_rate_pct", 0.0) - target_flag_rate)
    window_penalty = math.log1p(metrics.get("pred_windows", 0))
    duration_penalty = math.log1p(metrics.get("mean_window_points", 0.0))
    return float(separation - 0.35 * flag_penalty - 0.12 * window_penalty - 0.02 * duration_penalty)


def compute_ratio_metrics(
    df: pd.DataFrame,
    pred: np.ndarray,
    ratio: float,
    threshold: float,
) -> dict:
    y_score = df["anomaly_score"].to_numpy(dtype=float)
    y_true = df["is_anomaly_ground_truth"].to_numpy(dtype=int)
    has_gt = bool(y_true.sum() > 0)
    timestamps = df["timestamp"]

    pred_windows = contiguous_windows(pred, timestamps, y_score)
    flagged_scores = y_score[pred == 1]
    normal_scores = y_score[pred == 0]
    score_median = float(np.median(y_score))
    score_q75 = float(np.percentile(y_score, 75))
    score_q25 = float(np.percentile(y_score, 25))
    score_iqr = score_q75 - score_q25
    score_separation = 0.0
    if len(flagged_scores) and len(normal_scores):
        denom = float(np.std(normal_scores) + 1e-6)
        score_separation = float((np.mean(flagged_scores) - np.mean(normal_scores)) / denom)

    metrics = {
        "ratio": float(ratio),
        "threshold": float(threshold),
        "flagged_points": int(pred.sum()),
        "flagged_rate_pct": float(pred.mean() * 100.0),
        "pred_windows": int(len(pred_windows)),
        "mean_window_points": float(pred_windows["n_points"].mean()) if not pred_windows.empty else 0.0,
        "median_window_points": float(pred_windows["n_points"].median()) if not pred_windows.empty else 0.0,
        "max_window_points": int(pred_windows["n_points"].max()) if not pred_windows.empty else 0,
        "score_mean": float(np.mean(y_score)),
        "score_std": float(np.std(y_score)),
        "score_median": score_median,
        "score_iqr": float(score_iqr),
        "score_p95": float(np.percentile(y_score, 95)),
        "score_p99": float(np.percentile(y_score, 99)),
        "robust_tail_p99": float((np.percentile(y_score, 99) - score_median) / (score_iqr + 1e-6)),
        "score_separation": score_separation,
    }

    if not has_gt:
        metrics["heuristic_score"] = compute_operational_heuristic(metrics)
        return metrics

    adjusted_pred = point_adjust(y_true, pred)
    cm = confusion_matrix(y_true, adjusted_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    gt_windows = contiguous_windows(y_true, timestamps, y_score)
    window_precision, window_recall = compute_window_overlap_metrics(pred_windows, gt_windows)

    unique = np.unique(y_true)
    roc_auc = float("nan")
    avg_precision = float("nan")
    if len(unique) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))
        avg_precision = float(average_precision_score(y_true, y_score))

    metrics.update(
        {
            "accuracy": float(accuracy_score(y_true, adjusted_pred)),
            "precision": float(precision_score(y_true, adjusted_pred, zero_division=0)),
            "recall": float(recall_score(y_true, adjusted_pred, zero_division=0)),
            "f1": float(f1_score(y_true, adjusted_pred, zero_division=0)),
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "gt_points": int(y_true.sum()),
            "gt_rate_pct": float(y_true.mean() * 100.0),
            "gt_windows": int(len(gt_windows)),
            "window_precision": float(window_precision),
            "window_recall": float(window_recall),
        }
    )
    metrics["heuristic_score"] = compute_operational_heuristic(metrics)
    return metrics


def read_dataset_profile(mission: str, dataset_dir: Path) -> dict:
    train_path = dataset_dir / f"{mission}_train.npy"
    test_path = dataset_dir / f"{mission}_test.npy"
    channels_path = dataset_dir / f"{mission}_channels.txt"

    train_rows = test_rows = channels = 0
    if train_path.exists():
        train_arr = np.load(train_path, mmap_mode="r")
        train_rows = int(train_arr.shape[0])
        channels = int(train_arr.shape[1]) if train_arr.ndim == 2 else 0
    if test_path.exists():
        test_arr = np.load(test_path, mmap_mode="r")
        test_rows = int(test_arr.shape[0])
        if not channels and test_arr.ndim == 2:
            channels = int(test_arr.shape[1])
    channel_names = []
    if channels_path.exists():
        channel_names = channels_path.read_text().splitlines()
        if channel_names:
            channels = len(channel_names)

    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "channels": channels,
        "channel_names": channel_names,
    }


def select_best_ratio_state(ratio_df: pd.DataFrame, has_gt: bool) -> pd.Series:
    if has_gt:
        ordered = ratio_df.sort_values(
            ["f1", "recall", "precision", "flagged_rate_pct"],
            ascending=[False, False, False, True],
        )
    else:
        ordered = ratio_df.sort_values(
            ["heuristic_score", "score_separation", "flagged_rate_pct"],
            ascending=[False, False, True],
        )
    return ordered.iloc[0]


def summarize_run(
    manifest_row: pd.Series,
    ratios: list[float],
    progress_bins: int,
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mission = manifest_row["mission"]
    run_id = manifest_row["run_id"]
    points_path = Path(manifest_row["points_csv"])
    dataset_dir = Path(manifest_row["dataset_dir"])
    df = load_points(points_path)
    profile = read_dataset_profile(mission, dataset_dir)

    baseline_pred = df["is_anomaly_predicted"].to_numpy(dtype=int)
    baseline_ratio = float(manifest_row.get("base_anomaly_ratio") or 1.0)
    has_gt = bool(df["is_anomaly_ground_truth"].sum() > 0)

    ratio_rows = []
    for ratio in ratios:
        pred, threshold = apply_ratio_threshold(
            df["anomaly_score"].to_numpy(dtype=float),
            baseline_pred,
            ratio=ratio,
            baseline_ratio=baseline_ratio,
        )
        metrics = compute_ratio_metrics(df, pred, ratio=ratio, threshold=threshold)
        metrics.update(
            {
                "run_id": run_id,
                "mission": mission,
                "mode": manifest_row["mode"],
                "subsample_pct": float(manifest_row.get("subsample_pct") or 0.0),
                "prompt_tune_epoch": int(float(manifest_row.get("prompt_tune_epoch") or 0)),
                "train_epochs": int(float(manifest_row.get("train_epochs") or 0)),
                "points_csv": str(points_path),
                "dataset_dir": str(dataset_dir),
                "has_ground_truth": has_gt,
            }
        )
        ratio_rows.append(metrics)

    ratio_df = pd.DataFrame(ratio_rows)
    selected = select_best_ratio_state(ratio_df, has_gt)

    test_timestamps = df["timestamp"]
    progress_groups = pd.cut(
        np.arange(len(df)),
        bins=progress_bins,
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    ratio_for_profile = selected["ratio"]
    pred_for_profile, _ = apply_ratio_threshold(
        df["anomaly_score"].to_numpy(dtype=float),
        baseline_pred,
        ratio=ratio_for_profile,
        baseline_ratio=baseline_ratio,
    )
    progression = (
        pd.DataFrame({"bin": progress_groups, "pred": pred_for_profile})
        .groupby("bin", observed=False)["pred"]
        .mean()
        .reset_index()
    )
    progression["progress_bin"] = progression["bin"].astype(int) + 1
    progression["flagged_rate_pct"] = progression["pred"] * 100.0
    progression["run_id"] = run_id
    progression["mission"] = mission

    robust_scores = (
        (df["anomaly_score"].to_numpy(dtype=float) - selected["score_median"])
        / (selected["score_iqr"] + 1e-6)
    )
    if len(robust_scores) > sample_size:
        sample_idx = rng.choice(len(robust_scores), size=sample_size, replace=False)
        robust_scores = robust_scores[sample_idx]
    score_sample = pd.DataFrame(
        {
            "run_id": run_id,
            "mission": mission,
            "robust_score": robust_scores,
        }
    )

    summary = {
        "run_id": run_id,
        "mission": mission,
        "mode": manifest_row["mode"],
        "subsample_pct": float(manifest_row.get("subsample_pct") or 0.0),
        "prompt_tune_epoch": int(float(manifest_row.get("prompt_tune_epoch") or 0)),
        "train_epochs": int(float(manifest_row.get("train_epochs") or 0)),
        "points_csv": str(points_path),
        "dataset_dir": str(dataset_dir),
        "has_ground_truth": has_gt,
        "train_rows": profile["train_rows"],
        "test_rows": profile["test_rows"],
        "channels": profile["channels"],
        "test_start": test_timestamps.min(),
        "test_end": test_timestamps.max(),
        "test_days": float((test_timestamps.max() - test_timestamps.min()).total_seconds() / 86400.0),
        "base_anomaly_ratio": baseline_ratio,
        "selected_ratio": float(selected["ratio"]),
        "selected_threshold": float(selected["threshold"]),
        "selected_flagged_points": int(selected["flagged_points"]),
        "selected_flagged_rate_pct": float(selected["flagged_rate_pct"]),
        "selected_pred_windows": int(selected["pred_windows"]),
        "selected_mean_window_points": float(selected["mean_window_points"]),
        "selected_score_separation": float(selected["score_separation"]),
        "selected_score_p99": float(selected["score_p99"]),
        "selected_robust_tail_p99": float(selected["robust_tail_p99"]),
        "selected_heuristic_score": float(selected["heuristic_score"]),
        "selection_objective": "f1" if has_gt else "heuristic_score",
        "selection_value": float(selected["f1"] if has_gt else selected["heuristic_score"]),
    }

    if has_gt:
        summary.update(
            {
                "selected_accuracy": float(selected["accuracy"]),
                "selected_precision": float(selected["precision"]),
                "selected_recall": float(selected["recall"]),
                "selected_f1": float(selected["f1"]),
                "selected_roc_auc": float(selected["roc_auc"]),
                "selected_avg_precision": float(selected["avg_precision"]),
                "selected_gt_rate_pct": float(selected["gt_rate_pct"]),
                "selected_window_precision": float(selected["window_precision"]),
                "selected_window_recall": float(selected["window_recall"]),
                "selected_tp": int(selected["tp"]),
                "selected_fp": int(selected["fp"]),
                "selected_fn": int(selected["fn"]),
                "selected_tn": int(selected["tn"]),
            }
        )

    return summary, ratio_df, progression, score_sample


def compute_top_channels_for_reference_runs(
    reference_df: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    rows = []
    for _, row in reference_df.iterrows():
        mission = row["mission"]
        dataset_dir = Path(row["dataset_dir"])
        points_df = load_points(Path(row["points_csv"]))
        scores = points_df["anomaly_score"].to_numpy(dtype=float)
        baseline_pred = points_df["is_anomaly_predicted"].to_numpy(dtype=int)
        pred, _ = apply_ratio_threshold(scores, baseline_pred, row["selected_ratio"], row["base_anomaly_ratio"])

        test_path = dataset_dir / f"{mission}_test.npy"
        channels_path = dataset_dir / f"{mission}_channels.txt"
        if not test_path.exists():
            continue

        test_arr = np.load(test_path)
        n_common = min(len(pred), test_arr.shape[0])
        if n_common <= 0:
            continue
        pred = pred[:n_common]
        test_arr = test_arr[:n_common]
        if channels_path.exists():
            channel_names = channels_path.read_text().splitlines()
        else:
            channel_names = [f"channel_{idx}" for idx in range(test_arr.shape[1])]

        normal_data = test_arr[pred == 0]
        if len(normal_data) == 0:
            normal_data = test_arr

        medians = np.median(normal_data, axis=0)
        q75 = np.percentile(normal_data, 75, axis=0)
        q25 = np.percentile(normal_data, 25, axis=0)
        iqrs = q75 - q25

        window_df = contiguous_windows(pred, points_df["timestamp"].iloc[:n_common], scores[:n_common])
        if window_df.empty:
            continue

        channel_scores = np.zeros(test_arr.shape[1], dtype=float)
        window_count = 0
        for _, window in window_df.iterrows():
            start_idx = int(window["start_idx"])
            end_idx = int(window["end_idx"])
            window_mean = test_arr[start_idx:end_idx].mean(axis=0)
            deviation = np.abs(window_mean - medians) / (iqrs + 1e-6)
            channel_scores += deviation
            window_count += 1

        if window_count == 0:
            continue
        channel_scores /= window_count
        top_idx = np.argsort(channel_scores)[::-1][:top_n]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "mission": mission,
                    "run_id": row["run_id"],
                    "rank": rank,
                    "channel": channel_names[idx] if idx < len(channel_names) else f"channel_{idx}",
                    "mean_deviation_score": float(channel_scores[idx]),
                }
            )

    return pd.DataFrame(rows)


def choose_reference_runs(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    reference_rows = []
    for mission, mission_df in run_summary_df.groupby("mission", sort=False):
        mission_df = mission_df.copy()
        if bool(mission_df["has_ground_truth"].iloc[0]):
            mission_df = mission_df.sort_values(
                ["selected_f1", "selected_recall", "selected_precision", "selected_flagged_rate_pct"],
                ascending=[False, False, False, True],
            )
        else:
            mission_df = mission_df.sort_values(
                ["selected_heuristic_score", "selected_score_separation", "selected_flagged_rate_pct"],
                ascending=[False, False, True],
            )
        reference_rows.append(mission_df.iloc[0])
    return pd.DataFrame(reference_rows).reset_index(drop=True)


def ensure_output_dirs(study_dir: Path) -> dict[str, Path]:
    dirs = {
        "study": study_dir,
        "figures": study_dir / "figures",
        "tables": study_dir / "tables",
        "report": study_dir / "report",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reference_overview(reference_df: pd.DataFrame, out_path: Path) -> None:
    order = reference_df.sort_values("mission")["mission"].tolist()
    plot_df = reference_df.set_index("mission").loc[order]
    x = np.arange(len(order))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].bar(x, plot_df["channels"], color=PALETTE[: len(order)])
    axes[0, 0].set_xticks(x, order, rotation=20, ha="right")
    axes[0, 0].set_title("Channels per Mission")
    axes[0, 0].set_ylabel("Channels")

    axes[0, 1].bar(x, plot_df["selected_flagged_rate_pct"], color=COLOR_TEAL, label="Predicted")
    labelled = plot_df["has_ground_truth"].astype(bool)
    if labelled.any():
        gt_x = x[labelled.to_numpy()]
        gt_vals = plot_df.loc[labelled, "selected_gt_rate_pct"]
        axes[0, 1].plot(gt_x, gt_vals, color=COLOR_RED, marker="o", linewidth=2, label="Ground truth")
    axes[0, 1].set_xticks(x, order, rotation=20, ha="right")
    axes[0, 1].set_title("Selected-Ratio Anomaly Rate")
    axes[0, 1].set_ylabel("Percent of test points")
    axes[0, 1].legend(loc="upper right")

    axes[1, 0].bar(x, plot_df["selected_pred_windows"], color=COLOR_GOLD)
    axes[1, 0].set_xticks(x, order, rotation=20, ha="right")
    axes[1, 0].set_title("Predicted Windows")
    axes[1, 0].set_ylabel("Contiguous windows")

    axes[1, 1].bar(x, plot_df["selected_score_separation"], color=COLOR_NAVY)
    axes[1, 1].set_xticks(x, order, rotation=20, ha="right")
    axes[1, 1].set_title("Score Separation")
    axes[1, 1].set_ylabel("(flagged mean - normal mean) / normal std")

    fig.suptitle("Reference Run Overview", fontsize=16, fontweight="bold", y=1.01)
    save_figure(fig, out_path)


def plot_heatmaps(
    run_summary_df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
) -> None:
    missions = run_summary_df["mission"].drop_duplicates().tolist()
    if not missions:
        return

    n_cols = 2
    n_rows = int(math.ceil(len(missions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, max(4.5, 4.2 * n_rows)))
    axes = np.atleast_1d(axes).flatten()

    for ax, mission in zip(axes, missions):
        mission_df = run_summary_df[run_summary_df["mission"] == mission]
        pivot = mission_df.pivot_table(
            index="subsample_pct",
            columns="prompt_tune_epoch",
            values=value_col,
            aggfunc="max",
        ).sort_index().sort_index(axis=1)
        if pivot.empty:
            ax.axis("off")
            continue
        im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [str(int(c)) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{idx:.2f}" for idx in pivot.index])
        ax.set_xlabel("Prompt-tune epochs")
        ax.set_ylabel("Training fraction")
        ax.set_title(mission)
        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = pivot.iat[row_idx, col_idx]
                if pd.isna(value):
                    continue
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(missions) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    save_figure(fig, out_path)


def plot_ratio_sensitivity(
    ratio_metrics_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    out_path: Path,
) -> None:
    ref_ids = set(reference_df["run_id"])
    subset = ratio_metrics_df[ratio_metrics_df["run_id"].isin(ref_ids)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))

    for idx, (_, row) in enumerate(reference_df.iterrows()):
        mission = row["mission"]
        run_id = row["run_id"]
        color = PALETTE[idx % len(PALETTE)]
        mission_df = subset[subset["run_id"] == run_id].sort_values("ratio")
        axes[0].plot(
            mission_df["ratio"],
            mission_df["flagged_rate_pct"],
            marker="o",
            linewidth=2,
            color=color,
            label=mission,
        )
        metric_col = "f1" if row["has_ground_truth"] else "score_separation"
        axes[1].plot(
            mission_df["ratio"],
            mission_df[metric_col],
            marker="o",
            linewidth=2,
            color=color,
            label=mission,
        )

    axes[0].set_title("Flagged Rate vs Anomaly Ratio")
    axes[0].set_xlabel("Anomaly ratio")
    axes[0].set_ylabel("Flagged rate (%)")
    axes[1].set_title("Quality Metric vs Anomaly Ratio")
    axes[1].set_xlabel("Anomaly ratio")
    axes[1].set_ylabel("F1 (labeled) or score separation (unlabeled)")
    axes[0].legend(loc="best")
    fig.suptitle("Reference-Run Ratio Sensitivity", fontsize=16, fontweight="bold", y=1.03)
    save_figure(fig, out_path)


def plot_score_distributions(score_sample_df: pd.DataFrame, reference_df: pd.DataFrame, out_path: Path) -> None:
    ref_ids = set(reference_df["run_id"])
    subset = score_sample_df[score_sample_df["run_id"].isin(ref_ids)].copy()
    if subset.empty:
        return

    missions = reference_df["mission"].tolist()
    data = [subset.loc[subset["mission"] == mission, "robust_score"].to_numpy() for mission in missions]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    parts = ax.boxplot(data, patch_artist=True, labels=missions, showfliers=False)
    for patch, color in zip(parts["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axhline(0.0, color=COLOR_GRAY, linewidth=1, linestyle="--")
    ax.set_ylabel("Robust anomaly score")
    ax.set_title("Reference-Run Score Distributions")
    plt.xticks(rotation=18, ha="right")
    save_figure(fig, out_path)


def plot_progress_heatmap(
    progression_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    progress_bins: int,
    out_path: Path,
) -> None:
    ref_ids = set(reference_df["run_id"])
    subset = progression_df[progression_df["run_id"].isin(ref_ids)].copy()
    if subset.empty:
        return

    missions = reference_df["mission"].tolist()
    heatmap = np.zeros((len(missions), progress_bins), dtype=float)
    heatmap[:] = np.nan
    for row_idx, mission in enumerate(missions):
        mission_df = subset[subset["mission"] == mission]
        for _, row in mission_df.iterrows():
            bin_idx = int(row["progress_bin"]) - 1
            if 0 <= bin_idx < progress_bins:
                heatmap[row_idx, bin_idx] = float(row["flagged_rate_pct"])

    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(missions)), missions)
    ax.set_xticks(range(progress_bins), [str(idx + 1) for idx in range(progress_bins)])
    ax.set_xlabel("Normalized test-progress bin")
    ax.set_title("Flagged Rate Across Test Progress")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Flagged rate (%)")
    save_figure(fig, out_path)


def plot_top_channels(top_channels_df: pd.DataFrame, out_path: Path) -> None:
    if top_channels_df.empty:
        return

    missions = top_channels_df["mission"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(missions), 1, figsize=(12, max(4.0, 3.2 * len(missions))))
    axes = np.atleast_1d(axes)

    for ax, mission in zip(axes, missions):
        mission_df = top_channels_df[top_channels_df["mission"] == mission].sort_values(
            "mean_deviation_score", ascending=True
        )
        ax.barh(mission_df["channel"], mission_df["mean_deviation_score"], color=COLOR_TEAL)
        ax.set_title(mission)
        ax.set_xlabel("Average deviation score across predicted windows")

    fig.suptitle("Top Attributed Channels in Reference Runs", fontsize=16, fontweight="bold", y=1.01)
    save_figure(fig, out_path)


def make_key_findings(reference_df: pd.DataFrame) -> list[str]:
    findings = []
    labelled = reference_df[reference_df["has_ground_truth"].astype(bool)]
    if not labelled.empty:
        best = labelled.sort_values("selected_f1", ascending=False).iloc[0]
        findings.append(
            f"Best labeled configuration: {best['mission']} reached F1 {best['selected_f1']:.3f} "
            f"at training fraction {best['subsample_pct']:.2f}, prompt tuning {int(best['prompt_tune_epoch'])} epochs, "
            f"and anomaly ratio {best['selected_ratio']:.2f}."
        )

    widest = reference_df.sort_values("selected_flagged_rate_pct", ascending=False).iloc[0]
    findings.append(
        f"Highest selected anomaly burden: {widest['mission']} flagged {widest['selected_flagged_rate_pct']:.2f}% "
        f"of its test points across {int(widest['selected_pred_windows'])} windows."
    )

    sharpest = reference_df.sort_values("selected_score_separation", ascending=False).iloc[0]
    findings.append(
        f"Strongest score separation: {sharpest['mission']} produced a flagged-vs-normal separation score of "
        f"{sharpest['selected_score_separation']:.2f}, which is useful when ground truth is unavailable."
    )
    return findings


def relative_link(base_dir: Path, target: Path) -> str:
    return str(target.relative_to(base_dir))


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = df.fillna("").astype(str).values.tolist()
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(row) + " |")
    return "\n".join(table)


def write_readme(
    study_dir: Path,
    title: str,
    figure_catalog: list[tuple[str, str]],
    table_catalog: list[tuple[str, str]],
    reference_df: pd.DataFrame,
    findings: list[str],
) -> None:
    lines = [f"# {title}", "", "## Key Findings", ""]
    for finding in findings:
        lines.append(f"- {finding}")

    lines.extend(["", "## Reference Runs", ""])
    ref_table = reference_df.copy()
    display_cols = [
        "mission",
        "mode",
        "subsample_pct",
        "prompt_tune_epoch",
        "selected_ratio",
        "selected_flagged_rate_pct",
        "selected_pred_windows",
    ]
    if "selected_f1" in ref_table.columns:
        display_cols.append("selected_f1")
    ref_table = ref_table[display_cols]
    lines.append(dataframe_to_markdown(ref_table))

    lines.extend(["", "## Figures", ""])
    for rel_path, description in figure_catalog:
        lines.append(f"- [{Path(rel_path).name}]({rel_path}): {description}")

    lines.extend(["", "## Tables", ""])
    for rel_path, description in table_catalog:
        lines.append(f"- [{Path(rel_path).name}]({rel_path}): {description}")

    readme_path = study_dir / "README.md"
    readme_path.write_text("\n".join(lines) + "\n")


def _make_table(df: pd.DataFrame, max_rows: int = 12) -> Table:
    clipped = df.head(max_rows).copy()
    clipped = clipped.fillna("")
    clipped.columns = [str(col) for col in clipped.columns]
    rows = [list(clipped.columns)] + clipped.astype(str).values.tolist()
    col_width = 1.25 * inch
    table = Table(rows, colWidths=[col_width] * len(clipped.columns))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(COLOR_NAVY)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_pdf(
    title: str,
    study_dir: Path,
    figure_catalog: list[tuple[str, str]],
    mission_profiles_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    best_labelled_df: pd.DataFrame,
    best_unlabelled_df: pd.DataFrame,
    findings: list[str],
) -> None:
    pdf_path = study_dir / "report" / "paper_report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(letter),
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "paper_title",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor(COLOR_NAVY),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "paper_subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.HexColor(COLOR_GRAY),
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "paper_h1",
        parent=styles["Heading1"],
        fontSize=13,
        textColor=colors.HexColor(COLOR_NAVY),
        spaceBefore=10,
        spaceAfter=6,
    )
    note = ParagraphStyle(
        "paper_note",
        parent=styles["Normal"],
        fontSize=8.5,
        textColor=colors.HexColor(COLOR_GRAY),
        leftIndent=8,
        spaceAfter=8,
    )

    story = [
        Paragraph(title, title_style),
        Paragraph(f"Generated from {len(reference_df)} reference runs and {len(mission_profiles_df)} missions", subtitle_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=10),
        Paragraph("Key Findings", h1),
    ]
    for finding in findings:
        story.append(Paragraph(f"- {finding}", styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Mission Profiles", h1))
    mission_table = mission_profiles_df[
        ["mission", "channels", "train_rows", "test_rows", "test_days"]
    ].copy()
    mission_table["test_days"] = mission_table["test_days"].map(lambda x: f"{x:.1f}")
    story.append(_make_table(mission_table, max_rows=20))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Reference Configurations", h1))
    ref_cols = [
        "mission",
        "mode",
        "subsample_pct",
        "prompt_tune_epoch",
        "selected_ratio",
        "selected_flagged_rate_pct",
        "selected_pred_windows",
    ]
    if "selected_f1" in reference_df.columns:
        ref_cols.append("selected_f1")
    ref_table = reference_df[ref_cols].copy()
    for col in ["subsample_pct", "selected_ratio", "selected_flagged_rate_pct"]:
        if col in ref_table.columns:
            ref_table[col] = ref_table[col].map(lambda x: f"{float(x):.3f}")
    story.append(_make_table(ref_table, max_rows=20))
    story.append(PageBreak())

    existing_figures = [
        (rel_path, description)
        for rel_path, description in figure_catalog
        if (study_dir / rel_path).exists()
    ]
    for index, (rel_path, description) in enumerate(existing_figures):
        image_path = study_dir / rel_path
        story.append(Paragraph(Path(rel_path).stem.replace("_", " ").title(), h1))
        story.append(
            RLImage(
                str(image_path),
                width=9.35 * inch,
                height=5.0 * inch,
            )
        )
        story.append(Spacer(1, 6))
        story.append(Paragraph(description, note))
        if index < len(existing_figures) - 1:
            story.append(PageBreak())

    if not best_labelled_df.empty:
        story.append(Paragraph("Top Labelled Configurations", h1))
        labelled_view = best_labelled_df[
            [
                "mission",
                "subsample_pct",
                "prompt_tune_epoch",
                "selected_ratio",
                "selected_f1",
                "selected_precision",
                "selected_recall",
            ]
        ].copy()
        for col in labelled_view.columns:
            if col != "mission":
                labelled_view[col] = labelled_view[col].map(lambda x: f"{float(x):.3f}")
        story.append(_make_table(labelled_view, max_rows=12))
        story.append(Spacer(1, 10))

    if not best_unlabelled_df.empty:
        story.append(Paragraph("Top Unlabelled Configurations", h1))
        story.append(
            Paragraph(
                "These rows are ranked by a descriptive heuristic that rewards score separation and discourages overly noisy flag rates. "
                "They are for triage only, not for supervised claims.",
                note,
            )
        )
        unlabelled_view = best_unlabelled_df[
            [
                "mission",
                "subsample_pct",
                "prompt_tune_epoch",
                "selected_ratio",
                "selected_heuristic_score",
                "selected_flagged_rate_pct",
                "selected_score_separation",
            ]
        ].copy()
        for col in unlabelled_view.columns:
            if col != "mission":
                unlabelled_view[col] = unlabelled_view[col].map(lambda x: f"{float(x):.3f}")
        story.append(_make_table(unlabelled_view, max_rows=12))

    doc.build(story)


def main() -> None:
    args = parse_args()
    study_dir = Path(args.study_dir)
    dirs = ensure_output_dirs(study_dir)
    manifest_df = load_manifest(Path(args.manifest))

    rng = np.random.default_rng(42)
    run_summaries = []
    ratio_tables = []
    progress_tables = []
    score_samples = []

    for _, row in manifest_df.iterrows():
        points_path = Path(row["points_csv"])
        if not points_path.exists():
            continue
        summary, ratio_df, progression_df, sample_df = summarize_run(
            row,
            ratios=args.ratios,
            progress_bins=args.progress_bins,
            sample_size=args.score_sample_size,
            rng=rng,
        )
        run_summaries.append(summary)
        ratio_tables.append(ratio_df)
        progress_tables.append(progression_df)
        score_samples.append(sample_df)

    if not run_summaries:
        raise ValueError("No valid runs were found in the manifest.")

    run_summary_df = pd.DataFrame(run_summaries)
    ratio_metrics_df = pd.concat(ratio_tables, ignore_index=True)
    progression_df = pd.concat(progress_tables, ignore_index=True)
    score_sample_df = pd.concat(score_samples, ignore_index=True)
    reference_df = choose_reference_runs(run_summary_df)
    mission_profiles_df = (
        run_summary_df.sort_values(["mission", "selection_value"], ascending=[True, False])
        .drop_duplicates("mission")
        .loc[:, ["mission", "channels", "train_rows", "test_rows", "test_days", "has_ground_truth"]]
        .reset_index(drop=True)
    )
    top_channels_df = compute_top_channels_for_reference_runs(reference_df, top_n=args.top_channels)

    best_labelled_df = (
        run_summary_df[run_summary_df["has_ground_truth"].astype(bool)]
        .sort_values(["selected_f1", "selected_recall", "selected_precision"], ascending=[False, False, False])
        .groupby("mission", sort=False)
        .head(3)
        .reset_index(drop=True)
    )
    best_unlabelled_df = (
        run_summary_df[~run_summary_df["has_ground_truth"].astype(bool)]
        .sort_values(["selected_heuristic_score", "selected_score_separation"], ascending=[False, False])
        .groupby("mission", sort=False)
        .head(3)
        .reset_index(drop=True)
    )

    run_summary_path = dirs["tables"] / "run_summary.csv"
    ratio_metrics_path = dirs["tables"] / "ratio_metrics.csv"
    reference_path = dirs["tables"] / "reference_runs.csv"
    profiles_path = dirs["tables"] / "mission_profiles.csv"
    top_channels_path = dirs["tables"] / "reference_top_channels.csv"
    labelled_path = dirs["tables"] / "best_labelled_configs.csv"
    unlabelled_path = dirs["tables"] / "best_unlabelled_configs.csv"

    run_summary_df.to_csv(run_summary_path, index=False)
    ratio_metrics_df.to_csv(ratio_metrics_path, index=False)
    reference_df.to_csv(reference_path, index=False)
    mission_profiles_df.to_csv(profiles_path, index=False)
    if not top_channels_df.empty:
        top_channels_df.to_csv(top_channels_path, index=False)
    if not best_labelled_df.empty:
        best_labelled_df.to_csv(labelled_path, index=False)
    if not best_unlabelled_df.empty:
        best_unlabelled_df.to_csv(unlabelled_path, index=False)

    figures = []

    fig_path = dirs["figures"] / "figure_01_reference_overview.png"
    plot_reference_overview(reference_df, fig_path)
    figures.append(
        (
            relative_link(study_dir, fig_path),
            "Overview of the selected reference configuration for each mission: channel counts, selected anomaly rates, predicted windows, and score separation.",
        )
    )

    fig_path = dirs["figures"] / "figure_02_ratio_sensitivity.png"
    plot_ratio_sensitivity(ratio_metrics_df, reference_df, fig_path)
    figures.append(
        (
            relative_link(study_dir, fig_path),
            "Sensitivity of each reference run to anomaly-ratio thresholding. Left: flagged rate. Right: F1 for labeled missions or score separation for unlabeled missions.",
        )
    )

    fig_path = dirs["figures"] / "figure_03_score_distributions.png"
    plot_score_distributions(score_sample_df, reference_df, fig_path)
    figures.append(
        (
            relative_link(study_dir, fig_path),
            "Distribution of robust anomaly scores in the selected reference runs. Wider or higher-score boxes indicate heavier tails or sharper separation.",
        )
    )

    fig_path = dirs["figures"] / "figure_04_progress_heatmap.png"
    plot_progress_heatmap(progression_df, reference_df, args.progress_bins, fig_path)
    figures.append(
        (
            relative_link(study_dir, fig_path),
            "Flagged-rate heatmap across normalized test progress. This makes missions with different calendar dates directly comparable.",
        )
    )

    labelled_heatmap_df = run_summary_df[
        run_summary_df["has_ground_truth"].astype(bool) & (run_summary_df["mode"] == "prompt_tuning")
    ].copy()
    if not labelled_heatmap_df.empty:
        fig_path = dirs["figures"] / "figure_05_labelled_prompt_heatmaps.png"
        plot_heatmaps(
            labelled_heatmap_df,
            value_col="selected_f1",
            title="Best F1 Across Prompt-Tuning Settings",
            out_path=fig_path,
        )
        figures.append(
            (
                relative_link(study_dir, fig_path),
                "Heatmaps for labeled missions showing the best achievable F1 in each training-fraction and prompt-epoch cell after ratio selection.",
            )
        )

    unlabelled_heatmap_df = run_summary_df[
        ~run_summary_df["has_ground_truth"].astype(bool) & (run_summary_df["mode"] == "prompt_tuning")
    ].copy()
    if not unlabelled_heatmap_df.empty:
        fig_path = dirs["figures"] / "figure_06_unlabelled_prompt_heatmaps.png"
        plot_heatmaps(
            unlabelled_heatmap_df,
            value_col="selected_heuristic_score",
            title="Unlabelled Operational Heuristic Across Prompt-Tuning Settings",
            out_path=fig_path,
        )
        figures.append(
            (
                relative_link(study_dir, fig_path),
                "Heatmaps for unlabeled missions using a descriptive heuristic that favors sharp score separation and operationally manageable alert volumes.",
            )
        )

    if not top_channels_df.empty:
        fig_path = dirs["figures"] / "figure_07_top_channels.png"
        plot_top_channels(top_channels_df, fig_path)
        figures.append(
            (
                relative_link(study_dir, fig_path),
                "Average per-channel deviation in the selected reference runs. These are the leading telemetry channels behind predicted anomaly windows.",
            )
        )

    tables = [
        (
            relative_link(study_dir, run_summary_path),
            "One row per run with selected-ratio metrics, dataset sizes, and prompt-tuning settings.",
        ),
        (
            relative_link(study_dir, ratio_metrics_path),
            "Expanded threshold sweep table with one row per run and anomaly-ratio setting.",
        ),
        (
            relative_link(study_dir, reference_path),
            "The single reference configuration chosen for each mission for cross-mission figures.",
        ),
        (
            relative_link(study_dir, profiles_path),
            "Mission-level dataset profile with channel counts and train/test sizes.",
        ),
    ]
    if top_channels_path.exists():
        tables.append(
            (
                relative_link(study_dir, top_channels_path),
                "Top attributed telemetry channels for the selected reference runs.",
            )
        )
    if labelled_path.exists():
        tables.append(
            (
                relative_link(study_dir, labelled_path),
                "Top labeled configurations per mission, ranked by F1.",
            )
        )
    if unlabelled_path.exists():
        tables.append(
            (
                relative_link(study_dir, unlabelled_path),
                "Top unlabeled configurations per mission, ranked by the operational heuristic.",
            )
        )

    findings = make_key_findings(reference_df)
    write_readme(
        study_dir=study_dir,
        title=args.title,
        figure_catalog=figures,
        table_catalog=tables,
        reference_df=reference_df,
        findings=findings,
    )
    build_pdf(
        title=args.title,
        study_dir=study_dir,
        figure_catalog=figures,
        mission_profiles_df=mission_profiles_df,
        reference_df=reference_df,
        best_labelled_df=best_labelled_df,
        best_unlabelled_df=best_unlabelled_df,
        findings=findings,
    )

    print(f"Study report written to {study_dir}")
    print(f"README : {study_dir / 'README.md'}")
    print(f"PDF    : {study_dir / 'report' / 'paper_report.pdf'}")


if __name__ == "__main__":
    main()
