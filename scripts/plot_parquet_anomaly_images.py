#!/usr/bin/env python3
"""
Create useful anomaly-detection figures directly from telemetry Parquet exports.

The full pipeline writes telemetry Parquets with one row per test timestamp plus:
timestamp, anomaly_score, is_anomaly_predicted, is_anomaly_ground_truth.
Those columns are enough to make metric-aligned plots without rerunning UniTS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


COLOR_NAVY = "#16324f"
COLOR_TEAL = "#1f7a8c"
COLOR_GOLD = "#c58f00"
COLOR_RED = "#c0392b"
COLOR_GREEN = "#2e8b57"
COLOR_GRAY = "#6c757d"
COLOR_DARK = "#243b53"

METRIC_COLUMNS = [
    "timestamp",
    "anomaly_score",
    "is_anomaly_predicted",
    "is_anomaly_ground_truth",
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
        description="Generate per-mission anomaly figures from telemetry Parquet files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory used for default Parquet discovery and output.",
    )
    parser.add_argument(
        "--missions",
        nargs="*",
        help="Optional mission names to include when discovering Parquets from --results-dir.",
    )
    parser.add_argument(
        "--parquet",
        action="append",
        default=[],
        metavar="MISSION=PATH",
        help=(
            "Telemetry Parquet. Repeat for multiple missions. If omitted, discovers "
            "results/<MISSION>/telemetry/<MISSION>_telemetry.parquet."
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional reference_runs/run_summary CSV. If present, selected_ratio is reapplied.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Directory for generated PNGs and summary CSV. If omitted, per-mission PNGs go to "
            "results/<MISSION>/figures/parquet_anomaly_images and cross-mission files go to "
            "results/figures/parquet_anomaly_images."
        ),
    )
    parser.add_argument("--bins", type=int, default=80, help="Timeline bins for rate plots.")
    parser.add_argument(
        "--score-sample",
        type=int,
        default=250_000,
        help="Maximum points sampled for score histograms.",
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


def apply_ratio_threshold(
    scores: np.ndarray,
    baseline_pred: np.ndarray,
    ratio: float,
    baseline_ratio: float,
) -> np.ndarray:
    baseline_flagged = int(baseline_pred.sum())
    if baseline_flagged > 0:
        target_n = int(round(baseline_flagged * (ratio / max(baseline_ratio, 1e-9))))
    else:
        target_n = int(round(len(scores) * (ratio / 100.0)))
    target_n = max(1, min(target_n, len(scores)))
    threshold = float(np.partition(scores, -target_n)[-target_n])
    return (scores >= threshold).astype(int)


def parse_parquets(specs: list[str]) -> dict[str, Path]:
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--parquet must look like MISSION=PATH, got: {spec}")
        mission, path = spec.split("=", 1)
        parsed[mission.strip()] = Path(path).expanduser()
    return parsed


def discover_parquets(results_dir: Path, missions: list[str] | None) -> dict[str, Path]:
    if not results_dir.exists():
        return {}
    search_roots = [results_dir / mission for mission in missions] if missions else sorted(results_dir.iterdir())
    parsed = {}
    for root in search_roots:
        if not root.is_dir():
            continue
        mission = root.name
        candidates = sorted((root / "telemetry").glob("*_telemetry.parquet"))
        if candidates:
            parsed[mission] = candidates[-1]
    return parsed


def mission_out_dir(results_dir: Path, mission: str, shared_out_dir: Path | None) -> Path:
    if shared_out_dir is not None:
        return shared_out_dir / mission
    return results_dir / mission / "figures" / "parquet_anomaly_images"


def overview_out_dir(results_dir: Path, shared_out_dir: Path | None) -> Path:
    if shared_out_dir is not None:
        return shared_out_dir
    return results_dir / "figures" / "parquet_anomaly_images"


def load_summary(path: Path | None) -> dict[str, pd.Series]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    return {str(row["mission"]): row for _, row in df.iterrows()}


def selected_prediction(df: pd.DataFrame, mission: str, summary: dict[str, pd.Series]) -> np.ndarray:
    baseline_pred = df["is_anomaly_predicted"].to_numpy(dtype=int)
    row = summary.get(mission)
    if row is None or "selected_ratio" not in row.index or "base_anomaly_ratio" not in row.index:
        return baseline_pred
    return apply_ratio_threshold(
        df["anomaly_score"].to_numpy(dtype=float),
        baseline_pred,
        float(row["selected_ratio"]),
        float(row["base_anomaly_ratio"]),
    )


def compute_summary(mission: str, df: pd.DataFrame, raw_pred: np.ndarray) -> dict:
    y_true = df["is_anomaly_ground_truth"].to_numpy(dtype=int)
    adjusted_pred = point_adjust(y_true, raw_pred) if y_true.sum() else raw_pred
    total = len(df)
    tp = int(((y_true == 1) & (adjusted_pred == 1)).sum())
    fp = int(((y_true == 0) & (adjusted_pred == 1)).sum())
    fn = int(((y_true == 1) & (adjusted_pred == 0)).sum())
    return {
        "mission": mission,
        "n_points": total,
        "raw_pred_rate_pct": 100.0 * float(raw_pred.mean()),
        "adjusted_pred_rate_pct": 100.0 * float(adjusted_pred.mean()),
        "gt_rate_pct": 100.0 * float(y_true.mean()),
        "precision": float(precision_score(y_true, adjusted_pred, zero_division=0)),
        "recall": float(recall_score(y_true, adjusted_pred, zero_division=0)),
        "f1": float(f1_score(y_true, adjusted_pred, zero_division=0)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tp_rate_pct": 100.0 * tp / max(total, 1),
        "fp_rate_pct": 100.0 * fp / max(total, 1),
        "fn_rate_pct": 100.0 * fn / max(total, 1),
    }


def timeline_bins(df: pd.DataFrame, raw_pred: np.ndarray, bins: int) -> pd.DataFrame:
    y_true = df["is_anomaly_ground_truth"].to_numpy(dtype=int)
    adjusted_pred = point_adjust(y_true, raw_pred) if y_true.sum() else raw_pred
    groups = pd.cut(np.arange(len(df)), bins=bins, labels=False, include_lowest=True, duplicates="drop")
    timeline = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "raw_pred": raw_pred,
            "adjusted_pred": adjusted_pred,
            "ground_truth": y_true,
            "bin": groups,
        }
    )
    return (
        timeline.groupby("bin", observed=False)
        .agg(
            timestamp=("timestamp", "median"),
            raw_rate_pct=("raw_pred", lambda s: 100.0 * float(s.mean())),
            adjusted_rate_pct=("adjusted_pred", lambda s: 100.0 * float(s.mean())),
            gt_rate_pct=("ground_truth", lambda s: 100.0 * float(s.mean())),
        )
        .reset_index(drop=True)
    )


def save_timeline_plot(mission: str, df: pd.DataFrame, raw_pred: np.ndarray, bins: int, out_dir: Path) -> None:
    timeline = timeline_bins(df, raw_pred, bins)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(timeline["timestamp"], timeline["raw_rate_pct"], color=COLOR_GRAY, linewidth=1.8, label="Raw threshold")
    ax.plot(
        timeline["timestamp"],
        timeline["adjusted_rate_pct"],
        color=COLOR_TEAL,
        linewidth=2.2,
        label="Point-adjusted predicted",
    )
    ax.plot(timeline["timestamp"], timeline["gt_rate_pct"], color=COLOR_RED, linewidth=2.0, label="Ground truth")
    ax.set_title(f"{mission}: Anomaly Rate Over Test Timeline")
    ax.set_ylabel("Percent of bin")
    ax.set_xlabel("Time")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / f"{mission}_metric_aligned_timeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_score_plot(
    mission: str,
    df: pd.DataFrame,
    raw_pred: np.ndarray,
    sample_size: int,
    out_dir: Path,
) -> None:
    y_true = df["is_anomaly_ground_truth"].to_numpy(dtype=int)
    adjusted_pred = point_adjust(y_true, raw_pred) if y_true.sum() else raw_pred
    score = df["anomaly_score"].to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    if len(score) > sample_size:
        idx = rng.choice(len(score), size=sample_size, replace=False)
        score = score[idx]
        y_true = y_true[idx]
        adjusted_pred = adjusted_pred[idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(score[y_true == 0], bins=80, alpha=0.75, color=COLOR_NAVY, density=True, label="GT normal")
    if y_true.sum():
        axes[0].hist(score[y_true == 1], bins=80, alpha=0.65, color=COLOR_RED, density=True, label="GT anomaly")
    axes[0].set_title(f"{mission}: Score by Ground Truth")
    axes[0].set_xlabel("Anomaly score")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper right")

    axes[1].hist(score[adjusted_pred == 0], bins=80, alpha=0.75, color=COLOR_GREEN, density=True, label="Pred normal")
    axes[1].hist(score[adjusted_pred == 1], bins=80, alpha=0.65, color=COLOR_GOLD, density=True, label="Pred anomaly")
    axes[1].set_title(f"{mission}: Score by Point-Adjusted Prediction")
    axes[1].set_xlabel("Anomaly score")
    axes[1].set_ylabel("Density")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / f"{mission}_score_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metric_overview(summary_df: pd.DataFrame, out_dir: Path) -> None:
    summary_df = summary_df.sort_values("mission").reset_index(drop=True)
    x = np.arange(len(summary_df))
    missions = summary_df["mission"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.28
    axes[0].bar(x - width, summary_df["raw_pred_rate_pct"], width=width, color=COLOR_GRAY, label="Raw threshold")
    axes[0].bar(x, summary_df["adjusted_pred_rate_pct"], width=width, color=COLOR_TEAL, label="Point-adjusted pred")
    axes[0].bar(x + width, summary_df["gt_rate_pct"], width=width, color=COLOR_RED, label="Ground truth")
    axes[0].set_xticks(x, missions, rotation=20, ha="right")
    axes[0].set_title("Metric-Aligned Anomaly Rates")
    axes[0].set_ylabel("Percent of test points")
    axes[0].legend(loc="upper right")

    metric_width = 0.22
    axes[1].bar(x - metric_width, summary_df["precision"], width=metric_width, color=COLOR_NAVY, label="Precision")
    axes[1].bar(x, summary_df["recall"], width=metric_width, color=COLOR_GREEN, label="Recall")
    axes[1].bar(x + metric_width, summary_df["f1"], width=metric_width, color=COLOR_GOLD, label="F1")
    axes[1].set_xticks(x, missions, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Point-Adjusted Metrics")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_dir / "parquet_metric_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_rows = load_summary(args.summary)
    parquets = parse_parquets(args.parquet) if args.parquet else discover_parquets(args.results_dir, args.missions)
    if not parquets:
        raise ValueError(
            f"No telemetry Parquets found under {args.results_dir}. "
            "Expected results/<MISSION>/telemetry/<MISSION>_telemetry.parquet."
        )

    overview_dir = overview_out_dir(args.results_dir, args.out_dir)
    overview_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for mission, parquet_path in parquets.items():
        out_dir = mission_out_dir(args.results_dir, mission, args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(parquet_path, columns=METRIC_COLUMNS)
        missing = set(METRIC_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{parquet_path} is missing required columns: {sorted(missing)}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        raw_pred = selected_prediction(df, mission, summary_rows)
        mission_summary = compute_summary(mission, df, raw_pred)
        rows.append(mission_summary)
        save_timeline_plot(mission, df, raw_pred, args.bins, out_dir)
        save_score_plot(mission, df, raw_pred, args.score_sample, out_dir)
        pd.DataFrame([mission_summary]).to_csv(out_dir / "parquet_metric_summary.csv", index=False)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(overview_dir / "parquet_metric_summary.csv", index=False)
    save_metric_overview(summary_df, overview_dir)
    print(f"Saved Parquet anomaly images under {args.results_dir}")


if __name__ == "__main__":
    main()
