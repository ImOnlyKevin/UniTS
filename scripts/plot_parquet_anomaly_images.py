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


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return scores[np.isfinite(scores)]


def plot_hist(ax: plt.Axes, values: np.ndarray, *, bins: np.ndarray | int, color: str, label: str) -> None:
    values = finite_scores(values)
    if len(values) == 0:
        ax.text(
            0.5,
            0.5,
            f"No finite {label.lower()} scores",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=COLOR_GRAY,
        )
        return
    ax.hist(values, bins=bins, alpha=0.68, color=color, density=False, label=label)


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
    scores = np.asarray(scores, dtype=float)
    baseline_flagged = int(baseline_pred.sum())
    if baseline_flagged > 0:
        target_n = int(round(baseline_flagged * (ratio / max(baseline_ratio, 1e-9))))
    else:
        target_n = int(round(len(scores) * (ratio / 100.0)))
    finite_mask = np.isfinite(scores)
    finite_scores = scores[finite_mask]
    if len(finite_scores) == 0:
        return np.zeros(len(scores), dtype=int)
    target_n = max(1, min(target_n, len(finite_scores)))
    threshold = float(np.partition(finite_scores, -target_n)[-target_n])
    pred = np.zeros(len(scores), dtype=int)
    pred[finite_mask] = (scores[finite_mask] >= threshold).astype(int)
    return pred


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
    tn = int(((y_true == 0) & (adjusted_pred == 0)).sum())
    metrics = binary_metrics(y_true, adjusted_pred)
    return {
        "mission": mission,
        "n_points": total,
        "has_ground_truth": bool(y_true.sum() > 0),
        "raw_pred_rate_pct": 100.0 * float(raw_pred.mean()),
        "adjusted_pred_rate_pct": 100.0 * float(adjusted_pred.mean()),
        "gt_rate_pct": 100.0 * float(y_true.mean()),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
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
    if np.allclose(timeline["raw_rate_pct"], timeline["adjusted_rate_pct"], equal_nan=True):
        ax.plot(timeline["timestamp"], timeline["adjusted_rate_pct"], color=COLOR_TEAL, linewidth=2.2, label="Predicted")
    else:
        ax.plot(timeline["timestamp"], timeline["raw_rate_pct"], color=COLOR_GRAY, linewidth=1.8, label="Raw threshold")
        ax.plot(
            timeline["timestamp"],
            timeline["adjusted_rate_pct"],
            color=COLOR_TEAL,
            linewidth=2.2,
            label="Point-adjusted predicted",
        )
    if timeline["gt_rate_pct"].sum() > 0:
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
    finite_score = finite_scores(score)
    if len(finite_score):
        bins: np.ndarray | int = np.linspace(float(finite_score.min()), float(finite_score.max()), 81)
        if np.isclose(bins[0], bins[-1]):
            bins = 80
    else:
        bins = 80

    plot_hist(axes[0], score[y_true == 0], bins=bins, color=COLOR_NAVY, label="GT normal")
    if y_true.sum():
        plot_hist(axes[0], score[y_true == 1], bins=bins, color=COLOR_RED, label="GT anomaly")
    axes[0].set_title(f"{mission}: Score by Ground Truth")
    axes[0].set_xlabel("Anomaly score")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="upper right")

    plot_hist(axes[1], score[adjusted_pred == 0], bins=bins, color=COLOR_GREEN, label="Pred normal")
    plot_hist(axes[1], score[adjusted_pred == 1], bins=bins, color=COLOR_GOLD, label="Pred anomaly")
    axes[1].set_title(f"{mission}: Score by Point-Adjusted Prediction")
    axes[1].set_xlabel("Anomaly score")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / f"{mission}_score_distributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_missions(ax: plt.Axes, x: np.ndarray, missions: list[str]) -> None:
    ax.set_xticks(x, missions, rotation=20, ha="right")


def annotate_bars(ax: plt.Axes, decimals: int = 2) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        ax.annotate(
            f"{height:.{decimals}f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLOR_DARK,
            xytext=(0, 3),
            textcoords="offset points",
        )


def save_labeled_rate_plot(summary_df: pd.DataFrame, out_dir: Path) -> Path | None:
    labelled = summary_df[summary_df["has_ground_truth"].astype(bool)].sort_values("mission").reset_index(drop=True)
    if labelled.empty:
        return None
    x = np.arange(len(labelled))
    missions = labelled["mission"].tolist()
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, labelled["adjusted_pred_rate_pct"], width=width, color=COLOR_TEAL, label="Predicted")
    ax.bar(x + width / 2, labelled["gt_rate_pct"], width=width, color=COLOR_RED, label="Ground truth")
    format_missions(ax, x, missions)
    ax.set_title("Labeled Missions: Predicted vs Ground-Truth Rate")
    ax.set_ylabel("Percent of test points")
    ax.legend(loc="upper right")
    annotate_bars(ax)
    fig.tight_layout()
    path = out_dir / "labeled_predicted_vs_ground_truth_rate.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_labeled_metrics_plot(summary_df: pd.DataFrame, out_dir: Path) -> Path | None:
    labelled = summary_df[summary_df["has_ground_truth"].astype(bool)].sort_values("mission").reset_index(drop=True)
    if labelled.empty:
        return None
    x = np.arange(len(labelled))
    missions = labelled["mission"].tolist()
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(x - width, labelled["precision"], width=width, color=COLOR_NAVY, label="Precision")
    ax.bar(x, labelled["recall"], width=width, color=COLOR_GREEN, label="Recall")
    ax.bar(x + width, labelled["f1"], width=width, color=COLOR_GOLD, label="F1")
    format_missions(ax, x, missions)
    ax.set_ylim(0, 1.08)
    ax.set_title("Labeled Missions: Point-Adjusted Detection Metrics")
    ax.set_ylabel("Score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    annotate_bars(ax, decimals=3)
    fig.tight_layout()
    path = out_dir / "labeled_point_adjusted_detection_metrics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_unlabeled_predicted_rate_plot(summary_df: pd.DataFrame, out_dir: Path) -> Path | None:
    unlabeled = summary_df[~summary_df["has_ground_truth"].astype(bool)].sort_values(
        "adjusted_pred_rate_pct", ascending=False
    )
    if unlabeled.empty:
        return None
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.38 * len(unlabeled))))
    y = np.arange(len(unlabeled))
    ax.barh(y, unlabeled["adjusted_pred_rate_pct"], color=COLOR_TEAL)
    ax.set_yticks(y, unlabeled["mission"])
    ax.invert_yaxis()
    ax.set_title("Unlabeled Missions: Predicted Anomaly Burden")
    ax.set_xlabel("Predicted percent of test points")
    ax.set_ylabel("Mission")
    for idx, value in enumerate(unlabeled["adjusted_pred_rate_pct"]):
        ax.text(value, idx, f" {value:.3f}%", va="center", ha="left", fontsize=9, color=COLOR_DARK)
    fig.tight_layout()
    path = out_dir / "unlabeled_predicted_anomaly_burden.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_confusion_matrix(row: pd.Series, out_dir: Path) -> Path:
    matrix = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=float)
    total = matrix.sum()
    pct = matrix / max(total, 1) * 100.0
    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    image = ax.imshow(pct, cmap="Blues", vmin=0, vmax=max(float(pct.max()), 1.0))
    ax.set_title(f"{row['mission']} Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Ground truth label")
    ax.set_xticks([0, 1], ["Normal", "Anomaly"])
    ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    for i in range(2):
        for j in range(2):
            percent = pct[i, j]
            text_color = "white" if percent > pct.max() * 0.55 else COLOR_DARK
            ax.text(
                j,
                i,
                f"{labels[i, j]}\n{int(matrix[i, j]):,}\n{percent:.2f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Percent of test points")
    fig.tight_layout()
    mission_slug = str(row["mission"]).lower().replace(" ", "_").replace("/", "_")
    path = out_dir / f"{mission_slug}_confusion_matrix.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_metric_overview(summary_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for maybe_path in [
        save_labeled_rate_plot(summary_df, out_dir),
        save_labeled_metrics_plot(summary_df, out_dir),
        save_unlabeled_predicted_rate_plot(summary_df, out_dir),
    ]:
        if maybe_path is not None:
            outputs.append(maybe_path)
    labelled = summary_df[summary_df["has_ground_truth"].astype(bool)].sort_values("mission")
    for _, row in labelled.iterrows():
        outputs.append(save_confusion_matrix(row, out_dir))
    return outputs


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
    overview_outputs = save_metric_overview(summary_df, overview_dir)
    print("Saved Parquet anomaly images:")
    print(f"  {overview_dir / 'parquet_metric_summary.csv'}")
    for path in overview_outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
