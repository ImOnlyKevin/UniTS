#!/usr/bin/env python3
"""
Build a metric-aligned replacement for figure_01_reference_overview.png.

The original paper overview mixed raw selected-threshold rates with
point-adjusted F1/precision/recall values. This script plots the rates and
confusion components used by those metrics instead.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


COLOR_NAVY = "#16324f"
COLOR_TEAL = "#1f7a8c"
COLOR_GOLD = "#c58f00"
COLOR_RED = "#c0392b"
COLOR_GREEN = "#2e8b57"
COLOR_GRAY = "#6c757d"
COLOR_LIGHT = "#eef3f8"
COLOR_DARK = "#243b53"

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
        description="Create a metric-aligned reference overview figure from study tables or telemetry Parquets."
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
        "--study-dir",
        type=Path,
        help="Study directory containing tables/reference_runs*.csv or run_summary*.csv.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Explicit reference_runs/run_summary CSV. Overrides --study-dir table discovery.",
    )
    parser.add_argument(
        "--parquet",
        action="append",
        default=[],
        metavar="MISSION=PATH",
        help="Telemetry Parquet for one mission. Can be repeated.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help=(
            "Output PNG path. Defaults to <study-dir>/figures/figure_01_metric_aligned_overview.png "
            "when --study-dir is used, otherwise results/figures/metric_aligned_overview/."
        ),
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
    baseline_pred = np.asarray(baseline_pred).astype(int)
    baseline_flagged = int(baseline_pred.sum())
    if baseline_flagged > 0:
        target_n = int(round(baseline_flagged * (ratio / max(baseline_ratio, 1e-9))))
    else:
        target_n = int(round(len(scores) * (ratio / 100.0)))
    target_n = max(1, min(target_n, len(scores)))
    threshold = float(np.partition(scores, -target_n)[-target_n])
    return (scores >= threshold).astype(int)


def discover_summary(study_dir: Path | None) -> Path | None:
    if study_dir is None:
        return None
    table_dir = study_dir / "tables"
    candidates = sorted(glob.glob(str(table_dir / "reference_runs*.csv")))
    if not candidates:
        candidates = sorted(glob.glob(str(table_dir / "run_summary*.csv")))
    if not candidates:
        return None
    return Path(candidates[-1])


def discover_parquets(results_dir: Path, missions: list[str] | None) -> dict[str, Path]:
    if not results_dir.exists():
        return {}
    search_roots = [results_dir / mission for mission in missions] if missions else sorted(results_dir.iterdir())
    parquets = {}
    for root in search_roots:
        if not root.is_dir():
            continue
        mission = root.name
        candidates = sorted((root / "telemetry").glob("*_telemetry.parquet"))
        if candidates:
            parquets[mission] = candidates[-1]
    return parquets


def load_summary_rows(summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError(f"Summary table is empty: {summary_path}")
    required = {"mission", "test_rows", "channels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{summary_path} is missing required columns: {sorted(missing)}")

    metric_cols = {
        "selected_precision",
        "selected_recall",
        "selected_f1",
        "selected_gt_rate_pct",
        "selected_tp",
        "selected_fp",
        "selected_fn",
        "selected_tn",
    }
    if not metric_cols.issubset(df.columns):
        raise ValueError(
            f"{summary_path} does not contain selected point-adjusted metrics. "
            "Use telemetry Parquets instead."
        )

    rows = []
    for _, row in df.iterrows():
        total = int(row["selected_tp"] + row["selected_fp"] + row["selected_fn"] + row["selected_tn"])
        adjusted_pred_rate = 100.0 * float(row["selected_tp"] + row["selected_fp"]) / max(total, 1)
        raw_rate = float(row.get("selected_flagged_rate_pct", np.nan))
        rows.append(
            {
                "mission": row["mission"],
                "channels": int(row["channels"]),
                "n_points": total,
                "raw_pred_rate_pct": raw_rate,
                "adjusted_pred_rate_pct": adjusted_pred_rate,
                "gt_rate_pct": float(row["selected_gt_rate_pct"]),
                "precision": float(row["selected_precision"]),
                "recall": float(row["selected_recall"]),
                "f1": float(row["selected_f1"]),
                "accuracy": float(row.get("selected_accuracy", np.nan)),
                "tp_rate_pct": 100.0 * float(row["selected_tp"]) / max(total, 1),
                "fp_rate_pct": 100.0 * float(row["selected_fp"]) / max(total, 1),
                "fn_rate_pct": 100.0 * float(row["selected_fn"]) / max(total, 1),
            }
        )
    return pd.DataFrame(rows)


def parse_parquet_specs(specs: list[str]) -> dict[str, Path]:
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--parquet must look like MISSION=PATH, got: {spec}")
        mission, path = spec.split("=", 1)
        parsed[mission.strip()] = Path(path).expanduser()
    return parsed


def load_parquet_rows(parquets: dict[str, Path], summary_df: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    summary_by_mission = {}
    if summary_df is not None and "mission" in summary_df.columns:
        summary_by_mission = {str(row["mission"]): row for _, row in summary_df.iterrows()}

    for mission, path in parquets.items():
        df = pd.read_parquet(path)
        required = {"anomaly_score", "is_anomaly_predicted", "is_anomaly_ground_truth"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

        y_true = df["is_anomaly_ground_truth"].to_numpy(dtype=int)
        baseline_pred = df["is_anomaly_predicted"].to_numpy(dtype=int)
        scores = df["anomaly_score"].to_numpy(dtype=float)
        raw_pred = baseline_pred
        summary_row = summary_by_mission.get(mission)
        if summary_row is not None and {"selected_ratio", "base_anomaly_ratio"}.issubset(summary_row.index):
            raw_pred = apply_ratio_threshold(
                scores,
                baseline_pred,
                float(summary_row["selected_ratio"]),
                float(summary_row["base_anomaly_ratio"]),
            )

        adjusted_pred = point_adjust(y_true, raw_pred) if y_true.sum() > 0 else raw_pred
        tp = int(((y_true == 1) & (adjusted_pred == 1)).sum())
        fp = int(((y_true == 0) & (adjusted_pred == 1)).sum())
        fn = int(((y_true == 1) & (adjusted_pred == 0)).sum())
        total = len(df)
        anomaly_cols = {"timestamp", "anomaly_score", "is_anomaly_predicted", "is_anomaly_ground_truth"}
        channels = [col for col in df.columns if col not in anomaly_cols and not str(col).endswith("_z")]

        rows.append(
            {
                "mission": mission,
                "channels": len(channels),
                "n_points": total,
                "raw_pred_rate_pct": 100.0 * float(raw_pred.mean()),
                "adjusted_pred_rate_pct": 100.0 * float(adjusted_pred.mean()),
                "gt_rate_pct": 100.0 * float(y_true.mean()),
                "precision": float(precision_score(y_true, adjusted_pred, zero_division=0)),
                "recall": float(recall_score(y_true, adjusted_pred, zero_division=0)),
                "f1": float(f1_score(y_true, adjusted_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, adjusted_pred)),
                "tp_rate_pct": 100.0 * tp / max(total, 1),
                "fp_rate_pct": 100.0 * fp / max(total, 1),
                "fn_rate_pct": 100.0 * fn / max(total, 1),
            }
        )
    return pd.DataFrame(rows)


def plot_overview(df: pd.DataFrame, out_path: Path) -> None:
    df = df.sort_values("mission").reset_index(drop=True)
    missions = df["mission"].tolist()
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].bar(x, df["channels"], color=COLOR_TEAL)
    axes[0, 0].set_xticks(x, missions, rotation=20, ha="right")
    axes[0, 0].set_title("Channels per Mission")
    axes[0, 0].set_ylabel("Channels")

    width = 0.28
    axes[0, 1].bar(x - width, df["raw_pred_rate_pct"], width=width, color=COLOR_GRAY, label="Raw threshold")
    axes[0, 1].bar(
        x,
        df["adjusted_pred_rate_pct"],
        width=width,
        color=COLOR_TEAL,
        label="Point-adjusted predicted",
    )
    axes[0, 1].bar(x + width, df["gt_rate_pct"], width=width, color=COLOR_RED, label="Ground truth")
    axes[0, 1].set_xticks(x, missions, rotation=20, ha="right")
    axes[0, 1].set_title("Metric-Aligned Anomaly Rate")
    axes[0, 1].set_ylabel("Percent of test points")
    axes[0, 1].legend(loc="upper right")

    metric_width = 0.22
    axes[1, 0].bar(x - metric_width, df["precision"], width=metric_width, color=COLOR_NAVY, label="Precision")
    axes[1, 0].bar(x, df["recall"], width=metric_width, color=COLOR_GREEN, label="Recall")
    axes[1, 0].bar(x + metric_width, df["f1"], width=metric_width, color=COLOR_GOLD, label="F1")
    axes[1, 0].set_xticks(x, missions, rotation=20, ha="right")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("Point-Adjusted Detection Metrics")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend(loc="lower right")

    axes[1, 1].bar(x, df["tp_rate_pct"], color=COLOR_GREEN, label="TP")
    axes[1, 1].bar(x, df["fp_rate_pct"], bottom=df["tp_rate_pct"], color=COLOR_RED, label="FP")
    axes[1, 1].bar(
        x,
        df["fn_rate_pct"],
        bottom=df["tp_rate_pct"] + df["fp_rate_pct"],
        color=COLOR_GRAY,
        label="FN",
    )
    axes[1, 1].set_xticks(x, missions, rotation=20, ha="right")
    axes[1, 1].set_title("Metric Confusion Components")
    axes[1, 1].set_ylabel("Percent of test points")
    axes[1, 1].legend(loc="upper right")

    fig.suptitle("Reference Run Overview: Point-Adjusted Metrics", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_default_mission_copies(df: pd.DataFrame, results_dir: Path) -> None:
    for mission in df["mission"].drop_duplicates():
        mission_df = df[df["mission"] == mission]
        out_path = (
            results_dir
            / str(mission)
            / "figures"
            / "metric_aligned_overview"
            / "figure_01_metric_aligned_overview.png"
        )
        plot_overview(mission_df, out_path)


def main() -> None:
    args = parse_args()
    summary_path = args.summary or discover_summary(args.study_dir)
    summary_raw = pd.read_csv(summary_path) if summary_path else None

    if args.parquet:
        rows = load_parquet_rows(parse_parquet_specs(args.parquet), summary_raw)
    elif summary_path:
        rows = load_summary_rows(summary_path)
    else:
        parquets = discover_parquets(args.results_dir, args.missions)
        if not parquets:
            raise ValueError(
                f"No telemetry Parquets found under {args.results_dir}. "
                "Expected results/<MISSION>/telemetry/<MISSION>_telemetry.parquet."
            )
        rows = load_parquet_rows(parquets, summary_raw)

    default_out = None
    if args.study_dir:
        default_out = args.study_dir / "figures" / "figure_01_metric_aligned_overview.png"
    else:
        default_out = args.results_dir / "figures" / "metric_aligned_overview" / "figure_01_metric_aligned_overview.png"
    out_path = args.out or default_out or Path("figure_01_metric_aligned_overview.png")
    plot_overview(rows, out_path)
    if args.out is None and args.study_dir is None:
        plot_default_mission_copies(rows, args.results_dir)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
