#!/usr/bin/env python3
"""
ratio_sweep.py — Re-threshold existing anomaly scores at multiple ratios and
produce a comparison PDF showing how flagging rate and score distribution
change across ratios.

Does NOT re-run UniTS — works entirely from the saved points CSV.

Usage:
    python scripts/ratio_sweep.py \
        --mission STPSat4-TCS \
        --points  checkpoints/.../STPSat4-TCS_points.csv \
        --ratios  0.1 0.25 0.5 1.0 2.0 \
        --out     results/STPSat4-TCS/ratio_sweep/
"""

import argparse
import os
import io
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak,
)

# ── Plot palette ──────────────────────────────────────────────────────────────
BG = "#0d1117"; FG = "#e6edf3"
PALETTE = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6",
           "#1abc9c", "#e67e22", "#e91e63", "#00bcd4", "#ff5722"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": "#30363d", "grid.color": "#21262d",
    "font.family": "monospace",
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mission",  required=True)
    p.add_argument("--points",   required=True)
    p.add_argument("--ratios",   nargs="+", type=float,
                   default=[0.1, 0.25, 0.5, 1.0, 2.0])
    p.add_argument("--out",      default="results/ratio_sweep")
    return p.parse_args()


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def apply_ratio_threshold(scores: np.ndarray, baseline_threshold: float,
                           baseline_flagged: int, ratio: float,
                           baseline_ratio: float) -> tuple:
    """
    Re-threshold by scaling the flagged count proportionally to ratio.
    Anchors to the original UniTS threshold (training-derived) at baseline_ratio,
    then scales up/down for other ratios.
    """
    if baseline_flagged <= 0:
        target_n = int(round(len(scores) * (ratio / 100.0)))
    else:
        target_n = int(len(scores) * (ratio / baseline_ratio) * (baseline_flagged / len(scores)))
    target_n   = max(1, min(target_n, len(scores)))
    threshold  = np.sort(scores)[::-1][target_n - 1]
    preds      = (scores >= threshold).astype(int)
    return preds, threshold


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Point-adjust evaluation (standard for time-series anomaly detection).
    If any point within a true anomaly window is predicted as anomaly,
    the entire window is credited as detected.
    """
    y_pred_pa = y_pred.copy()
    in_anomaly = False
    window_start = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            in_anomaly = True
            window_start = i
        if (y_true[i] == 0 or i == len(y_true) - 1) and in_anomaly:
            window_end = i if y_true[i] == 0 else i + 1
            if y_pred[window_start:window_end].sum() > 0:
                y_pred_pa[window_start:window_end] = 1
            in_anomaly = False
    return y_pred_pa


def compute_summary(df: pd.DataFrame, ratio: float,
                    baseline_ratio: float, baseline_threshold: float,
                    baseline_flagged: int) -> dict:
    preds, threshold = apply_ratio_threshold(
        df["anomaly_score"].values, baseline_threshold,
        baseline_flagged, ratio, baseline_ratio)
    n_flag = int(preds.sum())
    rate   = float(preds.mean() * 100)

    # Count contiguous windows
    changes = np.diff(np.concatenate([[0], preds, [0]]))
    n_windows = int((changes == 1).sum())

    has_gt = df["is_anomaly_ground_truth"].sum() > 0
    metrics = {}
    if has_gt:
        from sklearn.metrics import precision_score, recall_score, f1_score
        y_true = df["is_anomaly_ground_truth"].values
        # Point-adjust (PA): if any point in a true anomaly window is predicted,
        # the entire window is credited as detected. This matches UniTS evaluation.
        preds_pa = point_adjust(y_true, preds)
        metrics["precision"] = precision_score(y_true, preds_pa, zero_division=0)
        metrics["recall"]    = recall_score(y_true, preds_pa, zero_division=0)
        metrics["f1"]        = f1_score(y_true, preds_pa, zero_division=0)

    return {
        "ratio":      ratio,
        "threshold":  float(threshold),
        "n_flagged":  n_flag,
        "rate_pct":   rate,
        "n_windows":  n_windows,
        "preds":      preds,
        **metrics,
    }


def plot_flagging_rates(summaries: list, mission: str) -> bytes:
    ratios    = [s["ratio"]    for s in summaries]
    rates     = [s["rate_pct"] for s in summaries]
    n_windows = [s["n_windows"] for s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    bars1 = ax1.bar([str(r) for r in ratios], rates,
                    color=PALETTE[:len(ratios)], alpha=0.85, edgecolor="#30363d")
    ax1.set_xlabel("anomaly_ratio parameter")
    ax1.set_ylabel("Flagged points (%)")
    ax1.set_title("Flagging Rate by Ratio", fontsize=12)
    ax1.grid(True, alpha=0.3, axis="y")
    for bar, rate in zip(bars1, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{rate:.2f}%", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar([str(r) for r in ratios], n_windows,
                    color=PALETTE[:len(ratios)], alpha=0.85, edgecolor="#30363d")
    ax2.set_xlabel("anomaly_ratio parameter")
    ax2.set_ylabel("Number of anomaly windows")
    ax2.set_title("Anomaly Windows by Ratio", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, n in zip(bars2, n_windows):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{n:,}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"{mission} — Ratio Sweep Summary", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig_to_bytes(fig)


def plot_score_distributions(df: pd.DataFrame, summaries: list, mission: str) -> bytes:
    scores = df["anomaly_score"].values
    cap    = np.percentile(scores, 99.5)
    bins   = np.linspace(0, cap, 100)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.hist(scores.clip(max=cap), bins=bins, color="#555", alpha=0.4,
            density=True, label="All scores")

    for s, color in zip(summaries, PALETTE):
        threshold = s["threshold"]
        if threshold <= cap:
            ax.axvline(threshold, color=color, linewidth=1.5, linestyle="--",
                       label=f"ratio={s['ratio']}  ({s['rate_pct']:.2f}%  |  "
                             f"{s['n_windows']:,} windows)")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"{mission} — Score Distribution with Thresholds", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_bytes(fig)


def plot_timelines(df: pd.DataFrame, summaries: list, mission: str) -> bytes:
    n = len(summaries)
    fig, axes = plt.subplots(n, 1, figsize=(13, max(2.0, 2.2 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    ts = pd.to_datetime(df["timestamp"])

    for ax, s, color in zip(axes, summaries, PALETTE):
        preds = pd.Series(s["preds"], index=ts)
        monthly = preds.resample("D").mean() * 100
        ax.fill_between(monthly.index, monthly.values, color=color, alpha=0.75)
        ax.set_ylabel("Flag %")
        ax.set_title(f"ratio={s['ratio']}  →  {s['rate_pct']:.2f}% flagged  |  "
                     f"{s['n_windows']:,} windows", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.xticks(rotation=45, ha="right")
    fig.suptitle(f"{mission} — Daily Flagging Rate by Ratio", fontsize=12, y=1.01)
    fig.tight_layout()
    return fig_to_bytes(fig)


def build_pdf(df: pd.DataFrame, summaries: list, mission: str, out_path: str):
    doc = SimpleDocTemplate(
        out_path, pagesize=landscape(letter),
        leftMargin=0.6*inch, rightMargin=0.6*inch,
        topMargin=0.6*inch,  bottomMargin=0.6*inch,
    )
    styles   = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                 fontSize=20, spaceAfter=4, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("sub", parent=styles["Normal"],
                                    fontSize=10, textColor=colors.grey,
                                    alignment=TA_CENTER, spaceAfter=16)
    h1   = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=13,
                           spaceBefore=14, spaceAfter=6,
                           textColor=colors.HexColor("#1a1a2e"))
    note = ParagraphStyle("note", parent=styles["Normal"], fontSize=8,
                           textColor=colors.grey, leftIndent=8, spaceAfter=8)

    W = 9.5 * inch
    story = []

    has_gt = "f1" in summaries[0]

    # ── Page 1: summary table ─────────────────────────────────────────────────
    story.append(Paragraph("UniTS Anomaly Ratio Sweep", title_style))
    story.append(Paragraph(
        f"{mission}  |  Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#cccccc"), spaceAfter=14))

    story.append(Paragraph("Results by Ratio", h1))

    header = ["anomaly_ratio", "Threshold", "Flagged Points", "Flag Rate", "Windows"]
    if has_gt:
        header += ["Precision", "Recall", "F1"]
    rows = [header]

    for s in summaries:
        row = [
            str(s["ratio"]),
            f"{s['threshold']:.6f}",
            f"{s['n_flagged']:,}",
            f"{s['rate_pct']:.3f}%",
            f"{s['n_windows']:,}",
        ]
        if has_gt:
            row += [
                f"{s.get('precision', 0):.4f}",
                f"{s.get('recall', 0):.4f}",
                f"{s.get('f1', 0):.4f}",
            ]
        rows.append(row)

    col_widths = [1.2*inch, 1.4*inch, 1.4*inch, 1.1*inch, 1.1*inch]
    if has_gt:
        col_widths += [1.0*inch, 1.0*inch, 1.0*inch]

    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "anomaly_ratio controls the detection threshold: UniTS flags the top ratio% of "
        "reconstruction error scores as anomalies. Lower values = stricter threshold = "
        "fewer flags. The threshold column shows the exact score cutoff for each ratio.",
        note))

    story.append(Spacer(1, 14))
    story.append(Paragraph("Flagging Rate & Window Count", h1))
    story.append(RLImage(io.BytesIO(plot_flagging_rates(summaries, mission)),
                         width=W, height=W * 0.35))

    story.append(PageBreak())

    # ── Page 2: score distributions with threshold lines ─────────────────────
    story.append(Paragraph("Score Distribution with Thresholds", h1))
    story.append(RLImage(io.BytesIO(plot_score_distributions(df, summaries, mission)),
                         width=W, height=W * 0.45))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Dashed vertical lines show where each ratio sets its threshold. "
        "Choose a ratio whose threshold sits clearly in the tail of the distribution "
        "rather than in the main body of normal scores.",
        note))

    story.append(PageBreak())

    # ── Page 3: timelines ─────────────────────────────────────────────────────
    story.append(Paragraph("Daily Flagging Rate by Ratio", h1))
    max_h = 7.0 * inch
    timeline_h = min(max_h, 2.2 * len(summaries) * inch)
    story.append(RLImage(io.BytesIO(plot_timelines(df, summaries, mission)),
                         width=W, height=timeline_h))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "A realistic anomaly rate for a nominal satellite is typically 0.1–1%. "
        "Look for a ratio that shows isolated spikes rather than sustained elevated flagging.",
        note))

    doc.build(story)
    print(f"  Saved: {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.points} ...")
    df = pd.read_csv(args.points)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} timesteps")

    # Derive baseline from the original UniTS is_anomaly_predicted column
    # (which reflects the training-derived threshold, not test percentile)
    baseline_ratio     = 1.0
    baseline_flagged   = int(df["is_anomaly_predicted"].sum())
    if baseline_flagged > 0:
        baseline_threshold = float(df.loc[df["is_anomaly_predicted"] == 1, "anomaly_score"].min())
    else:
        baseline_threshold = float(df["anomaly_score"].max())
    baseline_rate      = baseline_flagged / len(df) * 100
    print(f"  Baseline (original UniTS): flagged={baseline_flagged:,} "
          f"({baseline_rate:.3f}%)  threshold={baseline_threshold:.6f}")

    print(f"\nSweeping ratios: {args.ratios}")
    summaries = []
    for ratio in sorted(args.ratios):
        s = compute_summary(df, ratio, baseline_ratio, baseline_threshold, baseline_flagged)
        summaries.append(s)
        print(f"  ratio={ratio:<5}  threshold={s['threshold']:.6f}  "
              f"flagged={s['n_flagged']:,} ({s['rate_pct']:.3f}%)  "
              f"windows={s['n_windows']:,}")

    # Save summary CSV
    csv_path = os.path.join(args.out, f"{args.mission}_ratio_sweep.csv")
    summary_df = pd.DataFrame([{k: v for k, v in s.items() if k != "preds"}
                                for s in summaries])
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV: {csv_path}")

    print("\nBuilding comparison PDF ...")
    pdf_path = os.path.join(args.out, f"{args.mission}_ratio_comparison.pdf")
    build_pdf(df, summaries, args.mission, pdf_path)
    print(f"\nDone.  PDF: {pdf_path}")


if __name__ == "__main__":
    main()
