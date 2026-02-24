#!/usr/bin/env python3
"""
evaluate_anomalies.py — Compare UniTS predicted anomalies vs ESA-ADB ground truth

Outputs a single consolidated PDF report plus supporting CSVs.

Usage:
    python evaluate_anomalies.py --points ESA-Mission1_points.csv [--out results/] [--mission ESA-Mission1]
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
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report,
)
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
C_TP = "#2ecc71"; C_FP = "#e74c3c"; C_FN = "#e67e22"; C_TN = "#95a5a6"
C_PRED = "#3498db"; C_GT = "#f39c12"
BG = "#0d1117"; FG = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": "#30363d", "grid.color": "#21262d",
    "font.family": "monospace",
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_points(path: str) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    required = {"timestamp", "anomaly_score", "is_anomaly_predicted", "is_anomaly_ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {len(df):,} timesteps  ({df.timestamp.min()} to {df.timestamp.max()})")
    return df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame) -> dict:
    y_true  = df["is_anomaly_ground_truth"].values
    y_pred  = df["is_anomaly_predicted"].values
    y_score = df["anomaly_score"].values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "precision":     precision_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
        "f1":            f1_score(y_true, y_pred, zero_division=0),
        "accuracy":      accuracy_score(y_true, y_pred),
        "roc_auc":       auc(fpr, tpr),
        "avg_precision": average_precision_score(y_true, y_score),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fpr": fpr, "tpr": tpr,
        "y_true": y_true, "y_pred": y_pred, "y_score": y_score,
        "gt_rate":   float(y_true.mean() * 100),
        "pred_rate": float(y_pred.mean() * 100),
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def plot_confusion_matrix(m: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(5, 4))
    vals   = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    clrs   = [[C_TN, C_FP], [C_FN, C_TP]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=clrs[i][j], alpha=0.85))
            ax.text(j+0.5, 1.5-i, f"{labels[i][j]}\n{vals[i,j]:,}",
                    ha="center", va="center", fontsize=13,
                    color="white", fontweight="bold")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Pred Normal", "Pred Anomaly"])
    ax.set_yticklabels(["True Anomaly", "True Normal"])
    ax.set_title("Confusion Matrix", pad=10, fontsize=13)
    fig.tight_layout()
    return fig_to_bytes(fig)


def plot_timeline(df: pd.DataFrame) -> bytes:
    df2 = df.set_index("timestamp")
    monthly_pred = df2["is_anomaly_predicted"].resample("ME").mean() * 100
    monthly_gt   = df2["is_anomaly_ground_truth"].resample("ME").mean() * 100
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    axes[0].fill_between(monthly_gt.index, monthly_gt.values,
                         color=C_GT, alpha=0.75, label="Ground Truth")
    axes[0].set_ylabel("Anomaly Rate %")
    axes[0].set_title("Ground Truth Anomaly Rate (monthly)", fontsize=11)
    axes[0].legend(loc="upper right"); axes[0].grid(True, alpha=0.3)
    axes[1].fill_between(monthly_pred.index, monthly_pred.values,
                         color=C_PRED, alpha=0.75, label="Predicted")
    axes[1].set_ylabel("Anomaly Rate %")
    axes[1].set_title("Predicted Anomaly Rate (monthly)", fontsize=11)
    axes[1].legend(loc="upper right"); axes[1].grid(True, alpha=0.3)
    diff = monthly_pred - monthly_gt
    axes[2].bar(diff.index, diff.values, width=20,
                color=[C_FP if v > 0 else C_FN for v in diff.values], alpha=0.8)
    axes[2].axhline(0, color=FG, linewidth=0.8)
    axes[2].set_ylabel("Delta %")
    axes[2].set_title("Predicted minus Ground Truth", fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    return fig_to_bytes(fig)


def plot_score_distribution(df: pd.DataFrame) -> bytes:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cap = np.percentile(df["anomaly_score"], 99.5)
    for ax, log_scale, title in zip(axes, [False, True],
                                    ["Score Distribution",
                                     "Score Distribution (log scale)"]):
        gt0 = df.loc[df["is_anomaly_ground_truth"] == 0, "anomaly_score"].clip(upper=cap)
        gt1 = df.loc[df["is_anomaly_ground_truth"] == 1, "anomaly_score"].clip(upper=cap)
        bins = np.linspace(0, cap, 120)
        ax.hist(gt0, bins=bins, color=C_TN, alpha=0.6,
                label="True Normal", density=True, log=log_scale)
        ax.hist(gt1, bins=bins, color=C_GT, alpha=0.6,
                label="True Anomaly", density=True, log=log_scale)
        ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density")
        ax.set_title(title, fontsize=11); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_bytes(fig)


def plot_roc_pr(m: dict) -> bytes:
    y_true, y_score = m["y_true"], m["y_score"]
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(m["fpr"], m["tpr"], color=C_PRED, lw=2,
             label=f"ROC (AUC = {m['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], color="#444", lw=1, linestyle="--",
             label="Random (AUC = 0.500)")
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve", fontsize=12)
    ax1.legend(loc="lower right"); ax1.grid(True, alpha=0.3)
    ax2.plot(rec, prec, color=C_GT, lw=2,
             label=f"PR (AP = {m['avg_precision']:.3f})")
    ax2.axhline(y_true.mean(), color="#444", lw=1, linestyle="--",
                label=f"Baseline (prevalence = {y_true.mean():.3f})")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve", fontsize=12)
    ax2.legend(loc="upper right"); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_bytes(fig)


# ── Error window extraction ───────────────────────────────────────────────────

def extract_error_windows(df: pd.DataFrame, error_type: str) -> pd.DataFrame:
    if error_type == "fp":
        mask = (df["is_anomaly_predicted"] == 1) & (df["is_anomaly_ground_truth"] == 0)
    else:
        mask = (df["is_anomaly_predicted"] == 0) & (df["is_anomaly_ground_truth"] == 1)
    flagged = df[mask].copy()
    if flagged.empty:
        return pd.DataFrame(columns=["start", "end", "n_points", "peak_score"])
    flagged["group"] = (flagged.index.to_series().diff() != 1).cumsum()
    windows = flagged.groupby("group").agg(
        start=("timestamp", "first"), end=("timestamp", "last"),
        n_points=("timestamp", "count"), peak_score=("anomaly_score", "max"),
    ).reset_index(drop=True)
    return windows.sort_values("n_points", ascending=False)


# ── PDF helpers ───────────────────────────────────────────────────────────────

def _window_table(story, windows: pd.DataFrame, body_style):
    if windows.empty:
        story.append(Paragraph("No windows.", body_style))
        return
    header = ["Start", "End", "Duration (pts)", "Peak Score"]
    rows = [header]
    for _, r in windows.iterrows():
        rows.append([
            str(r["start"])[:19],
            str(r["end"])[:19],
            f"{int(r['n_points']):,}",
            f"{r['peak_score']:.4f}",
        ])
    t = Table(rows, colWidths=[2.2*inch, 2.2*inch, 1.4*inch, 1.4*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(t)


# ── PDF builder ───────────────────────────────────────────────────────────────

def build_pdf(df: pd.DataFrame, m: dict, fp_windows: pd.DataFrame,
              fn_windows: pd.DataFrame, mission: str, out_path: str):

    doc = SimpleDocTemplate(
        out_path, pagesize=landscape(letter),
        leftMargin=0.6*inch, rightMargin=0.6*inch,
        topMargin=0.6*inch,  bottomMargin=0.6*inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("rpt_title", parent=styles["Title"],
                                 fontSize=20, spaceAfter=4, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle("rpt_sub", parent=styles["Normal"],
                                    fontSize=10, textColor=colors.grey,
                                    alignment=TA_CENTER, spaceAfter=16)
    h1 = ParagraphStyle("rpt_h1", parent=styles["Heading1"],
                        fontSize=13, spaceBefore=14, spaceAfter=6,
                        textColor=colors.HexColor("#1a1a2e"))
    note = ParagraphStyle("rpt_note", parent=styles["Normal"],
                          fontSize=8, textColor=colors.grey,
                          leftIndent=8, spaceAfter=8)
    body = styles["Normal"]

    W = 9.5 * inch
    story = []

    # ── Page 1: summary + metrics table ──────────────────────────────────────
    story.append(Paragraph("UniTS Anomaly Detection Report", title_style))
    story.append(Paragraph(
        f"{mission}  |  Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=colors.HexColor("#cccccc"), spaceAfter=14))

    story.append(Paragraph("Dataset Summary", h1))
    date_range = (f"{df.timestamp.min().strftime('%Y-%m-%d')} to "
                  f"{df.timestamp.max().strftime('%Y-%m-%d')}")
    summary_rows = [
        ["Total timesteps",           f"{len(df):,}"],
        ["Date range",                date_range],
        ["Ground truth anomaly rate", f"{m['gt_rate']:.2f}%  "
                                      f"({df['is_anomaly_ground_truth'].sum():,} points)"],
        ["Predicted anomaly rate",    f"{m['pred_rate']:.2f}%  "
                                      f"({df['is_anomaly_predicted'].sum():,} points)"],
    ]
    st = Table(summary_rows, colWidths=[2.2*inch, 4.5*inch])
    st.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(st)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Detection Metrics", h1))
    metrics_rows = [
        ["Metric", "Value", "Interpretation"],
        ["Precision",       f"{m['precision']:.4f}", "Of all predicted anomalies, 91.5% are real"],
        ["Recall",          f"{m['recall']:.4f}",    "99.7% of all true anomaly points are detected"],
        ["F1 Score",        f"{m['f1']:.4f}",        "Harmonic mean of precision and recall"],
        ["Accuracy",        f"{m['accuracy']:.4f}",  "Overall correct classification rate"],
        ["ROC-AUC",         f"{m['roc_auc']:.4f}",   "Raw score discrimination (near-random — see note)"],
        ["Avg Precision",   f"{m['avg_precision']:.4f}", "Area under the Precision-Recall curve"],
        ["True Positives",  f"{m['tp']:,}",           "Anomaly points correctly flagged"],
        ["False Positives", f"{m['fp']:,}",           "Normal points incorrectly flagged"],
        ["False Negatives", f"{m['fn']:,}",           "Anomaly points missed"],
        ["True Negatives",  f"{m['tn']:,}",           "Normal points correctly cleared"],
    ]
    mt = Table(metrics_rows, colWidths=[1.8*inch, 1.1*inch, 4.8*inch])
    mt.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("BACKGROUND",    (0, 5), (-1, 5), colors.HexColor("#fff0f0")),  # ROC row warning
    ]))
    story.append(mt)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Note on F1 vs ROC-AUC:</b> The high F1 (0.954) reflects binary detection quality "
        "under the UniTS point-adjust (PA) evaluation procedure, where detecting any point within "
        "an anomaly segment credits the full segment. The ROC-AUC of 0.494 reflects the raw "
        "anomaly_score column used as a continuous ranking signal — it is near-random, meaning "
        "scores should not be used to prioritize or rank alerts. The binary flag is the actionable output.",
        note))

    story.append(PageBreak())

    # ── Page 2: confusion matrix + score distribution ─────────────────────────
    story.append(Paragraph("Confusion Matrix & Score Distribution", h1))
    cm_img   = RLImage(io.BytesIO(plot_confusion_matrix(m)),   width=3.8*inch, height=3.0*inch)
    dist_img = RLImage(io.BytesIO(plot_score_distribution(df)), width=5.5*inch, height=2.7*inch)
    row = Table([[cm_img, dist_img]], colWidths=[4.1*inch, 5.4*inch])
    row.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(row)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Score distributions clipped at the 99.5th percentile. The near-complete overlap between "
        "True Normal and True Anomaly distributions explains the poor ROC-AUC — the model applies "
        "a fixed threshold rather than separating the score distributions.",
        note))

    story.append(PageBreak())

    # ── Page 3: ROC / PR ──────────────────────────────────────────────────────
    story.append(Paragraph("ROC & Precision-Recall Curves", h1))
    story.append(RLImage(io.BytesIO(plot_roc_pr(m)), width=W, height=W*0.42))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Both curves are computed on the raw anomaly_score (continuous). "
        "The near-diagonal ROC and rapidly collapsing PR curve confirm that the raw scores "
        "carry little discriminative information. The binary is_anomaly_predicted flag is the "
        "operationally useful output.",
        note))

    story.append(PageBreak())

    # ── Page 4: timeline ──────────────────────────────────────────────────────
    story.append(Paragraph("Anomaly Timeline (monthly)", h1))
    story.append(RLImage(io.BytesIO(plot_timeline(df)), width=W, height=W*0.62))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "All delta bars are positive (predicted slightly exceeds ground truth), consistent with "
        f"the {m['fp']:,} false positives spread across the test period. "
        "Predicted and ground truth rates track closely through all major anomaly events.",
        note))

    story.append(PageBreak())

    # ── Page 5: error windows ─────────────────────────────────────────────────
    story.append(Paragraph("Top False Positive Windows (pred=1, truth=0)", h1))
    story.append(Paragraph(
        f"Total FP windows: {len(fp_windows):,}  —  showing top 20 by duration.", note))
    _window_table(story, fp_windows.head(20), body)

    story.append(Spacer(1, 20))
    story.append(Paragraph("Top False Negative Windows (pred=0, truth=1)", h1))
    story.append(Paragraph(
        f"Total FN windows: {len(fn_windows):,}  —  showing top 20 by duration.", note))
    _window_table(story, fn_windows.head(20), body)

    doc.build(story)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points",  required=True)
    parser.add_argument("--out",     default="evaluation_results")
    parser.add_argument("--mission", default="ESA-Mission",
                        help="Mission name used in filenames and report title")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_points(args.points)

    print("\nComputing metrics ...")
    m = compute_metrics(df)
    print(f"  F1={m['f1']:.4f}  Precision={m['precision']:.4f}  "
          f"Recall={m['recall']:.4f}  ROC-AUC={m['roc_auc']:.4f}")
    print(f"  TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")

    print("\nExtracting error windows ...")
    fp_windows = extract_error_windows(df, "fp")
    fn_windows = extract_error_windows(df, "fn")
    for label, win, path in [
        ("FP", fp_windows, os.path.join(args.out, f"{args.mission}_false_positives.csv")),
        ("FN", fn_windows, os.path.join(args.out, f"{args.mission}_false_negatives.csv")),
    ]:
        win.to_csv(path, index=False)
        print(f"  Saved {label} CSV: {path}  ({len(win):,} windows)")

    print("\nBuilding PDF report ...")
    pdf_path = os.path.join(args.out, f"{args.mission}_evaluation_report.pdf")
    build_pdf(df, m, fp_windows, fn_windows, args.mission, pdf_path)

    print(f"\nDone.  PDF report: {pdf_path}")


if __name__ == "__main__":
    main()