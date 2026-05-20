#!/usr/bin/env python3
"""Summarize UniTS result artifacts for impact-paper figure planning."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "figures" / "impact_paper"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "nan", "None"} else 0.0


def ratio_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(RESULTS.glob("*/ratio_sweep/*_ratio_sweep.csv")):
        mission = path.parts[-3]
        for row in read_csv(path):
            row = dict(row)
            row["mission"] = mission
            rows.append(row)
    return rows


def svg_wrap(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<style>'
        'text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}'
        '.title{font-size:22px;font-weight:700}.subtitle{font-size:13px;fill:#52606d}'
        '.axis{stroke:#9aa5b1;stroke-width:1}.grid{stroke:#e4e7eb;stroke-width:1}'
        '.label{font-size:12px;fill:#52606d}.tick{font-size:11px;fill:#52606d}'
        '.legend{font-size:12px;fill:#323f4b}'
        '</style>\n'
        f"{body}\n</svg>\n"
    )


def line_chart_labeled_tradeoff(rows: list[dict[str, str]], path: Path) -> None:
    labeled = [r for r in rows if r["mission"].startswith("ESA")]
    width, height = 980, 560
    left, right, top, bottom = 82, 36, 78, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    max_x = max(fnum(r, "rate_pct") for r in labeled) * 1.05
    colors = {
        ("ESA-Mission1", "precision"): "#2563eb",
        ("ESA-Mission1", "recall"): "#059669",
        ("ESA-Mission1", "f1"): "#dc2626",
        ("ESA-Mission2", "precision"): "#60a5fa",
        ("ESA-Mission2", "recall"): "#34d399",
        ("ESA-Mission2", "f1"): "#f87171",
    }

    def x(v: float) -> float:
        return left + (v / max_x) * plot_w

    def y(v: float) -> float:
        return top + (1.0 - v) * plot_h

    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text class="title" x="34" y="34">Threshold Tradeoff on Labeled ESA Missions</text>',
        '<text class="subtitle" x="34" y="56">Lower selected-rate thresholds maximize F1; Mission 2 exposes a recall-sensitive operating point.</text>',
    ]
    for pct in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = y(pct)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 12}" y="{yy + 4:.1f}" text-anchor="end">{pct:.2f}</text>')
    for idx in range(6):
        val = max_x * idx / 5
        xx = x(val)
        parts.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="tick" x="{xx:.1f}" y="{top + plot_h + 22}" text-anchor="middle">{val:.1f}%</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="label" x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle">Selected anomaly rate</text>')
    parts.append(f'<text class="label" transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">Score</text>')

    for mission in ["ESA-Mission1", "ESA-Mission2"]:
        mission_rows = sorted([r for r in labeled if r["mission"] == mission], key=lambda r: fnum(r, "rate_pct"))
        for metric in ["precision", "recall", "f1"]:
            points = " ".join(f'{x(fnum(r, "rate_pct")):.1f},{y(fnum(r, metric)):.1f}' for r in mission_rows)
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[(mission, metric)]}" '
                f'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
            )
            for r in mission_rows:
                parts.append(
                    f'<circle cx="{x(fnum(r, "rate_pct")):.1f}" cy="{y(fnum(r, metric)):.1f}" '
                    f'r="4" fill="{colors[(mission, metric)]}"/>'
                )

    lx, ly = left + 18, top + 20
    legend = [
        ("ESA1 precision", colors[("ESA-Mission1", "precision")]),
        ("ESA1 recall", colors[("ESA-Mission1", "recall")]),
        ("ESA1 F1", colors[("ESA-Mission1", "f1")]),
        ("ESA2 precision", colors[("ESA-Mission2", "precision")]),
        ("ESA2 recall", colors[("ESA-Mission2", "recall")]),
        ("ESA2 F1", colors[("ESA-Mission2", "f1")]),
    ]
    for i, (label, color) in enumerate(legend):
        x0 = lx + (i % 3) * 145
        y0 = ly + (i // 3) * 22
        parts.append(f'<line x1="{x0}" x2="{x0 + 24}" y1="{y0}" y2="{y0}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="legend" x="{x0 + 32}" y="{y0 + 4}">{html.escape(label)}</text>')

    path.write_text(svg_wrap(width, height, "\n".join(parts)))


def bar_chart_burden(summary: list[dict[str, str]], path: Path) -> None:
    unlabeled = [r for r in summary if r["has_ground_truth"] == "False"]
    unlabeled.sort(key=lambda r: fnum(r, "raw_pred_rate_pct"), reverse=True)
    width, height = 980, 560
    left, right, top, bottom = 170, 38, 78, 64
    plot_w, plot_h = width - left - right, height - top - bottom
    max_v = max(fnum(r, "raw_pred_rate_pct") for r in unlabeled) or 1.0
    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text class="title" x="34" y="34">Unlabeled STPSat Anomaly Burden</text>',
        '<text class="subtitle" x="34" y="56">The transferred detector flags sparse STPSat4 events and no STPSat7 events at the Parquet threshold.</text>',
    ]
    for idx in range(5):
        val = max_v * idx / 4
        xx = left + (val / max_v) * plot_w
        parts.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="{top}" y2="{top + plot_h}"/>')
        parts.append(f'<text class="tick" x="{xx:.1f}" y="{top + plot_h + 20}" text-anchor="middle">{val:.2f}%</text>')
    bar_h = min(28, plot_h / len(unlabeled) * 0.62)
    gap = (plot_h - bar_h * len(unlabeled)) / max(len(unlabeled) - 1, 1)
    for i, r in enumerate(unlabeled):
        y0 = top + i * (bar_h + gap)
        value = fnum(r, "raw_pred_rate_pct")
        bw = (value / max_v) * plot_w if max_v else 0
        color = "#0f766e" if value > 0 else "#cbd5e1"
        parts.append(f'<text class="tick" x="{left - 12}" y="{y0 + bar_h * .68:.1f}" text-anchor="end">{html.escape(r["mission"])}</text>')
        parts.append(f'<rect x="{left}" y="{y0:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>')
        label_x = left + bw + 8 if bw < plot_w - 70 else left + bw - 8
        anchor = "start" if bw < plot_w - 70 else "end"
        parts.append(f'<text class="tick" x="{label_x:.1f}" y="{y0 + bar_h * .68:.1f}" text-anchor="{anchor}">{value:.3f}%</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="label" x="{left + plot_w / 2:.1f}" y="{height - 20}" text-anchor="middle">Predicted anomaly rate</text>')
    path.write_text(svg_wrap(width, height, "\n".join(parts)))


def error_decomposition(summary: list[dict[str, str]], path: Path) -> None:
    labeled = [r for r in summary if r["has_ground_truth"] == "True"]
    width, height = 860, 430
    left, top = 120, 92
    bar_w, bar_h = 520, 58
    max_total = max(fnum(r, "gt_rate_pct") + fnum(r, "fp_rate_pct") for r in labeled)
    colors = {"tp_rate_pct": "#059669", "fn_rate_pct": "#f59e0b", "fp_rate_pct": "#dc2626"}
    labels = {"tp_rate_pct": "Detected GT", "fn_rate_pct": "Missed GT", "fp_rate_pct": "False positives"}
    parts = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text class="title" x="34" y="34">Labeled Error Composition</text>',
        '<text class="subtitle" x="34" y="56">Mission 1 is near-complete; Mission 2 is precision-heavy but misses longer anomaly windows.</text>',
    ]
    for i, r in enumerate(labeled):
        y = top + i * 112
        x = left
        parts.append(f'<text class="label" x="{left - 16}" y="{y + 36}" text-anchor="end">{html.escape(r["mission"])}</text>')
        for key in ["tp_rate_pct", "fn_rate_pct", "fp_rate_pct"]:
            v = fnum(r, key)
            w = (v / max_total) * bar_w
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{colors[key]}"/>')
            if w > 28:
                parts.append(f'<text class="tick" x="{x + w / 2:.1f}" y="{y + 36}" text-anchor="middle" fill="#fff">{v:.2f}%</text>')
            x += w
        parts.append(f'<text class="tick" x="{left + bar_w + 22}" y="{y + 22}">F1 {fnum(r, "f1"):.3f}</text>')
        parts.append(f'<text class="tick" x="{left + bar_w + 22}" y="{y + 42}">P {fnum(r, "precision"):.3f} / R {fnum(r, "recall"):.3f}</text>')
    for i, key in enumerate(["tp_rate_pct", "fn_rate_pct", "fp_rate_pct"]):
        x = left + i * 150
        parts.append(f'<rect x="{x}" y="{height - 58}" width="16" height="16" fill="{colors[key]}"/>')
        parts.append(f'<text class="legend" x="{x + 24}" y="{height - 45}">{labels[key]}</text>')
    path.write_text(svg_wrap(width, height, "\n".join(parts)))


def write_report(summary: list[dict[str, str]], rows: list[dict[str, str]], path: Path) -> None:
    labeled = [r for r in summary if r["has_ground_truth"] == "True"]
    unlabeled = [r for r in summary if r["has_ground_truth"] == "False"]
    nonzero = [r for r in unlabeled if fnum(r, "raw_pred_rate_pct") > 0]
    zero = [r for r in unlabeled if fnum(r, "raw_pred_rate_pct") == 0]
    best_ratio = {}
    for mission in sorted({r["mission"] for r in rows if r["mission"].startswith("ESA")}):
        mission_rows = [r for r in rows if r["mission"] == mission]
        best_ratio[mission] = max(mission_rows, key=lambda r: fnum(r, "f1"))

    lines = [
        "# Impact Paper Result Triage",
        "",
        "## Figures Generated",
        "",
        "- `labeled_threshold_tradeoff.svg`: precision/recall/F1 against selected anomaly rate for ESA-Mission1 and ESA-Mission2.",
        "- `unlabeled_stpsat_anomaly_burden.svg`: predicted anomaly burden for each unlabeled STPSat subsystem.",
        "- `labeled_error_composition.svg`: detected ground truth, missed ground truth, and false-positive rates on labeled ESA missions.",
        "",
        "## High-Signal Findings",
        "",
    ]
    for r in labeled:
        lines.append(
            f"- {r['mission']}: F1 {fnum(r, 'f1'):.3f}, precision {fnum(r, 'precision'):.3f}, "
            f"recall {fnum(r, 'recall'):.3f}, predicted rate {fnum(r, 'raw_pred_rate_pct'):.2f}% "
            f"versus GT rate {fnum(r, 'gt_rate_pct'):.2f}%."
        )
    for mission, r in best_ratio.items():
        lines.append(
            f"- {mission} ratio sweep best F1 occurs at ratio {fnum(r, 'ratio'):.2f}, "
            f"flagging {fnum(r, 'rate_pct'):.2f}% of points "
            f"(P={fnum(r, 'precision'):.3f}, R={fnum(r, 'recall'):.3f}, F1={fnum(r, 'f1'):.3f})."
        )
    if nonzero:
        top = sorted(nonzero, key=lambda r: fnum(r, "raw_pred_rate_pct"), reverse=True)
        lines.append(
            "- Unlabeled STPSat4 has sparse predicted anomaly burden, led by "
            + ", ".join(f"{r['mission']} {fnum(r, 'raw_pred_rate_pct'):.3f}%" for r in top[:5])
            + "."
        )
    lines.append(
        f"- {len(zero)} unlabeled subsystems show zero flagged points at the Parquet threshold: "
        + ", ".join(r["mission"] for r in zero)
        + "."
    )
    lines.extend(
        [
            "",
            "## Error Audit",
            "",
        ]
    )
    for mission in ["ESA-Mission1", "ESA-Mission2"]:
        fn_path = RESULTS / mission / "evaluation" / f"{mission}_false_negatives.csv"
        fp_path = RESULTS / mission / "evaluation" / f"{mission}_false_positives.csv"
        false_negatives = read_csv(fn_path)
        false_positives = read_csv(fp_path)
        longest_fn = max(false_negatives, key=lambda r: int(r["n_points"])) if false_negatives else None
        longest_fp = max(false_positives, key=lambda r: int(r["n_points"])) if false_positives else None
        lines.append(
            f"- {mission}: {len(false_negatives)} false-negative windows and "
            f"{len(false_positives)} false-positive windows."
        )
        if longest_fn:
            lines.append(
                f"  Longest missed window spans {longest_fn['start']} to {longest_fn['end']} "
                f"({int(longest_fn['n_points']):,} points, peak score {float(longest_fn['peak_score']):.3f})."
            )
        if longest_fp:
            lines.append(
                f"  Longest false-positive island spans {longest_fp['start']} to {longest_fp['end']} "
                f"({int(longest_fp['n_points']):,} points, peak score {float(longest_fp['peak_score']):.3f})."
            )

    lines.extend(
        [
            "",
            "## Draft Hooks",
            "",
            "- Use ESA-Mission1 as the clean validation example: near-perfect precision and recall, with false positives totaling only 0.068% of all points.",
            "- Use ESA-Mission2 as the limitation/operating-point example: precision remains very high, but recall falls because the selected threshold misses a large share of labeled windows.",
            "- Use STPSat4/STPSat7 burden as an impact framing figure: deployed missions can be triaged by anomaly burden, with most unlabeled streams quiet and a small number of subsystems prioritized.",
            "- Avoid presenting unlabeled `precision`, `recall`, or `F1` as performance metrics; the current table encodes them as zeros because no ground truth exists.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_csv(RESULTS / "figures" / "parquet_anomaly_images" / "parquet_metric_summary.csv")
    ratios = ratio_rows()
    line_chart_labeled_tradeoff(ratios, OUT_DIR / "labeled_threshold_tradeoff.svg")
    bar_chart_burden(summary, OUT_DIR / "unlabeled_stpsat_anomaly_burden.svg")
    error_decomposition(summary, OUT_DIR / "labeled_error_composition.svg")
    write_report(summary, ratios, OUT_DIR / "impact_paper_result_triage.md")
    print(f"Wrote impact-paper triage to {OUT_DIR}")


if __name__ == "__main__":
    main()
