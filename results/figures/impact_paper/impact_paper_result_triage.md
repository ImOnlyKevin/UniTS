# Impact Paper Result Triage

## Figures Generated

- `labeled_threshold_tradeoff.svg`: precision/recall/F1 against selected anomaly rate for ESA-Mission1 and ESA-Mission2.
- `unlabeled_stpsat_anomaly_burden.svg`: predicted anomaly burden for each unlabeled STPSat subsystem.
- `labeled_error_composition.svg`: detected ground truth, missed ground truth, and false-positive rates on labeled ESA missions.

## High-Signal Findings

- ESA-Mission1: F1 0.981, precision 0.994, recall 0.969, predicted rate 10.83% versus GT rate 11.10%.
- ESA-Mission2: F1 0.612, precision 0.997, recall 0.442, predicted rate 1.85% versus GT rate 4.19%.
- ESA-Mission1 ratio sweep best F1 occurs at ratio 0.10, flagging 1.21% of points (P=0.921, R=0.997, F1=0.957).
- ESA-Mission2 ratio sweep best F1 occurs at ratio 0.10, flagging 0.61% of points (P=0.927, R=0.945, F1=0.936).
- Unlabeled STPSat4 has sparse predicted anomaly burden, led by STPSat4-ADCS 0.592%, STPSat4-PCE2 0.498%, STPSat4-TCS 0.254%, STPSat4-MRR 0.057%, STPSat4-HRR 0.034%.
- 7 unlabeled subsystems show zero flagged points at the Parquet threshold: STPSat4-PCE1, STPSat7-ADCS, STPSat7-EPS, STPSat7-HRR, STPSat7-MRR, STPSat7-TC, STPSat7-TO.

## Error Audit

- ESA-Mission1: 34 false-negative windows and 4274 false-positive windows.
  Longest missed window spans 2009-10-13 06:39:30 to 2009-10-15 02:27:00 (5,256 points, peak score 0.194).
  Longest false-positive island spans 2009-05-25 12:48:00 to 2009-05-25 13:09:00 (43 points, peak score 686251.100).
- ESA-Mission2: 71 false-negative windows and 20 false-positive windows.
  Longest missed window spans 2003-01-03 04:40:00 to 2003-01-04 12:34:00 (3,829 points, peak score 0.284).
  Longest false-positive island spans 2002-12-19 07:32:30 to 2002-12-19 07:33:30 (3 points, peak score 59.787).

## Draft Hooks

- Use ESA-Mission1 as the clean validation example: near-perfect precision and recall, with false positives totaling only 0.068% of all points.
- Use ESA-Mission2 as the limitation/operating-point example: precision remains very high, but recall falls because the selected threshold misses a large share of labeled windows.
- Use STPSat4/STPSat7 burden as an impact framing figure: deployed missions can be triaged by anomaly burden, with most unlabeled streams quiet and a small number of subsystems prioritized.
- Avoid presenting unlabeled `precision`, `recall`, or `F1` as performance metrics; the current table encodes them as zeros because no ground truth exists.
