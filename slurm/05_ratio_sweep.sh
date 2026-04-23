#!/bin/bash
#SBATCH --job-name=units-ratio-sweep
#SBATCH --output=logs/units_ratio_sweep_%j.out
#SBATCH --error=logs/units_ratio_sweep_%j.err
#SBATCH --partition=xeon-g6430-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Sweep anomaly_ratio thresholds across existing points CSVs — no GPU needed.
# Re-thresholds the saved anomaly scores without re-running UniTS.
#
# Run from UniTS root:
#   sbatch slurm/05_ratio_sweep.sh
#
# To sweep specific missions:
#   MISSIONS="STPSat4-TCS STPSat4-HRR" sbatch slurm/05_ratio_sweep.sh
#
# To use custom ratios:
#   RATIOS="0.1 0.25 0.5 1.0 2.0" sbatch slurm/05_ratio_sweep.sh

set -euo pipefail

mkdir -p logs

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

pip install --quiet reportlab scikit-learn pyarrow

# ── Config ────────────────────────────────────────────────────────────────────
MISSIONS=${MISSIONS:-"STPSat4-TCS STPSat4-HRR STPSat4-MRR STPSat4-PCE1 STPSat4-PCE2 STPSat4-ADCS"}
RATIOS=${RATIOS:-"0.1 0.25 0.5 1.0 2.0"}

echo "Missions : $MISSIONS"
echo "Ratios   : $RATIOS"

for MISSION in $MISSIONS; do
    echo ""
    echo "============================================================"
    echo "=== Ratio sweep: $MISSION ==="
    echo "============================================================"

    MISSION_LOWER=$(echo "$MISSION" | tr '[:upper:]' '[:lower:]')

    # Find points CSV
    CKPT_DIR=$(ls -td checkpoints/ALL_task_${MISSION_LOWER}_UniTS_* \
                         checkpoints/ALL_task_esa_${MISSION_LOWER}_UniTS_* \
               2>/dev/null | head -1) || true

    if [ -z "$CKPT_DIR" ]; then
        echo "ERROR: no checkpoint found for $MISSION — skipping"
        continue
    fi

    POINTS="${CKPT_DIR}/anomaly_results/${MISSION}_points.csv"
    if [ ! -f "$POINTS" ]; then
        echo "ERROR: points CSV not found at $POINTS — skipping"
        continue
    fi

    SWEEP_DIR="results/${MISSION}/ratio_sweep"
    mkdir -p "$SWEEP_DIR"

    echo "Points CSV : $POINTS"
    echo "Output dir : $SWEEP_DIR"

    # ── Run the Python sweep ──────────────────────────────────────────────────
    python3 scripts/ratio_sweep.py \
        --mission  "$MISSION" \
        --points   "$POINTS" \
        --ratios   $RATIOS \
        --out      "$SWEEP_DIR"

    echo "=== $MISSION sweep complete ==="
    echo "Comparison chart : $SWEEP_DIR/${MISSION}_ratio_comparison.pdf"
done

echo ""
echo "All sweeps complete."
echo ""
echo "Comparison charts:"
for MISSION in $MISSIONS; do
    echo "  results/${MISSION}/ratio_sweep/${MISSION}_ratio_comparison.pdf"
done