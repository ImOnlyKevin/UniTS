#!/bin/bash
#SBATCH --job-name=units-eval
#SBATCH --output=logs/units_eval_%j.out
#SBATCH --error=logs/units_eval_%j.err
#SBATCH --partition=xeon-g6430-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Run from UniTS root:
#   sbatch slurm/03_evaluate.sh
#
# To evaluate specific missions:
#   MISSIONS="STPSat4-TCS STPSat4-HRR" sbatch slurm/03_evaluate.sh
#   MISSIONS="ESA-Mission1 ESA-Mission2" sbatch slurm/03_evaluate.sh

set -euo pipefail

mkdir -p logs

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

pip install --quiet reportlab scikit-learn

MISSIONS=${MISSIONS:-"STPSat4-TCS STPSat4-HRR STPSat4-MRR STPSat4-PCE1 STPSat4-PCE2 STPSat4-ADCS"}

echo "Missions to evaluate: $MISSIONS"

for MISSION in $MISSIONS; do
    echo ""
    echo "============================================================"
    echo "=== Evaluating $MISSION ==="
    echo "============================================================"

    MISSION_LOWER=$(echo "$MISSION" | tr '[:upper:]' '[:lower:]')

    # Find the most recent checkpoint dir for this mission
    CKPT_DIR=$(ls -td checkpoints/ALL_task_${MISSION_LOWER}_UniTS_* checkpoints/ALL_task_esa_${MISSION_LOWER}_UniTS_* checkpoints/ALL_task_units_anomaly_${MISSION_LOWER}_UniTS_* 2>/dev/null | head -1) || true

    if [ -z "$CKPT_DIR" ]; then
        echo "ERROR: no checkpoint directory found for $MISSION"
        echo "       Expected pattern: checkpoints/ALL_task_${MISSION_LOWER}_UniTS_*"
        echo "       Run sbatch slurm/02_run_anomaly.sh first"
        exit 1
    fi

    POINTS="${CKPT_DIR}/anomaly_results/${MISSION}_points.csv"

    if [ ! -f "$POINTS" ]; then
        echo "ERROR: points CSV not found at $POINTS"
        exit 1
    fi

    echo "Checkpoint : $CKPT_DIR"
    echo "Points CSV : $POINTS"

    # ── PDF evaluation report ─────────────────────────────────────────────────
    mkdir -p "results/${MISSION}/evaluation"

    python scripts/evaluate_anomalies.py \
        --points "$POINTS" \
        --out    "results/${MISSION}/evaluation" \
        --mission "$MISSION"

    echo "Report: results/${MISSION}/evaluation/${MISSION}_evaluation_report.pdf"

    # ── Full telemetry CSV export ─────────────────────────────────────────────
    echo ""
    echo "Exporting telemetry CSV for $MISSION ..."
    mkdir -p "results/${MISSION}/telemetry"

    python scripts/export_telemetry.py \
        --mission "$MISSION" \
        --points  "$POINTS" \
        --out     "results/${MISSION}/telemetry/${MISSION}_telemetry.parquet"

    echo "Telemetry CSV: results/${MISSION}/telemetry/${MISSION}_telemetry.parquet"
    echo "=== $MISSION complete ==="
done

echo ""
echo "All evaluations and exports complete."
echo ""
echo "Reports:"
for MISSION in $MISSIONS; do
    echo "  results/${MISSION}/evaluation/${MISSION}_evaluation_report.pdf"
done
echo ""
echo "Telemetry CSVs:"
for MISSION in $MISSIONS; do
    echo "  results/${MISSION}/telemetry/${MISSION}_telemetry.parquet"
done