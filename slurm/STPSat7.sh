#!/bin/bash
#SBATCH --job-name=stpsat7-pipeline
#SBATCH --output=logs/stpsat7_pipeline_%j.out
#SBATCH --error=logs/stpsat7_pipeline_%j.err
#SBATCH --partition=xeon-g6430-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# Full STPSat-7 pipeline: prep → train → evaluate → export
#
# Run from UniTS root:
#   sbatch slurm/06_stpsat7_pipeline.sh
#
# To run specific subsystems only:
#   SUBSYSTEMS="EPS TC" sbatch slurm/06_stpsat7_pipeline.sh
#
# To skip prep (data already prepared):
#   SKIP_PREP=1 sbatch slurm/06_stpsat7_pipeline.sh

set -euo pipefail

mkdir -p logs

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

pip install --quiet reportlab scikit-learn pyarrow

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"

# ── Config ────────────────────────────────────────────────────────────────────
SUBSYSTEMS=${SUBSYSTEMS:-"EPS ADCS TO HRR MRR TC"}
SKIP_PREP=${SKIP_PREP:-0}
ANOMALY_RATIO=0.1
CKPT="newcheckpoints/units_x32_pretrain_checkpoint.pth"
YAML_PATH="data_provider/anomaly_detection_stpsat7.yaml"
MIN_TRAIN_ROWS=96   # UniTS needs at least seq_len rows to form one window

echo ""
echo "============================================================"
echo "STPSat-7 Pipeline"
echo "Subsystems : $SUBSYSTEMS"
echo "Skip prep  : $SKIP_PREP"
echo "============================================================"

# ── Validate checkpoint ───────────────────────────────────────────────────────
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

# ── Step 1: Data Preparation ──────────────────────────────────────────────────
if [ "$SKIP_PREP" = "0" ]; then
    echo ""
    echo "============================================================"
    echo "=== Step 1: Data Preparation ==="
    echo "============================================================"
    python scripts/prepare_stpsat7_data.py --subsystems $SUBSYSTEMS
    echo "Prep complete: $(date)"
else
    echo ""
    echo "Skipping prep (SKIP_PREP=1)"
fi

# ── Step 2: Train + Evaluate each subsystem ───────────────────────────────────
for SUBSYSTEM in $SUBSYSTEMS; do
    MISSION="STPSat7-${SUBSYSTEM}"
    MISSION_LOWER=$(echo "$MISSION" | tr '[:upper:]' '[:lower:]')

    echo ""
    echo "============================================================"
    echo "=== Step 2: Training $MISSION ==="
    echo "============================================================"

    # Check dataset exists
    TRAIN_NPY="dataset/${MISSION}/${MISSION}_train.npy"
    if [ ! -f "$TRAIN_NPY" ]; then
        echo "WARNING: $TRAIN_NPY not found — skipping $MISSION"
        continue
    fi

    # Check train AND test sets have enough rows for at least one window
    TRAIN_ROWS=$(python3 -c "import numpy as np; a=np.load('$TRAIN_NPY'); print(a.shape[0])")
    TEST_NPY="dataset/${MISSION}/${MISSION}_test.npy"
    TEST_ROWS=$(python3 -c "import numpy as np; a=np.load('$TEST_NPY'); print(a.shape[0])")

    if [ "$TRAIN_ROWS" -lt "$MIN_TRAIN_ROWS" ]; then
        echo "WARNING: $MISSION only has $TRAIN_ROWS training rows (need >=$MIN_TRAIN_ROWS)"
        echo "         Skipping — upload more data and re-run"
        continue
    fi

    if [ "$TEST_ROWS" -lt "$MIN_TRAIN_ROWS" ]; then
        echo "WARNING: $MISSION only has $TEST_ROWS test rows (need >=$MIN_TRAIN_ROWS)"
        echo "         Skipping — upload more data and re-run"
        continue
    fi

    echo "Train rows : $TRAIN_ROWS  |  Test rows : $TEST_ROWS"
    mkdir -p "results/${MISSION}"

    # Generate temp single-mission YAML
    TEMP_YAML="data_provider/tmp_${MISSION_LOWER}.yaml"
    python3 - <<PYEOF
import yaml
with open("${YAML_PATH}") as f:
    full = yaml.safe_load(f)
entry = full["task_dataset"]["${MISSION}"]
out = {"task_dataset": {"${MISSION}": entry}}
with open("${TEMP_YAML}", "w") as f:
    yaml.dump(out, f, default_flow_style=False, sort_keys=False)
print(f"  Temp YAML : ${TEMP_YAML}  (enc_in={entry['enc_in']})")
PYEOF

    PORT=$((RANDOM % 9000 + 1000))

    torchrun --nnodes 1 --nproc-per-node=1 --master_port $PORT run.py \
        --fix_seed 2021 \
        --is_training 1 \
        --subsample_pct 0.05 \
        --model_id ${MISSION_LOWER} \
        --pretrained_weight "$CKPT" \
        --model UniTS \
        --prompt_num 10 \
        --patch_len 16 \
        --stride 16 \
        --e_layers 3 \
        --d_model 32 \
        --des Exp \
        --itr 1 \
        --lradj prompt_tuning \
        --learning_rate 5e-5 \
        --weight_decay 1e-2 \
        --train_epochs 0 \
        --prompt_tune_epoch 10 \
        --batch_size 64 \
        --acc_it 8 \
        --dropout 0.0 \
        --debug offline \
        --project_name units_anomaly \
        --clip_grad 100 \
        --anomaly_ratio $ANOMALY_RATIO \
        --task_data_config_path "$TEMP_YAML"

    rm -f "$TEMP_YAML"
    echo "Training complete: $MISSION  $(date)"

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    echo ""
    echo "=== Step 3: Evaluating $MISSION ==="

    CKPT_DIR=$(ls -td checkpoints/ALL_task_${MISSION_LOWER}_UniTS_* 2>/dev/null | head -1) || true
    if [ -z "$CKPT_DIR" ]; then
        echo "WARNING: no checkpoint found for $MISSION — skipping evaluation"
        continue
    fi

    POINTS="${CKPT_DIR}/anomaly_results/${MISSION}_points.csv"
    if [ ! -f "$POINTS" ]; then
        echo "WARNING: points CSV not found at $POINTS — skipping evaluation"
        continue
    fi

    echo "Checkpoint : $CKPT_DIR"
    echo "Points CSV : $POINTS"

    # PDF report
    mkdir -p "results/${MISSION}/evaluation"
    python scripts/evaluate_anomalies.py \
        --points  "$POINTS" \
        --out     "results/${MISSION}/evaluation" \
        --mission "$MISSION"
    echo "Report: results/${MISSION}/evaluation/${MISSION}_evaluation_report.pdf"

    # Parquet export
    mkdir -p "results/${MISSION}/telemetry"
    python scripts/export_telemetry.py \
        --mission "$MISSION" \
        --points  "$POINTS" \
        --out     "results/${MISSION}/telemetry/${MISSION}_telemetry.parquet"
    echo "Parquet: results/${MISSION}/telemetry/${MISSION}_telemetry.parquet"

    # Attribution parquet
    python scripts/attribute_anomalies.py \
        --mission "$MISSION" \
        --points  "$POINTS" \
        --out     "results/${MISSION}/telemetry/${MISSION}_attribution.parquet"
    echo "Attribution: results/${MISSION}/telemetry/${MISSION}_attribution.parquet"

    echo "=== $MISSION complete: $(date) ==="
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "STPSat-7 Pipeline Complete: $(date)"
echo "============================================================"
echo ""
echo "Reports:"
for SUBSYSTEM in $SUBSYSTEMS; do
    MISSION="STPSat7-${SUBSYSTEM}"
    PDF="results/${MISSION}/evaluation/${MISSION}_evaluation_report.pdf"
    [ -f "$PDF" ] && echo "  OK  $PDF" || echo "  --  $PDF (not generated)"
done
echo ""
echo "Parquet files:"
for SUBSYSTEM in $SUBSYSTEMS; do
    MISSION="STPSat7-${SUBSYSTEM}"
    PQ="results/${MISSION}/telemetry/${MISSION}_telemetry.parquet"
    AT="results/${MISSION}/telemetry/${MISSION}_attribution.parquet"
    [ -f "$PQ" ] && echo "  OK  $PQ" || echo "  --  $PQ (not generated)"
    [ -f "$AT" ] && echo "  OK  $AT" || echo "  --  $AT (not generated)"
done