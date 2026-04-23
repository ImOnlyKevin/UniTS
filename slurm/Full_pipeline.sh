#!/bin/bash
#SBATCH --job-name=units-pipeline
#SBATCH --output=logs/units_pipeline_%j.out
#SBATCH --error=logs/units_pipeline_%j.err
#SBATCH --partition=xeon-g6430-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

# Full UniTS pipeline: prep → train → evaluate → export → attribute
# Supports ESA, STPSat-4, and STPSat-7 missions.
#
# Run from UniTS root:
#   sbatch slurm/07_full_pipeline.sh
#
# Examples:
#   # All STPSat-7 subsystems (default)
#   sbatch slurm/07_full_pipeline.sh
#
#   # Specific satellite
#   SATELLITE=STPSat4 sbatch slurm/07_full_pipeline.sh
#   SATELLITE=ESA sbatch slurm/07_full_pipeline.sh
#   SATELLITE=STPSat7 sbatch slurm/07_full_pipeline.sh
#
#   # Specific missions (overrides SATELLITE)
#   MISSIONS="STPSat7-EPS STPSat7-TC" sbatch slurm/07_full_pipeline.sh
#   MISSIONS="ESA-Mission1 ESA-Mission2" sbatch slurm/07_full_pipeline.sh
#   MISSIONS="STPSat4-TCS STPSat7-EPS" sbatch slurm/07_full_pipeline.sh
#
#   # Skip data prep (datasets already built)
#   SKIP_PREP=1 SATELLITE=STPSat7 sbatch slurm/07_full_pipeline.sh
#
#   # Skip training (re-evaluate/export only)
#   SKIP_TRAIN=1 MISSIONS="STPSat7-EPS" sbatch slurm/07_full_pipeline.sh

set -euo pipefail

mkdir -p logs

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

pip install --quiet reportlab scikit-learn pyarrow

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Started: $(date)"

# ── Config ────────────────────────────────────────────────────────────────────
SATELLITE=${SATELLITE:-"STPSat7"}
SKIP_PREP=${SKIP_PREP:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
ANOMALY_RATIO=0.1
CKPT="newcheckpoints/units_x32_pretrain_checkpoint.pth"
MIN_TRAIN_ROWS=96

# ── Resolve MISSIONS from SATELLITE if not explicitly set ─────────────────────
if [ -z "${MISSIONS:-}" ]; then
    case "$SATELLITE" in
        ESA)
            MISSIONS="ESA-Mission1 ESA-Mission2" ;;
        STPSat4)
            MISSIONS="STPSat4-TCS STPSat4-HRR STPSat4-MRR STPSat4-PCE1 STPSat4-PCE2 STPSat4-ADCS" ;;
        STPSat7)
            MISSIONS="STPSat7-EPS STPSat7-ADCS STPSat7-TO STPSat7-HRR STPSat7-MRR STPSat7-TC" ;;
        *)
            echo "ERROR: Unknown SATELLITE='$SATELLITE'"
            echo "       Valid options: ESA, STPSat4, STPSat7"
            echo "       Or set MISSIONS directly: MISSIONS='ESA-Mission1 STPSat7-EPS' sbatch ..."
            exit 1 ;;
    esac
fi

# ── Resolve YAML per mission ──────────────────────────────────────────────────
get_yaml() {
    local mission=$1
    case "$mission" in
        ESA*)     echo "data_provider/anomaly_detection_esa.yaml" ;;
        STPSat4*) echo "data_provider/anomaly_detection_stpsat4.yaml" ;;
        STPSat7*) echo "data_provider/anomaly_detection_stpsat7.yaml" ;;
        *)
            echo "ERROR: Cannot auto-detect YAML for mission '$mission'" >&2
            exit 1 ;;
    esac
}

# ── Resolve prep script per mission ──────────────────────────────────────────
get_prep_script() {
    local mission=$1
    case "$mission" in
        ESA*)     echo "scripts/prepare_esa_data.py" ;;
        STPSat4*) echo "scripts/prepare_stpsat4_data.py" ;;
        STPSat7*) echo "scripts/prepare_stpsat7_data.py" ;;
    esac
}

# ── Resolve subsystem name for prep scripts ───────────────────────────────────
get_subsystem() {
    # Extracts the part after the dash: STPSat7-EPS → EPS, ESA-Mission1 → Mission1
    echo "$1" | cut -d'-' -f2-
}

echo ""
echo "============================================================"
echo "UniTS Full Pipeline"
echo "Satellite  : $SATELLITE"
echo "Missions   : $MISSIONS"
echo "Skip prep  : $SKIP_PREP"
echo "Skip train : $SKIP_TRAIN"
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

    # Group missions by prep script to avoid running the same script multiple times
    ESA_MISSIONS=""
    SAT4_MISSIONS=""
    SAT7_MISSIONS=""
    for MISSION in $MISSIONS; do
        case "$MISSION" in
            ESA*)     ESA_MISSIONS="$ESA_MISSIONS $MISSION" ;;
            STPSat4*) SAT4_MISSIONS="$SAT4_MISSIONS $(get_subsystem $MISSION)" ;;
            STPSat7*) SAT7_MISSIONS="$SAT7_MISSIONS $(get_subsystem $MISSION)" ;;
        esac
    done

    if [ -n "$ESA_MISSIONS" ]; then
        echo "Running ESA data prep ..."
        python scripts/prepare_esa_data.py
    fi
    if [ -n "$SAT4_MISSIONS" ]; then
        echo "Running STPSat-4 data prep for:$SAT4_MISSIONS ..."
        python scripts/prepare_stpsat4_data.py --subsystems $SAT4_MISSIONS
    fi
    if [ -n "$SAT7_MISSIONS" ]; then
        echo "Running STPSat-7 data prep for:$SAT7_MISSIONS ..."
        python scripts/prepare_stpsat7_data.py --subsystems $SAT7_MISSIONS
    fi

    echo "Prep complete: $(date)"
else
    echo "Skipping prep (SKIP_PREP=1)"
fi

# ── Step 2 + 3: Train → Evaluate → Export per mission ────────────────────────
for MISSION in $MISSIONS; do
    MISSION_LOWER=$(echo "$MISSION" | tr '[:upper:]' '[:lower:]')
    YAML_PATH=$(get_yaml "$MISSION")

    echo ""
    echo "============================================================"
    echo "=== Processing $MISSION ==="
    echo "============================================================"

    # ── Train ─────────────────────────────────────────────────────────────────
    if [ "$SKIP_TRAIN" = "0" ]; then
        TRAIN_NPY="dataset/${MISSION}/${MISSION}_train.npy"
        if [ ! -f "$TRAIN_NPY" ]; then
            echo "WARNING: $TRAIN_NPY not found — skipping $MISSION"
            continue
        fi

        TRAIN_ROWS=$(python3 -c "import numpy as np; a=np.load('$TRAIN_NPY'); print(a.shape[0])")
        TEST_NPY="dataset/${MISSION}/${MISSION}_test.npy"
        TEST_ROWS=$(python3 -c "import numpy as np; a=np.load('$TEST_NPY'); print(a.shape[0])")

        if [ "$TRAIN_ROWS" -lt "$MIN_TRAIN_ROWS" ]; then
            echo "WARNING: $MISSION only has $TRAIN_ROWS training rows (need >=$MIN_TRAIN_ROWS) — skipping"
            continue
        fi
        if [ "$TEST_ROWS" -lt "$MIN_TRAIN_ROWS" ]; then
            echo "WARNING: $MISSION only has $TEST_ROWS test rows (need >=$MIN_TRAIN_ROWS) — skipping"
            continue
        fi

        echo "Train rows : $TRAIN_ROWS  |  Test rows : $TEST_ROWS"
        mkdir -p "results/${MISSION}"

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
            --subsample_pct 0.2 \
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
            --prompt_tune_epoch 30 \
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
    else
        echo "Skipping training (SKIP_TRAIN=1)"
    fi

    # ── Evaluate ──────────────────────────────────────────────────────────────
    CKPT_DIR=$(ls -td checkpoints/ALL_task_${MISSION_LOWER}_UniTS_* \
                        checkpoints/ALL_task_esa_${MISSION_LOWER}_UniTS_* \
               2>/dev/null | head -1) || true

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

    # Telemetry parquet
    mkdir -p "results/${MISSION}/telemetry"
    python scripts/export_telemetry.py \
        --mission "$MISSION" \
        --points  "$POINTS" \
        --out     "results/${MISSION}/telemetry/${MISSION}_telemetry.parquet"
    echo "Telemetry: results/${MISSION}/telemetry/${MISSION}_telemetry.parquet"

    echo "=== $MISSION complete: $(date) ==="
done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Pipeline Complete: $(date)"
echo "============================================================"
echo ""
echo "Reports:"
for MISSION in $MISSIONS; do
    PDF="results/${MISSION}/evaluation/${MISSION}_evaluation_report.pdf"
    [ -f "$PDF" ] && echo "  OK  $PDF" || echo "  --  $PDF (not generated)"
done
echo ""
echo "Telemetry parquets:"
for MISSION in $MISSIONS; do
    PQ="results/${MISSION}/telemetry/${MISSION}_telemetry.parquet"
    [ -f "$PQ" ] && echo "  OK  $PQ" || echo "  --  $PQ (not generated)"
done