#!/bin/bash
#SBATCH --job-name=units-anomaly
#SBATCH --output=logs/units_anomaly_%j.out
#SBATCH --error=logs/units_anomaly_%j.err
#SBATCH --partition=xeon-g6430-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

# Run from UniTS root:
#   sbatch slurm/02_run_anomaly.sh
#
# To run specific missions:
#   MISSIONS="ESA-Mission1 ESA-Mission2" sbatch slurm/02_run_anomaly.sh
#   MISSIONS="STPSat4-TCS STPSat4-PCE1" sbatch slurm/02_run_anomaly.sh
#   MISSIONS="STPSat7-EPS STPSat7-ADCS" sbatch slurm/02_run_anomaly.sh
#
# To override YAML manually:
#   YAML_PATH=data_provider/anomaly_detection_esa.yaml sbatch slurm/02_run_anomaly.sh

set -euo pipefail

mkdir -p logs

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

CKPT="newcheckpoints/units_x32_pretrain_checkpoint.pth"
ANOMALY_RATIO=0.1

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    echo "Download on a login node with:"
    echo "  wget -O $CKPT https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x32_pretrain_checkpoint.pth"
    exit 1
fi

# Allow caller to override which missions to run
MISSIONS=${MISSIONS:-"STPSat4-TCS STPSat4-PCE1 STPSat4-PCE2 STPSat4-HRR STPSat4-MRR STPSat4-ADCS"}

# Auto-select source YAML based on first mission name, or allow manual override
if [ -z "${YAML_PATH:-}" ]; then
    FIRST_MISSION=$(echo "$MISSIONS" | awk '{print $1}')
    case "$FIRST_MISSION" in
        ESA*)
            YAML_PATH="data_provider/anomaly_detection_esa.yaml" ;;
        STPSat4*)
            YAML_PATH="data_provider/anomaly_detection_stpsat4.yaml" ;;
        STPSat7*)
            YAML_PATH="data_provider/anomaly_detection_stpsat7.yaml" ;;
        *)
            echo "ERROR: Cannot auto-detect YAML for mission '$FIRST_MISSION'"
            echo "       Set manually: YAML_PATH=data_provider/your.yaml sbatch slurm/02_run_anomaly.sh"
            exit 1 ;;
    esac
fi

if [ ! -f "$YAML_PATH" ]; then
    echo "ERROR: YAML config not found at $YAML_PATH"
    exit 1
fi

echo "Source YAML : $YAML_PATH"

for MISSION in $MISSIONS; do
    echo ""
    echo "============================================================"
    echo "=== Running UniTS on $MISSION ==="
    echo "============================================================"

    if [ ! -f "dataset/${MISSION}/${MISSION}_train.npy" ]; then
        echo "ERROR: dataset/${MISSION}/ not found — run prep script first"
        exit 1
    fi

    # Create results dir for this mission
    mkdir -p "results/${MISSION}"

    # Derive a clean lowercase slug for model_id
    MISSION_SLUG=$(echo "$MISSION" | tr '[:upper:]' '[:lower:]')

    # Generate a temporary single-mission YAML so UniTS only trains on this mission.
    # UniTS loads ALL entries in the YAML regardless of --model_id, so we must
    # extract just the one entry we want.
    TEMP_YAML="data_provider/tmp_${MISSION_SLUG}.yaml"
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
        --model_id ${MISSION_SLUG} \
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
    echo "=== $MISSION complete ==="
done

echo ""
echo "All missions done."