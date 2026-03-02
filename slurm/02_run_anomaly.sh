#!/bin/bash
#SBATCH --job-name=esa-units
#SBATCH --output=logs/esa_units_%j.out
#SBATCH --error=logs/esa_units_%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

# Run from UniTS root:  sbatch slurm/02_run_anomaly.sh
# To run a single mission:  MISSIONS=ESA-Mission1 sbatch slurm/02_run_anomaly.sh
set -euo pipefail
mkdir -p logs results/ESA-Mission1 results/ESA-Mission2

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

CKPT="newcheckpoints/units_x32_pretrain_checkpoint.pth"
ANOMALY_RATIO=1.0

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    echo "Download on a login node with:"
    echo "  wget -O $CKPT https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x32_pretrain_checkpoint.pth"
    exit 1
fi

# Allow caller to override which missions to run, e.g.:
#   MISSIONS=ESA-Mission2 sbatch slurm/02_run_anomaly.sh
MISSIONS=${MISSIONS:-"ESA-Mission1 ESA-Mission2"}

for MISSION in $MISSIONS; do
    echo ""
    echo "============================================================"
    echo "=== Running UniTS on $MISSION ==="
    echo "============================================================"

    if [ ! -f "dataset/${MISSION}/${MISSION}_train.npy" ]; then
        echo "ERROR: dataset/${MISSION}/ not found — run sbatch slurm/01_prep_data.sh first"
        exit 1
    fi

    PORT=$((RANDOM % 9000 + 1000))

    torchrun --nnodes 1 --nproc-per-node=1 --master_port $PORT run.py \
        --fix_seed 2021 \
        --is_training 1 \
        --subsample_pct 0.05 \
        --model_id esa_${MISSION,,} \
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
        --project_name esa_anomaly \
        --clip_grad 100 \
        --anomaly_ratio $ANOMALY_RATIO \
        --task_data_config_path data_provider/anomaly_detection_esa.yaml

    echo "=== $MISSION complete ==="
done

echo ""
echo "All missions done."