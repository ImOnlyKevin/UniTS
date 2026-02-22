#!/bin/bash
#SBATCH --job-name=esa_adb_units
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=xeon-g6-volta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

# Required environment variables:
#   UNITS_REPO=/path/to/UniTS
#   UNITS_CKPT=/path/to/checkpoint.pth
# Optional:
#   RAW_ROOT=$UNITS_REPO/data/ESA-ADB-raw
#   OUT_DIR=$UNITS_REPO/outputs/esa_adb_units
#   MISSIONS="ESA-Mission1 ESA-Mission2"
#   SEQ_LEN=96
#   BATCH_SIZE=32
#   ANOMALY_RATIO=1.0
#   EXTRA_ARGS="--max-channels 64"

: "${UNITS_REPO:?Please set UNITS_REPO}"
: "${UNITS_CKPT:?Please set UNITS_CKPT}"

RAW_ROOT="${RAW_ROOT:-$UNITS_REPO/data/ESA-ADB-raw}"
OUT_DIR="${OUT_DIR:-$UNITS_REPO/outputs/esa_adb_units}"
MISSIONS="${MISSIONS:-ESA-Mission1 ESA-Mission2}"
SEQ_LEN="${SEQ_LEN:-96}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ANOMALY_RATIO="${ANOMALY_RATIO:-1.0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$UNITS_REPO"
mkdir -p logs "$OUT_DIR"

echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA devices: ${CUDA_VISIBLE_DEVICES:-not-set}"
python --version

python scripts/esa_adb/run_esa_adb_units.py \
  --checkpoint "$UNITS_CKPT" \
  --raw-root "$RAW_ROOT" \
  --missions $MISSIONS \
  --output-dir "$OUT_DIR" \
  --seq-len "$SEQ_LEN" \
  --batch-size "$BATCH_SIZE" \
  --anomaly-ratio "$ANOMALY_RATIO" \
  $EXTRA_ARGS


echo "[INFO] Completed ESA-ADB UniTS run. Outputs in $OUT_DIR"
