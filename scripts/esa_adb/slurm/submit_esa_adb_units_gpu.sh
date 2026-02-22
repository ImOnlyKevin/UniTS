#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /path/to/UniTS /path/to/checkpoint.pth [extra sbatch args...]"
  exit 1
fi

UNITS_REPO="$1"
UNITS_CKPT="$2"
shift 2

export UNITS_REPO
export UNITS_CKPT

mkdir -p "$UNITS_REPO/logs"

sbatch "$@" "$UNITS_REPO/scripts/esa_adb/slurm/esa_adb_units_gpu.sbatch"
