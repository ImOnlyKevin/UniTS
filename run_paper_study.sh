#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

mkdir -p logs results/paper_study

STUDY_TAG=${STUDY_TAG:-paper_$(date '+%Y%m%d_%H%M%S')}
submit_output=$(sbatch --export=ALL,STUDY_TAG="$STUDY_TAG",UNITS_ROOT_DIR="$ROOT_DIR" slurm/06_paper_study.sh)

printf '%s\n' "$submit_output"
printf 'Study tag : %s\n' "$STUDY_TAG"
printf 'Results   : %s\n' "$ROOT_DIR/results/paper_study/$STUDY_TAG"
printf 'Logs      : %s\n' "$ROOT_DIR/logs/units_paper_study_<jobid>.out"
