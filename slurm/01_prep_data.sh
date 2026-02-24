#!/bin/bash
#SBATCH --job-name=esa-prep
#SBATCH --output=logs/esa_prep_%j.out
#SBATCH --error=logs/esa_prep_%j.err
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Run from UniTS root:  sbatch slurm/01_prep_data.sh
set -euo pipefail
mkdir -p logs

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

pip install --quiet tqdm pandas numpy scikit-learn

echo "=== Step 1: Prepare ESA-ADB data ==="
python scripts/prepare_esa_data.py

# Count channels — python used instead of wc -l to avoid off-by-one on files without trailing newline
CHAN_COUNT=$(python -c "
with open('dataset/ESA-Mission1/ESA-Mission1_channels.txt') as f:
    print(len([l for l in f.read().splitlines() if l.strip()]))
")
echo "Channel count: $CHAN_COUNT"
