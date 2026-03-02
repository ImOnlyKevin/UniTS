#!/bin/bash
#SBATCH --job-name=esa-iforest
#SBATCH --output=logs/esa_iforest_%j.out
#SBATCH --error=logs/esa_iforest_%j.err
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Run from UniTS root:  sbatch slurm/03_baseline.sh
# No GPU or checkpoint needed. Good first sanity check of the data pipeline.
set -euo pipefail
mkdir -p logs

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

echo "=== Isolation Forest Baseline ==="
python scripts/infer_anomalies.py --iforest --anomaly_ratio 1.0

echo ""
echo "Results in results/ESA-Mission1/"
cat results/ESA-Mission1/ESA-Mission1_metrics.csv