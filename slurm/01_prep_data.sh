#!/bin/bash
#SBATCH --job-name=esa-prep
#SBATCH --output=logs/esa_prep_%j.out
#SBATCH --error=logs/esa_prep_%j.err
#SBATCH --partition=xeon-p8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Run from UniTS root:  sbatch slurm/01_prep_data.sh
set -euo pipefail
mkdir -p logs

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

pip install --quiet tqdm pandas numpy scikit-learn

# Allow caller to target one or both missions, e.g.:
#   MISSIONS=ESA-Mission2 sbatch slurm/01_prep_data.sh
MISSIONS=${MISSIONS:-"ESA-Mission1 ESA-Mission2"}
echo "Missions to prep: $MISSIONS"

for MISSION in $MISSIONS; do
    echo ""
    echo "=== Preparing $MISSION ==="

    case "$MISSION" in
        ESA-Mission1)
            TRAIN_END="2006-10-01"
            TEST_START="2007-01-01"
            ;;
        ESA-Mission2)
            TRAIN_END="2002-09-01"
            TEST_START="2002-10-01"
            ;;
        *)
            echo "ERROR: unknown mission '$MISSION' — add its date range to this script"
            exit 1
            ;;
    esac

    python scripts/prepare_esa_data.py \
        --raw_dir  "data/ESA-ADB-raw/${MISSION}" \
        --out_dir  "dataset/${MISSION}" \
        --name     "$MISSION" \
        --train_end  "$TRAIN_END" \
        --test_start "$TEST_START"

    echo "${MISSION} prep done."
    ls -lh "dataset/${MISSION}/"
done

echo ""
echo "=== All prep complete ==="