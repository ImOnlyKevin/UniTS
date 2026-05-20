#!/bin/bash
#SBATCH --job-name=units-paper-study
#SBATCH --output=logs/units_paper_study_%j.out
#SBATCH --error=logs/units_paper_study_%j.err
#SBATCH --partition=xeon-g6430-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=128:00:00

# Paper-oriented anomaly-detection study:
#   - prepares requested missions
#   - sweeps prompt-tuning and/or finetune settings
#   - saves per-run evaluation bundles in one study directory
#   - builds a centralized report with figures, tables, README, and PDF
#
# Typical usage from the UniTS repo root:
#   sbatch slurm/06_paper_study.sh
#
# Mission selection follows slurm/Full_pipeline.sh:
#   # All STPSat-7 subsystems (default)
#   sbatch slurm/06_paper_study.sh
#
#   # Specific satellite
#   SATELLITE=STPSat4 sbatch slurm/06_paper_study.sh
#   SATELLITE=ESA sbatch slurm/06_paper_study.sh
#   SATELLITE=STPSat7 sbatch slurm/06_paper_study.sh
#
#   # Specific missions (overrides SATELLITE)
#   MISSIONS="STPSat7-EPS STPSat7-TC" sbatch slurm/06_paper_study.sh
#   MISSIONS="ESA-Mission1 ESA-Mission2" sbatch slurm/06_paper_study.sh
#   MISSIONS="STPSat4-TCS STPSat7-EPS" sbatch slurm/06_paper_study.sh
#
# Sweep controls:
#   SUBSAMPLE_PCTS="0.05 0.10 0.20 0.30" PROMPT_TUNE_EPOCHS="5 10 20 40" sbatch slurm/06_paper_study.sh
#   STUDY_MODES="prompt_tuning finetune" FINETUNE_EPOCHS="5 10" sbatch slurm/06_paper_study.sh
#   STUDY_PRESET=fast NUM_WORKERS=6 SATELLITE=ESA sbatch slurm/06_paper_study.sh
#   SKIP_PREP=1 SKIP_TRAIN=1 STUDY_DIR=/abs/path/to/existing_study sbatch slurm/06_paper_study.sh

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
elif [[ -n "${UNITS_ROOT_DIR:-}" ]]; then
    ROOT_DIR="$UNITS_ROOT_DIR"
else
    ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fi
cd "$ROOT_DIR"
mkdir -p logs

module unload anaconda/2023b 2>/dev/null || true
module load anaconda/2023a-pytorch
source activate ARGUS

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

check_python_deps() {
    python3 - <<'PY'
import importlib.util
modules = ["numpy", "pandas", "matplotlib", "sklearn", "reportlab", "yaml"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
PY
}

normalize_token() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | tr '. ' 'p_'
}

get_satellite() {
    local mission=$1
    case "$mission" in
        ESA*) echo "ESA" ;;
        STPSat4*) echo "STPSat4" ;;
        STPSat7*) echo "STPSat7" ;;
        *) echo "unknown" ;;
    esac
}

get_yaml() {
    local mission=$1
    case "$mission" in
        ESA*) echo "$ROOT_DIR/data_provider/anomaly_detection_esa.yaml" ;;
        STPSat4*) echo "$ROOT_DIR/data_provider/anomaly_detection_stpsat4.yaml" ;;
        STPSat7*) echo "$ROOT_DIR/data_provider/anomaly_detection_stpsat7.yaml" ;;
        *)
            echo "Unsupported mission: $mission" >&2
            return 1
            ;;
    esac
}

get_subsystem() {
    echo "$1" | cut -d'-' -f2-
}

append_manifest() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" "${18}" \
        >> "$MANIFEST_PATH"
}

write_manifest_header() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "run_id" "mission" "satellite" "mode" "subsample_pct" "prompt_tune_epoch" "train_epochs" \
        "base_anomaly_ratio" "ratio_grid" "status" "dataset_dir" "points_csv" "windows_csv" \
        "checkpoint_dir" "evaluation_pdf" "ratio_sweep_csv" "ratio_sweep_pdf" "notes" \
        > "$MANIFEST_PATH"
}

make_single_mission_yaml() {
    local template_path=$1
    local mission=$2
    local out_path=$3
    python3 - "$template_path" "$mission" "$out_path" <<'PY'
import sys
import yaml

template_path, mission, out_path = sys.argv[1:4]
with open(template_path, "r") as handle:
    data = yaml.safe_load(handle) or {}

task_dataset = data.get("task_dataset", data)
if mission not in task_dataset:
    raise SystemExit(f"Mission '{mission}' not found in {template_path}")

payload = {"task_dataset": {mission: task_dataset[mission]}}
with open(out_path, "w") as handle:
    yaml.safe_dump(payload, handle, sort_keys=False)
PY
}

validate_dataset() {
    local mission=$1
    local yaml_path=${2:-}
    local train_npy="$ROOT_DIR/dataset/${mission}/${mission}_train.npy"
    local test_npy="$ROOT_DIR/dataset/${mission}/${mission}_test.npy"

    if [[ ! -f "$train_npy" || ! -f "$test_npy" ]]; then
        log "Dataset files missing for $mission"
        return 1
    fi

    python3 - "$mission" "$train_npy" "$test_npy" "$yaml_path" "$MIN_TRAIN_ROWS" <<'PY'
import sys
import numpy as np
import yaml

mission, train_npy, test_npy, yaml_path, min_rows_raw = sys.argv[1:6]
min_rows = int(min_rows_raw)

train_shape = np.load(train_npy, mmap_mode="r").shape
test_shape = np.load(test_npy, mmap_mode="r").shape

if train_shape[0] < min_rows or test_shape[0] < min_rows:
    raise SystemExit(
        f"Dataset for {mission} is too small: "
        f"train={train_shape[0]} test={test_shape[0]}"
    )

if train_shape[1] != test_shape[1]:
    raise SystemExit(
        f"Dataset column mismatch for {mission}: "
        f"train={train_shape[1]} test={test_shape[1]}"
    )

if yaml_path:
    with open(yaml_path, "r") as handle:
        config = yaml.safe_load(handle) or {}
    entry = config.get("task_dataset", {}).get(mission)
    if entry is None:
        raise SystemExit(f"Mission {mission} missing from YAML {yaml_path}")
    enc_in = int(entry["enc_in"])
    if train_shape[1] != enc_in:
        raise SystemExit(
            f"YAML/dataset channel mismatch for {mission}: "
            f"enc_in={enc_in} train_columns={train_shape[1]}"
        )
PY
}

prepare_requested_data() {
    [[ "$SKIP_PREP" == "1" ]] && {
        log "Skipping data preparation (SKIP_PREP=1)"
        return 0
    }

    local esa_missions=()
    local stpsat4_subsystems=()
    local stpsat7_subsystems=()
    local mission

    for mission in $MISSIONS; do
        case "$(get_satellite "$mission")" in
            ESA) esa_missions+=("$mission") ;;
            STPSat4) stpsat4_subsystems+=("$(get_subsystem "$mission")") ;;
            STPSat7) stpsat7_subsystems+=("$(get_subsystem "$mission")") ;;
        esac
    done

    if (( ${#esa_missions[@]} )); then
        log "Running ESA data prep ..."
        local esa_cmd=(python3 scripts/prepare_esa_data.py)
        [[ -n "$ESA_TRAIN_END" ]] && esa_cmd+=(--train_end "$ESA_TRAIN_END")
        [[ -n "$ESA_TEST_START" ]] && esa_cmd+=(--test_start "$ESA_TEST_START")
        "${esa_cmd[@]}"
    fi

    if (( ${#stpsat4_subsystems[@]} )); then
        log "Preparing STPSat-4 subsystems: ${stpsat4_subsystems[*]}"
        if [[ -n "$STPSAT4_SPLIT_DATE" ]]; then
            python3 scripts/prepare_stpsat4_data.py \
                --subsystems "${stpsat4_subsystems[@]}" \
                --split_date "$STPSAT4_SPLIT_DATE"
        else
            python3 scripts/prepare_stpsat4_data.py \
                --subsystems "${stpsat4_subsystems[@]}"
        fi
    fi

    if (( ${#stpsat7_subsystems[@]} )); then
        log "Preparing STPSat-7 subsystems: ${stpsat7_subsystems[*]}"
        if [[ -n "$STPSAT7_SPLIT_DATE" ]]; then
            python3 scripts/prepare_stpsat7_data.py \
                --subsystems "${stpsat7_subsystems[@]}" \
                --split_date "$STPSAT7_SPLIT_DATE"
        else
            python3 scripts/prepare_stpsat7_data.py \
                --subsystems "${stpsat7_subsystems[@]}"
        fi
    fi
}

run_torch_experiment() {
    local run_id=$1
    local mission=$2
    local mode=$3
    local subsample_pct=$4
    local prompt_tune_epoch=$5
    local train_epochs=$6
    local yaml_path=$7
    local run_dir=$8

    local learning_rate=$PROMPT_LR
    local weight_decay=$PROMPT_WEIGHT_DECAY
    local lradj="prompt_tuning"
    if [[ "$mode" == "finetune" ]]; then
        learning_rate=$FINETUNE_LR
        weight_decay=$FINETUNE_WEIGHT_DECAY
        lradj="finetune_anl"
    fi

    local train_log="$run_dir/train.log"
    local port=$((RANDOM % 9000 + 1000))

    torchrun --nnodes 1 --nproc-per-node=1 --master_port "$port" run.py \
        --fix_seed "$FIX_SEED" \
        --is_training 1 \
        --subsample_pct "$subsample_pct" \
        --model_id "$run_id" \
        --pretrained_weight "$CKPT" \
        --model UniTS \
        --prompt_num "$PROMPT_NUM" \
        --patch_len "$PATCH_LEN" \
        --stride "$STRIDE" \
        --e_layers "$E_LAYERS" \
        --d_model "$D_MODEL" \
        --des "$DES" \
        --itr 1 \
        --lradj "$lradj" \
        --learning_rate "$learning_rate" \
        --weight_decay "$weight_decay" \
        --train_epochs "$train_epochs" \
        --prompt_tune_epoch "$prompt_tune_epoch" \
        --batch_size "$BATCH_SIZE" \
        --acc_it "$ACC_IT" \
        --num_workers "$NUM_WORKERS" \
        --dropout "$DROPOUT" \
        --debug "$WANDB_MODE" \
        --project_name "$PROJECT_NAME" \
        --clip_grad 100 \
        --anomaly_ratio "$BASE_ANOMALY_RATIO" \
        --task_data_config_path "$yaml_path" \
        > "$train_log" 2>&1
}

process_run() {
    local mission=$1
    local mode=$2
    local subsample_pct=$3
    local prompt_tune_epoch=$4
    local train_epochs=$5

    local satellite
    satellite=$(get_satellite "$mission")
    local subsample_tag
    subsample_tag=$(normalize_token "$subsample_pct")
    local run_id
    run_id="$(normalize_token "${mission}__${mode}__pct${subsample_tag}__pt${prompt_tune_epoch}__ft${train_epochs}")"
    local run_dir="$STUDY_DIR/runs/$mission/$run_id"
    local tmp_yaml="$TMP_DIR/${run_id}.yaml"
    local dataset_dir="$ROOT_DIR/dataset/$mission"
    local template_yaml
    template_yaml=$(get_yaml "$mission") || return 1

    mkdir -p "$run_dir"
    make_single_mission_yaml "$template_yaml" "$mission" "$tmp_yaml" || {
        append_manifest "$run_id" "$mission" "$satellite" "$mode" "$subsample_pct" "$prompt_tune_epoch" "$train_epochs" \
            "$BASE_ANOMALY_RATIO" "$RATIOS" "failed_yaml" "$dataset_dir" "" "" "" "" "" "" "Could not create single-mission YAML"
        return 1
    }

    if ! validate_dataset "$mission" "$tmp_yaml"; then
        append_manifest "$run_id" "$mission" "$satellite" "$mode" "$subsample_pct" "$prompt_tune_epoch" "$train_epochs" \
            "$BASE_ANOMALY_RATIO" "$RATIOS" "failed_dataset" "$dataset_dir" "" "" "" "" "" "" "Dataset validation failed"
        rm -f "$tmp_yaml"
        return 1
    fi

    local setting="ALL_task_${run_id}_UniTS_All_ftM_dm${D_MODEL}_el${E_LAYERS}_${DES}_0"
    local checkpoint_dir="$ROOT_DIR/${CHECKPOINTS_DIR}/${setting}"
    local points_dst="$run_dir/${mission}_points.csv"
    local windows_dst="$run_dir/${mission}_windows.csv"
    local eval_pdf="$run_dir/evaluation/${mission}_evaluation_report.pdf"
    local ratio_csv="$run_dir/ratio_sweep/${mission}_ratio_sweep.csv"
    local ratio_pdf="$run_dir/ratio_sweep/${mission}_ratio_comparison.pdf"
    local notes="ok"

    if [[ "$SKIP_TRAIN" != "1" ]]; then
        log "Training $mission | mode=$mode | pct=$subsample_pct | pt=$prompt_tune_epoch | ft=$train_epochs"
        if ! run_torch_experiment "$run_id" "$mission" "$mode" "$subsample_pct" "$prompt_tune_epoch" "$train_epochs" "$tmp_yaml" "$run_dir"; then
            notes="torchrun_failed"
            append_manifest "$run_id" "$mission" "$satellite" "$mode" "$subsample_pct" "$prompt_tune_epoch" "$train_epochs" \
                "$BASE_ANOMALY_RATIO" "$RATIOS" "failed_train" "$dataset_dir" "" "" "$checkpoint_dir" "" "" "" "$notes"
            rm -f "$tmp_yaml"
            return 1
        fi
    else
        log "Skipping training for $run_id (SKIP_TRAIN=1)"
    fi

    local points_src="$checkpoint_dir/anomaly_results/${mission}_points.csv"
    local windows_src="$checkpoint_dir/anomaly_results/${mission}_windows.csv"
    if [[ -f "$points_dst" ]]; then
        points_src="$points_dst"
    fi
    if [[ -f "$windows_dst" ]]; then
        windows_src="$windows_dst"
    fi
    if [[ ! -f "$points_src" ]]; then
        notes="missing_points_csv"
        append_manifest "$run_id" "$mission" "$satellite" "$mode" "$subsample_pct" "$prompt_tune_epoch" "$train_epochs" \
            "$BASE_ANOMALY_RATIO" "$RATIOS" "failed_points" "$dataset_dir" "" "" "$checkpoint_dir" "" "" "" "$notes"
        rm -f "$tmp_yaml"
        return 1
    fi

    if [[ "$points_src" != "$points_dst" ]]; then
        cp -f "$points_src" "$points_dst"
    fi
    if [[ -f "$windows_src" && "$windows_src" != "$windows_dst" ]]; then
        cp -f "$windows_src" "$windows_dst"
    fi
    [[ -f "$checkpoint_dir/finetune_output.log" ]] && cp -f "$checkpoint_dir/finetune_output.log" "$run_dir/finetune_output.log"

    if [[ "$RUN_PER_MISSION_EVAL" == "1" ]]; then
        mkdir -p "$run_dir/evaluation"
        if ! python3 scripts/evaluate_anomalies.py \
            --points "$points_dst" \
            --out "$run_dir/evaluation" \
            --mission "$mission" \
            > "$run_dir/evaluation.log" 2>&1; then
            notes="${notes};evaluation_failed"
            eval_pdf=""
        fi
    fi

    if [[ "$RUN_RATIO_SWEEP" == "1" ]]; then
        mkdir -p "$run_dir/ratio_sweep"
        if ! python3 scripts/ratio_sweep.py \
            --mission "$mission" \
            --points "$points_dst" \
            --ratios $RATIOS \
            --out "$run_dir/ratio_sweep" \
            > "$run_dir/ratio_sweep.log" 2>&1; then
            notes="${notes};ratio_sweep_failed"
            ratio_csv=""
            ratio_pdf=""
        fi
    fi

    append_manifest "$run_id" "$mission" "$satellite" "$mode" "$subsample_pct" "$prompt_tune_epoch" "$train_epochs" \
        "$BASE_ANOMALY_RATIO" "$RATIOS" "complete" "$dataset_dir" "$points_dst" "$windows_dst" "$checkpoint_dir" "$eval_pdf" "$ratio_csv" "$ratio_pdf" "$notes"
    rm -f "$tmp_yaml"
    return 0
}

check_python_deps

SATELLITE=${SATELLITE:-"STPSat7"}
if [[ -z "${MISSIONS:-}" ]]; then
    case "$SATELLITE" in
        ESA)
            MISSIONS="ESA-Mission1 ESA-Mission2"
            ;;
        STPSat4)
            MISSIONS="STPSat4-TCS STPSat4-HRR STPSat4-MRR STPSat4-PCE1 STPSat4-PCE2 STPSat4-ADCS"
            ;;
        STPSat7)
            MISSIONS="STPSat7-EPS STPSat7-ADCS STPSat7-TO STPSat7-HRR STPSat7-MRR STPSat7-TC"
            ;;
        *)
            log "ERROR: Unknown SATELLITE='$SATELLITE'"
            log "       Valid options: ESA, STPSat4, STPSat7"
            log "       Or set MISSIONS directly: MISSIONS='ESA-Mission1 STPSat7-EPS' sbatch ..."
            exit 1
            ;;
    esac
fi

STUDY_TAG=${STUDY_TAG:-paper_$(date '+%Y%m%d_%H%M%S')}
STUDY_DIR=${STUDY_DIR:-"$ROOT_DIR/results/paper_study/$STUDY_TAG"}
TMP_DIR="$STUDY_DIR/tmp"
MANIFEST_PATH="$STUDY_DIR/study_manifest.tsv"
mkdir -p "$STUDY_DIR" "$TMP_DIR"
write_manifest_header

STUDY_PRESET=${STUDY_PRESET:-standard}
NUM_WORKERS=${NUM_WORKERS:-}
if [[ -z "$NUM_WORKERS" ]]; then
    if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" -gt 2 ]]; then
        NUM_WORKERS=$((SLURM_CPUS_PER_TASK - 2))
    else
        NUM_WORKERS=4
    fi
fi

SUBSAMPLE_PCTS_DEFAULT="0.05 0.10 0.20"
PROMPT_TUNE_EPOCHS_DEFAULT="5 10 20"
RUN_PER_MISSION_EVAL_DEFAULT=1
RUN_RATIO_SWEEP_DEFAULT=1

case "$STUDY_PRESET" in
    standard)
        ;;
    fast)
        SUBSAMPLE_PCTS_DEFAULT="0.10 0.20"
        PROMPT_TUNE_EPOCHS_DEFAULT="5 10"
        RUN_PER_MISSION_EVAL_DEFAULT=0
        RUN_RATIO_SWEEP_DEFAULT=0
        ;;
    *)
        log "ERROR: Unknown STUDY_PRESET='$STUDY_PRESET'"
        log "       Valid options: standard, fast"
        exit 1
        ;;
esac

STUDY_MODES=${STUDY_MODES:-prompt_tuning}
SUBSAMPLE_PCTS=${SUBSAMPLE_PCTS:-"$SUBSAMPLE_PCTS_DEFAULT"}
PROMPT_TUNE_EPOCHS=${PROMPT_TUNE_EPOCHS:-"$PROMPT_TUNE_EPOCHS_DEFAULT"}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-"5 10"}
RATIOS=${RATIOS:-"0.1 0.25 0.5 1.0 2.0"}
BASE_ANOMALY_RATIO=${BASE_ANOMALY_RATIO:-1.0}
CKPT=${CKPT:-"$ROOT_DIR/newcheckpoints/units_x32_pretrain_checkpoint.pth"}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-checkpoints}
SKIP_PREP=${SKIP_PREP:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}
RUN_PER_MISSION_EVAL=${RUN_PER_MISSION_EVAL:-$RUN_PER_MISSION_EVAL_DEFAULT}
RUN_RATIO_SWEEP=${RUN_RATIO_SWEEP:-$RUN_RATIO_SWEEP_DEFAULT}
RUN_STUDY_REPORT=${RUN_STUDY_REPORT:-1}
PROJECT_NAME=${PROJECT_NAME:-units_anomaly_paper}
WANDB_MODE=${WANDB_MODE:-offline}
DES=${DES:-Study}
FIX_SEED=${FIX_SEED:-2021}
MIN_TRAIN_ROWS=${MIN_TRAIN_ROWS:-96}
BATCH_SIZE=${BATCH_SIZE:-64}
ACC_IT=${ACC_IT:-8}
D_MODEL=${D_MODEL:-32}
E_LAYERS=${E_LAYERS:-3}
PATCH_LEN=${PATCH_LEN:-16}
STRIDE=${STRIDE:-16}
PROMPT_NUM=${PROMPT_NUM:-10}
DROPOUT=${DROPOUT:-0.0}
PROMPT_LR=${PROMPT_LR:-5e-5}
PROMPT_WEIGHT_DECAY=${PROMPT_WEIGHT_DECAY:-1e-2}
FINETUNE_LR=${FINETUNE_LR:-5e-4}
FINETUNE_WEIGHT_DECAY=${FINETUNE_WEIGHT_DECAY:-1e-3}
ESA_TRAIN_END=${ESA_TRAIN_END:-}
ESA_TEST_START=${ESA_TEST_START:-}
STPSAT4_SPLIT_DATE=${STPSAT4_SPLIT_DATE:-}
STPSAT7_SPLIT_DATE=${STPSAT7_SPLIT_DATE:-}
REPORT_TITLE=${REPORT_TITLE:-"UniTS Anomaly Detection Paper Study"}

log "Starting paper study"
log "Study dir        : $STUDY_DIR"
log "Satellite        : $SATELLITE"
log "Missions         : $MISSIONS"
log "Study preset     : $STUDY_PRESET"
log "Study modes      : $STUDY_MODES"
log "Subsample pcts   : $SUBSAMPLE_PCTS"
log "Prompt epochs    : $PROMPT_TUNE_EPOCHS"
log "Finetune epochs  : $FINETUNE_EPOCHS"
log "Num workers      : $NUM_WORKERS"
log "Anomaly ratios   : $RATIOS"
log "Checkpoint       : $CKPT"

if [[ "$SKIP_TRAIN" != "1" && ! -f "$CKPT" ]]; then
    log "Checkpoint not found: $CKPT"
    exit 1
fi

prepare_requested_data

mission_count=0
run_count=0
failed_count=0

for mission in $MISSIONS; do
    mission_count=$((mission_count + 1))
    if ! validate_dataset "$mission"; then
        log "Skipping mission with invalid dataset: $mission"
        continue
    fi

    for mode in $STUDY_MODES; do
        for subsample_pct in $SUBSAMPLE_PCTS; do
            case "$mode" in
                prompt_tuning)
                    for prompt_epoch in $PROMPT_TUNE_EPOCHS; do
                        run_count=$((run_count + 1))
                        if ! process_run "$mission" "$mode" "$subsample_pct" "$prompt_epoch" 0; then
                            failed_count=$((failed_count + 1))
                            log "Continuing after failed run: $mission $mode pct=$subsample_pct pt=$prompt_epoch"
                        fi
                    done
                    ;;
                finetune)
                    for train_epoch in $FINETUNE_EPOCHS; do
                        run_count=$((run_count + 1))
                        if ! process_run "$mission" "$mode" "$subsample_pct" 0 "$train_epoch"; then
                            failed_count=$((failed_count + 1))
                            log "Continuing after failed run: $mission $mode pct=$subsample_pct ft=$train_epoch"
                        fi
                    done
                    ;;
                *)
                    log "Unknown study mode '$mode' - skipping"
                    ;;
            esac
        done
    done
done

log "Study sweep finished: attempted=$run_count failed=$failed_count"

if [[ "$RUN_STUDY_REPORT" == "1" ]]; then
    log "Building centralized report bundle"
    python3 scripts/paper_anomaly_report.py \
        --manifest "$MANIFEST_PATH" \
        --study-dir "$STUDY_DIR" \
        --title "$REPORT_TITLE" \
        --ratios $RATIOS \
        > "$STUDY_DIR/report_build.log" 2>&1
    log "Report bundle: $STUDY_DIR"
    log "README       : $STUDY_DIR/README.md"
    log "PDF          : $STUDY_DIR/report/paper_report.pdf"
fi

printf '%s\n' "$STUDY_DIR" > "$ROOT_DIR/results/paper_study/latest_path.txt"
log "Done"
