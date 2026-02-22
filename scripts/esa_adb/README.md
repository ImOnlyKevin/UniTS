# ESA-ADB -> UniTS anomaly inference

This folder contains tooling to:
1. Read ESA-ADB raw mission data from `data/ESA-ADB-raw/ESA-Mission1` and `data/ESA-ADB-raw/ESA-Mission2`.
2. Build aligned multivariate train/test arrays from channel `*.zip` pickled files.
3. Run UniTS reconstruction-based anomaly scoring.
4. Export predicted anomaly timestamps.

## Local usage

```bash
python scripts/esa_adb/run_esa_adb_units.py \
  --checkpoint /path/to/units_checkpoint.pth \
  --raw-root data/ESA-ADB-raw \
  --missions ESA-Mission1 ESA-Mission2 \
  --output-dir outputs/esa_adb_units
```

## Outputs

For each mission, the script writes:

- `predicted_anomaly_points.csv`: timestamp-level energy, ground-truth labels, and predicted labels.
- `predicted_anomaly_ranges.csv`: contiguous predicted anomaly intervals (`start_timestamp`, `end_timestamp`).
- `run_summary.json`: run metadata (threshold, sample count, feature count).

## Default split dates

Default train/test time split dates follow ESA-ADB preprocessing conventions:

- Mission 1: `2007-01-01`
- Mission 2: `2001-10-01`

You can override split date with `--split-override`.

## MIT SuperCloud SLURM usage

Two helper scripts are provided:

- `scripts/esa_adb/slurm/esa_adb_units_gpu.sbatch` (actual SLURM job script)
- `scripts/esa_adb/slurm/submit_esa_adb_units_gpu.sh` (convenience submit wrapper)

### Submit a job

```bash
scripts/esa_adb/slurm/submit_esa_adb_units_gpu.sh \
  /path/to/UniTS \
  /path/to/checkpoint.pth
```

### Optional runtime controls (export before submission)

```bash
export RAW_ROOT=/path/to/UniTS/data/ESA-ADB-raw
export OUT_DIR=/path/to/UniTS/outputs/esa_adb_units
export MISSIONS="ESA-Mission1 ESA-Mission2"
export SEQ_LEN=96
export BATCH_SIZE=32
export ANOMALY_RATIO=1.0
export EXTRA_ARGS="--max-channels 64"
```

Then submit as above.

### Current default SLURM resources

- Partition: `xeon-g6-volta`
- Nodes: `1`
- CPU cores: `16` (`--cpus-per-task=16`)
- GPUs: `1` (`--gres=gpu:1`)
- Walltime: `24:00:00`

Adjust `#SBATCH` lines in `esa_adb_units_gpu.sbatch` as needed.
