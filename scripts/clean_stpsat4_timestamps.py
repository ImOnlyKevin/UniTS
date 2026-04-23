#!/usr/bin/env python3
"""
clean_stpsat4_timestamps.py — One-time cleanup script

Removes rows with year 2000 timestamps from all STPSat-4 health & status CSVs.
These occur during GPS acquisition before the receiver has a valid time lock.

Edits files IN PLACE. Run once before prepare_stpsat4_data.py.

Usage:
    python scripts/clean_stpsat4_timestamps.py

    # Dry run — shows what would be removed without changing any files
    python scripts/clean_stpsat4_timestamps.py --dry_run

    # Custom raw data directory
    python scripts/clean_stpsat4_timestamps.py --raw_dir data/STPSat-4-raw/Sat-4_HS_data
"""

import argparse
import glob
import os
import pandas as pd
from pathlib import Path


TIMESTAMP_COL = "UTC_Time_String"
BAD_YEAR      = 2000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/STPSat-4-raw/Sat-4_HS_data")
    p.add_argument("--dry_run", action="store_true",
                   help="Report what would be removed without modifying any files")
    p.add_argument("--bad_year", type=int, default=BAD_YEAR,
                   help="Year to treat as invalid GPS acquisition rows (default: 2000)")
    return p.parse_args()


def main():
    args = parse_args()

    pattern = os.path.join(args.raw_dir, "*__HS_Data*", "*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"No CSV files found in {args.raw_dir}")
        return

    print(f"{'DRY RUN — ' if args.dry_run else ''}Scanning {len(files)} CSV files "
          f"for year-{args.bad_year} GPS acquisition rows ...")
    print(f"Raw data dir : {args.raw_dir}")
    print()

    total_removed = 0
    total_rows    = 0
    files_touched = 0

    for fpath in files:
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception as e:
            print(f"  ERROR reading {fpath}: {e}")
            continue

        if TIMESTAMP_COL not in df.columns:
            continue

        total_rows += len(df)

        ts   = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
        mask = ts.dt.year == args.bad_year
        n    = mask.sum()

        if n == 0:
            continue

        files_touched += 1
        total_removed += n
        fname = os.path.relpath(fpath, args.raw_dir)

        if args.dry_run:
            print(f"  [DRY RUN] {fname}: would remove {n:,} rows "
                  f"(of {len(df):,} total)")
        else:
            df_clean = df[~mask].reset_index(drop=True)
            df_clean.to_csv(fpath, index=False)
            print(f"  Cleaned {fname}: removed {n:,} rows "
                  f"({len(df_clean):,} remaining)")

    print()
    print("=" * 60)
    if args.dry_run:
        print(f"DRY RUN COMPLETE — no files were modified")
        print(f"  Files with bad rows : {files_touched:,} / {len(files):,}")
        print(f"  Rows that would be removed : {total_removed:,} / {total_rows:,}")
    else:
        print(f"CLEANUP COMPLETE")
        print(f"  Files modified : {files_touched:,} / {len(files):,}")
        print(f"  Rows removed   : {total_removed:,} / {total_rows:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()