#!/usr/bin/env python3
"""
prepare_stpsat7_data.py — Prepare STPSat-7 telemetry for UniTS ingestion.

Reads raw CSV files from data/STPSat-7-raw/HS_data/, one file per day per
subsystem. Files may be named in either order:
    09A5 2026-04-08.csv
    2026-04-08 09A5.csv

MID → Subsystem mapping:
    09A5 → EPS
    09B5 → ADCS
    0808 → TO
    0884 → CI
    0901 → HRR
    0903 → MRR

For each subsystem:
  1. Concatenates all daily files
  2. Parses dtSAT_Receipt_Time as timestamp
  3. Drops GPS-acquisition rows (timestamps before GPS_EPOCH)
  4. Drops configured unnecessary columns
  5. Resamples to 60-second grid
  6. Ordinal-encodes categorical columns
  7. Forward-fills then mean-fills NaNs
  8. Splits into train/test by SPLIT_DATE
  9. Saves .npy arrays + channels.txt to dataset/STPSat7-<subsystem>/
 10. Updates anomaly_detection_stpsat7.yaml

Usage:
    python scripts/prepare_stpsat7_data.py
    python scripts/prepare_stpsat7_data.py --subsystems EPS ADCS
    python scripts/prepare_stpsat7_data.py --split_date 2026-03-01 --dry_run
"""

import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

RAW_DIR    = Path("data/STPSat-7-raw/HS_data")
DATASET_DIR = Path("dataset")
YAML_PATH  = Path("data_provider/anomaly_detection_stpsat7.yaml")

TIMESTAMP_COL = "dtSAT_Receipt_Time"

# Rows before this date are GPS-acquisition artifacts (satellite clock = Jan 1 2000)
GPS_EPOCH = pd.Timestamp("2000-01-02", tz="UTC")

# Train/test split date — set to None to auto-calculate at 80/20 from data
SPLIT_DATE = None  # auto-calculated per subsystem from actual timestamp range

# MID → subsystem name mapping
MID_MAP = {
    "09A5": "EPS",
    "09B5": "ADCS",
    "0880": "TO",
    # "0884": "CI",  # Excluded: CI is purely command routing counters and padding.
    #                # Zero physics channels after cleaning — not suitable for UniTS.
    "0901": "HRR",
    "0903": "MRR",
    "0923": "TC",
}

# ── COLUMN DROP CONFIG ────────────────────────────────────────────────────────
# Add column names to drop for each subsystem once you have the headers.
# These are typically: redundant index columns, duplicate timestamp columns,
# checksum/CRC fields, and any columns that are entirely constant.
#
# Example:
#   "EPS": ["Unnamed: 0", "scEpochTime", "CRC16"],
#
COLUMNS_TO_DROP = {
    # ── EPS (09A5) ────────────────────────────────────────────────────────────
    # Drop: packet padding fields
    "EPS": [
        "PpdABoardHk.Output/padding/LCL4 LARADO SURV HTR",
        "PpdABoardHk.Output/padding/LCL6 GARI-1C",
        "PpdABoardHk.Output/padding/LCL10 LARADO",
        "PpdABoardHk.Output/padding/LCL15 GOSAS",
        "PpdABoardHk.Output/padding/LCL18 NanoUHF",
        "PpdABoardHk.Output/padding/LCL19 NanoUHF Deploy",
        "PpdBBoardHk.Output/padding/LCL4 SFPE",
    ],

    # ── ADCS (09B5) ───────────────────────────────────────────────────────────
    # Drop: packet structure, scheduler metadata, raw bytes, software management,
    #       monotonic counters, and raw tracker/wheel internals
    "ADCS": [
        # Packet structure / protocol
        "PSC",
        "Level 0/padding", "Level 0/Padding", "Level 0/Sync Word",
        "Checksum", "CRC", "Request Echo",
        "AttCmd/spare_1", "AttCmd/spare_2", "AttCmd/spare_3",
        "AttCtrl/spare_1", "AttCtrl/spare_2", "AttCtrl/spare_3",
        "AttDet/spare",
        "Momentum/spare_1", "Momentum/spare_2", "Momentum/spare_3", "Momentum/spare_4",
        "Imu/spare_1", "Imu/spare_2", "Imu/spare_3",
        "Css/spare",
        "Gps/spare",
        "Time/spare",
        "Mag/spare",

        # Scheduler / high-rate timing metadata (not physics)
        "ClockSync/HR_CycleNum", "ClockSync/VHR_CycleNum",
        "ClockSync/HR_ExecTimeMs_1 [ms]", "ClockSync/HR_ExecTimeMs_2 [ms]",
        "ClockSync/HR_ExecTimeMs_3 [ms]", "ClockSync/HR_ExecTimeMs_4 [ms]",
        "ClockSync/HR_ExecTimeMs_5 [ms]",
        "ClockSync/HR_RunCount", "ClockSync/HR_TimeUsec [µs]",
        "ExtTracker/HR_ExecTimeMs_1", "ExtTracker/HR_ExecTimeMs_2",
        "ExtTracker/HR_ExecTimeMs_3", "ExtTracker/HR_ExecTimeMs_4",
        "ExtTracker/HR_ExecTimeMs_5", "ExtTracker/HR_RunCount",
        "ExtTracker/HR_TimeUsec [µs]",
        "ExtTracker2/HR_ExecTimeMs_1", "ExtTracker2/HR_ExecTimeMs_2",
        "ExtTracker2/HR_ExecTimeMs_3", "ExtTracker2/HR_ExecTimeMs_4",
        "ExtTracker2/HR_ExecTimeMs_5", "ExtTracker2/HR_RunCount",
        "ExtTracker2/HR_TimeUsec [µs]",
        *[f"ExtWheel{i}/HR_RunCount" for i in range(1, 5)],
        *[f"ExtWheel{i}/HR_TimeUsec [µs]" for i in range(1, 5)],
        "RwDrive/RwTime [µs]",
        "Imu/AvgTimeTag [µs]",
        "Tracker/AttTimeTag [µs]", "Tracker2/AttTimeTag [µs]",

        # Raw packet bytes (L0_1..L0_32 across all peripheral subsections)
        *[f"ExtWheel{i}/L0_{j}" for i in range(1, 5) for j in range(1, 33)],
        *[f"ExtTracker/L0_{j}" for j in range(1, 33)],
        *[f"ExtTracker2/L0_{j}" for j in range(1, 33)],

        # Flash / filesystem / software management
        "General/VersionCode_1", "General/VersionCode_2", "General/VersionType",
        "General/FlashCmdFailCount", "General/FlashCmdSuccCount",
        "General/FlashLastCheck", "General/FlashLastOff",
        "General/FlashLastPart", "General/FlashPowerState", "General/FlashSelectState",
        "General/FsLastCheck", "General/FsLastLen", "General/FsLastOff",
        "General/FsLastOp", "General/FsMounted", "General/FsOpStatus",
        "General/ImageAutoFailover", "General/ImageBooted",
        "General/ScrubStatusPart0", "General/ScrubStatusPart1",
        "General/ScrubStatusPart2", "General/ScrubStatusPart3",
        "General/ScrubStatusPart4", "General/ScrubStatusPart5",
        "General/AsyncRunning",
        *[f"ExtWheel{i}/FlashBurnArmed" for i in range(1, 5)],
        *[f"ExtWheel{i}/Image" for i in range(1, 5)],
        *[f"ExtWheel{i}/Length32" for i in range(1, 5)],
        *[f"ExtWheel{i}/Offset32" for i in range(1, 5)],
        *[f"ExtWheel{i}/TableUploadStatus" for i in range(1, 5)],
        *[f"ExtWheel{i}/WhichTable" for i in range(1, 5)],
        "ExtTracker/FlashBurnArmed", "ExtTracker/Image",
        "ExtTracker/Length32", "ExtTracker/Offset32",
        "ExtTracker/TableUploadStatus", "ExtTracker/WhichTable",
        "ExtTracker2/FlashBurnArmed", "ExtTracker2/Image",
        "ExtTracker2/Length32", "ExtTracker2/Offset32",
        "ExtTracker2/TableUploadStatus", "ExtTracker2/WhichTable",
        "Tables/FlashBurnArmed", "Tables/Image",
        "Tables/Length32", "Tables/Offset32",
        "Tables/TableUploadStatus", "Tables/WhichTable",
        "TlmProc/TlmMapId", "TlmProc/TlmMapSize", "TlmProc/TlmTableUsed",

        # Monotonic counters (non-stationary, uninformative for reconstruction)
        "Level 0/Cmd Accept Counter", "Level 0/Cmd Reject Counter",
        "Level 0/Watchdog 2 Second Counter",
        "Level 0/Time Tag of Last Incoming Command [s]",
        *[f"ExtWheel{i}/CmdAcceptCount" for i in range(1, 5)],
        *[f"ExtWheel{i}/CmdRejectCount" for i in range(1, 5)],
        *[f"ExtWheel{i}/CyclesSincePps" for i in range(1, 5)],
        "ExtTracker/CmdAcceptCount", "ExtTracker/CmdRejectCount",
        "ExtTracker/CyclesSincePps", "ExtTracker/CountsPerSec",
        "ExtTracker2/CmdAcceptCount", "ExtTracker2/CmdRejectCount",
        "ExtTracker2/CyclesSincePps", "ExtTracker2/CountsPerSec",
        "ClockSync/CyclesSincePps", "ClockSync/CountsPerSec [counts/s]",
        "CommandTlm/RealtimeCmdAcceptCount", "CommandTlm/RealtimeCmdRejectCount",
        "CommandTlm/StoredCmdAcceptCount", "CommandTlm/StoredCmdRejectCount",
        "CommandTlm/MacroCmdsExpired", "CommandTlm/MacroCmdsQueued",
        "TrackerCtrl/NstResetCount_1_Wake", "TrackerCtrl/NstResetCount_2_Ram",
        "TrackerCtrl/CyclesSinceValidNstData_1_Wake",
        "TrackerCtrl/CyclesSinceValidNstData_2_Ram",
        "TrackerCtrl/CyclesSinceValidNstSol_1_Wake",
        "TrackerCtrl/CyclesSinceValidNstSol_2_Ram",
        "AttDet/ReinitCount",
        "AttDet/BadAttTimer [cycles]", "AttDet/BadRateTimer [cycles]",
        "AttDet/GoodAttRateTimer [cycles]",
        "Imu/ImuInvalidCount", "Imu/ImuReinitCount",
        "Imu/ImuValidPackets", "Imu/NewPacketCount", "Imu/FirstPacketID",
        *[f"RwDrive/ExtRwResetCount_{i}" for i in range(1, 5)],
        # Note: RwDrive/CyclesSinceValidRwData kept — resets on fault, useful signal
        "Gps/GpsCyclesSinceCrcData", "Gps/GpsCyclesSinceLatestData",
        "Gps/GpsLockCount",
        "General/RamSingleBitErrorCount", "General/ScrubCount",
        "General/ScrubSecsSinceLastScrub", "General/ScrubStatusOverall",
        "Cal/RejectedTrackerEstCount",
        "Tracker/DetTimeoutCount", "Tracker/NumOfAttLoops",
        "Tracker/NumIdPatternsTried", "Tracker/NumTrackBlocksIssued",
        "Tracker/FSWcounter",
        "Tracker2/DetTimeoutCount", "Tracker2/NumOfAttLoops",
        "Tracker2/NumIdPatternsTried", "Tracker2/NumTrackBlocksIssued",
        "Tracker2/FSWcounter",

        # Raw tracker internals (no physics value)
        "Tracker/ImageIndex", "Tracker/StoreSeqImages",
        "Tracker/RefIndex", "Tracker/VideoAddress",
        "Tracker2/ImageIndex", "Tracker2/StoreSeqImages",
        "Tracker2/RefIndex", "Tracker2/VideoAddress",

        # Raw GPS satellite-level data (too low level)
        *[f"Gps/MsgSatPrn_{i}" for i in range(1, 6)],
        *[f"Gps/MsgSatPsr_{i} [m]" for i in range(1, 6)],

        # Raw reaction wheel counts (RPM already captured in Filtered_Speed_RPM)
        *[f"RwDrive/Motor_Tach_Counts_{i}" for i in range(1, 5)],
        *[f"RwDrive/PWM_Counts_{i}" for i in range(1, 5)],

        # Time internals and monotonically increasing time fields
        "Time/GpsUpdateCycleCount", "Time/GpsUpdateCyclePeriod",
        "Time/RtcSyncCyclePeriod", "Time/RtcOscRstCount",
        "Time/TaiSeconds [s]",         # monotonically increasing TAI time
        "Time/RtcTaiSecond [s]",       # monotonically increasing RTC time
        "Time/RtcTaiSubsec",           # sub-second increment
        "Time/JdOfNowWrtTai [day]",    # Julian date, monotonic
        "Time/RtcAlive",               # always 1 in normal ops
        "Time/RtcInitTimeAtBoot",      # constant per boot
        "Time/RtcPower",               # constant
        "Time/RtcSyncStat",            # constant
        "Time/TimeValid",              # constant TRUE in normal ops

        # GPS time fields (monotonically increasing)
        "Gps/GpsSolutionTimeTag [µs]", # large monotonically increasing timestamp
        "Gps/GpsTime",                 # GPS time string e.g. "58:16.0", increasing

        # Raw packet bytes — checksums and last cmd bytes across all peripherals
        *[f"ExtWheel{i}/Checksum" for i in range(1, 5)],
        *[f"ExtWheel{i}/LastAccCmdBytes_1" for i in range(1, 5)],
        *[f"ExtWheel{i}/LastAccCmdBytes_2" for i in range(1, 5)],
        *[f"ExtWheel{i}/LastRejCmdBytes_1" for i in range(1, 5)],
        *[f"ExtWheel{i}/LastRejCmdBytes_2" for i in range(1, 5)],
        "ExtTracker/LastAccCmdBytes_1", "ExtTracker/LastAccCmdBytes_2",
        "ExtTracker/LastRejCmdBytes_1", "ExtTracker/LastRejCmdBytes_2",
        "ExtTracker2/LastAccCmdBytes_1", "ExtTracker2/LastAccCmdBytes_2",
        "ExtTracker2/LastRejCmdBytes_1", "ExtTracker2/LastRejCmdBytes_2",
        *[f"CommandTlm/LastAccCmdBytes_{i}" for i in range(1, 9)],
        *[f"CommandTlm/LastRejCmdBytes_{i}" for i in range(1, 9)],
        "Tables/Checksum",

        # Constant configuration fields
        "CommandTlm/HealthStatusInterval [s]",  # always 300
        "CommandTlm/NavTlmInterval [s]",         # always 1
        "CommandTlm/MacrosEnabled",              # constant flag
        *[f"CommandTlm/MacrosExecutingPack_{i}" for i in range(1, 13)],

        # Unused CSS diode channels (always 0)
        "Css/NumDiodesUsed_4_Unused",
        "Css/RawSunSensorData13_Unused", "Css/RawSunSensorData14_Unused",
        "Css/RawSunSensorData15_Unused", "Css/RawSunSensorData16_Unused",

        # CSS counters
        "Css/CssInvalidCount", "Css/CssReinitCount",

        # Level 0 Hardware seconds counter (monotonically increasing)
        "Level 0/Hardware Seconds Counter [s]",
    ],

    # ── TO (0880) ─────────────────────────────────────────────────────────────
    "TO": [
        # Packet metadata and monotonic counters — nearly all of TO is counters.
        # Remaining meaningful channels: CurrentOpsRadio, DataFramesPerSecond,
        # DownlinkEnabled, hrr_stopBitValue, mrr_stopBitValue, IgnoreStopBit (6 total)
        "PSC",
        "CmdCnt",
        "CmdErrCnt",
        "hrr_stopBitCount",
        "mrr_stopBitCount",
        "RadioWriteCnt",
        "RadioWriteErrCnt",
        "hrr_cmd_data_frame_counter",
        "hrr_cmd_enable_radio_commands_counter",
        "hrr_cmd_get_gryphon_telemetry_counter",
        "hrr_cmd_get_telemetry_counter",
        "hrr_cmd_get_telemetry_legacy_counter",
        "hrr_cmd_manual_sync_counter",
        "hrr_cmd_peek_counter",
        "hrr_cmd_ping_counter",
        "hrr_cmd_poke_counter",
        "hrr_cmd_radio_command_port1_counter",
        "hrr_cmd_radio_command_port2_counter",
        "hrr_cmd_reset_gryphon_counter",
        "hrr_cmd_select_dl_key_counter",
        "hrr_cmd_select_dl_otar_key_counter",
        "hrr_cmd_select_kek_counter",
        "hrr_cmd_select_ul_key_counter",
        "hrr_cmd_select_ul_otar_key_counter",
        "hrr_cmd_set_aes_modes_counter",
        "hrr_cmd_set_crypto_config_counter",
        "hrr_cmd_set_data_clock_counter",
        "hrr_cmd_set_datapath_config_counter",
        "hrr_cmd_set_flow_polarity_counter",
        "hrr_cmd_set_gcm_block_length_counter",
        "hrr_cmd_set_mode_counter",
        "hrr_cmd_set_radio_interface_counter",
        "hrr_cmd_set_sync_period_counter",
        "hrr_cmd_set_transmitter_power_counter",
        "hrr_cmd_store_otar_key_counter",
        "hrr_cmd_use_dl_otar_key_counter",
        "hrr_cmd_use_ul_otar_key_counter",
        "hrr_radio_cmd_get_telemetry_counter",
        "hrr_radio_cmd_load_defaults_counter",
        "hrr_radio_cmd_ping_counter",
        "hrr_radio_cmd_program_defaults_counter",
        "hrr_radio_cmd_set_coherence_ratio_counter",
        "hrr_radio_cmd_set_datapath_interface_counter",
        "hrr_radio_cmd_set_mode_counter",
        "hrr_radio_cmd_set_ranging_counter",
        "hrr_radio_cmd_set_ranging_mod_index_counter",
        "hrr_radio_cmd_set_ranging_pn_params_counter",
        "hrr_radio_cmd_set_rx_data_params_counter",
        "hrr_radio_cmd_set_rx_demod_params_counter",
        "hrr_radio_cmd_set_rx_fec_counter",
        "hrr_radio_cmd_set_rx_freq_chan_counter",
        "hrr_radio_cmd_set_rx_rf_params_counter",
        "hrr_radio_cmd_set_rx_subcarrier_params_counter",
        "hrr_radio_cmd_set_rx_sym_rate_counter",
        "hrr_radio_cmd_set_subcarrier_counter",
        "hrr_radio_cmd_set_tx_data_params_counter",
        "hrr_radio_cmd_set_tx_freq_chan_counter",
        "hrr_radio_cmd_set_tx_mod_params_counter",
        "hrr_radio_cmd_set_tx_rf_params_counter",
        "hrr_radio_cmd_set_tx_subcarrier_params_counter",
        "hrr_radio_cmd_set_tx_sym_rate_counter",
        "hrr_radio_cmd_transmit_enable_counter",
        "hrr_radio_cmd_transmit_key_counter",
        "mrr_cmd_data_frame_counter",
        "mrr_cmd_enable_radio_commands_counter",
        "mrr_cmd_get_gryphon_telemetry_counter",
        "mrr_cmd_get_telemetry_counter",
        "mrr_cmd_get_telemetry_legacy_counter",
        "mrr_cmd_manual_sync_counter",
        "mrr_cmd_peek_counter",
        "mrr_cmd_ping_counter",
        "mrr_cmd_poke_counter",
        "mrr_cmd_radio_command_port1_counter",
        "mrr_cmd_radio_command_port2_counter",
        "mrr_cmd_reset_gryphon_counter",
        "mrr_cmd_select_dl_key_counter",
        "mrr_cmd_select_dl_otar_key_counter",
        "mrr_cmd_select_kek_counter",
        "mrr_cmd_select_ul_key_counter",
        "mrr_cmd_select_ul_otar_key_counter",
        "mrr_cmd_set_aes_modes_counter",
        "mrr_cmd_set_crypto_config_counter",
        "mrr_cmd_set_data_clock_counter",
        "mrr_cmd_set_datapath_config_counter",
        "mrr_cmd_set_flow_polarity_counter",
        "mrr_cmd_set_gcm_block_length_counter",
        "mrr_cmd_set_mode_counter",
        "mrr_cmd_set_radio_interface_counter",
        "mrr_cmd_set_sync_period_counter",
        "mrr_cmd_set_transmitter_power_counter",
        "mrr_cmd_store_otar_key_counter",
        "mrr_cmd_use_dl_otar_key_counter",
        "mrr_cmd_use_ul_otar_key_counter",
        "mrr_radio_cmd_get_telemetry_counter",
        "mrr_radio_cmd_load_defaults_counter",
        "mrr_radio_cmd_ping_counter",
        "mrr_radio_cmd_program_defaults_counter",
        "mrr_radio_cmd_set_coherence_ratio_counter",
        "mrr_radio_cmd_set_datapath_interface_counter",
        "mrr_radio_cmd_set_mode_counter",
        "mrr_radio_cmd_set_ranging_counter",
        "mrr_radio_cmd_set_ranging_mod_index_counter",
        "mrr_radio_cmd_set_ranging_pn_params_counter",
        "mrr_radio_cmd_set_rx_data_params_counter",
        "mrr_radio_cmd_set_rx_demod_params_counter",
        "mrr_radio_cmd_set_rx_fec_counter",
        "mrr_radio_cmd_set_rx_freq_chan_counter",
        "mrr_radio_cmd_set_rx_rf_params_counter",
        "mrr_radio_cmd_set_rx_subcarrier_params_counter",
        "mrr_radio_cmd_set_rx_sym_rate_counter",
        "mrr_radio_cmd_set_subcarrier_counter",
        "mrr_radio_cmd_set_tx_data_params_counter",
        "mrr_radio_cmd_set_tx_freq_chan_counter",
        "mrr_radio_cmd_set_tx_mod_params_counter",
        "mrr_radio_cmd_set_tx_rf_params_counter",
        "mrr_radio_cmd_set_tx_subcarrier_params_counter",
        "mrr_radio_cmd_set_tx_sym_rate_counter",
        "mrr_radio_cmd_transmit_enable_counter",
        "mrr_radio_cmd_transmit_key_counter",
    ],

    # ── CI (0884) ─────────────────────────────────────────────────────────────
    # CI excluded from pipeline: purely command routing counters and padding.
    # No physics channels remain after cleaning. See MID_MAP comment above.

    # ── HRR (0901) ────────────────────────────────────────────────────────────
    # Drop: packet metadata, counters, version strings, test/config constants
    # Keeps: all voltages, currents, temperatures, RF params, signal quality (~85 channels)
    "HRR": [
        # Packet metadata
        "PSC", "spare",
        # Monotonic counters
        "ADCErrorCount", "HostInvalidPackets", "HostValidPackets",
        "UPLINK.INVALID_COUNT", "UPLINK.VALID_COUNT",
        # Version strings (constant)
        "FE.Version_FrontEndFabric", "FE.Version_ModemHDLRelease",
        "VERSION_FABRIC", "VERSION_MSS",
        # Test/config constants
        "Ping_Bit_Param", "PRBS_TEST_MODE",
    ],

    # ── MRR (0903) ────────────────────────────────────────────────────────────
    # Identical schema to HRR — same drop list applies
    "MRR": [
        "PSC", "spare",
        "ADCErrorCount", "HostInvalidPackets", "HostValidPackets",
        "UPLINK.INVALID_COUNT", "UPLINK.VALID_COUNT",
        "FE.Version_FrontEndFabric", "FE.Version_ModemHDLRelease",
        "VERSION_FABRIC", "VERSION_MSS",
        "Ping_Bit_Param", "PRBS_TEST_MODE",
    ],

    # ── TC (0923) ─────────────────────────────────────────────────────────────
    # Drop: packet metadata and monotonic counters
    # Keeps: GPIO states, all spiAdcData (analog readings), all spiTempData (temps) — 34 channels
    "TC": [
        "PSC",
        "Command Count", "Command Error Count",
        "SPI Error Count", "SPI Transmission Count",
    ],
}

# ── UniTS YAML template values ────────────────────────────────────────────────
SEQ_LEN  = 96
LABEL_LEN = 0
PRED_LEN  = 0

# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subsystems", nargs="*", default=list(MID_MAP.values()),
                   help="Subsystems to process (default: all). "
                        "E.g. --subsystems EPS ADCS")
    p.add_argument("--split_date", default=None,
                   help=f"Train/test split date (default: {SPLIT_DATE})")
    p.add_argument("--raw_dir", default=str(RAW_DIR),
                   help=f"Raw data directory (default: {RAW_DIR})")
    p.add_argument("--dry_run", action="store_true",
                   help="Parse and report without writing any output files")
    return p.parse_args()


def find_mid_for_subsystem(subsystem: str) -> str:
    """Reverse lookup: subsystem name → MID."""
    for mid, name in MID_MAP.items():
        if name == subsystem:
            return mid
    raise ValueError(f"Unknown subsystem: {subsystem}")


def find_files_for_mid(raw_dir: Path, mid: str) -> list:
    """
    Find all CSV files for a given MID, handling both filename orderings:
        09A5 2026-04-08.csv
        2026-04-08 09A5.csv
    """
    pattern_a = str(raw_dir / f"{mid} *.csv")
    pattern_b = str(raw_dir / f"* {mid}.csv")
    files = sorted(glob.glob(pattern_a) + glob.glob(pattern_b))
    # Deduplicate (shouldn't happen but be safe)
    files = sorted(set(files))
    return files


def load_subsystem(raw_dir: Path, mid: str, subsystem: str,
                   drop_cols: list) -> pd.DataFrame:
    """Load and concatenate all daily CSVs for one subsystem."""
    files = find_files_for_mid(raw_dir, mid)
    if not files:
        raise FileNotFoundError(
            f"No CSV files found for MID={mid} ({subsystem}) in {raw_dir}\n"
            f"Expected filenames like: '{mid} 2026-04-08.csv' or "
            f"'2026-04-08 {mid}.csv'")

    print(f"  Found {len(files)} files for {mid} ({subsystem})")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: could not read {f}: {e}")

    if not dfs:
        raise RuntimeError(f"All files failed to load for {subsystem}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} raw rows, {df.shape[1]} columns")
    return df


def clean_timestamps(df: pd.DataFrame, subsystem: str) -> pd.DataFrame:
    """Parse timestamp column and remove GPS-acquisition rows."""
    if TIMESTAMP_COL not in df.columns:
        raise KeyError(
            f"Timestamp column '{TIMESTAMP_COL}' not found in {subsystem}.\n"
            f"Available columns: {list(df.columns)}")

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")

    # Drop rows where timestamp couldn't be parsed
    n_unparseable = df[TIMESTAMP_COL].isna().sum()
    if n_unparseable > 0:
        print(f"  Dropping {n_unparseable:,} rows with unparseable timestamps")
        df = df.dropna(subset=[TIMESTAMP_COL])

    # Drop GPS-acquisition rows (clock = Jan 1 2000 before GPS fix)
    gps_mask = df[TIMESTAMP_COL] < GPS_EPOCH
    n_gps = gps_mask.sum()
    if n_gps > 0:
        print(f"  Dropping {n_gps:,} GPS-acquisition rows (before {GPS_EPOCH.date()})")
        df = df[~gps_mask]

    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    print(f"  {len(df):,} rows after timestamp cleaning "
          f"({df[TIMESTAMP_COL].min()} → {df[TIMESTAMP_COL].max()})")
    return df


def drop_unnecessary_columns(df: pd.DataFrame, subsystem: str,
                              drop_cols: list) -> pd.DataFrame:
    """Drop configured unnecessary columns plus any fully-unnamed columns."""
    # Always drop Unnamed index columns
    unnamed = [c for c in df.columns if re.match(r"^Unnamed", c)]
    to_drop = list(set(drop_cols + unnamed))
    to_drop = [c for c in to_drop if c in df.columns]

    if to_drop:
        print(f"  Dropping {len(to_drop)} columns: {to_drop}")
        df = df.drop(columns=to_drop)

    return df


def encode_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode categoricals, fill NaNs."""
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == "category":
            df[col] = pd.Categorical(df[col]).codes.astype(float)
            df[col] = df[col].replace(-1, np.nan)

    # Forward-fill then fill remaining with column mean
    df = df.ffill()
    df = df.fillna(df.mean(numeric_only=True))
    return df


def resample_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 60-second uniform grid, taking mean within each bucket."""
    df = df.set_index(TIMESTAMP_COL)
    df = df.resample("60s").mean()
    df = df.reset_index()
    return df


def prepare_subsystem(raw_dir: Path, subsystem: str, split_date: str,
                       dry_run: bool) -> dict:
    """Full prep pipeline for one subsystem. Returns metadata for YAML."""
    print(f"\n{'='*60}")
    print(f"Processing {subsystem}")
    print(f"{'='*60}")

    mid = find_mid_for_subsystem(subsystem)
    drop_cols = COLUMNS_TO_DROP.get(subsystem, [])

    # Load
    df = load_subsystem(raw_dir, mid, subsystem, drop_cols)

    # Clean timestamps
    df = clean_timestamps(df, subsystem)

    # Drop unnecessary columns
    df = drop_unnecessary_columns(df, subsystem, drop_cols)

    # Encode and fill BEFORE resampling so object columns are numeric
    # (resample.mean() fails on string/bool columns)
    ts_col = df[[TIMESTAMP_COL]].copy()
    feature_df = df.drop(columns=[TIMESTAMP_COL])
    feature_df = encode_and_fill(feature_df)
    df = pd.concat([ts_col, feature_df], axis=1)

    # Resample
    df = resample_to_grid(df)
    print(f"  After resampling: {len(df):,} rows")

    # Separate timestamp from features
    timestamps = df[TIMESTAMP_COL].copy()
    feature_df = df.drop(columns=[TIMESTAMP_COL])
    channel_names = list(feature_df.columns)
    n_channels = len(channel_names)
    print(f"  Channels: {n_channels}")

    # Train/test split — auto-calculate 80/20 if no date provided
    if split_date is None:
        n = len(timestamps)
        split_idx = int(n * 0.80)
        split_ts  = timestamps.iloc[split_idx]
        print(f"  Auto split at 80/20: {split_ts.date()} "
              f"(row {split_idx:,} of {n:,})")
    else:
        split_ts = pd.Timestamp(split_date, tz="UTC")
    train_mask = timestamps < split_ts
    test_mask  = timestamps >= split_ts

    train_arr  = feature_df[train_mask].values.astype(np.float32)
    test_arr   = feature_df[test_mask].values.astype(np.float32)
    test_ts    = timestamps[test_mask].values
    test_labels = np.zeros(len(test_arr), dtype=np.int32)

    print(f"  Train: {train_arr.shape}  "
          f"({timestamps[train_mask].min().date()} → "
          f"{timestamps[train_mask].max().date()})")
    print(f"  Test:  {test_arr.shape}  "
          f"({timestamps[test_mask].min().date()} → "
          f"{timestamps[test_mask].max().date()})")

    if dry_run:
        print("  [DRY RUN] Skipping file writes")
        return {
            "subsystem": subsystem, "n_channels": n_channels,
            "train_rows": len(train_arr), "test_rows": len(test_arr),
        }

    # Write dataset
    mission_name = f"STPSat7-{subsystem}"
    out_dir = Path("dataset") / mission_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f"{mission_name}_train.npy",       train_arr)
    np.save(out_dir / f"{mission_name}_test.npy",        test_arr)
    np.save(out_dir / f"{mission_name}_test_label.npy",  test_labels)
    np.save(out_dir / f"{mission_name}_test_timestamps.npy", test_ts)
    (out_dir / f"{mission_name}_channels.txt").write_text(
        "\n".join(channel_names))

    print(f"  Saved to {out_dir}/")

    return {
        "subsystem":   subsystem,
        "mission_name": mission_name,
        "n_channels":  n_channels,
        "train_rows":  len(train_arr),
        "test_rows":   len(test_arr),
    }


def update_yaml(results: list):
    """Write or update anomaly_detection_stpsat7.yaml."""
    # Load existing if present
    if YAML_PATH.exists():
        with open(YAML_PATH) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    if "task_dataset" not in config:
        config["task_dataset"] = {}

    for r in results:
        mission_name = r["mission_name"]
        n_ch = r["n_channels"]
        config["task_dataset"][mission_name] = {
            "task_name":    "anomaly_detection",
            "dataset_name": mission_name,
            "dataset":      mission_name,
            "data":         mission_name,
            "root_path":    f"dataset/{mission_name}/",
            "seq_len":      SEQ_LEN,
            "label_len":    LABEL_LEN,
            "pred_len":     PRED_LEN,
            "features":     "M",
            "embed":        "timeF",
            "enc_in":       n_ch,
            "dec_in":       n_ch,
            "c_out":        n_ch,
        }

    YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(YAML_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nUpdated {YAML_PATH}")


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)

    # Validate requested subsystems
    valid = set(MID_MAP.values())
    for s in args.subsystems:
        if s not in valid:
            print(f"ERROR: Unknown subsystem '{s}'. Valid: {sorted(valid)}")
            raise SystemExit(1)

    print(f"STPSat-7 Data Preparation")
    print(f"  Raw dir    : {raw_dir}")
    print(f"  Split date : {args.split_date}")
    print(f"  Subsystems : {args.subsystems}")
    print(f"  Dry run    : {args.dry_run}")

    if not raw_dir.exists():
        print(f"\nERROR: Raw data directory not found: {raw_dir}")
        print(f"Expected structure:")
        print(f"  {raw_dir}/09A5 2026-04-08.csv")
        print(f"  {raw_dir}/2026-04-08 09A5.csv  (either naming convention)")
        raise SystemExit(1)

    # Show what files are found before processing
    print(f"\nScanning {raw_dir} ...")
    for mid, subsystem in MID_MAP.items():
        if subsystem not in args.subsystems:
            continue
        files = find_files_for_mid(raw_dir, mid)
        print(f"  {mid} ({subsystem:4s}): {len(files)} files")

    results = []
    errors  = []

    for subsystem in args.subsystems:
        try:
            r = prepare_subsystem(raw_dir, subsystem, args.split_date, args.dry_run)
            results.append(r)
        except Exception as e:
            print(f"\nERROR processing {subsystem}: {e}")
            errors.append((subsystem, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    for r in results:
        status = "[DRY RUN]" if args.dry_run else "OK"
        print(f"  {status}  STPSat7-{r['subsystem']:6s}  "
              f"{r['n_channels']:4d} channels  "
              f"train={r['train_rows']:,}  test={r['test_rows']:,}")
    for subsystem, err in errors:
        print(f"  ERROR  {subsystem}: {err}")

    if not args.dry_run and results:
        update_yaml(results)
        print(f"\nNext steps:")
        print(f"  1. Update data_factory.py to register STPSat7 missions")
        print(f"  2. Run: MISSIONS=\"{' '.join(r['mission_name'] for r in results)}\" "
              f"sbatch slurm/02_run_anomaly.sh")


if __name__ == "__main__":
    main()