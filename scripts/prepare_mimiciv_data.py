#!/usr/bin/env python3
"""
MIMIC-IV data preparation script.
Converts raw MIMIC-IV files to format expected by preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime


def prepare_chartevents(
        chartevents_path: Path,
        admissions_df: pd.DataFrame,
        output_path: Path
) -> pd.DataFrame:
    """
    Process chartevents (vital signs) to hourly aggregated format.
    """
    print("\n[1/3] Processing chartevents...")

    vitals_itemids = {
        220045: 'heart_rate',
        220050: 'sbp',
        220051: 'dbp',
        220052: 'mbp',
        220210: 'resp_rate',
        223761: 'temperature',
        220277: 'spo2',
    }

    print(f"  Loading chartevents from: {chartevents_path}")
    chartevents = pd.read_csv(chartevents_path)

    chartevents = chartevents[chartevents['itemid'].isin(vitals_itemids.keys())]

    chartevents = chartevents.merge(
        admissions_df[['hadm_id', 'admittime']],
        on='hadm_id'
    )

    chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
    chartevents['admittime'] = pd.to_datetime(chartevents['admittime'])

    chartevents['time_hours'] = (
                                        chartevents['charttime'] - chartevents['admittime']
                                ).dt.total_seconds() / 3600

    chartevents = chartevents[chartevents['time_hours'] >= 0]

    chartevents['vital'] = chartevents['itemid'].map(vitals_itemids)

    chartevents_pivot = chartevents.pivot_table(
        index=['hadm_id', 'time_hours'],
        columns='vital',
        values='valuenum',
        aggfunc='mean'
    ).reset_index()

    output_file = output_path / 'chartevents_processed.csv'
    chartevents_pivot.to_csv(output_file, index=False)

    print(f"  Saved processed chartevents: {len(chartevents_pivot)} records")
    print(f"  Features: {list(chartevents_pivot.columns[2:])}")

    return chartevents_pivot


def prepare_labevents(
        labevents_path: Path,
        admissions_df: pd.DataFrame,
        output_path: Path
) -> pd.DataFrame:
    """
    Process labevents to format compatible with pipeline.
    """
    print("\n[2/3] Processing labevents...")

    lab_itemids = {
        50912: 'creatinine',
        50971: 'potassium',
        50983: 'sodium',
        50902: 'chloride',
        50882: 'bicarbonate',
        51006: 'bun',
        50931: 'glucose',
        51221: 'hematocrit',
        51301: 'wbc',
        51265: 'platelet',
    }

    print(f"  Loading labevents from: {labevents_path}")
    labevents = pd.read_csv(labevents_path)

    labevents = labevents[labevents['itemid'].isin(lab_itemids.keys())]

    labevents = labevents.merge(
        admissions_df[['hadm_id', 'admittime']],
        on='hadm_id'
    )

    labevents['charttime'] = pd.to_datetime(labevents['charttime'])
    labevents['admittime'] = pd.to_datetime(labevents['admittime'])

    labevents['time_hours'] = (
                                      labevents['charttime'] - labevents['admittime']
                              ).dt.total_seconds() / 3600

    labevents = labevents[labevents['time_hours'] >= 0]

    labevents['lab'] = labevents['itemid'].map(lab_itemids)

    labevents_pivot = labevents.pivot_table(
        index=['hadm_id', 'time_hours'],
        columns='lab',
        values='valuenum',
        aggfunc='mean'
    ).reset_index()

    output_file = output_path / 'labevents_processed.csv'
    labevents_pivot.to_csv(output_file, index=False)

    print(f"  Saved processed labevents: {len(labevents_pivot)} records")
    print(f"  Features: {list(labevents_pivot.columns[2:])}")

    return labevents_pivot


def validate_files(input_path: Path) -> bool:
    """
    Validate that required MIMIC-IV files exist.
    """
    print("\nValidating MIMIC-IV files...")

    required_files = [
        'hosp/admissions.csv',
        'hosp/patients.csv',
        'icu/chartevents.csv',
        'hosp/labevents.csv',
    ]

    missing_files = []
    for file in required_files:
        file_path = input_path / file
        if not file_path.exists():
            missing_files.append(file)
            print(f"  Missing: {file}")
        else:
            print(f"  Found: {file}")

    if missing_files:
        print(f"\nError: {len(missing_files)} required files missing")
        return False

    print("\nAll required files present")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MIMIC-IV data for preprocessing pipeline'
    )
    parser.add_argument('--input-dir', required=True,
                        help='Directory with raw MIMIC-IV data')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to save processed data')
    parser.add_argument('--max-hours', type=int, default=72,
                        help='Maximum hours to include from admission')

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / 'hosp').mkdir(exist_ok=True)
    (output_path / 'icu').mkdir(exist_ok=True)

    print("=" * 80)
    print("MIMIC-IV DATA PREPARATION")
    print("=" * 80)
    print(f"\nInput directory: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Max hours: {args.max_hours}")

    if not validate_files(input_path):
        return 1

    print("\n[0/3] Loading admissions and patients...")
    admissions = pd.read_csv(input_path / 'hosp' / 'admissions.csv')
    patients = pd.read_csv(input_path / 'hosp' / 'patients.csv')

    print(f"  Admissions: {len(admissions)}")
    print(f"  Patients: {len(patients)}")

    admissions.to_csv(output_path / 'hosp' / 'admissions.csv', index=False)
    patients.to_csv(output_path / 'hosp' / 'patients.csv', index=False)

    chartevents = prepare_chartevents(
        input_path / 'icu' / 'chartevents.csv',
        admissions,
        output_path / 'icu'
    )

    labevents = prepare_labevents(
        input_path / 'hosp' / 'labevents.csv',
        admissions,
        output_path / 'hosp'
    )

    diagnoses_path = input_path / 'hosp' / 'diagnoses_icd.csv'
    if diagnoses_path.exists():
        print("\n[3/3] Copying diagnoses_icd...")
        diagnoses = pd.read_csv(diagnoses_path)
        diagnoses.to_csv(output_path / 'hosp' / 'diagnoses_icd.csv', index=False)
        print(f"  Copied {len(diagnoses)} diagnosis records")

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nProcessed files saved to: {output_path}")
    print("\nNext steps:")
    print("1. Update configs/mimiciv_config.yaml with base_dir:")
    print(f"   base_dir: {output_path}")
    print("2. Run preprocessing:")
    print("   python scripts/run_preprocessing.py --config configs/mimiciv_config.yaml")


if __name__ == '__main__':
    main()