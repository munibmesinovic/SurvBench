# !/usr/bin/env python3
"""
Helper script to prepare MC-MED data in the format expected by the preprocessing pipeline.
Converts from raw ED data to the structured format needed.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Tuple


def prepare_mcmed_labels(
        edstays_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        output_path: Path
) -> pd.DataFrame:
    """
    Prepare labels file with competing risks outcomes.

    Expected input columns:
    - edstays_df: stay_id, intime, outtime
    - outcomes_df: stay_id, disposition (or similar outcome indicator)

    Output columns:
    - stay_id (index), ed_los_hours, outcome (0=censored, 1=ICU, 2=Hospital, 3=Obs)
    """
    print("\n[1/5] Preparing labels...")

    # Calculate ED length of stay
    edstays_df['intime'] = pd.to_datetime(edstays_df['intime'])
    edstays_df['outtime'] = pd.to_datetime(edstays_df['outtime'])
    edstays_df['ed_los_hours'] = (
                                         edstays_df['outtime'] - edstays_df['intime']
                                 ).dt.total_seconds() / 3600

    # Merge with outcomes
    labels = edstays_df[['stay_id', 'ed_los_hours']].copy()
    labels = labels.merge(outcomes_df, on='stay_id', how='left')

    # Map disposition to outcome codes
    # Adjust this mapping based on your data
    disposition_mapping = {
        'HOME': 0,  # Censored (discharged home)
        'ICU': 1,  # ICU admission
        'FLOOR': 2,  # Hospital admission
        'OBSERVATION': 3,  # Observation unit
        'DIED': 1,  # Consider as critical (ICU-level)
        'LEFT': 0,  # Left without being seen - censored
    }

    if 'disposition' in labels.columns:
        labels['outcome'] = labels['disposition'].map(disposition_mapping)
        labels['outcome'] = labels['outcome'].fillna(0).astype(int)
    else:
        raise ValueError("outcomes_df must have 'disposition' or similar column")

    # Keep only needed columns
    labels = labels[['stay_id', 'ed_los_hours', 'outcome']]
    labels = labels.set_index('stay_id')

    # Save
    labels.to_csv(output_path / 'labels.csv')

    print(f"  ✓ Labels saved: {len(labels)} stays")
    print(f"  Outcome distribution:")
    for outcome in sorted(labels['outcome'].unique()):
        count = (labels['outcome'] == outcome).sum()
        pct = count / len(labels) * 100
        print(f"    Outcome {outcome}: {count} ({pct:.1f}%)")

    return labels


def prepare_mcmed_static(
        patients_df: pd.DataFrame,
        edstays_df: pd.DataFrame,
        output_path: Path
) -> pd.DataFrame:
    """
    Prepare static features (demographics).

    Expected columns:
    - patients_df: subject_id, gender, anchor_age, ...
    - edstays_df: stay_id, subject_id
    """
    print("\n[2/5] Preparing static features...")

    # Merge patient info with ED stays
    static = edstays_df[['stay_id', 'subject_id']].copy()
    static = static.merge(patients_df, on='subject_id', how='left')

    # Select demographic features
    demographic_cols = ['gender', 'anchor_age', 'race', 'insurance']
    available_cols = [col for col in demographic_cols if col in static.columns]

    if 'anchor_age' in static.columns:
        static = static.rename(columns={'anchor_age': 'age'})
        available_cols = ['age' if c == 'anchor_age' else c for c in available_cols]

    # One-hot encode categoricals
    categorical_cols = ['gender', 'race', 'insurance']
    for col in categorical_cols:
        if col in static.columns:
            dummies = pd.get_dummies(static[col], prefix=col, drop_first=True)
            static = pd.concat([static, dummies], axis=1)
            static = static.drop(columns=[col])

    # Set index and keep only feature columns
    static = static.set_index('stay_id')
    static = static.drop(columns=['subject_id'], errors='ignore')

    # Save
    static.to_csv(output_path / 'static_features.csv')

    print(f"  ✓ Static features saved: {len(static)} stays, {len(static.columns)} features")

    return static


def prepare_mcmed_timeseries(
        vitals_df: pd.DataFrame,
        labs_df: pd.DataFrame,
        edstays_df: pd.DataFrame,  # <-- ADDED THIS ARGUMENT
        output_path: Path
) -> pd.DataFrame:
    """
    Prepare time-series vitals and labs.
    Expected columns:
    - vitals_df: stay_id, charttime, hr, sbp, dbp, temp, spo2, rr, ...
    - labs_df: stay_id, charttime, lab values, ...
    - edstays_df: stay_id, intime
    """
    print("\n[3/5] Preparing time-series...")

    # Get intime for all stays
    edstays_df['intime'] = pd.to_datetime(edstays_df['intime'])
    stay_times = edstays_df[['stay_id', 'intime']].drop_duplicates()

    ts_dfs = []

    # Process Vitals
    if 'charttime' in vitals_df.columns:
        vitals_df['charttime'] = pd.to_datetime(vitals_df['charttime'])
        vitals_df = vitals_df.merge(stay_times, on='stay_id', how='left')
        vitals_df['time_hours'] = (
                                          vitals_df['charttime'] - vitals_df['intime']
                                  ).dt.total_seconds() / 3600

        vital_cols = [col for col in vitals_df.columns
                      if col not in ['stay_id', 'charttime', 'intime', 'subject_id', 'time_hours']]
        vitals_df = vitals_df[['stay_id', 'time_hours'] + vital_cols]
        ts_dfs.append(vitals_df)
        print(f"  Processed {len(vitals_df)} vital measurements.")

    # Process Labs
    if 'charttime' in labs_df.columns:
        labs_df['charttime'] = pd.to_datetime(labs_df['charttime'])
        labs_df = labs_df.merge(stay_times, on='stay_id', how='left')
        labs_df['time_hours'] = (
                                        labs_df['charttime'] - labs_df['intime']
                                ).dt.total_seconds() / 3600

        lab_cols = [col for col in labs_df.columns
                    if col not in ['stay_id', 'charttime', 'intime', 'subject_id', 'time_hours']]
        labs_df = labs_df[['stay_id', 'time_hours'] + lab_cols]
        ts_dfs.append(labs_df)
        print(f"  Processed {len(labs_df)} lab measurements.")

    # Merge vitals and labs
    if not ts_dfs:
        print("  Warning: No time-series data found.")
        return pd.DataFrame()

    ts = pd.concat(ts_dfs, axis=0, ignore_index=True)

    # Pivot if data is in long format (e.g., itemid, value)
    # This example assumes data is already in wide format (one col per feature)
    # If not, a pivot or groupby.pivot is needed here.

    # Group by stay_id and time_hours if labs/vitals have same-time measurements
    ts = ts.groupby(['stay_id', 'time_hours']).mean().reset_index()

    ts = ts.sort_values(['stay_id', 'time_hours'])

    # Remove negative times (measurements before admission)
    ts = ts[ts['time_hours'] >= 0]

    value_cols = [col for col in ts.columns if col not in ['stay_id', 'time_hours']]

    # Save
    ts.to_csv(output_path / 'timeseries_vitals.csv', index=False)

    print(f"  ✓ Time-series saved: {len(ts)} measurements")
    print(f"  Features: {len(value_cols)}")
    print(f"  Unique stays: {ts['stay_id'].nunique()}")

    return ts


def prepare_mcmed_icd(
        diagnoses_df: pd.DataFrame,
        top_n: int,
        output_path: Path
) -> pd.DataFrame:
    """
    Prepare ICD codes (one-hot encoded or counts).

    Expected columns:
    - diagnoses_df: stay_id, icd_code
    """
    print("\n[4/5] Preparing ICD codes...")

    # Get top N most frequent codes
    code_counts = diagnoses_df['icd_code'].value_counts()
    top_codes = code_counts.head(top_n).index

    # Filter to top codes
    diagnoses_filtered = diagnoses_df[diagnoses_df['icd_code'].isin(top_codes)]

    # Create one-hot encoding
    icd_onehot = pd.crosstab(
        diagnoses_filtered['stay_id'],
        diagnoses_filtered['icd_code']
    )

    # Convert to binary (in case of multiple occurrences)
    icd_onehot = (icd_onehot > 0).astype(int)

    # Save
    icd_onehot.to_csv(output_path / 'icd_codes.csv')

    print(f"  ✓ ICD codes saved: {len(icd_onehot)} stays")
    print(f"  Top {top_n} codes encoded")

    return icd_onehot


def prepare_mcmed_radiology(
        radiology_df: pd.DataFrame,
        embeddings_df: pd.DataFrame,
        output_path: Path
) -> pd.DataFrame:
    """
    Prepare radiology report embeddings.

    If embeddings are pre-computed (from Clinical-Longformer), just format them.
    If not, this would require running the text through the embedding model.
    """
    print("\n[5/5] Preparing radiology embeddings...")

    if embeddings_df is not None:
        # Embeddings already computed
        embeddings_df = embeddings_df.set_index('stay_id')
        embeddings_df.to_csv(output_path / 'radiology_emb.csv')

        print(f"  ✓ Radiology embeddings saved: {len(embeddings_df)} stays")
        print(f"  Embedding dimension: {len(embeddings_df.columns)}")
    else:
        # Save text for later embedding
        radiology_df = radiology_df.set_index('stay_id')
        radiology_df.to_csv(output_path / 'radiology_text.csv')

        print(f"  ✓ Radiology text saved (embeddings need to be computed)")
        print(f"  Use Clinical-Longformer or similar model to create embeddings")

    return embeddings_df if embeddings_df is not None else radiology_df


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MC-MED data for preprocessing pipeline'
    )
    parser.add_argument('--input-dir', required=True, help='Directory with raw MC-MED data')
    parser.add_argument('--output-dir', required=True, help='Directory to save prepared data')
    parser.add_argument('--top-n-icd', type=int, default=500, help='Number of top ICD codes to keep')

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MC-MED DATA PREPARATION")
    print("=" * 80)
    print(f"\nInput directory: {input_path}")
    print(f"Output directory: {output_path}")

    # Load raw data
    print("\nLoading raw data files...")

    # Adjust these filenames based on your actual data
    edstays = pd.read_csv(input_path / 'edstays.csv')
    patients = pd.read_csv(input_path / 'patients.csv')
    outcomes = pd.read_csv(input_path / 'outcomes.csv')  # Or extract from edstays
    vitals = pd.read_csv(input_path / 'vitalsign.csv')
    labs = pd.read_csv(input_path / 'labevents.csv')
    diagnoses = pd.read_csv(input_path / 'diagnoses_icd.csv')

    print("✓ Raw data loaded")

    # Prepare each component
    labels = prepare_mcmed_labels(edstays, outcomes, output_path)
    static = prepare_mcmed_static(patients, edstays, output_path)
    timeseries = prepare_mcmed_timeseries(vitals, labs, edstays, output_path)
    icd = prepare_mcmed_icd(diagnoses, args.top_n_icd, output_path)

    # Radiology (if available)
    radiology_path = input_path / 'radiology.csv'
    if radiology_path.exists():
        radiology = pd.read_csv(radiology_path)

        # Check for pre-computed embeddings
        embeddings_path = input_path / 'radiology_embeddings.csv'
        embeddings = None
        if embeddings_path.exists():
            embeddings = pd.read_csv(embeddings_path)

        prepare_mcmed_radiology(radiology, embeddings, output_path)

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nPrepared files saved to: {output_path}")
    print("\nNext step:")
    print(f"  python scripts/run_preprocessing.py --config configs/mcmed_config.yaml")
    print("\n  (Update base_dir in config to point to this output directory)")


if __name__ == '__main__':
    main()