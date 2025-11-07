import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc
from typing import Dict, Tuple
from data.base_loader import BaseDataLoader
from tqdm import tqdm


class eICUDataLoader(BaseDataLoader):
    """
    Data loader for eICU Collaborative Research Database (raw files).

    This loader processes raw eICU 2.0 CSVs (patient, lab, vitalPeriodic,
    vitalAperiodic) to generate the cohort and feature sets.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.files = config['dataset']['files']
        self.max_horizon = config['preprocessing']['max_horizon_hours']
        self.max_hours_input = config['preprocessing']['max_hours']

        # Caching
        self._cohort_df_cache = None

    def _get_cohort_df(self) -> pd.DataFrame:
        """
        Loads and caches the core cohort from raw patient.csv.
        This file contains demographics, admission info, and outcomes.
        """
        if self._cohort_df_cache is not None:
            return self._cohort_df_cache

        patient_path = self.base_dir / self.files['patient']
        print(f"Loading core cohort from: {patient_path}")

        # Define columns to load
        cols_to_load = [
            'patientunitstayid', 'patienthealthsystemstayid', 'gender', 'age',
            'ethnicity', 'unitstaytype', 'unitdischargeoffset',
            'hospitaldischargestatus'
        ]

        cohort_df = pd.read_csv(patient_path, usecols=cols_to_load)

        # 1. Apply age filters (from notebook/config)
        cohort_df = cohort_df[cohort_df['age'] != '> 89']
        cohort_df['age'] = pd.to_numeric(cohort_df['age'], errors='coerce')
        cohort_df = cohort_df.dropna(subset=['age'])
        cohort_df = cohort_df[cohort_df['age'] >= 18]  # Standard adult cohort

        # 2. Calculate duration in hours
        cohort_df['duration_hours'] = cohort_df['unitdischargeoffset'] / 60.0

        # 3. Apply minimum stay filter (from config)
        min_hours = self.config['dataset']['cohort']['min_stay_hours']
        cohort_df = cohort_df[cohort_df['duration_hours'] >= min_hours]

        # 4. Rename for pipeline compatibility
        # 'patienthealthsystemstayid' is the unique patient identifier
        # 'patientunitstayid' is the unique stay identifier
        cohort_df = cohort_df.rename(columns={
            'patientunitstayid': 'admission_id',
            'patienthealthsystemstayid': 'subject_id'
        })

        cohort_df = cohort_df.set_index('admission_id')

        print(f"Loaded {len(cohort_df)} stays from patient.csv meeting criteria.")
        self._cohort_df_cache = cohort_df
        return cohort_df

    def load_labels(self) -> pd.DataFrame:
        """
        Extracts survival labels from the processed cohort_df.
        """
        cohort_df = self._get_cohort_df()

        # 1. Get duration
        durations = cohort_df['duration_hours'].values

        # 2. Get event status
        # 'EXPIRED' -> 1, 'ALIVE' -> 0
        events = (cohort_df['hospitaldischargestatus'] == 'Expired').astype(int)

        # 3. Censor events beyond max horizon (e.g., 240h)
        events[durations > self.max_horizon] = 0
        durations = np.clip(durations, a_min=None, a_max=self.max_horizon)

        labels_df = pd.DataFrame({
            'duration': durations,
            'event': events,
            'subject_id': cohort_df['subject_id']
        }, index=cohort_df.index)

        print(f"Processed labels: {events.mean() * 100:.2f}% event rate (within {self.max_horizon}h)")
        return labels_df

    def load_static_features(self) -> pd.DataFrame:
        """
        Extracts static (triage/demo) features from the cohort_df.
        """
        cohort_df = self._get_cohort_df()

        static_cols = ['age', 'gender', 'ethnicity', 'unitstaytype']
        static_df = cohort_df[static_cols].copy()

        # One-hot encode categorical features
        static_df = pd.get_dummies(
            static_df,
            columns=['gender', 'ethnicity', 'unitstaytype'],
            dtype=int
        )

        # Rename 'age' to match notebook/config expectation if needed
        # (Assuming 'age' is fine)

        print(f"Loaded {len(static_df.columns)} static features.")
        return static_df

    def load_timeseries(self) -> pd.DataFrame:
        """
        Loads and processes raw lab.csv, vitalPeriodic.csv, and
        vitalAperiodic.csv for the first max_hours_input (e.g., 24h).

        **This version is heavily optimized to aggregate *before* merging.**
        """
        cohort_ids = self.cohort_df.index.unique()
        train_ids = self.create_splits()['train']
        max_offset_minutes = self.max_hours_input * 60

        # We will aggregate each file into hourly bins *first*.

        # --- 1. Load and Pre-Aggregate VitalPeriodic ---
        vital_p_path = self.base_dir / self.files['vital_periodic']
        print(f"Loading and aggregating vitalPeriodic from: {vital_p_path}...")
        vital_p_cols = [
            'patientunitstayid', 'observationoffset', 'temperature', 'sao2',
            'heartrate', 'respiration', 'systemicsystolic', 'systemicdiastolic'
        ]
        vital_p_df = pd.read_csv(vital_p_path, usecols=vital_p_cols)
        vital_p_df = vital_p_df[vital_p_df['patientunitstayid'].isin(cohort_ids)]
        vital_p_df = vital_p_df[
            (vital_p_df['observationoffset'] >= 0) &
            (vital_p_df['observationoffset'] < max_offset_minutes)
            ]

        # *** NEW AGGREGATION STEP ***
        vital_p_df['hour_bin'] = (vital_p_df['observationoffset'] / 60.0).astype(int)
        vital_p_cols_to_agg = [col for col in vital_p_cols if col not in ['patientunitstayid', 'observationoffset']]

        vitals_p_hourly = vital_p_df.groupby(['patientunitstayid', 'hour_bin'])[vital_p_cols_to_agg].mean()
        print(f"  Aggregated {len(vital_p_df)} rows to {len(vitals_p_hourly)} hourly records.")
        del vital_p_df  # Free memory
        gc.collect()

        # --- 2. Load and Pre-Aggregate VitalAperiodic ---
        vital_a_path = self.base_dir / self.files['vital_aperiodic']
        print(f"Loading and aggregating vitalAperiodic from: {vital_a_path}...")
        vital_a_cols = [
            'patientunitstayid', 'observationoffset', 'noninvasivesystolic',
            'noninvasivediastolic'
        ]
        vital_a_df = pd.read_csv(vital_a_path, usecols=vital_a_cols)
        vital_a_df = vital_a_df[vital_a_df['patientunitstayid'].isin(cohort_ids)]
        vital_a_df = vital_a_df[
            (vital_a_df['observationoffset'] >= 0) &
            (vital_a_df['observationoffset'] < max_offset_minutes)
            ]

        # *** NEW AGGREGATION STEP ***
        vital_a_df['hour_bin'] = (vital_a_df['observationoffset'] / 60.0).astype(int)
        vital_a_cols_to_agg = [col for col in vital_a_cols if col not in ['patientunitstayid', 'observationoffset']]

        vitals_a_hourly = vital_a_df.groupby(['patientunitstayid', 'hour_bin'])[vital_a_cols_to_agg].mean()
        print(f"  Aggregated {len(vital_a_df)} rows to {len(vitals_a_hourly)} hourly records.")
        del vital_a_df  # Free memory
        gc.collect()

        # --- 3. Load and Pre-Aggregate Labs ---
        lab_path = self.base_dir / self.files['lab']
        print(f"Loading and aggregating labs from: {lab_path}...")
        labs_df = pd.read_csv(lab_path, usecols=[
            'patientunitstayid', 'labresultoffset', 'labname', 'labresult'
        ])
        labs_df = labs_df[labs_df['patientunitstayid'].isin(cohort_ids)]
        labs_df = labs_df.dropna(subset=['labresult'])
        labs_df = labs_df[
            (labs_df['labresultoffset'] >= 0) &
            (labs_df['labresultoffset'] < max_offset_minutes)
            ]

        # *** NEW AGGREGATION STEP ***
        # Pivot to wide format first
        labs_wide_raw = labs_df.pivot_table(
            index=['patientunitstayid', 'labresultoffset'],
            columns='labname',
            values='labresult'
        )
        del labs_df
        gc.collect()

        # Now aggregate the wide-format table by hour
        labs_wide_raw = labs_wide_raw.reset_index()
        labs_wide_raw['hour_bin'] = (labs_wide_raw['labresultoffset'] / 60.0).astype(int)
        lab_cols = [col for col in labs_wide_raw.columns if
                    col not in ['patientunitstayid', 'labresultoffset', 'hour_bin']]

        labs_hourly = labs_wide_raw.groupby(['patientunitstayid', 'hour_bin'])[lab_cols].mean()
        print(f"  Aggregated and pivoted lab rows to {len(labs_hourly)} hourly records.")
        del labs_wide_raw
        gc.collect()

        # --- 4. Merge Aggregated DataFrames ---
        print("Merging aggregated hourly data...")
        ts_df = pd.merge(vitals_p_hourly, vitals_a_hourly, left_index=True, right_index=True, how='outer')
        ts_df = pd.merge(ts_df, labs_hourly, left_index=True, right_index=True, how='outer')

        del vitals_p_hourly, vitals_a_hourly, labs_hourly
        gc.collect()

        # --- 5. Finalize for pipeline ---
        ts_df = ts_df.reset_index()
        ts_df = ts_df.rename(columns={'patientunitstayid': 'admission_id', 'hour_bin': 'time_hours'})
        ts_df = ts_df.set_index(['admission_id', 'time_hours'])
        ts_df = ts_df.sort_index()

        # --- 6. Missingness Filter (same as before) ---
        print(f"Filtering features by missingness...")
        missingness_threshold = self.config['preprocessing']['missingness_threshold']

        ts_train = ts_df[ts_df.index.get_level_values('admission_id').isin(train_ids)]
        missingness = ts_train.notna().mean()
        cols_to_keep = missingness[missingness >= missingness_threshold].index
        ts_df = ts_df[cols_to_keep]

        print(f"Kept {len(cols_to_keep)} dynamic features (>= {missingness_threshold * 100}% present in training)")

        del ts_train
        gc.collect()

        return ts_df

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        # All filtering was done in _get_cohort_df().
        if 'subject_id' not in df.columns:
            raise ValueError("Critical Error: 'subject_id' was not found.")
        print(f"Cohort criteria applied: {len(df)} stays remaining.")
        return df