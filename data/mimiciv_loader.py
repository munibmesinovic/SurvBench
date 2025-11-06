import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from .base_loader import BaseDataLoader


class MIMICIVDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC-IV database with multi-modal support.
    Supports: time-series (vitals/labs), static demographics, ICD codes, and radiology notes.
    Outcome: In-hospital mortality (single risk).
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.files = config['dataset']['files']
        self.modalities_config = config.get('modalities', {})
        self.admissions_df = None

        self.load_timeseries = self.modalities_config.get('timeseries', True)
        self.load_static = self.modalities_config.get('static', True)
        self.load_icd = self.modalities_config.get('icd', False)
        self.load_radiology = self.modalities_config.get('radiology', False)

        print(f"\nMIMIC-IV Modalities Configuration:")
        print(f"  Time-series: {self.load_timeseries}")
        print(f"  Static: {self.load_static}")
        print(f"  ICD codes: {self.load_icd}")
        print(f"  Radiology: {self.load_radiology}")
        print(f"  Outcome: In-hospital mortality")

    def _get_admissions_df(self) -> pd.DataFrame:
        """Loads admissions.csv once and caches it."""
        if self.admissions_df is None:
            admissions_path = self.base_dir / self.files['admissions']
            print(f"\n(Loading admissions from: {admissions_path})")
            self.admissions_df = pd.read_csv(admissions_path)
            self.admissions_df = self.admissions_df.rename(
                columns={'hadm_id': 'admission_id'}
            )
            self.admissions_df['admittime'] = pd.to_datetime(
                self.admissions_df['admittime']
            )
            self.admissions_df['dischtime'] = pd.to_datetime(
                self.admissions_df['dischtime']
            )
        return self.admissions_df

    def load_labels(self) -> pd.DataFrame:
        """Load in-hospital mortality labels."""
        # Use the helper function to get the cached admissions data
        admissions = self._get_admissions_df().copy()
        print(f"\nProcessing labels...")

        admissions['los_hours'] = (
                                          admissions['dischtime'] - admissions['admittime']
                                  ).dt.total_seconds() / 3600

        admissions['event'] = (admissions['hospital_expire_flag'] == 1).astype(int)
        admissions['duration'] = admissions['los_hours']

        labels = admissions[['admission_id', 'subject_id', 'duration', 'event']].copy()
        labels = labels.set_index('admission_id')

        min_hours = self.config['dataset']['cohort']['min_stay_hours']
        labels = labels[labels['duration'] >= min_hours]

        event_rate = labels['event'].mean() * 100
        print(f"Loaded {len(labels)} admissions (>={min_hours}h stay)")
        print(f"In-hospital mortality rate: {event_rate:.1f}%")

        return labels

    def load_static_features(self) -> pd.DataFrame:
        """Load static demographic and admission features."""
        if not self.load_static:
            print("Static features disabled")
            return pd.DataFrame(index=pd.Index([], name='admission_id'))

        patients_path = self.base_dir / self.files['patients']

        print(f"Loading static features...")

        patients = pd.read_csv(patients_path)
        # Use the helper function here
        admissions = self._get_admissions_df().copy()

        static_df = admissions.merge(patients, on='subject_id', how='left')

        # This line is already handled by _get_admissions_df, but it's safe to run again
        static_df['admittime'] = pd.to_datetime(static_df['admittime'])
        static_df['anchor_year_group'] = static_df['anchor_year_group'].astype(str)

        admit_year = static_df['admittime'].dt.year
        anchor_year_start = static_df['anchor_year_group'].str.split(' - ').str[0].astype(int)
        static_df['age'] = static_df['anchor_age'] + (admit_year - anchor_year_start)

        feature_cols = ['admission_id', 'age', 'gender', 'race', 'admission_type',
                        'admission_location', 'insurance', 'marital_status']
        available_cols = [col for col in feature_cols if col in static_df.columns]
        static_df = static_df[available_cols]

        categorical_cols = ['gender', 'race', 'admission_type', 'admission_location',
                            'insurance', 'marital_status']
        for col in categorical_cols:
            if col in static_df.columns:
                dummies = pd.get_dummies(static_df[col], prefix=col, drop_first=True)
                static_df = pd.concat([static_df, dummies], axis=1)
                static_df = static_df.drop(columns=[col])

        static_df = static_df.set_index('admission_id')

        print(f"Loaded {len(static_df)} admissions with {len(static_df.columns)} static features")
        return static_df

    def load_timeseries(self) -> pd.DataFrame:
        """Load time-series vitals and labs."""
        if not self.load_timeseries:
            print("Time-series disabled")
            return pd.DataFrame()

        cohort_ids = self.cohort_df.index.unique()
        train_ids = self.create_splits()['train']
        max_hours = self.config['preprocessing']['max_hours']
        missingness_threshold = self.config['preprocessing']['missingness_threshold']

        chartevents_path = self.base_dir / self.files.get('chartevents')
        labevents_path = self.base_dir / self.files.get('labevents')

        # Get admissions data ONCE using the helper function
        admissions = self._get_admissions_df()
        # We only need these two columns for merging
        admissions_time = admissions[['admission_id', 'admittime']]

        ts_dfs = []

        if chartevents_path and chartevents_path.exists():
            print(f"\nLoading chartevents from: {chartevents_path}")
            chartevents = pd.read_csv(chartevents_path)
            chartevents = chartevents.rename(columns={'hadm_id': 'admission_id'})
            chartevents = chartevents[chartevents['admission_id'].isin(cohort_ids)]

            chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])

            # Use the cached admissions_time dataframe
            chartevents = chartevents.merge(
                admissions_time,
                on='admission_id'
            )
            chartevents['time_hours'] = (
                                                chartevents['charttime'] - chartevents['admittime']
                                        ).dt.total_seconds() / 3600

            chartevents = chartevents[
                (chartevents['time_hours'] >= 0) &
                (chartevents['time_hours'] < max_hours)
                ]

            vital_features = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate',
                              'temperature', 'spo2', 'glucose']
            available_vitals = [col for col in vital_features if col in chartevents.columns]

            chartevents = chartevents[['admission_id', 'time_hours'] + available_vitals]
            chartevents = chartevents.set_index(['admission_id', 'time_hours'])
            ts_dfs.append(chartevents)

            print(f"Loaded {len(chartevents)} vital measurements")

        if labevents_path and labevents_path.exists():
            print(f"Loading labevents from: {labevents_path}")
            labevents = pd.read_csv(labevents_path)
            labevents = labevents.rename(columns={'hadm_id': 'admission_id'})
            labevents = labevents[labevents['admission_id'].isin(cohort_ids)]

            labevents['charttime'] = pd.to_datetime(labevents['charttime'])

            # Use the cached admissions_time dataframe again
            labevents = labevents.merge(
                admissions_time,
                on='admission_id'
            )
            labevents['time_hours'] = (
                                              labevents['charttime'] - labevents['admittime']
                                      ).dt.total_seconds() / 3600

            labevents = labevents[
                (labevents['time_hours'] >= 0) &
                (labevents['time_hours'] < max_hours)
                ]

            lab_features = ['creatinine', 'potassium', 'sodium', 'chloride', 'bicarbonate',
                            'bun', 'glucose', 'hematocrit', 'wbc', 'platelet']
            available_labs = [col for col in lab_features if col in labevents.columns]

            labevents = labevents[['admission_id', 'time_hours'] + available_labs]
            labevents = labevents.set_index(['admission_id', 'time_hours'])
            ts_dfs.append(labevents)

            print(f"Loaded {len(labevents)} lab measurements")

        if not ts_dfs:
            print("Warning: No time-series data found")
            return pd.DataFrame()

        ts_combined = pd.concat(ts_dfs, axis=1)
        ts_combined = ts_combined.groupby(level=[0, 1]).mean()

        print(f"Combined to {len(ts_combined.columns)} raw dynamic features")

        print(f"Filtering by missingness (threshold: {missingness_threshold})...")
        ts_train = ts_combined[ts_combined.index.get_level_values('admission_id').isin(train_ids)]
        missingness = ts_train.notna().mean()
        cols_to_keep = missingness[missingness >= missingness_threshold].index
        ts_combined = ts_combined[cols_to_keep]
        ts_combined = ts_combined.sort_index()

        print(f"Kept {len(cols_to_keep)} dynamic features (>={missingness_threshold * 100}% present in training)")
        return ts_combined

    def load_icd_codes(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Load ICD diagnosis codes."""
        if not self.load_icd:
            print("ICD codes disabled")
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        diagnoses_path = self.base_dir / self.files.get('diagnoses_icd')
        if not diagnoses_path or not diagnoses_path.exists():
            print("Warning: ICD codes enabled but file not found")
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        print(f"\nLoading ICD codes from: {diagnoses_path}")

        diagnoses = pd.read_csv(diagnoses_path)
        diagnoses = diagnoses.rename(columns={'hadm_id': 'admission_id'})

        cohort_ids = self.cohort_df.index.unique()
        diagnoses = diagnoses[diagnoses['admission_id'].isin(cohort_ids)]

        icd_config = self.config.get('icd_processing', {})
        top_n = icd_config.get('top_n_codes', 500)

        code_counts = diagnoses['icd_code'].value_counts()
        top_codes = code_counts.head(top_n).index

        diagnoses = diagnoses[diagnoses['icd_code'].isin(top_codes)]

        icd_df = pd.crosstab(diagnoses['admission_id'], diagnoses['icd_code'])
        icd_df = (icd_df > 0).astype(int)

        print(f"Loaded ICD codes for {len(icd_df)} admissions")
        print(f"Top {top_n} codes encoded")

        return icd_df, None

    def load_radiology_reports(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Load radiology report embeddings or text."""
        if not self.load_radiology:
            print("Radiology reports disabled")
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        radiology_path = self.base_dir / self.files.get('radiology_embeddings')
        if radiology_path and radiology_path.exists():
            print(f"\nLoading radiology embeddings from: {radiology_path}")

            embeddings_df = pd.read_csv(radiology_path)
            embeddings_df = embeddings_df.rename(columns={'hadm_id': 'admission_id'})
            embeddings_df = embeddings_df.set_index('admission_id')

            cohort_ids = self.cohort_df.index.unique()
            embeddings_df = embeddings_df[embeddings_df.index.isin(cohort_ids)]

            print(f"Loaded radiology embeddings for {len(embeddings_df)} admissions")
            print(f"Embedding dimension: {len(embeddings_df.columns)}")

            return embeddings_df, embeddings_df.values

        print("Warning: Radiology enabled but no embeddings file found")
        return pd.DataFrame(index=pd.Index([], name='admission_id')), None

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MIMIC-IV specific cohort criteria."""
        initial_n = len(df)

        max_age = self.config['dataset']['cohort'].get('max_age')
        if max_age and 'age' in df.columns:
            df = df[df['age'] <= max_age]
            print(f"  Age filter (<={max_age}): {len(df)} remains")

        if 'age' in df.columns:
            df = df[df['age'] >= 18]
            print(f"  Adult filter (>=18): {len(df)} remains")

        print(f"Cohort criteria applied: {initial_n} -> {len(df)} admissions")
        return df

    def get_modality_info(self) -> Dict:
        """Return information about loaded modalities."""
        return {
            'timeseries': self.load_timeseries,
            'static': self.load_static,
            'icd': self.load_icd,
            'radiology': self.load_radiology,
        }