import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
from .base_loader import BaseDataLoader


class eICUDataLoader(BaseDataLoader):

    def __init__(self, config: Dict):
        super().__init__(config)
        self.files = config['dataset']['files']

    def load_labels(self) -> pd.DataFrame:
        label_path = self.base_dir / self.files['labels']
        print(f"Loading labels from: {label_path}")

        labels = pd.read_csv(label_path, index_col='patient')
        labels = labels[['unitdischargeoffset', 'actualhospitalmortality']]
        labels['unitdischargeoffset_hours'] = labels['unitdischargeoffset'] / 60.0

        min_hours = self.config['dataset']['cohort']['min_stay_hours']
        labels = labels[labels['unitdischargeoffset_hours'] >= min_hours]

        print(f"Loaded {len(labels)} labels (≥{min_hours}h stay)")
        return labels

    def load_static_features(self) -> pd.DataFrame:
        flat_path = self.base_dir / self.files['flat']
        print(f"Loading static features from: {flat_path}")

        flats = pd.read_csv(flat_path, index_col='patient')

        exclude_cols = self.config['dataset']['cohort']['exclude_columns']

        if '> 89' in flats.columns:
            flats = flats[flats['> 89'] != 1]

        flats = flats.drop(columns=exclude_cols, errors='ignore')

        print(f"Loaded {len(flats)} patients with {len(flats.columns)} static features")
        return flats

    def load_timeseries(self) -> pd.DataFrame:
        cohort_ids = self.cohort_df.index.unique()
        train_ids = self.create_splits()['train']
        max_hours = self.config['preprocessing']['max_hours']
        missingness_threshold = self.config['preprocessing']['missingness_threshold']

        # Load lab data
        lab_path = self.base_dir / self.files['timeseries_lab']
        print(f"Loading lab time-series from: {lab_path}")

        ts_lab = pd.read_csv(lab_path, index_col=['patient', 'time'])
        ts_lab = ts_lab[ts_lab.index.get_level_values('patient').isin(cohort_ids)]
        ts_lab = ts_lab.reset_index(level='time')
        ts_lab['time'] = pd.to_timedelta(ts_lab['time'], errors='coerce')
        ts_lab['time_hours'] = ts_lab['time'].dt.total_seconds() / 3600.0
        ts_lab = ts_lab.drop(columns=['time'])
        ts_lab = ts_lab[(ts_lab['time_hours'] >= 0) & (ts_lab['time_hours'] < max_hours)]
        ts_lab = ts_lab.set_index('time_hours', append=True)

        print(f"Loaded {len(ts_lab)} lab measurements")

        # Load aperiodic vitals
        aperiodic_path = self.base_dir / self.files['timeseries_aperiodic']
        print(f"Loading aperiodic vitals from: {aperiodic_path}")

        ts_aperiodic = pd.read_csv(aperiodic_path, index_col=['patient', 'time'])
        ts_aperiodic = ts_aperiodic[ts_aperiodic.index.get_level_values('patient').isin(cohort_ids)]
        ts_aperiodic = ts_aperiodic.reset_index(level='time')
        ts_aperiodic['time'] = pd.to_timedelta(ts_aperiodic['time'], errors='coerce')
        ts_aperiodic['time_hours'] = ts_aperiodic['time'].dt.total_seconds() / 3600.0
        ts_aperiodic = ts_aperiodic.drop(columns=['time'])
        ts_aperiodic = ts_aperiodic[(ts_aperiodic['time_hours'] >= 0) & (ts_aperiodic['time_hours'] < max_hours)]
        ts_aperiodic = ts_aperiodic.set_index('time_hours', append=True)

        print(f"Loaded {len(ts_aperiodic)} aperiodic vital measurements")

        # Combine
        ts_combined = ts_lab.join(ts_aperiodic, how='outer')
        print(f"Combined to {len(ts_combined.columns)} raw dynamic features")

        # Missingness filter
        print(f"Filtering features by missingness (threshold: {missingness_threshold})...")
        ts_train = ts_combined[ts_combined.index.get_level_values('patient').isin(train_ids)]
        missingness = ts_train.notna().mean()
        cols_to_keep = missingness[missingness >= missingness_threshold].index
        ts_combined = ts_combined[cols_to_keep]
        ts_combined = ts_combined.sort_index()

        print(f"Kept {len(cols_to_keep)} dynamic features (≥{missingness_threshold * 100}% present in training)")
        return ts_combined

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        return df