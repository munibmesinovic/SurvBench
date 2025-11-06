from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class BaseDataLoader(ABC):

    def __init__(self, config: Dict):
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.base_dir = Path(config['dataset']['base_dir'])
        self.seed = config['splits']['seed']
        np.random.seed(self.seed)

        self.labels_df = None
        self.static_df = None
        self.timeseries_df = None
        self.cohort_df = None

    @abstractmethod
    def load_labels(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_static_features(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def load_timeseries(self) -> pd.DataFrame:
        pass

    def build_cohort(self) -> pd.DataFrame:
        self.labels_df = self.load_labels()
        self.static_df = self.load_static_features()

        cohort_df = self.static_df.merge(
            self.labels_df,
            left_index=True,
            right_index=True
        )

        cohort_df = self.apply_cohort_criteria(cohort_df)
        self.cohort_df = cohort_df
        return cohort_df

    @abstractmethod
    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def create_splits(self) -> Dict[str, pd.Index]:
        from sklearn.model_selection import train_test_split

        if self.cohort_df is None:
            raise ValueError("Must build cohort before creating splits")

        # Check for the patient identifier column
        if 'subject_id' not in self.cohort_df.columns:
            raise ValueError(
                "Data splitting requires a 'subject_id' column in the cohort_df. "
                "Please ensure your data loader (e.g., MIMICIVDataLoader) "
                "and preparation scripts (e.g., prepare_mcmed_data.py) "
                "load and retain 'subject_id'."
            )

        split_config = self.config['splits']
        train_size = split_config['train']
        val_size = split_config['val']
        test_size = split_config['test']

        # Split on the unique patient (subject_id)
        unique_patient_ids = self.cohort_df['subject_id'].unique()

        train_patient_ids, temp_patient_ids = train_test_split(
            unique_patient_ids,
            test_size=(val_size + test_size),
            random_state=self.seed
        )

        val_patient_ids, test_patient_ids = train_test_split(
            temp_patient_ids,
            test_size=(test_size / (val_size + test_size)),
            random_state=self.seed
        )

        # Get the dataframe indices (stay_id/admission_id) for each split
        train_ids = self.cohort_df[
            self.cohort_df['subject_id'].isin(train_patient_ids)
        ].index
        val_ids = self.cohort_df[
            self.cohort_df['subject_id'].isin(val_patient_ids)
        ].index
        test_ids = self.cohort_df[
            self.cohort_df['subject_id'].isin(test_patient_ids)
        ].index

        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }

        print(f"\n--- Data Splits (by 'subject_id') ---")
        print(f"Total unique patients: {len(unique_patient_ids)}")
        print(f"Train patients: {len(train_patient_ids)} ({len(train_ids)} stays/admissions)")
        print(f"Val patients:   {len(val_patient_ids)} ({len(val_ids)} stays/admissions)")
        print(f"Test patients:  {len(test_patient_ids)} ({len(test_ids)} stays/admissions)")

        return splits