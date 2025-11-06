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

        split_config = self.config['splits']
        train_size = split_config['train']
        val_size = split_config['val']
        test_size = split_config['test']

        unique_patients = self.cohort_df.index.unique()

        train_ids, temp_ids = train_test_split(
            unique_patients,
            test_size=(val_size + test_size),
            random_state=self.seed
        )

        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(test_size / (val_size + test_size)),
            random_state=self.seed
        )

        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }

        print(f"\\n--- Data Splits ---")
        print(f"Train: {len(train_ids)} ({train_size * 100:.0f}%)")
        print(f"Val:   {len(val_ids)} ({val_size * 100:.0f}%)")
        print(f"Test:  {len(test_ids)} ({test_size * 100:.0f}%)")

        return splits