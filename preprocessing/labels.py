import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


class SurvivalLabelsProcessor:

    def __init__(self, config: Dict):
        self.config = config
        self.max_horizon = config['preprocessing']['max_horizon_hours']
        self.n_bins = config['preprocessing']['n_time_bins']
        self.discretisation = config['preprocessing']['discretisation_method']
        self.event_col = config['preprocessing']['event_col']
        self.duration_col = config['preprocessing']['duration_col']

        self.cuts = None
        self.labtrans = None

    def process_labels(
            self,
            cohort_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

        y_data = {}
        all_durations_train = None
        all_events_train = None

        for split in ['train', 'val', 'test']:
            df = cohort_dfs[split]

            durations = df[self.duration_col].values
            events = df[self.event_col].values.astype(int)

            events[durations > self.max_horizon] = 0
            durations = np.clip(durations, a_min=None, a_max=self.max_horizon)

            y_data[split] = (durations, events)

            if split == 'train':
                all_durations_train = durations
                all_events_train = events

        self._create_time_bins(all_durations_train, all_events_train)
        self._log_statistics(y_data)

        return y_data

    def _create_time_bins(self, durations: np.ndarray, events: np.ndarray):
        print(f"Creating {self.n_bins} time bins using {self.discretisation} method...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.labtrans = LabTransDiscreteTime(self.n_bins, self.discretisation)
            self.cuts = self.labtrans.fit(durations, events).cuts

        print(f"Time bins created: Min={self.cuts.min():.1f}h, Max={self.cuts.max():.1f}h")

    def _log_statistics(self, y_data: Dict):
        print("\\n--- Label Statistics ---")
        for split, (durations, events) in y_data.items():
            event_rate = events.mean() * 100
            censoring_rate = (1 - events).mean() * 100
            mean_duration = durations.mean()
            median_duration = np.median(durations)

            print(f"\\n{split.upper()}:")
            print(f"  N: {len(durations)}")
            print(f"  Event rate: {event_rate:.2f}%")
            print(f"  Censoring rate: {censoring_rate:.2f}%")
            print(f"  Mean duration: {mean_duration:.1f}h")
            print(f"  Median duration: {median_duration:.1f}h")

    def get_cuts(self) -> np.ndarray:
        if self.cuts is None:
            raise ValueError("Must process labels before accessing cuts")
        return self.cuts

    def get_n_bins(self) -> int:
        return self.n_bins