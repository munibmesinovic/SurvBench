import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Optional
from pycox.preprocessing.label_transforms import LabTransDiscreteTime


class SurvivalLabelsProcessor:
    """
    Process survival labels for both single-risk and competing risks scenarios.
    Handles time discretisation and label transformation for neural network training.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.max_horizon = config['preprocessing']['max_horizon_hours']
        self.n_bins = config['preprocessing']['n_time_bins']
        self.discretisation = config['preprocessing']['discretisation_method']
        self.event_col = config['preprocessing'].get('event_col', 'event')
        self.duration_col = config['preprocessing'].get('duration_col', 'duration')

        # Competing risks configuration
        self.competing_risks_enabled = config['dataset'].get('competing_risks', {}).get('enabled', False)
        self.outcome_mapping = config['dataset'].get('competing_risks', {}).get('outcome_mapping', {})
        self.n_events = len([k for k in self.outcome_mapping.keys() if k != 0]) if self.competing_risks_enabled else 1

        self.cuts = None
        self.labtrans = None

        if self.competing_risks_enabled:
            print(f"\nCompeting Risks Mode: {self.n_events} competing outcomes")
            for code, name in self.outcome_mapping.items():
                print(f"  {code}: {name}")
        else:
            print(f"\nSingle Risk Mode")

    def process_labels(
            self,
            cohort_dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Process labels for all data splits.

        Args:
            cohort_dfs: Dictionary with 'train', 'val', 'test' DataFrames

        Returns:
            Dictionary with (durations, events) tuples for each split
        """

        y_data = {}
        all_durations_train = None
        all_events_train = None

        for split in ['train', 'val', 'test']:
            df = cohort_dfs[split]

            durations = df[self.duration_col].values
            events = df[self.event_col].values.astype(int)

            # Censor events beyond max horizon
            events[durations > self.max_horizon] = 0
            durations = np.clip(durations, a_min=None, a_max=self.max_horizon)

            y_data[split] = (durations, events)

            if split == 'train':
                all_durations_train = durations
                all_events_train = events

        # Create time bins based on training data
        self._create_time_bins(all_durations_train, all_events_train)

        # Log statistics
        self._log_statistics(y_data)

        return y_data

    def _create_time_bins(self, durations: np.ndarray, events: np.ndarray):
        """Create discretised time bins for neural network training."""
        print(f"\nCreating {self.n_bins} time bins using {self.discretisation} method...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)

            if self.competing_risks_enabled:
                # For competing risks, create bins based on any event (not just event=1)
                any_event = (events > 0).astype(int)
                self.labtrans = LabTransDiscreteTime(self.n_bins, self.discretisation)
                self.cuts = self.labtrans.fit(durations, any_event).cuts
            else:
                # Standard single-risk
                self.labtrans = LabTransDiscreteTime(self.n_bins, self.discretisation)
                self.cuts = self.labtrans.fit(durations, events).cuts

        print(f"Time bins: Min={self.cuts.min():.1f}h, Max={self.cuts.max():.1f}h")
        print(f"Bin edges: {self.cuts}")

    def _log_statistics(self, y_data: Dict):
        """Log label statistics for each split."""
        print("\n" + "=" * 80)
        print("LABEL STATISTICS")
        print("=" * 80)

        for split, (durations, events) in y_data.items():
            print(f"\n{split.upper()}:")
            print(f"  Total N: {len(durations)}")

            if self.competing_risks_enabled:
                # Statistics per competing risk
                censored_count = (events == 0).sum()
                censored_pct = censored_count / len(events) * 100
                print(f"  Censored (0): {censored_count} ({censored_pct:.1f}%)")

                for event_code, event_name in self.outcome_mapping.items():
                    if event_code == 0:  # Skip censoring
                        continue
                    count = (events == event_code).sum()
                    pct = count / len(events) * 100
                    mean_time = durations[events == event_code].mean() if count > 0 else 0
                    print(f"  {event_name} ({event_code}): {count} ({pct:.1f}%), mean time: {mean_time:.1f}h")
            else:
                # Single risk statistics
                event_rate = events.mean() * 100
                censoring_rate = (1 - events).mean() * 100
                mean_duration_event = durations[events == 1].mean() if events.sum() > 0 else 0
                mean_duration_censored = durations[events == 0].mean() if (events == 0).sum() > 0 else 0

                print(f"  Event rate: {event_rate:.2f}%")
                print(f"  Censoring rate: {censoring_rate:.2f}%")
                print(f"  Mean duration (events): {mean_duration_event:.1f}h")
                print(f"  Mean duration (censored): {mean_duration_censored:.1f}h")

            print(f"  Overall mean duration: {durations.mean():.1f}h")
            print(f"  Overall median duration: {np.median(durations):.1f}h")

    def transform_labels(self, durations: np.ndarray, events: np.ndarray) -> Tuple:
        """
        Transform continuous time labels to discrete time format for neural networks.

        Args:
            durations: Time to event or censoring
            events: Event indicators

        Returns:
            Transformed labels suitable for discrete-time models
        """
        if self.labtrans is None:
            raise ValueError("Must fit labtrans before transforming labels")

        return self.labtrans.transform(durations, events)

    def get_cuts(self) -> np.ndarray:
        """Get time bin boundaries."""
        if self.cuts is None:
            raise ValueError("Must process labels before accessing cuts")
        return self.cuts

    def get_n_bins(self) -> int:
        """Get number of time bins."""
        return self.n_bins

    def get_n_events(self) -> int:
        """Get number of competing events (1 for single risk)."""
        return self.n_events

    def is_competing_risks(self) -> bool:
        """Check if competing risks mode is enabled."""
        return self.competing_risks_enabled