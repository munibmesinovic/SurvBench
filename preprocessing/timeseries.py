import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm


class TimeSeriesAggregator:

    def __init__(self, config: Dict):
        self.config = config
        self.num_windows = config['preprocessing']['num_windows']
        self.window_size = config['preprocessing']['window_size_hours']
        self.max_hours = config['preprocessing']['max_hours']

        assert self.num_windows * self.window_size == self.max_hours

    def aggregate_to_windows(
            self,
            ts_df: pd.DataFrame,
            cohort_dfs: Dict[str, pd.DataFrame],
            static_df: pd.DataFrame
    ) -> Dict[str, np.ndarray]:

        dynamic_cols = list(ts_df.columns)
        static_cols = list(static_df.columns)

        print(f"\\n--- Time-Series Aggregation ---")
        print(f"Windows: {self.num_windows} × {self.window_size}h = {self.max_hours}h")
        print(f"Dynamic features: {len(dynamic_cols)}")
        print(f"Static features: {len(static_cols)}")

        all_feature_names = dynamic_cols + static_cols
        num_features = len(all_feature_names)

        split_data = {}

        for split in ['train', 'val', 'test']:
            print(f"\\nProcessing {split} split...")
            cohort = cohort_dfs[split]

            N = len(cohort)
            T = self.num_windows
            F = num_features

            data_array = np.full((N, T, F), np.nan, dtype=np.float32)
            static_data_split = static_df.loc[cohort.index]

            for idx, patient_id in enumerate(tqdm(cohort.index, desc=f"{split}")):

                if patient_id in ts_df.index:
                    patient_ts = ts_df.loc[patient_id]

                    if isinstance(patient_ts, pd.Series):
                        patient_ts = patient_ts.to_frame().T
                        patient_ts.index.name = 'time_hours'

                    patient_ts = patient_ts.reset_index()
                    patient_ts['window'] = (patient_ts['time_hours'] // self.window_size).astype(int)
                    patient_ts = patient_ts[(patient_ts['window'] >= 0) & (patient_ts['window'] < self.num_windows)]
                    ts_grouped = patient_ts.groupby('window')[dynamic_cols].mean()

                    for window_idx in ts_grouped.index:
                        data_array[idx, window_idx, :len(dynamic_cols)] = ts_grouped.loc[window_idx].values

                static_vector = static_data_split.loc[patient_id].values
                data_array[idx, :, len(dynamic_cols):] = static_vector

            split_data[split] = data_array
            print(f"{split} shape: {data_array.shape}")

        return split_data

    def get_feature_structure(
            self,
            dynamic_cols: List[str],
            static_cols: List[str]
    ) -> Dict:
        return {
            'num_dynamic': len(dynamic_cols),
            'num_static': len(static_cols),
            'num_total': len(dynamic_cols) + len(static_cols),
            'dynamic_names': dynamic_cols,
            'static_names': static_cols,
            'all_names': dynamic_cols + static_cols,
            'dynamic_indices': list(range(len(dynamic_cols))),
            'static_indices': list(range(len(dynamic_cols), len(dynamic_cols) + len(static_cols)))
        }