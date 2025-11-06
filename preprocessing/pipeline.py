import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict
from sklearn.preprocessing import StandardScaler

from ..data.eicu_loader import eICUDataLoader
from .labels import SurvivalLabelsProcessor
from .timeseries import TimeSeriesAggregator


class PreprocessingPipeline:

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output']['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = eICUDataLoader(config)
        self.label_processor = SurvivalLabelsProcessor(config)
        self.ts_aggregator = TimeSeriesAggregator(config)

        self.cohort_dfs = None
        self.split_data = None
        self.split_masks = None
        self.y_data = None
        self.scaler = None
        self.feature_structure = None

    def run(self):
        print("=" * 80)
        print(f"PREPROCESSING PIPELINE: {self.config['dataset']['name'].upper()}")
        print("=" * 80)

        print("\\n[1/7] Building cohort...")
        cohort_df = self.loader.build_cohort()

        print("\\n[2/7] Creating data splits...")
        splits = self.loader.create_splits()
        self.cohort_dfs = {
            split: cohort_df.loc[patient_ids]
            for split, patient_ids in splits.items()
        }

        print("\\n[3/7] Processing survival labels...")
        self.y_data = self.label_processor.process_labels(self.cohort_dfs)

        print("\\n[4/7] Loading and aggregating time-series...")
        self.loader.timeseries_df = self.loader.load_timeseries()

        self.split_data = self.ts_aggregator.aggregate_to_windows(
            self.loader.timeseries_df,
            self.cohort_dfs,
            self.loader.static_df
        )

        dynamic_cols = list(self.loader.timeseries_df.columns)
        static_cols = list(self.loader.static_df.columns)
        self.feature_structure = self.ts_aggregator.get_feature_structure(
            dynamic_cols, static_cols
        )

        print("\\n[5/7] Creating missingness masks...")
        self.split_masks = self._create_masks()

        print("\\n[6/7] Applying imputation...")
        self._apply_imputation()

        print("\\n[7/7] Applying feature scaling...")
        self._apply_scaling()

        self._save_outputs()

        print("\\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)

    def _create_masks(self) -> Dict[str, np.ndarray]:
        masks = {}
        for split in ['train', 'val', 'test']:
            data = self.split_data[split]
            masks[split] = (~np.isnan(data)).astype(np.float32)
        print("Binary missingness masks created")
        return masks

    def _apply_imputation(self):
        imputation_cfg = self.config['preprocessing']['imputation']
        static_method = imputation_cfg['static']
        dynamic_method = imputation_cfg['dynamic']

        print(f"Imputation strategy: Static={static_method}, Dynamic={dynamic_method}")

        dynamic_indices = self.feature_structure['dynamic_indices']
        static_indices = self.feature_structure['static_indices']

        train_static_data = self.split_data['train'][:, :, static_indices]
        train_static_means = np.nanmean(train_static_data, axis=(0, 1))
        train_static_means = np.nan_to_num(train_static_means, nan=0.0)

        for split in ['train', 'val', 'test']:
            data = self.split_data[split].copy()

            if static_method == 'mean':
                for i, col_idx in enumerate(static_indices):
                    col_data = data[:, :, col_idx]
                    col_data[np.isnan(col_data)] = train_static_means[i]
                    data[:, :, col_idx] = col_data

            if dynamic_method == 'zero':
                data = np.nan_to_num(data, nan=0.0)

            self.split_data[split] = data

        print("Imputation complete")

    def _apply_scaling(self):
        scaling_cfg = self.config['preprocessing']['scaling']

        if scaling_cfg['method'] != 'standard':
            raise NotImplementedError(f"Scaling method {scaling_cfg['method']} not implemented")

        num_dynamic = self.feature_structure['num_dynamic']

        train_data = self.split_data['train']
        x_train_dynamic_flat = train_data[:, :, :num_dynamic].reshape(-1, num_dynamic)

        self.scaler = StandardScaler()
        self.scaler.fit(x_train_dynamic_flat)

        print(f"Fitted StandardScaler on {num_dynamic} dynamic features")

        for split in ['train', 'val', 'test']:
            data = self.split_data[split]
            N, T, F = data.shape

            dynamic_part = data[:, :, :num_dynamic].reshape(-1, num_dynamic)
            static_part = data[:, :, num_dynamic:]

            dynamic_scaled = self.scaler.transform(dynamic_part).reshape(N, T, num_dynamic)

            self.split_data[split] = np.concatenate(
                [dynamic_scaled, static_part],
                axis=2
            ).astype(np.float32)

        print("Scaling complete")

    def _save_outputs(self):
        print(f"\\nSaving outputs to: {self.output_dir}")

        output_files = self.config['output']['files']

        for split in ['train', 'val', 'test']:
            x_path = self.output_dir / output_files[f'x_{split}']
            np.save(x_path, self.split_data[split])
            print(f"  Saved {x_path.name}")

            mask_path = str(x_path).replace('.npy', '_mask.npy')
            np.save(mask_path, self.split_masks[split])
            print(f"  Saved {Path(mask_path).name}")

        for split in ['train', 'val']:
            y_path = self.output_dir / output_files[f'y_{split}']
            with open(y_path, 'wb') as f:
                pickle.dump(self.y_data[split], f)
            print(f"  Saved {y_path.name}")

        durations_path = self.output_dir / output_files['durations_test']
        events_path = self.output_dir / output_files['events_test']
        np.save(durations_path, self.y_data['test'][0])
        np.save(events_path, self.y_data['test'][1])
        print(f"  Saved {durations_path.name}")
        print(f"  Saved {events_path.name}")

        cuts_path = self.output_dir / output_files['cuts']
        out_features_path = self.output_dir / output_files['out_features']
        np.save(cuts_path, self.label_processor.get_cuts())
        np.save(out_features_path, np.int64(self.label_processor.get_n_bins()))
        print(f"  Saved {cuts_path.name}")
        print(f"  Saved {out_features_path.name}")

        feature_names_path = self.output_dir / output_files['feature_names']
        with open(feature_names_path, 'wb') as f:
            pickle.dump(self.feature_structure, f)
        print(f"  Saved {feature_names_path.name}")

        scaler_path = self.output_dir / output_files['scaler']
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Saved {scaler_path.name}")

        self._print_summary()

    def _print_summary(self):
        print("\\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)

        for split in ['train', 'val', 'test']:
            data = self.split_data[split]
            mask = self.split_masks[split]
            durations, events = self.y_data[split]

            print(f"\\n{split.upper()}:")
            print(f"  Shape: {data.shape} [N, T, F]")
            print(f"  N (patients): {data.shape[0]}")
            print(f"  T (time windows): {data.shape[1]}")
            print(f"  F (features): {data.shape[2]}")
            print(f"    - Dynamic: {self.feature_structure['num_dynamic']}")
            print(f"    - Static: {self.feature_structure['num_static']}")
            print(f"  Missingness: {(mask == 0).mean() * 100:.2f}%")
            print(f"  Event rate: {events.mean() * 100:.2f}%")