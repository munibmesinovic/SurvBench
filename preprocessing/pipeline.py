import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.preprocessing import StandardScaler

from ..data.eicu_loader import eICUDataLoader
from ..data.mcmed_loader import MCMEDDataLoader
from .labels import SurvivalLabelsProcessor
from .timeseries import TimeSeriesAggregator


class PreprocessingPipeline:
    """
    Unified preprocessing pipeline for multi-modal survival analysis.
    Supports: eICU, MC-MED with time-series, static, ICD, and radiology modalities.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config['output']['dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize appropriate data loader
        dataset_name = config['dataset']['name'].lower()
        if dataset_name == 'eicu':
            self.loader = eICUDataLoader(config)
        elif dataset_name == 'mcmed':
            self.loader = MCMEDDataLoader(config)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.label_processor = SurvivalLabelsProcessor(config)
        self.ts_aggregator = TimeSeriesAggregator(config)

        self.cohort_dfs = None
        self.split_data = {}  # Will hold data for each modality
        self.split_masks = {}  # Missingness masks
        self.y_data = None
        self.scaler = None
        self.feature_structure = None

        # Modality configuration
        self.modalities = config.get('modalities', {})
        self.active_modalities = [k for k, v in self.modalities.items() if v]

        print(f"\n{'=' * 80}")
        print(f"MULTI-MODAL PREPROCESSING PIPELINE")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"Active modalities: {', '.join(self.active_modalities)}")
        print(f"{'=' * 80}")

    def run(self):
        """Execute the full preprocessing pipeline."""

        print("\n[1/9] Building cohort...")
        cohort_df = self.loader.build_cohort()

        print("\n[2/9] Creating data splits...")
        splits = self.loader.create_splits()
        self.cohort_dfs = {
            split: cohort_df.loc[patient_ids]
            for split, patient_ids in splits.items()
        }

        print("\n[3/9] Processing survival labels...")
        self.y_data = self.label_processor.process_labels(self.cohort_dfs)

        print("\n[4/9] Processing modalities...")
        self._process_all_modalities()

        print("\n[5/9] Creating missingness masks...")
        self._create_all_masks()

        print("\n[6/9] Applying imputation...")
        self._apply_imputation()

        print("\n[7/9] Applying feature scaling...")
        self._apply_scaling()

        print("\n[8/9] Combining modalities...")
        self._combine_modalities()

        print("\n[9/9] Saving outputs...")
        self._save_outputs()

        print(f"\n{'=' * 80}")
        print("PREPROCESSING COMPLETE!")
        print(f"{'=' * 80}")

    def _process_all_modalities(self):
        """Process each active modality."""

        # 1. Time-series
        if self.modalities.get('timeseries', False):
            print("\n  → Processing time-series modality...")
            self.loader.timeseries_df = self.loader.load_timeseries()

            if hasattr(self.loader, 'static_df') and self.loader.static_df is not None:
                # Include static features in time-series aggregation
                self.split_data['timeseries'] = self.ts_aggregator.aggregate_to_windows(
                    self.loader.timeseries_df,
                    self.cohort_dfs,
                    self.loader.static_df if self.modalities.get('static', False) else None
                )
            else:
                self.split_data['timeseries'] = self.ts_aggregator.aggregate_to_windows(
                    self.loader.timeseries_df,
                    self.cohort_dfs,
                    None
                )

            dynamic_cols = list(self.loader.timeseries_df.columns) if len(self.loader.timeseries_df) > 0 else []
            static_cols = list(self.loader.static_df.columns) if self.modalities.get('static', False) and hasattr(
                self.loader, 'static_df') else []

            self.feature_structure = self.ts_aggregator.get_feature_structure(
                dynamic_cols, static_cols
            )

            print(f"    Time-series shape: {self.split_data['timeseries']['train'].shape}")

        # 2. Static (if separate from time-series)
        elif self.modalities.get('static', False):
            print("\n  → Processing static-only modality...")
            # Create static-only arrays (no temporal dimension needed here as handled by aggregator)
            pass

        # 3. ICD codes
        if self.modalities.get('icd', False):
            print("\n  → Processing ICD codes modality...")
            icd_df, icd_embeddings = self.loader.load_icd_codes()

            if icd_embeddings is not None:
                # Use pre-computed embeddings
                self.split_data['icd'] = self._create_icd_splits(icd_df, icd_embeddings)
            elif len(icd_df) > 0:
                # Use one-hot or count encoding
                self.split_data['icd'] = self._create_icd_splits(icd_df, None)
            else:
                print("    Warning: No ICD data available")

        # 4. Radiology
        if self.modalities.get('radiology', False):
            print("\n  → Processing radiology modality...")
            reports_df, rad_embeddings = self.loader.load_radiology_reports()

            if rad_embeddings is not None or len(reports_df) > 0:
                self.split_data['radiology'] = self._create_radiology_splits(reports_df, rad_embeddings)
            else:
                print("    Warning: No radiology data available")

    def _create_icd_splits(self, icd_df: pd.DataFrame, embeddings: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
        """Create train/val/test splits for ICD codes."""
        icd_splits = {}

        for split in ['train', 'val', 'test']:
            cohort = self.cohort_dfs[split]
            split_ids = cohort.index

            # Get ICD data for this split
            split_icd = icd_df.loc[icd_df.index.isin(split_ids)]
            split_icd = split_icd.reindex(split_ids, fill_value=0)

            icd_splits[split] = split_icd.values.astype(np.float32)
            print(f"    {split} ICD shape: {icd_splits[split].shape}")

        return icd_splits

    def _create_radiology_splits(self, reports_df: pd.DataFrame, embeddings: Optional[np.ndarray]) -> Dict[
        str, np.ndarray]:
        """Create train/val/test splits for radiology embeddings."""
        rad_splits = {}

        for split in ['train', 'val', 'test']:
            cohort = self.cohort_dfs[split]
            split_ids = cohort.index

            # Get radiology data for this split
            split_rad = reports_df.loc[reports_df.index.isin(split_ids)]
            split_rad = split_rad.reindex(split_ids, fill_value=0)

            rad_splits[split] = split_rad.values.astype(np.float32)
            print(f"    {split} radiology shape: {rad_splits[split].shape}")

        return rad_splits

    def _create_all_masks(self):
        """Create missingness masks for all modalities."""
        for modality, data_dict in self.split_data.items():
            self.split_masks[modality] = {}
            for split in ['train', 'val', 'test']:
                if split in data_dict:
                    data = data_dict[split]
                    self.split_masks[modality][split] = (~np.isnan(data)).astype(np.float32)

        print("Binary missingness masks created for all modalities")

    def _apply_imputation(self):
        """Apply imputation strategies to all modalities."""
        imputation_cfg = self.config['preprocessing']['imputation']
        static_method = imputation_cfg.get('static', 'mean')
        dynamic_method = imputation_cfg.get('dynamic', 'zero')

        print(f"Imputation strategy: Static={static_method}, Dynamic={dynamic_method}")

        # Impute time-series modality
        if 'timeseries' in self.split_data and self.feature_structure:
            dynamic_indices = self.feature_structure['dynamic_indices']
            static_indices = self.feature_structure['static_indices']

            # Compute training statistics for static features
            train_static_data = self.split_data['timeseries']['train'][:, :, static_indices]
            train_static_means = np.nanmean(train_static_data, axis=(0, 1))
            train_static_means = np.nan_to_num(train_static_means, nan=0.0)

            for split in ['train', 'val', 'test']:
                data = self.split_data['timeseries'][split].copy()

                # Impute static features
                if static_method == 'mean':
                    for i, col_idx in enumerate(static_indices):
                        col_data = data[:, :, col_idx]
                        col_data[np.isnan(col_data)] = train_static_means[i]
                        data[:, :, col_idx] = col_data

                # Impute dynamic features
                if dynamic_method == 'zero':
                    data = np.nan_to_num(data, nan=0.0)

                self.split_data['timeseries'][split] = data

        # Impute other modalities (typically already complete or use zero-filling)
        for modality in ['icd', 'radiology']:
            if modality in self.split_data:
                for split in ['train', 'val', 'test']:
                    self.split_data[modality][split] = np.nan_to_num(
                        self.split_data[modality][split], nan=0.0
                    )

        print("Imputation complete for all modalities")

    def _apply_scaling(self):
        """Apply feature scaling to dynamic modalities."""
        scaling_cfg = self.config['preprocessing']['scaling']
        method = scaling_cfg.get('method', 'standard')

        if method == 'none':
            print("Scaling disabled")
            return

        # Scale time-series modality
        if 'timeseries' in self.split_data and self.feature_structure:
            num_dynamic = self.feature_structure['num_dynamic']

            if num_dynamic > 0:
                train_data = self.split_data['timeseries']['train']
                x_train_dynamic_flat = train_data[:, :, :num_dynamic].reshape(-1, num_dynamic)

                self.scaler = StandardScaler()
                self.scaler.fit(x_train_dynamic_flat)

                print(f"Fitted StandardScaler on {num_dynamic} dynamic features")

                for split in ['train', 'val', 'test']:
                    data = self.split_data['timeseries'][split]
                    N, T, F = data.shape

                    dynamic_part = data[:, :, :num_dynamic].reshape(-1, num_dynamic)
                    static_part = data[:, :, num_dynamic:]

                    dynamic_scaled = self.scaler.transform(dynamic_part).reshape(N, T, num_dynamic)

                    self.split_data['timeseries'][split] = np.concatenate(
                        [dynamic_scaled, static_part],
                        axis=2
                    ).astype(np.float32)

        print("Scaling complete")

    def _combine_modalities(self):
        """Combine all modalities into final data arrays."""
        # If only time-series (most common case), rename to 'combined'
        if len(self.active_modalities) == 1 and 'timeseries' in self.active_modalities:
            self.split_data['combined'] = self.split_data['timeseries']
            return

        # Otherwise, we keep them separate
        # The model will need to handle multiple input modalities
        print("Multiple modalities detected - keeping separate")
        # In a full implementation, you might concatenate or use separate encoders

    def _save_outputs(self):
        """Save all preprocessed data and metadata."""
        print(f"\nSaving outputs to: {self.output_dir}")

        output_files = self.config['output']['files']

        # Determine which data to save (combined or timeseries)
        data_key = 'combined' if 'combined' in self.split_data else 'timeseries'

        # Save main data arrays
        for split in ['train', 'val', 'test']:
            x_path = self.output_dir / output_files[f'x_{split}']
            np.save(x_path, self.split_data[data_key][split])
            print(f"  Saved {x_path.name}")

            # Save masks
            mask_key = 'combined' if 'combined' in self.split_masks else 'timeseries'
            mask_path = str(x_path).replace('.npy', '_mask.npy')
            np.save(mask_path, self.split_masks[mask_key][split])
            print(f"  Saved {Path(mask_path).name}")

        # Save additional modalities if present
        for modality in ['icd', 'radiology']:
            if modality in self.split_data:
                for split in ['train', 'val', 'test']:
                    mod_path = self.output_dir / f"x_{split}_{modality}.npy"
                    np.save(mod_path, self.split_data[modality][split])
                    print(f"  Saved {mod_path.name}")

        # Save labels
        for split in ['train', 'val']:
            y_path = self.output_dir / output_files[f'y_{split}']
            with open(y_path, 'wb') as f:
                pickle.dump(self.y_data[split], f)
            print(f"  Saved {y_path.name}")

        # Save test labels separately
        durations_path = self.output_dir / output_files['durations_test']
        events_path = self.output_dir / output_files['events_test']
        np.save(durations_path, self.y_data['test'][0])
        np.save(events_path, self.y_data['test'][1])
        print(f"  Saved {durations_path.name}")
        print(f"  Saved {events_path.name}")

        # Save time bins and metadata
        cuts_path = self.output_dir / output_files['cuts']
        out_features_path = self.output_dir / output_files['out_features']
        np.save(cuts_path, self.label_processor.get_cuts())
        np.save(out_features_path, np.int64(self.label_processor.get_n_bins()))
        print(f"  Saved {cuts_path.name}")
        print(f"  Saved {out_features_path.name}")

        # Save number of events (for competing risks)
        if 'n_events' in output_files:
            n_events_path = self.output_dir / output_files['n_events']
            np.save(n_events_path, np.int64(self.label_processor.get_n_events()))
            print(f"  Saved {n_events_path.name}")

        # Save feature structure
        feature_names_path = self.output_dir / output_files['feature_names']
        with open(feature_names_path, 'wb') as f:
            pickle.dump(self.feature_structure, f)
        print(f"  Saved {feature_names_path.name}")

        # Save modality info
        if 'modality_info' in output_files:
            modality_info_path = self.output_dir / output_files['modality_info']
            modality_info = {
                'active_modalities': self.active_modalities,
                'competing_risks': self.label_processor.is_competing_risks(),
                'n_events': self.label_processor.get_n_events(),
            }
            with open(modality_info_path, 'wb') as f:
                pickle.dump(modality_info, f)
            print(f"  Saved {modality_info_path.name}")

        # Save scaler
        if self.scaler is not None:
            scaler_path = self.output_dir / output_files['scaler']
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"  Saved {scaler_path.name}")

        self._print_summary()

    def _print_summary(self):
        """Print preprocessing summary."""
        print(f"\n{'-' * 80}")
        print("SUMMARY")
        print(f"{'-' * 80}")

        # Use the primary data (combined or timeseries)
        data_key = 'combined' if 'combined' in self.split_data else 'timeseries'

        for split in ['train', 'val', 'test']:
            data = self.split_data[data_key][split]
            mask_key = 'combined' if 'combined' in self.split_masks else 'timeseries'
            mask = self.split_masks[mask_key][split]
            durations, events = self.y_data[split]

            print(f"\n{split.upper()}:")
            print(f"  Shape: {data.shape} [N, T, F]")
            print(f"  N (patients): {data.shape[0]}")

            if len(data.shape) == 3:
                print(f"  T (time windows): {data.shape[1]}")
                print(f"  F (features): {data.shape[2]}")
                if self.feature_structure:
                    print(f"    - Dynamic: {self.feature_structure['num_dynamic']}")
                    print(f"    - Static: {self.feature_structure['num_static']}")

            print(f"  Missingness: {(mask == 0).mean() * 100:.2f}%")

            if self.label_processor.is_competing_risks():
                print(f"  Competing risks distribution:")
                for event_code in np.unique(events):
                    count = (events == event_code).sum()
                    pct = count / len(events) * 100
                    outcome_name = self.config['dataset']['competing_risks']['outcome_mapping'].get(int(event_code),
                                                                                                    f"Event {event_code}")
                    print(f"    {outcome_name}: {count} ({pct:.1f}%)")
            else:
                print(f"  Event rate: {events.mean() * 100:.2f}%")

        # Additional modalities summary
        for modality in ['icd', 'radiology']:
            if modality in self.split_data:
                print(f"\n{modality.upper()} modality:")
                for split in ['train', 'val', 'test']:
                    shape = self.split_data[modality][split].shape
                    print(f"  {split}: {shape}")