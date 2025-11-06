import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .base_loader import BaseDataLoader


class MCMEDDataLoader(BaseDataLoader):
    """
    Data loader for MC-MED (Emergency Department) dataset with multi-modal support.
    Supports: time-series vitals, static demographics, ICD codes, and radiology reports.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.files = config['dataset']['files']
        self.modalities_config = config.get('modalities', {})

        # Which modalities to load
        self.load_timeseries = self.modalities_config.get('timeseries', True)
        self.load_static = self.modalities_config.get('static', True)
        self.load_icd = self.modalities_config.get('icd', False)
        self.load_radiology = self.modalities_config.get('radiology', False)

        print(f"\nMC-MED Modalities Configuration:")
        print(f"  Time-series: {self.load_timeseries}")
        print(f"  Static: {self.load_static}")
        print(f"  ICD codes: {self.load_icd}")
        print(f"  Radiology: {self.load_radiology}")

    def load_labels(self) -> pd.DataFrame:
        """
        Load labels for competing risks outcomes.
        Expected columns:
          - stay_id (index)
          - ed_los_hours: ED length of stay in hours
          - outcome: competing risk outcome (0=censored, 1=ICU, 2=Hospital, 3=Observation)
        """
        label_path = self.base_dir / self.files['labels']
        print(f"\nLoading labels from: {label_path}")

        labels = pd.read_csv(label_path, index_col='stay_id')

        # Competing risks setup
        competing_risks_config = self.config['dataset'].get('competing_risks', {})

        if competing_risks_config.get('enabled', False):
            # For competing risks, we need outcome (event type) and duration
            required_cols = ['ed_los_hours', 'outcome']
            for col in required_cols:
                if col not in labels.columns:
                    raise ValueError(f"Required column '{col}' not found in labels file")

            # Rename for consistency
            labels['duration'] = labels['ed_los_hours']
            labels['event'] = labels['outcome']

            # Apply minimum stay filter
            min_hours = self.config['dataset']['cohort']['min_stay_hours']
            labels = labels[labels['duration'] >= min_hours]

            print(f"Loaded {len(labels)} labels (≥{min_hours}h stay)")
            print(f"Outcome distribution:")
            for outcome_val, outcome_name in competing_risks_config['outcome_mapping'].items():
                count = (labels['event'] == outcome_val).sum()
                pct = count / len(labels) * 100
                print(f"  {outcome_name}: {count} ({pct:.1f}%)")
        else:
            # Single risk - binary outcome
            if 'event' not in labels.columns or 'duration' not in labels.columns:
                raise ValueError("Labels must contain 'event' and 'duration' columns")

            min_hours = self.config['dataset']['cohort']['min_stay_hours']
            labels = labels[labels['duration'] >= min_hours]

            event_rate = labels['event'].mean() * 100
            print(f"Loaded {len(labels)} labels (≥{min_hours}h stay)")
            print(f"Event rate: {event_rate:.1f}%")

        return labels

    def load_static_features(self) -> pd.DataFrame:
        """Load static demographic features."""
        if not self.load_static:
            print("Static features disabled - returning empty DataFrame")
            return pd.DataFrame(index=pd.Index([], name='stay_id'))

        static_path = self.base_dir / self.files['static']
        print(f"Loading static features from: {static_path}")

        static_df = pd.read_csv(static_path, index_col='stay_id')

        # Exclude specified columns
        exclude_cols = self.config['dataset']['cohort'].get('exclude_columns', [])
        static_df = static_df.drop(columns=exclude_cols, errors='ignore')

        print(f"Loaded {len(static_df)} stays with {len(static_df.columns)} static features")
        return static_df

    def load_timeseries(self) -> pd.DataFrame:
        """Load time-series vital signs and labs."""
        if not self.load_timeseries:
            print("Time-series disabled - returning empty DataFrame")
            return pd.DataFrame()

        cohort_ids = self.cohort_df.index.unique()
        train_ids = self.create_splits()['train']
        max_hours = self.config['preprocessing']['max_hours']
        missingness_threshold = self.config['preprocessing']['missingness_threshold']

        ts_path = self.base_dir / self.files['timeseries']
        print(f"\nLoading time-series from: {ts_path}")

        ts_df = pd.read_csv(ts_path)

        # Expected format: stay_id, time_hours, feature columns
        if 'stay_id' not in ts_df.columns or 'time_hours' not in ts_df.columns:
            raise ValueError("Time-series must have 'stay_id' and 'time_hours' columns")

        # Filter to cohort
        ts_df = ts_df[ts_df['stay_id'].isin(cohort_ids)]

        # Filter time range
        ts_df = ts_df[(ts_df['time_hours'] >= 0) & (ts_df['time_hours'] < max_hours)]

        # Set multi-index
        ts_df = ts_df.set_index(['stay_id', 'time_hours'])

        print(f"Loaded {len(ts_df)} time-series measurements")
        print(f"Raw features: {len(ts_df.columns)}")

        # Missingness filtering
        print(f"Filtering features by missingness (threshold: {missingness_threshold})...")
        ts_train = ts_df[ts_df.index.get_level_values('stay_id').isin(train_ids)]
        missingness = ts_train.notna().mean()
        cols_to_keep = missingness[missingness >= missingness_threshold].index
        ts_df = ts_df[cols_to_keep]
        ts_df = ts_df.sort_index()

        print(f"Kept {len(cols_to_keep)} dynamic features (≥{missingness_threshold * 100}% present in training)")
        return ts_df

    def load_icd_codes(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Load ICD diagnosis codes and optionally pre-computed embeddings.

        Returns:
            - icd_df: DataFrame with stay_id index and ICD code columns (one-hot or counts)
            - embeddings: Optional pre-computed embeddings array
        """
        if not self.load_icd:
            print("ICD codes disabled")
            return pd.DataFrame(index=pd.Index([], name='stay_id')), None

        icd_path = self.base_dir / self.files.get('icd_codes')
        if icd_path is None:
            print("Warning: ICD codes enabled but no file path specified")
            return pd.DataFrame(index=pd.Index([], name='stay_id')), None

        print(f"\nLoading ICD codes from: {icd_path}")

        # Load ICD codes - expected format: stay_id, icd_code (or one-hot encoded)
        icd_df = pd.read_csv(icd_path, index_col='stay_id')

        cohort_ids = self.cohort_df.index.unique()
        icd_df = icd_df[icd_df.index.isin(cohort_ids)]

        # Check for pre-computed embeddings
        embeddings = None
        embeddings_path = self.base_dir / self.files.get('icd_embeddings', 'icd_embeddings.npy')
        if embeddings_path.exists():
            print(f"Loading pre-computed ICD embeddings from: {embeddings_path}")
            embeddings = np.load(embeddings_path)

        print(f"Loaded ICD codes for {len(icd_df)} stays with {len(icd_df.columns)} features")
        return icd_df, embeddings

    def load_radiology_reports(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        Load radiology report embeddings.

        Returns:
            - reports_df: DataFrame with metadata (if any)
            - embeddings: Pre-computed embeddings from Clinical-Longformer or similar
        """
        if not self.load_radiology:
            print("Radiology reports disabled")
            return pd.DataFrame(index=pd.Index([], name='stay_id')), None

        # Option 1: Pre-computed embeddings
        embeddings_path = self.base_dir / self.files.get('radiology_embeddings')
        if embeddings_path and embeddings_path.exists():
            print(f"\nLoading radiology embeddings from: {embeddings_path}")

            # Expected format: stay_id index, embedding columns
            embeddings_df = pd.read_csv(embeddings_path, index_col='stay_id')
            cohort_ids = self.cohort_df.index.unique()
            embeddings_df = embeddings_df[embeddings_df.index.isin(cohort_ids)]

            print(f"Loaded radiology embeddings for {len(embeddings_df)} stays")
            print(f"Embedding dimension: {len(embeddings_df.columns)}")

            return embeddings_df, embeddings_df.values

        # Option 2: Raw text (would need processing)
        reports_path = self.base_dir / self.files.get('radiology_text')
        if reports_path and reports_path.exists():
            print(f"\nLoading radiology text from: {reports_path}")
            print("Warning: Raw text loading - embeddings should be pre-computed")

            reports_df = pd.read_csv(reports_path, index_col='stay_id')
            cohort_ids = self.cohort_df.index.unique()
            reports_df = reports_df[reports_df.index.isin(cohort_ids)]

            print(f"Loaded reports for {len(reports_df)} stays")
            return reports_df, None

        print("Warning: Radiology enabled but no files found")
        return pd.DataFrame(index=pd.Index([], name='stay_id')), None

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MC-MED specific cohort criteria."""
        initial_n = len(df)

        # Age filtering if specified
        max_age = self.config['dataset']['cohort'].get('max_age')
        if max_age and 'age' in df.columns:
            df = df[df['age'] <= max_age]
            print(f"  Age filter (≤{max_age}): {len(df)} remains")

        # Additional filters can be added here

        print(f"Cohort criteria applied: {initial_n} → {len(df)} stays")
        return df

    def get_modality_info(self) -> Dict:
        """Return information about loaded modalities."""
        return {
            'timeseries': self.load_timeseries,
            'static': self.load_static,
            'icd': self.load_icd,
            'radiology': self.load_radiology,
        }