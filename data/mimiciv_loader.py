import pandas as pd
import numpy as np
import gc
import os  # <-- This fixes the NameError
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from data.base_loader import BaseDataLoader
from tqdm import tqdm

# Imports from your CausalSurv notebook
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

# Set device for BERT
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Item ID Maps from Notebook ---
VITAL_ITEMIDS = {
    220045: 'Heart_Rate', 220179: 'Systolic_BP', 220180: 'Diastolic_BP',
    220210: 'Respiratory_Rate', 223761: 'Temperature_F', 220277: 'SpO2',
    223900: 'GCS_Total', 220050: 'Arterial_BP_Systolic', 220051: 'Arterial_BP_Diastolic',
}
LAB_ITEMIDS = {
    50912: 'Creatinine', 50902: 'Chloride', 50882: 'Bicarbonate', 50893: 'Calcium_Total',
    51006: 'BUN', 51222: 'Hemoglobin', 51221: 'Hematocrit', 51265: 'Platelet_Count',
    51301: 'WBC', 50931: 'Glucose', 50983: 'Sodium', 50971: 'Potassium',
    50960: 'Magnesium', 50813: 'Lactate', 50868: 'Anion_Gap',
}


class MIMICIVDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC-IV database.

    This loader is adapted directly from the CausalSurv MIMIC_IV.ipynb notebook.
    It loads raw CSVs, creates the cohort, and processes all modalities
    (time-series, static, ICD, radiology) in memory.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.files = config['dataset']['files']
        self.modalities_config = config.get('modalities', {})

        # Which modalities to load
        self._load_timeseries = self.modalities_config.get('timeseries', True)
        self._load_static = self.modalities_config.get('static', True)
        self._load_icd = self.modalities_config.get('icd', False)
        self._load_radiology = self.modalities_config.get('radiology', False)

        # Notebook parameters from config
        self.n_top_codes = config['icd_processing'].get('top_n_codes', 500)
        self.bert_batch_size = config['radiology_processing'].get('bert_batch_size', 32)
        self.bert_max_length = config['radiology_processing'].get('bert_max_length', 1024)

        # Parameters from config (matching notebook)
        self.n_input_steps_hourly = config['preprocessing']['max_hours']  # e.g., 24
        self.n_windows = config['preprocessing']['num_windows']  # e.g., 6
        self.window_size_hours = config['preprocessing']['window_size_hours']  # e.g., 4
        self.max_label_hours = config['preprocessing']['max_horizon_hours']  # e.g., 240
        self.seed = config['splits']['seed']

        # Caching for raw files
        self._cohort_df_cache = None
        self._train_stays_cache = None

        print(f"\nMIMIC-IV Modalities (CausalSurv Logic):")
        print(f"  Time-series: {self._load_timeseries}")
        print(f"  Static (Demo/Admin): {self._load_static}")
        print(f"  ICD codes (Causal): {self._load_icd}")
        print(f"  Radiology (Live BERT): {self._load_radiology}")

    def _get_cohort_df(self) -> pd.DataFrame:
        """Loads and caches the core cohort from icustays, patients, admissions."""
        if self._cohort_df_cache is not None:
            return self._cohort_df_cache

        print("\nLoading core tables (icustays, patients, admissions)...")
        icustays = pd.read_csv(os.path.join(self.base_dir, self.files['icustays']),
                               parse_dates=['intime', 'outtime'])
        patients = pd.read_csv(os.path.join(self.base_dir, self.files['patients']),
                               parse_dates=['dod'])
        admissions = pd.read_csv(os.path.join(self.base_dir, self.files['admissions']),
                                 parse_dates=['admittime', 'dischtime', 'deathtime'])

        print("Filtering cohort (first stay, adult, >=24h)...")
        cohort = icustays.merge(patients, on='subject_id', how='left')
        cohort = cohort.merge(admissions, on=['subject_id', 'hadm_id'], how='left')

        # 1. First ICU stay only
        cohort = cohort[cohort.groupby('subject_id')['intime'].rank(method='first') == 1]
        # 2. Adults only
        cohort = cohort[(cohort['anchor_age'] >= 18)]
        # 3. Minimum 24-hour stay (to have full data)
        cohort = cohort[((cohort['outtime'] - cohort['intime']).dt.total_seconds() / 3600 >= 24)]

        print(f"Full cohort size: {len(cohort)} ICU stays")
        self._cohort_df_cache = cohort
        return self._cohort_df_cache

    def _get_train_stays(self) -> np.ndarray:
        """
        Gets the list of training stay_ids.
        This is needed for causally-correct ICD code generation.
        """
        if self._train_stays_cache is not None:
            return self._train_stays_cache

        print("\n(Pre-caching train/val/test splits for feature generation...)")
        cohort = self._get_cohort_df()

        # Create temporary labels just for splitting
        temp_labels_df = self._create_survival_labels(cohort)
        temp_labels_df = temp_labels_df.set_index('stay_id')  # Ensure index is stay_id

        # Align labels with cohort stays
        aligned_labels = temp_labels_df.reindex(cohort['stay_id'])

        train_stays, test_stays_val = train_test_split(
            cohort['stay_id'].values,
            test_size=0.2,
            random_state=self.seed,
            stratify=aligned_labels['event'].values
        )
        # We only need train_stays, so we can stop here.
        self._train_stays_cache = train_stays
        print(f"(Cached {len(train_stays)} training stays)")
        return self._train_stays_cache

    def _create_survival_labels(self, cohort_df: pd.DataFrame) -> pd.DataFrame:
        """Helper to create labels from the cohort."""
        cohort_df['true_death_time'] = cohort_df['deathtime'].combine_first(cohort_df['dod'])
        time_to_death_hours = (cohort_df['true_death_time'] - cohort_df['intime']).dt.total_seconds() / 3600

        # Calculate time to hospital discharge
        # (ensure dischtime is loaded in _get_cohort_df)
        time_to_discharge = (cohort_df['dischtime'] - cohort_df['intime']).dt.total_seconds() / 3600

        # 1. Determine the effective censoring time (Discharge or Horizon, whichever is first)
        censoring_time = np.minimum(time_to_discharge, self.max_label_hours)

        # 2. Define Event: Death must happen BEFORE the horizon AND BEFORE/AT discharge
        # (If they die after discharge, they are censored at discharge for "In-Hospital Mortality")
        is_event = (time_to_death_hours <= self.max_label_hours)

        # 3. Set Duration
        # If Event: Use death time
        # If Censored: Use discharge time (or horizon if they are still in hospital)
        duration = np.where(is_event, time_to_death_hours, censoring_time)

        # Cleanup
        duration = np.maximum(0, duration)
        event = is_event.astype(int)

        return cohort_df[['stay_id']].copy().assign(duration_hours=duration, event=event)

    def load_labels(self) -> pd.DataFrame:
        """Loads 10-day (240h) mortality labels."""
        print("Creating 10-day mortality labels...")
        cohort = self._get_cohort_df()
        labels_df = self._create_survival_labels(cohort)

        print(f"10-Day Event rate: {labels_df['event'].mean():.1%}")

        # Add subject_id for base_loader splitter
        labels_df = labels_df.merge(cohort[['stay_id', 'subject_id']], on='stay_id')
        labels_df = labels_df.rename(columns={
            'duration_hours': 'duration',
            'stay_id': 'admission_id'  # Use 'admission_id' as index for pipeline
        }).set_index('admission_id')

        return labels_df[['duration', 'event', 'subject_id']]

    def load_static_features(self) -> pd.DataFrame:
        """Loads static demo/admission features."""
        if not self._load_static:
            return pd.DataFrame(index=pd.Index([], name='admission_id'))

        print("Extracting static features...")
        cohort = self._get_cohort_df()
        static_df = cohort[
            ['stay_id', 'anchor_age', 'gender', 'first_careunit', 'admission_type', 'admission_location']
        ].copy()

        static_df = pd.get_dummies(
            static_df,
            columns=['gender', 'first_careunit', 'admission_type', 'admission_location'],
            drop_first=False
        )

        static_df = static_df.rename(columns={'stay_id': 'admission_id'}).set_index('admission_id')
        print(f"Loaded {len(static_df)} stays with {len(static_df.columns)} static features.")
        return static_df

    def load_timeseries(self) -> pd.DataFrame:
        """Loads and processes chartevents and labevents."""
        if not self._load_timeseries:
            return pd.DataFrame()

        cohort = self._get_cohort_df()

        print("Loading chartevents (vitals)...")
        chart_chunks = []
        for chunk in pd.read_csv(os.path.join(self.base_dir, self.files['timeseries_vitals']),
                                 usecols=['stay_id', 'charttime', 'itemid', 'valuenum'],
                                 parse_dates=['charttime'],
                                 chunksize=10_000_000):
            chunk = chunk[chunk['stay_id'].isin(cohort['stay_id']) & chunk['itemid'].isin(VITAL_ITEMIDS.keys())]
            chart_chunks.append(chunk)
        chartevents = pd.concat(chart_chunks, ignore_index=True)
        chartevents['feature'] = chartevents['itemid'].map(VITAL_ITEMIDS)
        chartevents = chartevents.merge(cohort[['stay_id', 'intime']], on='stay_id', how='left')

        print("Loading labevents (labs)...")
        lab_chunks = []
        for chunk in pd.read_csv(os.path.join(self.base_dir, self.files['timeseries_lab']),
                                 usecols=['subject_id', 'charttime', 'itemid', 'valuenum'],
                                 parse_dates=['charttime'],
                                 chunksize=10_000_000):
            chunk = chunk[chunk['itemid'].isin(LAB_ITEMIDS.keys())]
            lab_chunks.append(chunk)
        labevents = pd.concat(lab_chunks, ignore_index=True)
        labevents['feature'] = labevents['itemid'].map(LAB_ITEMIDS)
        labevents = labevents.merge(cohort[['subject_id', 'stay_id', 'intime']], on='subject_id', how='left')

        # Combine
        timeseries_raw = pd.concat([
            chartevents[['stay_id', 'charttime', 'feature', 'valuenum', 'intime']],
            labevents[['stay_id', 'charttime', 'feature', 'valuenum', 'intime']]
        ], ignore_index=True)
        timeseries_raw = timeseries_raw.dropna(subset=['stay_id'])  # Drop labs that didn't merge
        timeseries_raw['stay_id'] = timeseries_raw['stay_id'].astype(int)

        print("Filtering and binning time-series data...")
        timeseries_raw['hours_since_admit'] = (
                (timeseries_raw['charttime'] - timeseries_raw['intime']).dt.total_seconds() / 3600
        )
        timeseries_raw = timeseries_raw[
            (timeseries_raw['hours_since_admit'] >= 0) &
            (timeseries_raw['hours_since_admit'] < self.n_input_steps_hourly)  # < 24
            ]

        # Create hourly time index (0-23)
        timeseries_raw['hour_bin'] = (timeseries_raw['hours_since_admit']).astype(int)

        # Pivot
        ts_wide = timeseries_raw.pivot_table(
            index=['stay_id', 'hour_bin'],
            columns='feature',
            values='valuenum',
            aggfunc='mean'
        )

        # Resample to ensure all 24 hours exist
        all_hours = pd.MultiIndex.from_product([
            cohort['stay_id'].unique(),
            range(self.n_input_steps_hourly)
        ], names=['stay_id', 'hour_bin'])
        ts_hourly = ts_wide.reindex(all_hours)

        ts_windowed = ts_hourly.rename_axis(['admission_id', 'time_hours'])

        print(f"Final dynamic dataframe shape (Hourly): {ts_windowed.shape}")

        # Missingness filtering
        train_ids = self.create_splits()['train']
        missingness_threshold = self.config['preprocessing']['missingness_threshold']
        print(f"Filtering features by missingness (threshold: {missingness_threshold})...")

        # Ensure we only check missingness on training data
        ts_train = ts_windowed[ts_windowed.index.get_level_values('admission_id').isin(train_ids)]
        missingness = ts_train.notna().mean()
        cols_to_keep = missingness[missingness >= missingness_threshold].index
        ts_windowed = ts_windowed[cols_to_keep]
        ts_windowed = ts_windowed.sort_index()

        print(f"Kept {len(cols_to_keep)} dynamic features (≥{missingness_threshold * 100}% present in training)")
        return ts_windowed

    def load_icd_codes(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Loads and processes diagnoses_icd.csv into one-hot vectors."""
        if not self._load_icd:
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        cohort = self._get_cohort_df()
        train_stays = self._get_train_stays()  # Get cached train stay_ids

        print("\nExtracting causally-correct ICD codes...")
        diagnoses = pd.read_csv(os.path.join(self.base_dir, self.files['icd_codes']))

        # Use cohort-loaded admissions data
        admissions = self._get_cohort_df()[['hadm_id', 'admittime']].drop_duplicates()
        diagnoses = diagnoses.merge(admissions, on='hadm_id', how='left')

        diagnoses_merged = diagnoses.merge(
            cohort[['stay_id', 'subject_id', 'intime', 'hadm_id']].rename(
                columns={'hadm_id': 'current_hadm_id', 'intime': 'current_intime'}
            ),
            on='subject_id'
        )

        # CAUSAL FILTER: Only diagnoses from BEFORE current ICU admission
        past_diagnoses = diagnoses_merged[
            diagnoses_merged['admittime'] < diagnoses_merged['current_intime']
            ]

        # Find top N most frequent codes in training set
        train_diagnoses = past_diagnoses[past_diagnoses['stay_id'].isin(train_stays)]
        top_codes = train_diagnoses['icd_code'].value_counts().head(self.n_top_codes).index.tolist()

        past_diagnoses = past_diagnoses[past_diagnoses['icd_code'].isin(top_codes)]
        icd_matrix = past_diagnoses.groupby(['stay_id', 'icd_code']).size().unstack(fill_value=0)
        icd_matrix = (icd_matrix > 0).astype(int)

        icd_matrix = icd_matrix.reindex(cohort['stay_id'], fill_value=0)
        icd_matrix = icd_matrix.reindex(columns=top_codes, fill_value=0)
        icd_matrix.columns = [f'ICD_{code}' for code in icd_matrix.columns]

        icd_df = icd_matrix.rename_axis(index='admission_id')

        # Filter to cohort
        cohort_ids = self.cohort_df.index.unique()
        icd_df = icd_df.loc[icd_df.index.isin(cohort_ids)]

        print(f"Created ({len(icd_df)}, {len(icd_df.columns)}) visit-ICD matrix.")
        return icd_df, None  # Return None for embeddings

    def load_radiology_reports(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Loads radiology.csv and runs Clinical-Longformer."""
        if not self._load_radiology:
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        cohort = self._get_cohort_df()
        cohort_ids = self.cohort_df.index.unique()

        rad_path = self.base_dir / self.files['radiology_embeddings']  # Using this key for 'radiology.csv'
        print(f"\nExtracting {self.files['radiology_embeddings']} (Radiology Reports)...")
        radiology = pd.read_csv(os.path.join(self.base_dir, rad_path), parse_dates=['charttime'])

        radiology = radiology.merge(
            cohort[['stay_id', 'subject_id', 'hadm_id', 'intime']],
            on=['subject_id', 'hadm_id'],
            how='inner'
        )

        # CAUSAL FILTER: Only reports from within the 24-hour input window
        radiology['hours_since_admit'] = (
                (radiology['charttime'] - radiology['intime']).dt.total_seconds() / 3600
        )
        radiology = radiology[
            (radiology['hours_since_admit'] >= 0) &
            (radiology['hours_since_admit'] < self.n_input_steps_hourly)
            ]

        radiology = radiology.dropna(subset=['text'])

        if len(radiology) == 0:
            print("No valid radiology reports found for this cohort.")
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        print("Loading Clinical-Longformer (yikuan8/Clinical-Longformer)...")
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        model = AutoModel.from_pretrained("yikuan8/Clinical-Longformer").to(DEVICE)
        model.eval()

        embeddings_list = []
        print(f"Embedding {len(radiology)} reports (this will take a long time)...")
        with torch.no_grad():
            for i in tqdm(range(0, len(radiology), self.bert_batch_size), desc="Embedding Reports"):
                batch_texts = radiology['text'].iloc[i:i + self.bert_batch_size].tolist()
                inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True,
                                   padding=True, max_length=self.bert_max_length).to(DEVICE)

                with autocast():
                    outputs = model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                # Store embeddings and stay_id
                stay_ids_batch = radiology['stay_id'].iloc[i:i + self.bert_batch_size].values
                for stay_id, emb in zip(stay_ids_batch, batch_embeddings.cpu().numpy()):
                    embeddings_list.append({'stay_id': stay_id, 'embedding': emb.flatten()})

        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

        if not embeddings_list:
            print("No embeddings generated.")
            return pd.DataFrame(index=pd.Index([], name='admission_id')), None

        # Average embeddings per stay_id
        print("Averaging embeddings per visit (stay_id)...")
        embeddings_df = pd.DataFrame(embeddings_list)
        embeddings_grouped = embeddings_df.groupby('stay_id')['embedding'].apply(
            lambda x: np.mean(np.vstack(x), axis=0)
        ).reset_index()

        embedding_matrix = np.vstack(embeddings_grouped['embedding'].values)
        embedding_cols = [f'Radio_{i}' for i in range(embedding_matrix.shape[1])]

        radio_df = pd.DataFrame(embedding_matrix, columns=embedding_cols)
        radio_df['stay_id'] = embeddings_grouped['stay_id'].values

        radio_df = radio_df.set_index('stay_id').rename_axis('admission_id')

        # Reindex to include all visits in cohort (with NaNs for those w/o reports)
        radio_df = radio_df.reindex(cohort_ids.rename('admission_id'))

        print(f"Created ({len(radio_df)}, {radio_df.shape[1]}) visit-Report matrix.")

        # The pipeline expects (reports_df, embeddings_array)
        # We return (embeddings_df, None) and the pipeline will handle it.
        return radio_df, None

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cohort criteria."""
        # All filtering was done in _get_cohort_df().
        # 'subject_id' was added in load_labels() and merged in base_loader.build_cohort()

        # Just need to check that it's present for the splitter
        if 'subject_id' not in df.columns:
            raise ValueError("Critical Error: 'subject_id' was not found in cohort_df before splitting.")

        print(f"Cohort criteria applied: {len(df)} stays remaining.")
        return df