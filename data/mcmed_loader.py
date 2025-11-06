import pandas as pd
import numpy as np
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from data.base_loader import BaseDataLoader
from tqdm import tqdm

# Imports from your CausalSurv notebook
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModel

# Set device for BERT
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MCMEDDataLoader(BaseDataLoader):
    """
    Data loader for MC-MED (Emergency Department) dataset.

    This loader is adapted directly from the CausalSurv notebook.
    It loads raw CSVs (visits, numerics, labs, pmh, rads) and
    processes them in memory, including:
    - Pivoting and hourly resampling for time-series.
    - One-hot encoding for ICD codes.
    - Live Clinical-Longformer embedding for radiology reports.
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

        # Notebook parameters
        self.n_top_codes = config['icd_processing'].get('top_n_codes', 500)
        self.bert_batch_size = config['radiology_processing'].get('bert_batch_size', 32)
        self.bert_max_length = config['radiology_processing'].get('bert_max_length', 1024)

        # Caching for raw files
        self._visits_df_cache = None
        self._pmh_df_cache = None
        self._rad_df_cache = None

        print(f"\nMC-MED Modalities (CausalSurv Logic):")
        print(f"  Time-series: {self._load_timeseries}")
        print(f"  Static (Triage/Demo): {self._load_static}")
        print(f"  ICD codes: {self._load_icd}")
        print(f"  Radiology (Live BERT): {self._load_radiology}")

    def _get_visits_df(self) -> pd.DataFrame:
        """Loads and caches the main visits.csv file."""
        if self._visits_df_cache is not None:
            return self._visits_df_cache

        label_path = self.base_dir / self.files['labels']  # 'labels' points to 'visits.csv'
        print(f"\nProcessing {self.files['labels']} (Demographics, Triage, & Outcomes)...")
        visits = pd.read_csv(label_path)

        # Filter out rows with missing core data
        core_cols = ['Race', 'Ethnicity', 'Triage_Temp', 'Triage_HR', 'Triage_RR',
                     'Triage_SpO2', 'Triage_SBP', 'Triage_DBP', 'Triage_acuity', 'ED_LOS', 'ED_dispo']
        visits = visits.dropna(subset=core_cols)

        # Keep relevant columns
        keep_cols = ['MRN', 'CSN', 'Age', 'Gender', 'Race', 'Ethnicity',
                     'Triage_Temp', 'Triage_HR', 'Triage_RR', 'Triage_SpO2',
                     'Triage_SBP', 'Triage_DBP', 'Triage_acuity', 'ED_LOS', 'ED_dispo']
        visits = visits[keep_cols]

        # Map outcomes (0=Censor, 1=ICU, 2=Inpatient, 3=Observation)
        ed_dispo_mapping = {'Discharge': 0, 'ICU': 1, 'Inpatient': 2, 'Observation': 3}
        visits['Outcome'] = visits['ED_dispo'].map(ed_dispo_mapping)
        visits = visits.dropna(subset=['Outcome'])
        visits['Outcome'] = visits['Outcome'].astype(int)

        # Set CSN as index
        visits.set_index('CSN', inplace=True)

        # Filter to 24-hour label horizon
        max_label_hours = self.config['preprocessing']['max_horizon_hours']
        visits = visits[visits['ED_LOS'] <= max_label_hours]

        print(f"Loaded {len(visits)} valid visits <= {max_label_hours}h LOS.")

        self._visits_df_cache = visits
        return self._visits_df_cache

    def load_labels(self) -> pd.DataFrame:
        """Loads labels (Duration, Outcome) from visits.csv."""
        visits = self._get_visits_df().copy()

        # This loader expects 'subject_id' for splitting, so we use MRN
        labels = visits[['ED_LOS', 'Outcome', 'MRN']].rename(columns={
            'ED_LOS': 'duration',
            'Outcome': 'event',
            'MRN': 'subject_id'
        })

        # Apply minimum stay filter (from config)
        min_hours = self.config['dataset']['cohort']['min_stay_hours']
        labels = labels[labels['duration'] >= min_hours]

        return labels

    def load_static_features(self) -> pd.DataFrame:
        """Loads static triage/demographics from visits.csv."""
        if not self._load_static:
            return pd.DataFrame(index=pd.Index([], name='CSN'))

        visits = self._get_visits_df().copy()

        # One-hot encode categoricals
        static_df = pd.get_dummies(visits, columns=['Race', 'Gender', 'Ethnicity'], dtype=int)

        # Map triage acuity
        triage_acuity_mapping = {'1-Resuscitation': 5, '2-Emergent': 4, '3-Urgent': 3,
                                 '4-Semi-Urgent': 2, '5-Non-Urgent': 1}
        static_df['Triage_acuity_ordinal'] = static_df['Triage_acuity'].map(triage_acuity_mapping)
        static_df['Triage_acuity_ordinal'] = static_df['Triage_acuity_ordinal'].fillna(-1)

        # Drop raw/unneeded columns
        static_df = static_df.drop(['ED_dispo', 'Triage_acuity', 'ED_LOS', 'Outcome', 'MRN'], axis=1)

        print(f"Loaded {len(static_df)} stays with {len(static_df.columns)} static features.")
        return static_df

    def load_timeseries(self) -> pd.DataFrame:
        """Loads and processes numerics.csv and labs.csv."""
        if not self._load_timeseries:
            return pd.DataFrame()

        cohort_csns = self.cohort_df.index.unique()
        max_hours = self.config['preprocessing']['max_hours']

        print("Processing dynamic data (Labs & Vitals)...")

        # --- Labs ---
        labs_path = self.base_dir / self.files['timeseries_lab']
        print(f"  Loading {self.files['timeseries_lab']}...")
        labs = pd.read_csv(labs_path)
        labs = labs[labs['CSN'].isin(cohort_csns)]
        labs['Result_time'] = pd.to_datetime(labs['Result_time'], errors='coerce')
        labs = labs.dropna(subset=['Result_time', 'Component_name', 'Component_value'])

        df_labs = labs.pivot_table(index=['CSN', 'Result_time'],
                                   columns='Component_name',
                                   values='Component_value',
                                   aggfunc='first')

        threshold = 0.95
        df_labs = df_labs.loc[:, df_labs.isnull().sum() / len(df_labs) < threshold]
        for col in df_labs.columns:
            df_labs[col] = pd.to_numeric(df_labs[col], errors='coerce')
        df_labs = df_labs.dropna(how='all')

        # --- Vitals (Numerics) ---
        vitals_path = self.base_dir / self.files['timeseries_vitals']
        print(f"  Loading {self.files['timeseries_vitals']}...")
        vitals = pd.read_csv(vitals_path)
        vitals = vitals[vitals['CSN'].isin(cohort_csns)]
        vitals['Time'] = pd.to_datetime(vitals['Time'], errors='coerce')
        vitals = vitals.dropna(subset=['Time', 'Measure', 'Value'])

        df_vitals = vitals.pivot_table(index=['CSN', 'Time'],
                                       columns='Measure',
                                       values='Value',
                                       aggfunc='first')
        df_vitals = df_vitals.loc[:, df_vitals.isnull().sum() / len(df_vitals) < threshold]
        df_vitals = df_vitals.dropna(how='all')

        # --- Align and Resample ---
        print("  Aligning and resampling to hourly grid...")
        df_dynamic = pd.concat([df_labs, df_vitals.rename_axis(index={'Time': 'Result_time'})])

        df_dynamic.reset_index(level=1, inplace=True)
        first_event_time = df_dynamic.groupby('CSN')['Result_time'].min()
        df_dynamic = df_dynamic.merge(first_event_time.rename('first_time'), on='CSN', how='left')
        df_dynamic['Time_delta'] = df_dynamic['Result_time'] - df_dynamic['first_time']

        # Filter to our N_INPUT_STEPS (from config, e.g., 6h) window
        df_dynamic = df_dynamic[df_dynamic['Time_delta'] <= pd.Timedelta(hours=max_hours)]

        df_dynamic['Time'] = df_dynamic['Time_delta'].dt.ceil('h')
        df_dynamic = df_dynamic.drop(columns=['Result_time', 'first_time', 'Time_delta'])

        # Group by CSN and the new rounded 'Time'
        df_dynamic = df_dynamic.groupby(['CSN', 'Time']).mean()

        # Reindex to create full hourly grid (0h to max_hours)
        new_index = pd.MultiIndex.from_product(
            [df_dynamic.index.get_level_values(0).unique(),
             pd.to_timedelta(range(max_hours + 1), unit='h')],
            names=['CSN', 'Time']
        )
        df_dynamic = df_dynamic.reindex(new_index)

        # Keep only first N steps (e.g., 0-5h for max_hours=6)
        df_dynamic = df_dynamic.loc[
            df_dynamic.index.get_level_values('Time') < pd.Timedelta(hours=max_hours)
            ]

        # Rename index for pipeline compatibility
        df_dynamic.index = df_dynamic.index.set_levels(
            df_dynamic.index.levels[1].total_seconds() / 3600.0, level=1
        )
        df_dynamic.index = df_dynamic.index.rename(['stay_id', 'time_hours'])

        print(f"Final dynamic dataframe shape: {df_dynamic.shape}")

        # Missingness filtering (as in original repo logic)
        train_ids = self.create_splits()['train']
        missingness_threshold = self.config['preprocessing']['missingness_threshold']
        print(f"Filtering features by missingness (threshold: {missingness_threshold})...")
        ts_train = df_dynamic[df_dynamic.index.get_level_values('stay_id').isin(train_ids)]
        missingness = ts_train.notna().mean()
        cols_to_keep = missingness[missingness >= missingness_threshold].index
        df_dynamic = df_dynamic[cols_to_keep]
        df_dynamic = df_dynamic.sort_index()

        print(f"Kept {len(cols_to_keep)} dynamic features (≥{missingness_threshold * 100}% present in training)")
        return df_dynamic

    def load_icd_codes(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Loads and processes pmh.csv into one-hot vectors."""
        if not self._load_icd:
            return pd.DataFrame(index=pd.Index([], name='CSN')), None

        # We need MRNs from the visits.csv file
        visits_df = self._get_visits_df()
        valid_mrns = visits_df['MRN'].unique()

        icd_path = self.base_dir / self.files['icd_codes']
        print(f"\nProcessing {self.files['icd_codes']} (ICD Codes) for {len(valid_mrns)} patients...")
        pmh = pd.read_csv(icd_path)
        pmh = pmh[['MRN', 'Code']]
        pmh = pmh[pmh['MRN'].isin(valid_mrns)]

        top_codes = pmh['Code'].value_counts().nlargest(self.n_top_codes).index.tolist()
        pmh = pmh[pmh['Code'].isin(top_codes)]

        patient_icd_matrix = pd.crosstab(pmh['MRN'], pmh['Code'])
        patient_icd_matrix = patient_icd_matrix.reindex(columns=top_codes, fill_value=0)
        patient_icd_matrix.columns = [f'icd_{c}' for c in patient_icd_matrix.columns]

        # Map from MRN (patient) to CSN (visit)
        mrn_to_csn_map = visits_df[['MRN']].reset_index()
        icd_df = mrn_to_csn_map.merge(patient_icd_matrix, on='MRN', how='left')
        icd_df = icd_df.drop(columns='MRN').set_index('CSN')
        icd_df = icd_df.fillna(0).astype(int)

        # Filter to cohort
        cohort_ids = self.cohort_df.index.unique()
        icd_df = icd_df.loc[icd_df.index.isin(cohort_ids)]

        print(f"Created ({len(icd_df)}, {len(icd_df.columns)}) visit-ICD matrix.")
        return icd_df, None  # Return None for embeddings

    def load_radiology_reports(self) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Loads rads.csv and runs Clinical-Longformer."""
        if not self._load_radiology:
            return pd.DataFrame(index=pd.Index([], name='CSN')), None

        cohort_csns = self.cohort_df.index.unique()
        rad_path = self.base_dir / self.files['radiology_embeddings']  # Using this key for 'rads.csv'
        print(f"\nProcessing {self.files['radiology_embeddings']} (Radiology Reports) for {len(cohort_csns)} visits...")
        rads = pd.read_csv(rad_path)
        rads = rads[rads['CSN'].isin(cohort_csns)]
        rads = rads.dropna(subset=['Study', 'Impression'])
        rads['Text'] = rads['Study'] + ' ' + rads['Impression']

        if len(rads) == 0:
            print("No valid radiology reports found for this cohort.")
            return pd.DataFrame(index=pd.Index([], name='CSN')), None

        print("Loading Clinical-Longformer model (yikuan8/Clinical-Longformer)...")
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        model = AutoModel.from_pretrained("yikuan8/Clinical-Longformer").to(DEVICE)
        model.eval()

        print(f"Generating embeddings for {len(rads)} reports...")
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(rads), self.bert_batch_size), desc="Embedding Reports"):
                batch_texts = rads['Text'].iloc[i:i + self.bert_batch_size].tolist()
                inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True,
                                   padding=True, max_length=self.bert_max_length).to(DEVICE)

                with autocast():
                    outputs = model(**inputs)
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())

        X_bert_array = np.vstack(embeddings)
        bert_cols = [f'rad_{i}' for i in range(X_bert_array.shape[1])]
        X_bert_df = pd.DataFrame(X_bert_array, columns=bert_cols, index=rads['CSN'].iloc[:len(X_bert_array)])

        print("Averaging embeddings per visit (CSN)...")
        visit_rad_embeddings = X_bert_df.groupby('CSN').mean()

        del model, tokenizer, embeddings, X_bert_array, X_bert_df
        torch.cuda.empty_cache()
        gc.collect()

        # Reindex to include all visits in cohort (with NaNs for those w/o reports)
        visit_rad_embeddings = visit_rad_embeddings.reindex(cohort_csns)

        print(f"Created ({len(visit_rad_embeddings)}, {visit_rad_embeddings.shape[1]}) visit-Report matrix.")

        # The pipeline expects (reports_df, embeddings_array)
        # We return (embeddings_df, None) and the pipeline will handle it.
        return visit_rad_embeddings, None

    def apply_cohort_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MC-MED specific cohort criteria."""
        # All cohort criteria (LOS, valid outcomes) were already
        # applied in the load_labels() and load_static_features()
        # methods when loading 'visits.csv'.

        # The 'subject_id' (from MRN) was added in load_labels()
        # and correctly merged in base_loader.build_cohort().

        # We just need to confirm the final cohort size.
        print(f"Cohort criteria applied: {len(df)} stays remaining.")
        return df

    def get_modality_info(self) -> Dict:
        """Return information about loaded modalities."""
        return {
            'timeseries': self.load_timeseries,
            'static': self.load_static,
            'icd': self.load_icd,
            'radiology': self.load_radiology,
        }