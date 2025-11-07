Here is a professional README for your `SurvBench` repository. I have written it from the perspective of a researcher, focusing on clear, practical, and detailed instructions for setup and usage, as you requested.

-----

# SurvBench: A Standardized Preprocessing Pipeline for Multi-Modal EHR Survival Analysis

`SurvBench` is a Python-based preprocessing pipeline designed to bridge the gap between raw, complex Electronic Health Record (EHR) datasets and the clean, windowed, multi-modal tensors required by modern survival analysis models.

Reproducibility in EHR research is challenging, especially in the preprocessing phase. This repository provides a standardized, configurable, and open-source tool to convert raw EHR files into a consistent format suitable for training and evaluating deep learning survival models, including those that handle competing risks.

[cite\_start]This pipeline was developed to support the experiments in **MM-GraphSurv** [cite: 1] and related projects.

-----

## 🔬 Key Features

  * **Raw-to-Tensor:** Ingests raw CSVs directly from PhysioNet. No intermediate, pre-processed files are required.
  * **Multi-Dataset Support:** Provides data loaders for three major critical care and emergency datasets:
      * **MIMIC-IV** (v2.2)
      * **eICU** (v2.0)
      * **MC-MED** (Emergency Department dataset)
  * **Multi-Modal by Design:** Seamlessly loads, aligns, and aggregates features from different modalities:
      * **Time-Series:** Vitals (periodic and aperiodic) and Lab results.
      * **Static:** Demographics, admission details, and triage information.
      * **Structural:** ICD diagnoses histories (for MIMIC-IV and MC-MED).
  * **Survival-Specific:** Natively handles both **single-risk** (e.g., in-hospital mortality) and **competing-risk** (e.g., discharge, ICU admission, death) scenarios.
  * **Configurable Pipeline:** All parameters—time windows, horizons, feature selection, and paths—are controlled via simple YAML config files.

-----

## 1\. Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/munibmesinovic/SurvBench.git
    cd SurvBench
    ```

2.  **Install dependencies:**
    We recommend using a Python virtual environment (e.g., `conda` or `venv`) with Python 3.9+.

    ```bash
    pip install -r requirements.txt
    ```

    This will install `pandas`, `numpy`, `scikit-learn`, `pycox`, `torch`, `transformers`, and other necessary packages.

-----

## 2\. Dataset Setup

This is the most critical step. The pipeline reads raw CSVs. You must download them from their respective sources and place them in a directory.

### eICU (v2.0)

1.  **Download:** Obtain access and download the raw CSVs from the [eICU Collaborative Research Database on PhysioNet](https://physionet.org/content/eicu-crd/2.0/).
2.  **Required Files:** You only need the following four files. You can use the `.csv` or `.csv.gz` versions. The config `configs/eicu_config.yaml` is set up for the `.gz` versions, which is recommended.
      * `patient.csv` (or `patient.csv.gz`)
      * `lab.csv` (or `lab.csv.gz`)
      * `vitalPeriodic.csv` (or `vitalPeriodic.csv.gz`)
      * `vitalAperiodic.csv` (or `vitalAperiodic.csv.gz`)
3.  **Folder Structure:** Place these files in a single directory. Example:
    ```
    data/eicu_raw_data/
    ├── lab.csv.gz
    ├── patient.csv
    ├── vitalAperiodic.csv.gz
    └── vitalPeriodic.csv.gz
    ```
4.  **Configuration:** You will later update `configs/eicu_config.yaml` to point to this directory.

### MIMIC-IV (v2.2)

1.  **Download:** Obtain access and download the raw CSVs from the [MIMIC-IV dataset on PhysioNet](https://physionet.org/content/mimiciv/2.2/).
2.  **Required Files:** The pipeline requires files from the `icu` and `hosp` modules.
      * `icu/icustays.csv`
      * `hosp/patients.csv`
      * `hosp/admissions.csv`
      * `icu/chartevents.csv`
      * `hosp/labevents.csv`
      * `hosp/diagnoses_icd.csv`
      * `hosp/radiology.csv` (Optional, only needed if `radiology: true` in config)
3.  **Folder Structure:** Place these files in a directory *maintaining their module folders (`hosp`, `icu`)*.
    ```
    data/mimiciv_raw_data/
    ├── hosp/
    │   ├── admissions.csv
    │   ├── diagnoses_icd.csv
    │   ├── labevents.csv
    │   ├── patients.csv
    │   └── radiology.csv
    └── icu/
        ├── chartevents.csv
        └── icustays.csv
    ```
4.  **Configuration:** You will later update `configs/mimiciv_config.yaml` to point to this `data/mimiciv_raw_data` directory.

### MC-MED (Private)

This dataset was used in the CausalSurv and MM-GraphSurv papers and is not publicly available. If you have access to it, the `mcmed_loader.py` expects the following files:

  * `visits.csv` (for labels and static features)
  * `numerics.csv` (time-series vitals)
  * `labs.csv` (time-series labs)
  * `pmh.csv` (ICD code history)
  * `rads.csv` (Radiology report text)

-----

## 3\. 🚀 Running the Pipeline

The preprocessing is executed using the `scripts/run_preprocessing.py` script.

**Step 1: Edit the Configuration File**

Choose the config file for the dataset you want to process (e.g., `configs/eicu_config.yaml`). You **must** edit the following paths:

  * **`dataset.base_dir`**: Set this to the full path of your raw data directory (e.g., `/Users/munib/data/eicu_raw_data`).
  * **`output.dir`**: Set this to the full path where you want the processed files to be saved (e.g., `/Users/munib/data/eicu_processed`).

**Step 2: Run the Script**

From the `SurvBench` root directory, run the script pointing to your chosen config file.

**To process eICU:**

```bash
python scripts/run_preprocessing.py --config configs/eicu_config.yaml
```

**To process MIMIC-IV:**

```bash
python scripts/run_preprocessing.py --config configs/mimiciv_config.yaml
```

**To process MC-MED:**

```bash
python scripts/run_preprocessing.py --config configs/mcmed_config.yaml
```

The script will execute the full pipeline: loading the cohort, splitting by patient, processing all modalities, aggregating into time windows, applying imputation and scaling, and saving the final files.

-----

## 4\. 💾 Output Files

After running, your specified `output.dir` will contain the following files (using `eicu` as an example prefix):

  * **`x_train_eicu.npy`**: The main training data tensor.
      * **Shape:** `(N_train, T, F)` where `N` is \# of patients, `T` is \# of time windows (e.g., 6), `F` is \# of features.
  * **`x_val_eicu.npy` / `x_test_eicu.npy`**: Validation and test tensors.
  * **`x_train_eicu_mask.npy`**: Binary missingness mask for the training data.
      * **Shape:** `(N_train, T, F)`. `1.0` if a value was present *before* imputation, `0.0` if it was missing.
  * **`y_train_surv_eicu.p` / `y_val_surv_eicu.p`**: Pickled tuples `(durations, events)` for training and validation.
  * **`durations_test_eicu.npy` / `events_test_eicu.npy`**: Separate numpy arrays for test labels.
  * **`feature_names.pkl`**: A critical metadata dictionary. It contains:
      * `'dynamic_names'`: List of feature names for the dynamic (time-series) part of the tensor.
      * `'static_names'`: List of feature names for the static part.
      * `'dynamic_indices'` / `'static_indices'`: Index slices for the feature dimension `F`.
  * **`cuts.npy`**: The time-bin edges (quantiles) learned from the training data, used for discrete-time survival models.
  * **`modality_info.pkl`**: A dictionary specifying which modalities were active (e.g., `['timeseries', 'static']`).
  * **`scaler.pkl`**: The saved `StandardScaler` object fit on the training data.

(For MIMIC-IV/MC-MED with `icd: true`, you will also see `x_train_icd.npy`, etc.)

-----

## 5\. Example: Using the Processed Data

Here is a simple example of how to load and use the preprocessed data for model training.

```python
import numpy as np
import pickle
from pathlib import Path

# Path to your processed output directory
data_dir = Path("data/eicu_processed_final/")

# 1. Load data and masks
x_train = np.load(data_dir / "x_train_eicu.npy")
m_train = np.load(data_dir / "x_train_eicu_mask.npy")

# 2. Load labels
with open(data_dir / "y_train_surv_eicu.p", 'rb') as f:
    (durations_train, events_train) = pickle.load(f)

# 3. Load feature metadata
with open(data_dir / "feature_names.pkl", 'rb') as f:
    feature_info = pickle.load(f)

print(f"Loaded training data: {x_train.shape}")
print(f"Loaded training mask: {m_train.shape}")
print(f"Loaded labels: {durations_train.shape}, {events_train.shape}")

print(f"\\n--- Feature Info ---")
print(f"Total features: {feature_info['num_total']}")
print(f"Dynamic features: {feature_info['num_dynamic']}")
print(f"Static features: {feature_info['num_static']}")

print(f"\\nFirst 5 dynamic feature names: {feature_info['dynamic_names'][:5]}")
print(f"First 5 static feature names: {feature_info['static_names'][:5]}")

# Example: Get all static data
static_indices = feature_info['static_indices']
# Static data is broadcast across all time windows, so just take the first window
x_train_static = x_train[:, 0, static_indices] 
print(f"\\nShape of static data block: {x_train_static.shape}")

# Example: Get all dynamic data
dynamic_indices = feature_info['dynamic_indices']
x_train_dynamic = x_train[:, :, dynamic_indices]
print(f"Shape of dynamic data block: {x_train_dynamic.shape}")

# Example: Get time bins for discrete models
time_bins = np.load(data_dir / "cuts_eicu.npy")
print(f"\\nTime bins for discrete survival: {time_bins}")
```

If you use `SurvBench` or its underlying data loaders in your research, please cite our AISTATS 2024 paper:

> Munib Mesinovic, Peter Watkinson, Tingting Zhu. (2024). *MM-GraphSurv: Interpretable Multi-Modal Graph for Survival Prediction with Electronic Health Records*. [cite\_start]In Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS). [cite: 1, 3, 5, 15]
