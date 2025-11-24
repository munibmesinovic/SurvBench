# SurvBench: A Standardised Preprocessing Pipeline for Multi-Modal EHR Survival Analysis

`SurvBench` is a Python-based preprocessing pipeline designed to bridge the gap between raw, complex Electronic Health Record (EHR) datasets and the clean, windowed, multi-modal tensors required by deep learning survival analysis models.

This repository provides a standardised, configurable, and open-source tool to convert raw EHR files into a consistent format suitable for training and evaluating deep learning survival models, including those that handle competing risks.

-----

## Key features include

  * **Raw-to-Tensor:** Ingests raw CSVs directly from PhysioNet. No intermediate, pre-processed files are required.
  * **Multi-dataset support:** Provides data loaders for three major critical care and emergency datasets:
      * **MIMIC-IV** (v3.1)
      * **eICU** (v2.0)
      * **MC-MED** (v1.0.1)
  * **Multi-Modal:** Seamlessly loads, aligns, and aggregates features from different modalities:
      * **Time-Series:** Vitals (periodic and aperiodic) and Lab results.
      * **Static:** Demographics, admission details, and triage information.
      * **Structural:** ICD diagnoses histories (for MIMIC-IV and MC-MED).
      * **Radiography:** clinical notes of radiography scans (for MIMIC-IV and MC-MED).
  * **Survival-specific:** Natively handles both single-risk (e.g., in-hospital mortality) and competing-risk (e.g., discharge, ED observation, ICU admission, death) scenarios.
  * **Configurable pipeline:** All parameters—time windows, horizons, feature selection, and paths—are controlled via YAML config files.

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

## 2\. Dataset setup

The pipeline reads raw CSVs. You must download them from their respective sources and place them in a directory.

### eICU (v2.0)

1.  **Download:** Obtain access and download the raw CSVs from the [eICU Collaborative Research Database on PhysioNet](https://physionet.org/content/eicu-crd/2.0/).
2.  **Required files:** You only need the following four files. You can use the `.csv` or `.csv.gz` versions. The config `configs/eicu_config.yaml` is set up for the `.gz` versions, which is recommended.
      * `patient.csv` (or `patient.csv.gz`)
      * `lab.csv` (or `lab.csv.gz`)
      * `vitalPeriodic.csv` (or `vitalPeriodic.csv.gz`)
      * `vitalAperiodic.csv` (or `vitalAperiodic.csv.gz`)
3.  **Folder structure:** Place these files in a single directory. Example:
    ```
    data/eicu_raw_data/
    ├── lab.csv.gz
    ├── patient.csv
    ├── vitalAperiodic.csv.gz
    └── vitalPeriodic.csv.gz
    ```
4.  **Configuration:** You will later update `configs/eicu_config.yaml` to point to this directory.

### MIMIC-IV (v3.1)

1.  **Download:** Obtain access and download the raw CSVs from the [MIMIC-IV dataset on PhysioNet](https://physionet.org/content/mimiciv/3.1/) and [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/).
2.  **Required Files:** The pipeline requires files from the `icu`, `hosp`, and `note` modules.
      * `icu/icustays.csv`
      * `hosp/patients.csv`
      * `hosp/admissions.csv`
      * `icu/chartevents.csv`
      * `hosp/labevents.csv`
      * `hosp/diagnoses_icd.csv`
      * `note/radiology.csv` (Optional, only needed if `radiology: true` in config)
3.  **Folder structure:** Place these files in a directory maintaining their module folders (`hosp`, `icu`).
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

### MC-MED (v.1.0.1)

Obtain access and download the raw CSVs from the [MC-MED dataset on PhysioNet](https://physionet.org/content/mc-med/1.0.1/):

  * `data/visits.csv` (for labels and static features)
  * `data/numerics.csv` (time-series vitals)
  * `data/labs.csv` (time-series labs)
  * `data/pmh.csv` (ICD code history)
  * `data/rads.csv` (Radiology report text)

-----

## 3\. Running the pipeline

The preprocessing is executed using the `scripts/run_preprocessing.py` script.

### 3.1. Configuration

Before running, you can customise the entire preprocessing logic by editing the `.yaml` config file for your chosen dataset (e.g., `configs/mimiciv_config.yaml`). Here are the most important parameters you can change:

**`dataset`**

  * **`base_dir`**: **(Required)** The full path to your raw data directory.
  * **`files`**: Maps the pipeline's internal keys (like `timeseries_vitals`) to the actual filenames in your `base_dir` (e.g., `chartevents.csv`).
  * **`cohort`**: Defines the initial patient filters, such as `min_stay_hours` or `max_age`.
  * **`competing_risks`**: Specific to `mcmed_config.yaml`, this enables competing risk logic and maps event codes (e.g., 1, 2, 3) to human-readable names.

**`preprocessing`**

  * **`max_hours`**: Defines the length of the input observation window (e.g., `24` for the first 24 hours of data).
  * **`num_windows`** & **`window_size_hours`**: Controls the temporal aggregation. The `max_hours` will be divided into `num_windows` bins, each `window_size_hours` long. For example, `max_hours: 24`, `num_windows: 6`, and `window_size_hours: 4` will create a tensor of shape `(N, 6, F)` where each of the 6 time steps represents a 4-hour average.
  * **`max_horizon_hours`**: Sets the output survival horizon. Any event or censoring occurring after this time (e.g., `240` hours / 10 days) will be censored at this timestamp.
  * **`n_time_bins`**: The number of discrete bins to create for discrete-time survival models (e.g., 10).
  * **`missingness_threshold`**: Any time-series feature (e.g., a specific lab test) that is present in less than this percentage of the training set (e.g., `0.01` or 1%) will be discarded entirely.
  * **`imputation`** & **`scaling`**: Lets you control the imputation method (`static: mean`, `dynamic: zero`) and scaling (`standard`).

**`modalities`**

  * These are boolean flags (`true`/`false`) that let you turn entire modalities on or off. For example, you can set `icd: false` and `radiology: false` in `mimiciv_config.yaml` to run a "vitals + static" experiment without the multi-hour embedding time.

**`output`**

  * **`dir`**: **(Required)** The full path where you want the processed files to be saved.
  * **`prefix`**: The prefix for all output filenames (e.g., `mimiciv` results in `x_train_mimiciv.npy`).

### 3.2. How to run

**Step 1: Edit the configuration file**

Open the config file for the dataset you want to process (e.g., `configs/eicu_config.yaml`). At a minimum, you must edit the following paths:

  * **`dataset.base_dir`**: Set this to the full path of your raw data directory.
  * **`output.dir`**: Set this to the full path where you want the processed files to be saved.

Tweak any other parameters (like `max_hours` or `modalities`) to fit your experiment.

**Step 2: Run the script**

From the `SurvBench` root directory, run the script pointing to your chosen config file.

**To process eICU:**

```bash
python scripts/run_preprocessing.py --config configs/eicu_config.yaml
```

**To process MIMIC-IV:**

```bash
python scripts/run_preprocessing.py --config configs/mimiciv_config.yaml
```

The script will execute the full pipeline: loading the cohort, splitting by patient, processing all modalities, aggregating into time windows, applying imputation and scaling, and saving the final files.

-----

### 3.5. Visualising the data

After running the pipeline, you can generate plots to understand and validate your processed dataset. We provide a `visualize_data.py` script for this.

Point the script to the same config file you used for preprocessing. It will find your `output.dir` automatically.

```bash
python scripts/visualize_data.py --config configs/eicu_config.yaml
```

**What it does:**
The script will load your processed `train` data and metadata to generate and save several key plots inside your `output.dir/figures/` folder:

1.  **`survival_curve.png`**: A Kaplan-Meier plot (for single-risk) or Aalen-Johansen plots (for competing risks) showing the survival/incidence probability over time.
2.  **`duration_histogram.png`**: A stacked histogram showing the distribution of event times for both censored patients and those who experienced an event.
3.  **`feature_trajectories.png`**: A line plot showing the mean value (± std dev) of 10 random time-series features across the processed time windows.

-----

## 4\. Output

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

print(f"\\nFeature info")
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

## Citation

If you use SurvBench in your research, please cite:

```
https://arxiv.org/pdf/2511.11935
```
