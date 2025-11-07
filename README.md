# SurvBench: A Standardised Preprocessing Pipeline for Multi-Modal EHR Survival Analysis

`SurvBench` is a Python-based preprocessing pipeline designed to bridge the gap between raw, complex Electronic Health Record (EHR) datasets and the clean, windowed, multi-modal tensors required by deep learning survival analysis models.

Reproducibility in EHR research is challenging, especially in the preprocessing phase. This repository provides a standardised, configurable, and open-source tool to convert raw EHR files into a consistent format suitable for training and evaluating deep learning survival models, including those that handle competing risks.

-----

## Key features include

  * **Raw-to-Tensor:** Ingests raw CSVs directly from PhysioNet. No intermediate, pre-processed files are required.
  * **Multi-dataset support:** Provides data loaders for three major critical care and emergency datasets:
      * **MIMIC-IV** (v3.2)
      * **eICU** (v2.0)
      * **MC-MED** (Emergency Department)
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

### MIMIC-IV (v2.2)

1.  **Download:** Obtain access and download the raw CSVs from the [MIMIC-IV dataset on PhysioNet](https://physionet.org/content/mimiciv/3.2/).
2.  **Required Files:** The pipeline requires files from the `icu` and `hosp` modules.
      * `icu/icustays.csv`
      * `hosp/patients.csv`
      * `hosp/admissions.csv`
      * `icu/chartevents.csv`
      * `hosp/labevents.csv`
      * `hosp/diagnoses_icd.csv`
      * `hosp/radiology.csv` (Optional, only needed if `radiology: true` in config)
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

### MC-MED (Private)

This dataset was used in the CausalSurv and MM-GraphSurv papers and is not publicly available. If you have access to it, the `mcmed_loader.py` expects the following files:

  * `visits.csv` (for labels and static features)
  * `numerics.csv` (time-series vitals)
  * `labs.csv` (time-series labs)
  * `pmh.csv` (ICD code history)
  * `rads.csv` (Radiology report text)

You're right, that's a great idea. A user needs to know *how* to customize the pipeline beyond just changing file paths.

Here is the revised "Running the Pipeline" section for your `README.md`. It now includes a detailed breakdown of the key configuration parameters you can tune.

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

**Step 2: Run the ccript**

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

That's an excellent idea. A visualization script is a great utility for users to quickly understand the dataset they've just processed. It helps build intuition and verify that the preprocessing steps (like aggregation and label censoring) worked as expected.

Since you've already processed the eICU data, this script will work out-of-the-box for you. For other users, it will work on any dataset processed by the `SurvBench` pipeline.

Here's the plan:

1.  **Create `scripts/visualize_data.py`**: This new script will load the processed `.npy` and `.pkl` files and generate figures.
2.  **Update `requirements.txt`**: This script will require `matplotlib`, `seaborn`, and the `lifelines` library (for Kaplan-Meier survival curves). We need to add these.
3.  **Update `README.md`**: Add a new section explaining how to run the visualization script.

-----

### 1\. New Script: `scripts/visualize_data.py`

Create this new file inside your `scripts/` directory.

```python
#!/usr/bin/env python3
"""
Data Visualization Script for SurvBench.

This script loads the final preprocessed output files (.npy, .p, .pkl)
and generates a few key plots to help understand the processed dataset:
1. A Kaplan-Meier survival curve (or competing risks CIF).
2. A histogram of event/censoring durations.
3. A plot of mean time-series feature trajectories.

Requires: lifelines, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
import pickle
import yaml
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, AalenJohansenFitter

# This tells Python to look for packages in your "SurvBench" folder
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)


def load_config(config_path: str) -> dict:
    """Loads the specified config YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def plot_survival_curves(durations, events, is_competing_risks, outcome_mapping, save_path):
    """Plots and saves the Kaplan-Meier or Aalen-Johansen CIF curves."""
    print(f"Plotting survival curves...")
    plt.figure(figsize=(10, 7))
    
    if not is_competing_risks:
        # Single risk: Use Kaplan-Meier
        kmf = KaplanMeierFitter()
        kmf.fit(durations, event_observed=events, label="Kaplan-Meier Estimate (Event: 1)")
        kmf.plot_survival_function()
        plt.title("Kaplan-Meier Survival Curve")
        plt.xlabel("Hours Since Admission")
        plt.ylabel("Survival Probability")
    
    else:
        # Competing risks: Use Aalen-Johansen for Cumulative Incidence
        # Get event names, skipping 'Censored'
        event_names = {k: v for k, v in outcome_mapping.items() if k != 0}
        
        for event_code, event_name in event_names.items():
            ajf = AalenJohansenFitter(calculate_variance=True)
            # Fit for this specific event vs. all other events
            ajf.fit(durations, events, event_of_interest=event_code)
            ajf.plot(label=f"Event: {event_name}")

        plt.title("Competing Risks Cumulative Incidence Functions (Aalen-Johansen)")
        plt.xlabel("Hours Since Admission")
        plt.ylabel("Cumulative Incidence Probability")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  ✓ Saved to {save_path.name}")
    plt.close()


def plot_duration_histogram(durations, events, save_path):
    """Plots and saves a histogram of event/censoring durations."""
    print(f"Plotting duration histogram...")
    plt.figure(figsize=(10, 7))
    
    df = pd.DataFrame({
        'duration': durations,
        'event_type': pd.Series(events).map({0: 'Censored', 1: 'Event (1+)'})
    })
    
    sns.histplot(data=df, x='duration', hue='event_type', multiple='stack', bins=50, kde=True)
    
    plt.title("Histogram of Event and Censoring Durations")
    plt.xlabel("Hours Since Admission")
    plt.ylabel("Count")
    plt.legend(title='Outcome')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  ✓ Saved to {save_path.name}")
    plt.close()


def plot_feature_trajectories(x_train, feature_info, num_features_to_plot=10, save_path=None):
    """Plots and saves the mean trajectories of top dynamic features."""
    print(f"Plotting feature trajectories...")
    dynamic_indices = feature_info.get('dynamic_indices', [])
    dynamic_names = feature_info.get('dynamic_names', [])
    
    if not dynamic_indices:
        print("  ! No dynamic features found to plot.")
        return
        
    # Get the dynamic feature block
    x_dynamic = x_train[:, :, dynamic_indices]
    
    # Select features to plot (e.g., first N or random)
    if num_features_to_plot > len(dynamic_names):
        num_features_to_plot = len(dynamic_names)
    
    plot_indices = np.random.choice(len(dynamic_names), num_features_to_plot, replace=False)
    plot_names = [dynamic_names[i] for i in plot_indices]
    
    mean_trajectories = np.nanmean(x_dynamic[:, :, plot_indices], axis=0)
    std_trajectories = np.nanstd(x_dynamic[:, :, plot_indices], axis=0)
    
    num_windows = x_train.shape[1]
    time_steps = np.arange(num_windows)
    
    plt.figure(figsize=(15, 10))
    for i in range(num_features_to_plot):
        mean = mean_trajectories[:, i]
        std = std_trajectories[:, i]
        plt.plot(time_steps, mean, label=f"{plot_names[i]} (mean)", marker='o')
        plt.fill_between(time_steps, mean - std, mean + std, alpha=0.2)
        
    plt.title(f"Mean Trajectories of {num_features_to_plot} Random Dynamic Features (Standard Scaled)")
    plt.xlabel("Time Window")
    plt.ylabel("Standardized Value")
    plt.xticks(time_steps)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  ✓ Saved to {save_path.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize preprocessed SurvBench data'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file (e.g., configs/eicu_config.yaml)'
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    output_dir = Path(config['output']['dir'])
    files = config['output']['files']
    
    # Create a new directory for figures
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    print(f"Figures will be saved to: {figures_dir}")

    # --- 1. Load Data ---
    print("\n[1/3] Loading processed data...")
    try:
        # Load labels
        with open(output_dir / files['y_train'], 'rb') as f:
            durations, events = pickle.load(f)

        # Load features (only needed for trajectory plot)
        x_train = np.load(output_dir / files['x_train'])

        # Load metadata
        with open(output_dir / files['feature_names'], 'rb') as f:
            feature_info = pickle.load(f)
        
        with open(output_dir / files['modality_info'], 'rb') as f:
            modality_info = pickle.load(f)
            
        is_competing_risks = modality_info.get('competing_risks', False)
        outcome_mapping = config.get('dataset', {}).get('competing_risks', {}).get('outcome_mapping', {0: "Censored", 1: "Event"})
        
        print("  ✓ All data loaded.")
    
    except FileNotFoundError as e:
        print(f"  ✗ Error loading file: {e}")
        print("  Please run the preprocessing pipeline first.")
        sys.exit(1)

    # --- 2. Generate Plots ---
    print("\n[2/3] Generating visualizations...")
    
    # Plot 1: Survival Curve(s)
    plot_survival_curves(
        durations, events, is_competing_risks, outcome_mapping,
        figures_dir / "survival_curve.png"
    )

    # Plot 2: Duration Histogram
    plot_duration_histogram(
        durations, events,
        figures_dir / "duration_histogram.png"
    )
    
    # Plot 3: Feature Trajectories
    plot_feature_trajectories(
        x_train, feature_info,
        save_path=figures_dir / "feature_trajectories.png"
    )

    print("\n[3/3] Visualization complete!")


if __name__ == '__main__':
    main()

```

-----

### 2\. Update `requirements.txt`

The libraries `lifelines`, `matplotlib`, and `seaborn` are needed for the script above. You should add them to your `requirements.txt` file.

```text
# Core data processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Survival analysis
pycox>=0.2.3
torchtuples>=0.2.2
torch>=2.0.0
lifelines>=0.28.0  # <-- ADD THIS

# Configuration
pyyaml>=6.0

# Progress bars
tqdm>=4.65.0

# Visualization (now required for visualize_data.py)
matplotlib>=3.7.0  # <-- ADD THIS (or move from optional)
seaborn>=0.12.0   # <-- ADD THIS (or move from optional)

# Testing (optional)
pytest>=7.4.0
pytest-cov>=4.1.0

# Jupyter notebooks (optional)
jupyter>=1.0.0
notebook>=7.0.0

transformers>=4.0.0
```

Your users will need to run `pip install -r requirements.txt` again to install these new packages.

-----

### 3\. Update `README.md`

Finally, let's add a new section to the `README.md` to tell users about this new script. You can add this **after** the "Running the Pipeline" section and **before** the "Output Files" section.

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

If you use `SurvBench` or its underlying data loaders in your research, please cite our paper:

> TBD


