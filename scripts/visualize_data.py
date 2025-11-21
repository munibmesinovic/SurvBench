#!/usr/bin/env python3
"""
Data Visualization Script for SurvBench.

This script loads the final preprocessed output files (.npy, .p, .pkl)
and generates a few key plots to help understand the processed dataset:
1. A Kaplan-Meier survival curve (or competing risks CIF).
2. A histogram of event/censoring durations.
3. A plot of mean time-series feature trajectories.
4. (New) ICD Code statistics (if available, e.g., MIMIC-IV).
5. (New) Radiology Embedding statistics (if available).

Requires: lifelines, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
import pickle
import yaml
import argparse
import sys
import warnings  # Added for clean output
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
    print(f"  Saved to {save_path.name}")
    plt.close()


def plot_duration_histogram(durations, events, save_path):
    """Plots and saves a histogram of event/censoring durations."""
    print(f"Plotting duration histogram...")
    plt.figure(figsize=(10, 7))

    df = pd.DataFrame({
        'duration': durations,
        'event_type': pd.Series(events).map({0: 'Censored', 1: 'Event (1+)'})
    })

    # Seaborn automatically adds the legend when using 'hue'
    sns.histplot(data=df, x='duration', hue='event_type', multiple='stack', bins=50, kde=True)

    plt.title("Histogram of Event and Censoring Durations")
    plt.xlabel("Hours Since Admission")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Saved to {save_path.name}")
    plt.close()


def plot_feature_trajectories(x_train, m_train, feature_info, num_features_to_plot=10, save_path=None):
    """
    Plots and saves the mean trajectories of top dynamic features,
    using the mask to calculate stats only on observed data.
    """
    print(f"Plotting feature trajectories (observed data only)...")
    dynamic_indices = feature_info.get('dynamic_indices', [])
    dynamic_names = feature_info.get('dynamic_names', [])

    if not dynamic_indices:
        print("  No dynamic features found to plot.")
        return

    # Get the dynamic feature blocks
    x_dynamic = x_train[:, :, dynamic_indices]
    m_dynamic = m_train[:, :, dynamic_indices]

    # Apply the mask: set imputed values to NaN so np.nanmean/nanstd ignore them
    x_dynamic_observed_only = np.where(m_dynamic == 1, x_dynamic, np.nan)

    # Select features to plot
    if num_features_to_plot > len(dynamic_names):
        num_features_to_plot = len(dynamic_names)

    # Use a fixed seed for reproducibility of random feature selection
    rng = np.random.default_rng(42)
    plot_indices = rng.choice(len(dynamic_names), num_features_to_plot, replace=False)
    plot_names = [dynamic_names[i] for i in plot_indices]

    # Calculate stats on the *observed-only* data
    # We suppress warnings because some features might be completely missing in a window, which is fine.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_trajectories = np.nanmean(x_dynamic_observed_only[:, :, plot_indices], axis=0)
        std_trajectories = np.nanstd(x_dynamic_observed_only[:, :, plot_indices], axis=0)

    num_windows = x_train.shape[1]
    time_steps = np.arange(num_windows)

    plt.figure(figsize=(15, 10))
    for i in range(num_features_to_plot):
        mean = mean_trajectories[:, i]
        std = std_trajectories[:, i]

        # Only plot if we have valid data
        if not np.isnan(mean).all():
            plt.plot(time_steps, mean, label=f"{plot_names[i]} (mean)", marker='o')
            # Use a much lighter alpha
            plt.fill_between(time_steps, mean - std, mean + std, alpha=0.1)

    plt.title(f"Mean Trajectories of {num_features_to_plot} Random Dynamic Features (Standard Scaled, Observed Only)")
    plt.xlabel("Time Window")
    plt.ylabel("Standardized Value")
    plt.xticks(time_steps)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Saved to {save_path.name}")
    plt.close()


def plot_icd_statistics(x_icd, save_path):
    """
    Plots statistics for ICD codes (MIMIC-IV specific).
    x_icd: (N, F_icd) binary matrix.
    """
    print(f"Plotting ICD code statistics...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Codes per patient (Row sums)
    codes_per_patient = x_icd.sum(axis=1)
    sns.histplot(codes_per_patient, bins=30, kde=False, ax=ax1, color='purple')
    ax1.set_title("Distribution of ICD Codes per Patient")
    ax1.set_xlabel("Number of Codes")
    ax1.set_ylabel("Count of Patients")

    # 2. Code Frequencies (Column sums)
    # We plot the Top 50 most common codes
    code_freqs = x_icd.sum(axis=0)
    # Sort descending
    code_freqs = np.sort(code_freqs)[::-1]
    top_n = min(50, len(code_freqs))

    ax2.bar(range(top_n), code_freqs[:top_n], color='teal')
    ax2.set_title(f"Frequency of Top {top_n} Most Common Codes")
    ax2.set_xlabel("Rank of ICD Code")
    ax2.set_ylabel("Frequency (Count)")
    ax2.set_yscale('log')  # Log scale often helps with long-tail distributions

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Saved to {save_path.name}")
    plt.close()


def plot_radiology_embedding_stats(x_rad, save_path):
    """
    Plots basic stats for radiology embeddings (MIMIC-IV specific).
    x_rad: (N, Embedding_Dim) matrix.
    """
    print(f"Plotting radiology embedding statistics...")
    plt.figure(figsize=(10, 6))

    # Flatten to see distribution of embedding values (sanity check for normalization)
    sample_values = x_rad.flatten()
    # Downsample for plotting if too large
    if len(sample_values) > 10000:
        rng = np.random.default_rng(42)
        sample_values = rng.choice(sample_values, 10000, replace=False)

    sns.histplot(sample_values, bins=50, kde=True, color='orange')
    plt.title(f"Distribution of Radiology Embedding Values (Dimension: {x_rad.shape[1]})")
    plt.xlabel("Value")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Saved to {save_path.name}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize preprocessed SurvBench data'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file (e.g., configs/mimiciv_config.yaml)'
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

        # Load features
        x_train = np.load(output_dir / files['x_train'])

        # Load mask (Robust loading)
        mask_file_key = 'train_mask'
        if mask_file_key in files:
            mask_path = output_dir / files[mask_file_key]
        else:
            # Fallback if key missing in config
            mask_path = output_dir / str(files['x_train']).replace('.npy', '_mask.npy')
            print(f"  (Note: '{mask_file_key}' key not found, trying fallback: {mask_path.name})")

        if not mask_path.exists():
            # Fallback for eICU where mask might be named slightly differently
            mask_path = output_dir / "x_train_eicu_mask.npy"

        m_train = np.load(mask_path)

        # Load metadata
        with open(output_dir / files['feature_names'], 'rb') as f:
            feature_info = pickle.load(f)

        with open(output_dir / files['modality_info'], 'rb') as f:
            modality_info = pickle.load(f)

        is_competing_risks = modality_info.get('competing_risks', False)
        outcome_mapping = config.get('dataset', {}).get('competing_risks', {}).get('outcome_mapping',
                                                                                   {0: "Censored", 1: "Event"})

        print("  ✓ Core data loaded.")

    except FileNotFoundError as e:
        print(f"  ✗ Error loading file: {e}")
        print("  Please run the preprocessing pipeline first.")
        sys.exit(1)

    # --- 2. Check for Additional Modalities ---
    x_icd = None
    icd_path = output_dir / "x_train_icd.npy"
    if icd_path.exists():
        print("  ✓ Found ICD data.")
        x_icd = np.load(icd_path)

    x_rad = None
    rad_path = output_dir / "x_train_radiology.npy"
    if rad_path.exists():
        print("  ✓ Found Radiology data.")
        x_rad = np.load(rad_path)

    # --- 3. Generate Plots ---
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
        x_train, m_train, feature_info,
        save_path=figures_dir / "feature_trajectories.png"
    )

    # Plot 4: ICD Statistics
    if x_icd is not None:
        plot_icd_statistics(x_icd, figures_dir / "icd_statistics.png")

    # Plot 5: Radiology Statistics
    if x_rad is not None:
        plot_radiology_embedding_stats(x_rad, figures_dir / "radiology_embedding_stats.png")

    print("\n[3/3] Visualization complete!")


if __name__ == '__main__':
    main()