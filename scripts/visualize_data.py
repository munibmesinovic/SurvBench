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

    plot_indices = np.random.choice(len(dynamic_names), num_features_to_plot, replace=False)
    plot_names = [dynamic_names[i] for i in plot_indices]

    # Calculate stats on the *observed-only* data
    mean_trajectories = np.nanmean(x_dynamic_observed_only[:, :, plot_indices], axis=0)
    std_trajectories = np.nanstd(x_dynamic_observed_only[:, :, plot_indices], axis=0)

    num_windows = x_train.shape[1]
    time_steps = np.arange(num_windows)

    plt.figure(figsize=(15, 10))
    for i in range(num_features_to_plot):
        mean = mean_trajectories[:, i]
        std = std_trajectories[:, i]
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

        # Load features
        x_train = np.load(output_dir / files['x_train'])

        # --- THIS IS THE CRITICAL NEW LINE ---
        m_train = np.load(output_dir / files['train_mask'])
        # --- END NEW LINE ---

        # Load metadata
        with open(output_dir / files['feature_names'], 'rb') as f:
            feature_info = pickle.load(f)

        with open(output_dir / files['modality_info'], 'rb') as f:
            modality_info = pickle.load(f)

        is_competing_risks = modality_info.get('competing_risks', False)
        outcome_mapping = config.get('dataset', {}).get('competing_risks', {}).get('outcome_mapping',
                                                                                   {0: "Censored", 1: "Event"})

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
        x_train, m_train, feature_info,
        save_path=figures_dir / "feature_trajectories.png"
    )

    print("\n[3/3] Visualization complete!")


if __name__ == '__main__':
    main()