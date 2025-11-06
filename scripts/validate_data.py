#!/usr/bin/env python3
"""
Data Validation Script for SurvBench Pipeline.

This script loads the final preprocessed output files (.npy, .p)
and prints a comprehensive statistical summary suitable for
verification and inclusion in a research paper.

It checks:
- Cohort sizes (Train/Val/Test)
- Outcome distributions (Competing Risks)
- Time-to-event statistics
- Tensor shapes (N, T, F)
- Feature counts by modality
- Missingness percentage
- Discretization bins
"""

import numpy as np
import pickle
import yaml
import argparse
import sys
from pathlib import Path

# Ensure the project root is on the path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)


def load_config(config_path: str) -> dict:
    """Loads the specified config YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def print_label_stats(label_data: dict, outcome_mapping: dict):
    """Prints formatted stats for a single data split."""
    durations, events = label_data
    n = len(durations)
    if n == 0:
        print("  No data found.")
        return

    print(f"  Total Stays (N): {n}")

    print("\n  Outcome Distribution:")
    for code, name in outcome_mapping.items():
        count = (events == code).sum()
        pct = (count / n) * 100
        print(f"    {code} - {name}: {count} ({pct:.1f}%)")

    print("\n  Time-to-Event Statistics (raw hours):")
    print(f"    Overall Mean Duration:   {np.mean(durations):.2f}h")
    print(f"    Overall Median Duration: {np.median(durations):.2f}h")

    for code, name in outcome_mapping.items():
        if code == 0:  # Skip censored
            # Get stats for censored group
            censored_durations = durations[events == code]
            if len(censored_durations) > 0:
                print(f"    Mean Duration for '{name}':   {np.mean(censored_durations):.2f}h")
        else:
            # Get stats for event group(s)
            event_durations = durations[events == code]
            if len(event_durations) > 0:
                print(f"    Mean Duration for '{name}':   {np.mean(event_durations):.2f}h")
                print(f"    Median Duration for '{name}': {np.median(event_durations):.2f}h")


def main():
    parser = argparse.ArgumentParser(
        description='Validate preprocessed SurvBench data'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file (e.g., configs/mcmed_config.yaml)'
    )
    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    output_dir = Path(config['output']['dir'])
    files = config['output']['files']
    dataset_name = config['dataset']['name'].upper()

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        print("Please run the preprocessing pipeline first.")
        sys.exit(1)

    print(f"\n{'-' * 80}")
    print(f" VALIDATING PREPROCESSED DATA FOR: {dataset_name}")
    print(f" Output Directory: {output_dir}")
    print(f"{'-' * 80}")

    # --- 1. Load Metadata ---
    print("\n[1/4] Loading Metadata...")
    try:
        with open(output_dir / files['feature_names'], 'rb') as f:
            feature_structure = pickle.load(f)
        print(f"  ✓ Loaded {files['feature_names']} (feature structure dict)")

        with open(output_dir / files['modality_info'], 'rb') as f:
            modality_info = pickle.load(f)
        print(f"  ✓ Loaded {files['modality_info']} (active modalities list)")

        cuts = np.load(output_dir / files['cuts'])
        print(f"  ✓ Loaded {files['cuts']} (discretization bins)")

        outcome_mapping = config.get('dataset', {}).get('competing_risks', {}).get('outcome_mapping', {})
        if not outcome_mapping:
            outcome_mapping = {0: "Censored", 1: "Event"}
        print("  ✓ Loaded outcome mapping from config")

    except FileNotFoundError as e:
        print(f"  ✗ Error loading metadata: {e}")
        print("  Cannot proceed without metadata files.")
        sys.exit(1)

    # --- 2. Load Label Data ---
    print("\n[2/4] Loading Label Data (Train/Val/Test)...")
    try:
        with open(output_dir / files['y_train'], 'rb') as f:
            y_train = pickle.load(f)

        with open(output_dir / files['y_val'], 'rb') as f:
            y_val = pickle.load(f)

        y_test_dur = np.load(output_dir / files['durations_test'])
        y_test_evt = np.load(output_dir / files['events_test'])
        y_test = (y_test_dur, y_test_evt)

        all_labels = {'Train': y_train, 'Validation': y_val, 'Test': y_test}
        print("  ✓ All label files loaded successfully.")

    except FileNotFoundError as e:
        print(f"  ✗ Error loading labels: {e}")
        sys.exit(1)

    # --- 3. Print Label Statistics ---
    print(f"\n[3/4] Cohort and Label Statistics")
    print(f"{'-' * 80}")
    for split, label_data in all_labels.items():
        print(f"\n--- {split.upper()} SET ---")
        print_label_stats(label_data, outcome_mapping)
    print(f"{'-' * 80}")

    # --- 4. Print Tensor Statistics ---
    print(f"\n[4/4] Feature Tensor Shapes and Missingness")
    print(f"{'-' * 80}")
    print(f"Time Discretization Bins: {cuts}")
    print(f"Active Modalities: {modality_info['active_modalities']}")

    # Check Time-series + Static modality
    if 'timeseries' in modality_info['active_modalities']:
        try:
            x_train = np.load(output_dir / files['x_train'])
            m_train = np.load(output_dir / files['train_mask'])
            print("\n--- Time-series + Static (x_...npy) ---")
            print(f"  Train Shape (N, T, F): {x_train.shape}")
            print(f"    N (Stays):         {x_train.shape[0]}")
            print(f"    T (Time Windows):  {x_train.shape[1]}")
            print(f"    F (Features):      {x_train.shape[2]}")

            f_dyn = feature_structure.get('num_dynamic', 'N/A')
            f_stat = feature_structure.get('num_static', 'N/A')
            print(f"    Features (Dynamic):  {f_dyn}")
            print(f"    Features (Static):   {f_stat}")

            missing_pct = (1 - m_train.mean()) * 100
            print(f"  Train Missingness: {missing_pct:.2f}% (percent of 0s in mask)")

        except FileNotFoundError:
            print("\n--- Time-series + Static ---")
            print("  ✓ SKIPPED (files not found)")

    # Check ICD modality
    if 'icd' in modality_info['active_modalities']:
        try:
            x_train_icd = np.load(output_dir / "x_train_icd.npy")
            print("\n--- ICD (x_train_icd.npy) ---")
            print(f"  Train Shape (N, F_icd): {x_train_icd.shape}")
        except FileNotFoundError:
            print("\n--- ICD ---")
            print("  ✓ SKIPPED (files not found)")

    # Check Radiology modality
    if 'radiology' in modality_info['active_modalities']:
        try:
            x_train_rad = np.load(output_dir / "x_train_radiology.npy")
            print("\n--- Radiology (x_train_radiology.npy) ---")
            print(f"  Train Shape (N, F_rad): {x_train_rad.shape}")
        except FileNotFoundError:
            print("\n--- Radiology ---")
            print("  ✓ SKIPPED (files not found, as expected)")

    print(f"{'-' * 80}")
    print("Validation Complete.")


if __name__ == '__main__':
    main()