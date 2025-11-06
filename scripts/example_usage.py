Copy

# !/usr/bin/env python3
"""
Example usage of the survival analysis preprocessing pipeline.
Demonstrates both single-risk (eICU) and competing risks (MC-MED) scenarios.
"""

import numpy as np
import pickle
from pathlib import Path


def load_eicu_data(output_dir: str):
    """Load preprocessed eICU data (single risk - ICU mortality)."""
    print("\n" + "=" * 80)
    print("LOADING eICU DATA (Single Risk)")
    print("=" * 80)

    output_path = Path(output_dir)

    # Load training data
    x_train = np.load(output_path / 'x_train_eicu.npy')
    print(f"\n✓ Training data shape: {x_train.shape}")
    print(f"  [N={x_train.shape[0]} patients, T={x_train.shape[1]} windows, F={x_train.shape[2]} features]")

    # Load validation data
    x_val = np.load(output_path / 'x_val_eicu.npy')
    print(f"✓ Validation data shape: {x_val.shape}")

    # Load test data
    x_test = np.load(output_path / 'x_test_eicu.npy')
    print(f"✓ Test data shape: {x_test.shape}")

    # Load labels (training)
    with open(output_path / 'y_train_surv_eicu.p', 'rb') as f:
        durations_train, events_train = pickle.load(f)

    print(f"\n✓ Training labels loaded:")
    print(f"  Durations shape: {durations_train.shape}")
    print(f"  Events shape: {events_train.shape}")
    print(f"  Event rate: {events_train.mean() * 100:.1f}%")
    print(f"  Mean survival time: {durations_train.mean():.1f} hours")

    # Load test labels
    durations_test = np.load(output_path / 'durations_test_eicu.npy')
    events_test = np.load(output_path / 'events_test_eicu.npy')

    print(f"\n✓ Test labels loaded:")
    print(f"  Event rate: {events_test.mean() * 100:.1f}%")

    # Load metadata
    cuts = np.load(output_path / 'cuts_eicu.npy')
    n_bins = np.load(output_path / 'out_features_eicu.npy')

    print(f"\n✓ Metadata:")
    print(f"  Number of time bins: {n_bins}")
    print(f"  Time bin boundaries: {cuts}")

    # Load feature info
    with open(output_path / 'feature_names_eicu.pkl', 'rb') as f:
        feature_structure = pickle.load(f)

    print(f"\n✓ Feature structure:")
    print(f"  Dynamic features: {feature_structure['num_dynamic']}")
    print(f"  Static features: {feature_structure['num_static']}")
    print(f"  Total features: {feature_structure['num_total']}")

    # Load missingness mask
    mask_train = np.load(output_path / 'x_train_eicu_mask.npy')
    missing_pct = (mask_train == 0).mean() * 100
    print(f"\n✓ Missingness: {missing_pct:.2f}%")

    return {
        'x_train': x_train,
        'x_val': x_val,
        'x_test': x_test,
        'y_train': (durations_train, events_train),
        'durations_test': durations_test,
        'events_test': events_test,
        'cuts': cuts,
        'n_bins': n_bins,
        'mask_train': mask_train
    }


def load_mcmed_data(output_dir: str):
    """Load preprocessed MC-MED data (competing risks)."""
    print("\n" + "=" * 80)
    print("LOADING MC-MED DATA (Competing Risks)")
    print("=" * 80)

    output_path = Path(output_dir)

    # Load training data
    x_train = np.load(output_path / 'x_train_mcmed.npy')
    print(f"\n✓ Training data shape: {x_train.shape}")
    print(f"  [N={x_train.shape[0]} patients, T={x_train.shape[1]} windows, F={x_train.shape[2]} features]")

    # Load test data
    x_test = np.load(output_path / 'x_test_mcmed.npy')
    print(f"✓ Test data shape: {x_test.shape}")

    # Load labels
    with open(output_path / 'y_train_surv_mcmed.p', 'rb') as f:
        durations_train, events_train = pickle.load(f)

    print(f"\n✓ Training labels loaded:")
    print(f"  Durations shape: {durations_train.shape}")
    print(f"  Events shape: {events_train.shape}")

    # Competing risk outcome distribution
    outcome_mapping = {
        0: "Censored",
        1: "ICU Admission",
        2: "Hospital Admission",
        3: "Observation"
    }

    print(f"\n✓ Outcome distribution:")
    for outcome_code, outcome_name in outcome_mapping.items():
        count = (events_train == outcome_code).sum()
        pct = count / len(events_train) * 100
        print(f"  {outcome_name} ({outcome_code}): {count} ({pct:.1f}%)")

    # Load test labels
    durations_test = np.load(output_path / 'durations_test_mcmed.npy')
    events_test = np.load(output_path / 'events_test_mcmed.npy')

    # Load metadata
    cuts = np.load(output_path / 'cuts_mcmed.npy')
    n_bins = np.load(output_path / 'out_features_mcmed.npy')
    n_events = np.load(output_path / 'n_events_mcmed.npy')

    print(f"\n✓ Metadata:")
    print(f"  Number of time bins: {n_bins}")
    print(f"  Number of competing events: {n_events}")
    print(f"  Time bin boundaries: {cuts}")

    # Load modality info
    with open(output_path / 'modality_info_mcmed.pkl', 'rb') as f:
        modality_info = pickle.load(f)

    print(f"\n✓ Active modalities: {', '.join(modality_info['active_modalities'])}")

    # Check for additional modalities
    if (output_path / 'x_train_icd.npy').exists():
        x_train_icd = np.load(output_path / 'x_train_icd.npy')
        print(f"\n✓ ICD codes loaded: {x_train_icd.shape}")

    if (output_path / 'x_train_radiology.npy').exists():
        x_train_rad = np.load(output_path / 'x_train_radiology.npy')
        print(f"✓ Radiology embeddings loaded: {x_train_rad.shape}")

    return {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': (durations_train, events_train),
        'durations_test': durations_test,
        'events_test': events_test,
        'cuts': cuts,
        'n_bins': n_bins,
        'n_events': n_events
    }


def demonstrate_data_usage(data: dict, dataset_name: str):
    """Demonstrate how to use the loaded data."""
    print("\n" + "=" * 80)
    print(f"DATA USAGE EXAMPLE - {dataset_name.upper()}")
    print("=" * 80)

    x_train = data['x_train']
    durations_train, events_train = data['y_train']

    print("\n1. Accessing temporal data:")
    print(f"   First patient, first time window: {x_train[0, 0, :5]}")
    print(f"   First patient, last time window: {x_train[0, -1, :5]}")

    print("\n2. Accessing labels:")
    print(f"   First 5 durations: {durations_train[:5]}")
    print(f"   First 5 events: {events_train[:5]}")

    print("\n3. Creating mini-batch for training:")
    batch_size = 32
    batch_x = x_train[:batch_size]
    batch_durations = durations_train[:batch_size]
    batch_events = events_train[:batch_size]
    print(f"   Batch shape: {batch_x.shape}")
    print(f"   Batch labels: {batch_durations.shape}, {batch_events.shape}")

    print("\n4. Discretizing continuous time for neural network:")
    cuts = data['cuts']
    # Find which bin each duration falls into
    discrete_times = np.digitize(durations_train[:5], cuts)
    print(f"   Continuous times: {durations_train[:5]}")
    print(f"   Discrete time bins: {discrete_times}")

    if dataset_name == 'mcmed' and 'n_events' in data:
        print("\n5. Competing risks specific:")
        n_events = data['n_events']
        print(f"   Number of competing outcomes: {n_events}")
        print(f"   You'll need {n_events} output heads in your model")


def main():
    """Main example function."""

    print("=" * 80)
    print("SURVIVAL ANALYSIS PREPROCESSING - USAGE EXAMPLES")
    print("=" * 80)

    # Example 1: Load eICU data (single risk)
    print("\n\nEXAMPLE 1: Single Risk Scenario")
    print("-" * 80)

    eicu_dir = "/path/to/output/eicu_processed"  # Update this

    try:
        eicu_data = load_eicu_data(eicu_dir)
        demonstrate_data_usage(eicu_data, 'eicu')
    except FileNotFoundError:
        print(f"\n⚠ eICU data not found at {eicu_dir}")
        print("  Run: python scripts/run_preprocessing.py --config configs/eicu_config.yaml")

    # Example 2: Load MC-MED data (competing risks)
    print("\n\nEXAMPLE 2: Competing Risks Scenario")
    print("-" * 80)

    mcmed_dir = "/path/to/output/mcmed_processed"  # Update this

    try:
        mcmed_data = load_mcmed_data(mcmed_dir)
        demonstrate_data_usage(mcmed_data, 'mcmed')
    except FileNotFoundError:
        print(f"\n⚠ MC-MED data not found at {mcmed_dir}")
        print("  Run: python scripts/run_preprocessing.py --config configs/mcmed_config.yaml")

    print("\n\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use this data to train your survival model")
    print("2. For competing risks, create separate output heads for each event type")
    print("3. Use time bins (cuts) for discrete-time survival models")
    print("4. Apply missingness masks during training if needed")


if __name__ == '__main__':
    main()