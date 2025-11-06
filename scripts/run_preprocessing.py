#!/usr/bin/env python3

import argparse
import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from survival_benchmark.preprocessing.pipeline import PreprocessingPipeline


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Run EHR survival analysis preprocessing pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    base_dir = Path(config['dataset']['base_dir'])
    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset base directory not found: {base_dir}")

    print(f"\\nInitialising preprocessing pipeline for: {config['dataset']['name']}")
    pipeline = PreprocessingPipeline(config)

    try:
        pipeline.run()
        print("\\n✓ Preprocessing completed successfully!")

    except Exception as e:
        print(f"\\n✗ Preprocessing failed with error:")
        print(f"  {type(e).__name__}: {e}")

        if args.verbose:
            import traceback
            traceback.print_exc()

        sys.exit(1)


if __name__ == '__main__':
    main()