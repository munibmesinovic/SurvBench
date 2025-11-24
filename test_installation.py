#!/usr/bin/env python3

import sys


def test_imports():
    print("Testing package imports...")

    required_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
        ('pycox', 'pycox'),
        ('torch', 'torch'),
        ('yaml', 'pyyaml'),
        ('tqdm', 'tqdm'),
    ]

    failed = []
    for module, package in required_packages:
        try:
            __import__(module)
            print(f"  Yes {module}")
        except ImportError:
            print(f"  No {module} (install with: pip install {package})")
            failed.append(package)

    if failed:
        print(f"\\nMissing packages: {', '.join(failed)}")
        print(f"Install them with: pip install {' '.join(failed)}")
        return False

    print("\\nAll required packages installed")
    return True


def test_survbench():
    print("\\nTesting SurvBench package...")

    try:
        sys.path.insert(0, '.')

        from preprocessing.pipeline import PreprocessingPipeline
        print("  PreprocessingPipeline imported")

        from data.base_loader import BaseDataLoader
        from data.eicu_loader import eICUDataLoader
        print("  Data loaders imported")

        from preprocessing.labels import SurvivalLabelsProcessor
        print("  Preprocessing modules imported")

        print("\\nPackage structure verified")
        return True

    except ImportError as e:
        print(f"\\nFailed to import SurvBench modules: {e}")
        print("\\nMake sure you're in the SurvBench directory")
        return False


def main():
    print("=" * 60)
    print("SURVBENCH - INSTALLATION TEST")
    print("=" * 60)

    success = True

    if not test_imports():
        success = False

    if not test_survbench():
        success = False

    print("\\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED!")
        print("\\nYou're ready to use the preprocessing pipeline:")
        print("  python scripts/run_preprocessing.py --config configs/eicu_config.yaml")
    else:
        print("SOME TESTS FAILED")
        print("\\nPlease install missing dependencies.")
    print("=" * 60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())