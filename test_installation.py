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


def test_survival_benchmark():
    print("\\nTesting survival_benchmark package...")

    try:
        import survival_benchmark
        print(f"  Package version: {survival_benchmark.__version__}")

        from survival_benchmark import PreprocessingPipeline
        print("  PreprocessingPipeline imported")

        from survival_benchmark.data import BaseDataLoader, eICUDataLoader
        print("  Data loaders imported")

        from survival_benchmark.preprocessing import SurvivalLabelsProcessor
        print("  Preprocessing modules imported")

        print("\\nPackage structure verified")
        return True

    except ImportError as e:
        print(f"\\nFailed to import survival_benchmark: {e}")
        print("\\nTry installing with: pip install -e .")
        return False


def main():
    print("=" * 60)
    print("SURVIVAL BENCHMARK - INSTALLATION TEST")
    print("=" * 60)

    success = True

    if not test_imports():
        success = False

    if not test_survival_benchmark():
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