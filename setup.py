from setuptools import setup, find_packages
from pathlib import Path

readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

requirements_file = Path(__file__).parent / 'requirements.txt'
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

setup(
    name='SurvBench',
    version='0.1.0',
    author='Munib Mesinovic',
    author_email='munib.mesinovic@jesus.ox.ac.uk',
    description='Standardised preprocessing pipeline for multi-modal survival analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/SurvBench',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)