"""
EquiMed-DSS: A Comprehensive Library for Clinical AI Fairness Assessment

This package provides 19 novel metrics for evaluating reliability, equity,
governance, and intersectionality in clinical AI systems.
"""

import os
from pathlib import Path

from setuptools import find_packages, setup

# Read the README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from package
about = {}
with open(
    os.path.join(this_directory, "equimed_dss", "__version__.py"), encoding="utf-8"
) as f:
    exec(f.read(), about)

setup(
    name="equimed-dss",
    version=about["__version__"],
    author="John Muteba",
    author_email="johnmuteba@example.com",  # Update with actual email
    description="A comprehensive Python library for clinical AI fairness assessment with 19 novel metrics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnmuteba/EquiMed_DSS",
    project_urls={
        "Bug Tracker": "https://github.com/johnmuteba/EquiMed_DSS/issues",
        "Documentation": "https://github.com/johnmuteba/EquiMed_DSS#readme",
        "Source Code": "https://github.com/johnmuteba/EquiMed_DSS",
        "Changelog": "https://github.com/johnmuteba/EquiMed_DSS/blob/master/CHANGELOG.md",
    },
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "scipy>=1.7.0,<2.0.0",
        "scikit-learn>=1.0.0,<2.0.0",
        "matplotlib>=3.4.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        "networkx>=2.6.0,<4.0.0",
        "statsmodels>=0.13.0,<1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pylint>=2.17.0",
            "bandit>=1.7.0",
            "twine>=4.0.0",
            "build>=0.10.0",
        ],
        "mysql": [
            "mysql-connector-python>=8.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords=[
        "clinical-ai",
        "fairness",
        "equity",
        "healthcare",
        "machine-learning",
        "bias-detection",
        "medical-ai",
        "ethics",
        "governance",
        "metrics",
        "reliability",
        "calibration",
    ],
    include_package_data=True,
    zip_safe=False,
)
