from setuptools import setup, find_packages

setup(
    name="EquiMed_DSS",
    version="0.1.0",
    description="A Python library for 19 novel clinical fairness metrics.",
    author="EquiMed Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "networkx>=2.6.0",
        "statsmodels>=0.13.0"
    ],
    python_requires=">=3.8",
)
