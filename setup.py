from setuptools import setup, find_packages

setup(
    name="EquiMed_DSS",
    version="0.1.0",
    description="A Python library for 19 novel clinical fairness metrics.",
    author="EquiMed Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "networkx",
        "statsmodels"
    ],
    python_requires=">=3.8",
)
