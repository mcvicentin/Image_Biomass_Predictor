from setuptools import setup, find_packages

setup(
    name="biomass_prediction",
    version="0.1.0",
    description="Deep learning model for predicting pasture biomass from drone images.",
    author="Marcelo Ciani Vicentin",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "tqdm",
        "matplotlib",
    ],
    python_requires=">=3.8",
)

