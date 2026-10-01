from setuptools import setup, find_packages

setup(
    name="hydragnn-gfm-finetuning",
    version="1.0.0",
    description="Fine-tuning utilities for the HydraGNN Graph Foundation Model ensemble for materials science",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["hydragnn_gfm_finetuning"]),
)
