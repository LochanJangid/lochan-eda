from typing import List
from pathlib import Path
from setuptools import setup, find_packages

def require_packages(filepath: str) -> List[str]:
    return [pckg.replace("\n", "").strip() for pckg in open(filepath).readlines() if "-e ." not in pckg]

setup(
    name="lochan-eda",
    version="0.2.0",
    author="Lochan Jangid",
    description="An automated, leakage-free data preprocessing pipeline for machine learning.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=require_packages("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)