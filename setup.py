from setuptools import setup, find_packages


setup(
    name="lochan-eda",
    version="0.1.3",
    author="Lochan Jangid",
    description="An automated, leakage-free data preprocessing pipeline for machine learning.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "pandas==3.0.6",
        "numpy==2.5.3",
        "matplotlib==3.11.2",
        "seaborn==0.13.2",
        "scikit-learn==1.9.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)