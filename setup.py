#!/usr/bin/env python
"""Setup script for AD-DAN package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ad-dan",
    version="1.0.0",
    author="John K. Arthur",
    author_email="jkinarthur@example.com",
    description="Affect–Decision Dual Alignment Network for inconsistency detection in dialogue",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jkinarthur/modeling-affective-decision",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "tensorboard>=2.11.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ad-dan-train=train:main",
            "ad-dan-eval=evaluate:main",
        ]
    },
)
