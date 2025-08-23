#!/usr/bin/env python3
"""
NeuralScript Programming Language
A high-performance language for data science, machine learning, and scientific computing.
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    raise RuntimeError("NeuralScript requires Python 3.8 or later")

# Read version from __init__.py
here = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(here, "compiler", "__init__.py")
version = {}
if os.path.exists(version_file):
    with open(version_file) as f:
        exec(f.read(), version)
else:
    version["__version__"] = "0.1.0-alpha"

# Read README
readme_file = os.path.join(here, "README.md")
with open(readme_file, "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="neuralscript",
    version=version.get("__version__", "0.1.0-alpha"),
    description="A high-performance programming language for data science, ML, and scientific computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="xwest",
    author_email="dev@neuralscript.org",
    url="https://github.com/xwest/neuralscript",
    project_urls={
        "Bug Reports": "https://github.com/xwest/neuralscript/issues",
        "Source": "https://github.com/xwest/neuralscript",
        "Documentation": "https://docs.neuralscript.org",
    },
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        # Core has no external dependencies for from-scratch implementation
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0", 
            "pytest-benchmark>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
        ],
        "llvm": [
            "llvmlite>=0.40.0",
        ],
        "gpu": [
            "numba>=0.57.0",
            "cupy>=12.0.0",
        ],
        "scientific": [
            "numpy>=1.24.0",
            "scipy>=1.10.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.2.2",
            "myst-parser>=2.0.0",
        ],
        "all": [
            "pytest>=7.4.0", "pytest-cov>=4.1.0", "pytest-benchmark>=4.0.0",
            "black>=23.3.0", "flake8>=6.0.0", "mypy>=1.4.0", "isort>=5.12.0",
            "llvmlite>=0.40.0",
            "numba>=0.57.0", "cupy>=12.0.0", 
            "numpy>=1.24.0", "scipy>=1.10.0",
            "sphinx>=7.1.0", "sphinx-rtd-theme>=1.2.2", "myst-parser>=2.0.0",
            "rich>=13.4.0", "click>=8.1.0", "colorama>=0.4.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "nsc=compiler.cli:main",           # NeuralScript Compiler
            "nspm=tools.nspm.cli:main",        # NeuralScript Package Manager
            "ns-repl=tools.repl:main",         # Interactive REPL
            "ns-debug=tools.debugger.cli:main", # Debugger
            "ns-profile=tools.profiler.cli:main", # Profiler
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "programming-language", "compiler", "machine-learning", "data-science",
        "scientific-computing", "tensor", "neural-networks", "automatic-differentiation",
        "performance", "native-code", "gpu-computing", "mathematical-computing"
    ],
    zip_safe=False,
    platforms=["Windows", "Linux", "macOS"],
)
