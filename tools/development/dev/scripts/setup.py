"""
QuantoniumOS Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="quantoniumos",
    version="1.0.0",
    description="Quantum Operating System with RFT Kernel",
    author="QuantoniumOS Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "PyQt5>=5.15.0",
        "numpy>=1.20.0", 
        "scipy>=1.7.0",
        "qtawesome>=1.0.0",
        "pytz>=2021.1"
    ],
    entry_points={
        "console_scripts": [
            "quantonium=quantonium:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Scientists",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.8+",
    ]
)
