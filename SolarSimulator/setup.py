from setuptools import setup, find_packages
import os

setup(
    name="SolarSimulator",
    version="0.1.0",
    author="Brian Epstein",
    author_email="bepstein8@gatech.edu",
    description="Solar-powered seaplane simulation for whale observation missions",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/bantno/Solar-Simulator",
    packages=find_packages(include=["SolarSimulator", "SolarSimulator.*"]),
    install_requires=[
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "scipy>=1.13.1",
        "matplotlib>=3.9.0",
        "pvlib>=0.10.5",
        "ephem>=4.1.5",
        "suntime>=1.2.5",
        "timezonefinder>=6.2.0",
        "pytz>=2024.1",
        "tqdm>=4.66.4",
        "trimesh>=4.4.0",
        "statsmodels>=0.14.2",
        "shapely>=1.8.5",
        "openmeteo-requests>=1.1.0",
        "requests-cache>=1.1.0",
        "retry-requests>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "solar-sim=SolarSimulator.Scripts.run:main",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    package_data={
        "SolarSimulator": ["Data/*", "Figures/*"],
    },
    include_package_data=True,
)
