# Cool-Earth Navigation Software Package

## Project Overview

Cool-Earth is a technology demonstrator focused on deep-space navigation and climate-engineering mission concepts near the Sun-Earth L1 point. The broader mission concept is to deploy a large solar sail, validate navigation and station-keeping capabilities using solar radiation pressure, and explore the feasibility of a future "solar shade" architecture that can reduce incoming solar flux at Earth.

This repository specifically contains the software workflow for:

- Generating synthetic star images for controlled navigation testing.
- Running optical navigation (OpNav) processing using NASA GIANT.
- Estimating spacecraft attitude from stellar observations.
- Comparing estimated attitude against truth scenarios.

## Why This Exists

In deep space, there is no direct GNSS-style navigation infrastructure (GPS/GLONASS/Galileo) like on Earth. A practical navigation stack typically combines:

- Inertial sensing (IMU).
- Ground-assisted radiometric tracking (for example DSN/Transponder workflows).
- Optical sensing (star-based attitude and image-based navigation).
- Fault-safe coarse sun-direction sensing (Sun Sensor).

This project is the software-side demonstration for the optical part of that stack, including synthetic data generation and algorithm validation loops.

## High-Level Architecture

- `main.py`: End-to-end orchestrator. Loads scenarios, triggers image synthesis and processing, computes errors, and reports results.
- `config.py`: Central configuration (camera intrinsics, distortion, timestamps, magnitude limits, paths).
- `Synthesize_Image.py`: Synthetic image generator using GIANT catalog projection logic; writes FITS files for each scenario.
- `giant_processor.py`: GIANT integration layer for star identification and attitude estimation.
- `visualization.py`: Diagnostic plots for catalog, detections, and matches.
- `data/scenarios.csv`: Truth scenarios (image filename + quaternion values).
- `data/generated/`: Generated FITS outputs.

## Prerequisites

### System

- Python 3.11 (recommended)
- Git
- Optional IDE: VS Code / PyCharm / Spyder

### External Libraries

- [GIANT](https://github.com/nasa/giant) (installed from source in editable mode)
- numpy
- matplotlib
- astropy
- Optional: pandas

### Suggested Environment Setup (Conda)

```bash
conda create -n giant-py311 python=3.11
conda activate giant-py311
pip install numpy matplotlib astropy pandas
```

### Install GIANT

Clone or place the GIANT source folder in your workspace, then install:

```bash
cd giant
pip install -e .
```

Optional GIANT sanity check:

```bash
cd unittests
python -m unittest discover
```

References:

- GIANT installation docs: https://aliounis.github.io/giant_documentation/installation.html
- GIANT getting started: https://aliounis.github.io/giant_documentation/getting_started.html
- NASA software entry: https://software.nasa.gov/software/GSC-18758-1

## Step-by-Step Usage (Main Pipeline)

### 1. Configure the project

Edit `config.py` and verify:

- Camera model and intrinsics.
- Distortion coefficients.
- Observation date/time settings.
- Magnitude threshold.
- Input/output paths.

### 2. Prepare scenario truth cases

Populate `data/scenarios.csv` with:

- `filename`
- `qx`
- `qy`
- `qz`
- `qw`

Each row defines one synthetic image and its truth attitude quaternion.

### 3. Run the pipeline

From the repository root:

```bash
python main.py
```

The run performs:

- Synthetic star image generation (FITS files).
- GIANT star identification.
- Attitude estimation.
- Error computation against truth.
- Optional diagnostic visualization.

### 4. Inspect outputs

- Generated images are written to `data/generated/`.
- Console output includes match counts and attitude error metrics.
- Use `visualization.py` for additional visual checks.

## Synthetic Image Workflow Notes

The synthesis routine leverages GIANT catalog projection to create image-space star locations for an a priori quaternion. Those projected catalog points are then rasterized into synthetic FITS images for controlled testing of the downstream attitude solution.

Scenario batch mode is driven by `data/scenarios.csv`, making it straightforward to benchmark many pointing cases in a single run.

## Project Scope and Evolution

This repository is intended to be an evolving engineering package, not a fixed final product. As the project grows, keep documentation updated for each major capability added (new estimators, hardware interfaces, calibration flows, and validation tooling).

## Navigation Context References

- IMU background: https://en.wikipedia.org/wiki/Inertial_measurement_unit
- Deep Space Network overview: https://www.nasa.gov/communicating-with-missions/dsn/
- Sun sensor background: https://solar-mems.com/blog-news/how-sun-sensors-work/

## Repository Hygiene

- Commit source code, docs, configs, and small essential datasets.
- Do not commit generated large binaries (for example bulk FITS outputs).
- Do not commit environment artifacts (`__pycache__`, `.pyc`, local IDE caches).

