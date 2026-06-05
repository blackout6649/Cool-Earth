# Cool-Earth Navigation Software Package

## Project Overview

Cool-Earth is a technology demonstrator focused on deep-space navigation and climate-engineering mission concepts near the Sun-Earth L1 point. The broader mission concept is to deploy a large solar sail, validate navigation and station-keeping capabilities using solar radiation pressure, and explore the feasibility of a future "solar shade" architecture that can reduce incoming solar flux at Earth.

This repository contains the software workflow for:

- Generating synthetic star images with known ground-truth orientation.
- Displaying images on a monitor for real-camera capture (hardware-in-the-loop testing).
- Running optical navigation (OpNav) processing using NASA GIANT.
- Estimating spacecraft attitude from stellar observations.
- Comparing estimated attitude against ground truth with statistical analysis.

## Why This Exists

In deep space, there is no GNSS-style infrastructure (GPS/GLONASS/Galileo). A practical navigation stack typically combines:

- Inertial sensing (IMU).
- Ground-assisted radiometric tracking (DSN/Transponder).
- Optical sensing (star-based attitude and image-based navigation).
- Fault-safe coarse sun-direction sensing (Sun Sensor).

This project is the software-side demonstration for the optical part of that stack, covering synthetic data generation, hardware-in-the-loop validation with a real camera, and algorithm benchmarking.

---

## High-Level Architecture

```
config.py                  Central configuration (all parameters in one place)
Synthesize_Image.py        Synthetic star image generator (GIANT catalog projection -> FITS)
display_single_image.py    Monitor presentation (single or slideshow from scenarios.csv)
process_star_image.py      Star-ID + attitude estimation (single or batch mode)
batch_attitude_analysis.py Standalone batch evaluator with statistical summary

data/
  scenarios.csv            Truth scenarios (filename + quaternion)
  generated/               Synthetic FITS outputs
  captured/                Camera-captured FITS frames
  batch_attitude_results.csv  Batch evaluation output
```

---

## Hardware-in-the-Loop Test Setup

### Equipment

| Item | Model / Spec |
|------|------|
| Camera | IDS UI-3240ML-c-HQ (1.3 MP, 1280x1024, 5:4 aspect, USB3) |
| Lens | C-mount with adjustable focal length |
| Display monitor | 1920x1080 laptop screen (Dell Precision 3581) |
| Environment | Dark room, camera on tripod perpendicular to monitor |

### Test Concept

1. Generate synthetic star field images with known quaternion orientations.
2. Display them on a monitor at true 1:1 pixel mapping with black padding.
3. Capture the displayed image with the real camera.
4. Process the captured frame through the star-ID and attitude estimation pipeline.
5. Compare estimated quaternion against ground truth.

This validates the full end-to-end pipeline including real sensor noise, optics, and image processing under controlled conditions.

---

## IDS uEye Camera Settings (Required for Testing)

These settings must be applied in the IDS uEye Cockpit software before capturing test frames:

### Critical Settings

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Pixel format** | Mono12 (preferred) or Mono10/Mono8 | Grayscale only; avoid RGB modes |
| **Resolution** | 1280 x 1024 (native) | Must match calibration and config |
| **Auto Exposure** | OFF | Fixed, repeatable capture conditions |
| **Auto Gain** | OFF | Prevents variable noise floor |
| **Auto White Balance** | OFF (irrelevant in Mono) | Avoid color processing |
| **Gamma** | OFF or 1.0 | Linear sensor response required |
| **Gain** | 0 (minimum) | Lowest noise floor |
| **Black level** | 0–5 (fixed, low) | Stable dark pedestal |
| **Exposure time** | ~15–50 ms (manual) | Tune so brightest stars are 60–80% of full scale |
| **Color saturation** | OFF | Not applicable in Mono |
| **IR filter correction** | OFF | Not needed for monitor display |

### Tuning Guidance

- Target: brightest star peaks around DN 170–220 (8-bit) or equivalent in 12-bit.
- Background should remain low and stable (dark room required).
- Slight lens defocus is acceptable to achieve star FWHM of 2–3 pixels.
- Do NOT change lens focus/aperture after calibration.

### Important

The camera must be calibrated **in the exact same mode** (Mono, same resolution, same lens state, gamma off) that will be used for star image capture. If calibration was performed under different settings, **recalibrate**.

---

## Configuration (`config.py`)

All pipeline parameters are controlled from `config.py`. Key sections:

### Mode Selector

```python
PROC_MODE = 'single'  # 'single' | 'batch'
```

Controls both display and processing behavior:
- `'single'`: Display one image / process one captured frame.
- `'batch'`: Display slideshow from scenarios.csv / process all scenarios with statistics.

### 1) Synthetic Generation

- Image dimensions, focal length, principal point, distortion (zero for display tests).
- Max star magnitude and PSF sigma for rendering.

### 2) Display/Capture Settings

- Monitor selection (auto-detect with interactive prompt).
- Gamma pre-correction for monitor transfer function.
- Series slideshow parameters.

### 3) Processing Pipeline

- Calibrated camera model (focal length, principal point, distortion from OpenCV calibration).
- Star-ID tuning: magnitude limit, tolerance, RANSAC parameters.
- Point-of-interest extraction: threshold (in noise sigma units), blob size limits, centroid window.
- Initial quaternion: can be set manually or auto-resolved from scenarios.csv.
- Ground-truth source for error reporting.
- Console output controls for cleaner terminal logs:
  - `PROC_VERBOSE = False` keeps only key run status/results.
  - `PROC_SUPPRESS_GIANT_RUNTIME_WARNINGS = True` suppresses GIANT/NumPy overflow runtime warning spam.
  - `PROC_SUPPRESS_ALL_RUNTIME_WARNINGS = False` can be enabled only if you want to hide all runtime warnings.

### 4) Batch Evaluation

- Capture directory and filename template.
- Output CSV path for per-scenario results.
- Minimum match count requirement.

---

## Step-by-Step Usage

### Prerequisites

- Python 3.11, Conda environment `giant-py311`
- GIANT installed in editable mode (`pip install -e .` from `giant/` folder)
- numpy, matplotlib, astropy, scipy

```bash
conda create -n giant-py311 python=3.11
conda activate giant-py311
pip install numpy matplotlib astropy scipy
cd giant && pip install -e .
```

### 1. Define test scenarios

Edit `data/scenarios.csv`:

```csv
filename,qx,qy,qz,qw
Scenario_01.fits,-0.704416026,0.061628417,0,0.707106781
Scenario_02.fits,-0.704416026,0.061628417,0.005,0.707106781
```

### 2. Generate synthetic images

Set `PROC_MODE = 'batch'` in `config.py`, then:

```bash
python Synthesize_Image.py
```

Outputs FITS files to `data/generated/`.

### 3. Display images for camera capture

```bash
python display_single_image.py
```

- In batch mode: slideshow with N/P keys to navigate, Q to quit.
- In single mode: shows `DISPLAY_IMAGE_PATH`.
- Script auto-detects monitors and prompts for selection.

### 4. Capture frames

- Use IDS uEye Cockpit with settings above.
- Save captured frames to `data/captured/`.
- Naming convention: match scenario stem (e.g., `Scenario_01.fits`) or configure `BATCH_CAPTURE_FILENAME_TEMPLATE`.

### 5. Process captured images

```bash
python process_star_image.py
```

- In single mode: processes `PROCESS_IMAGE_PATH`, reports attitude error vs GT.
- In batch mode: runs `batch_attitude_analysis.py`, processes all scenarios, writes results CSV, prints summary statistics.
- Runtime warning note: warning lines such as GIANT gaussians overflow/jacobian messages are generated by GIANT internals and are filtered by default via `PROC_SUPPRESS_GIANT_RUNTIME_WARNINGS`.

### 6. Review results

Batch output example:

```
=== Batch Summary ===
Total scenarios: 4
Successful solutions: 4
Failed/Skipped: 0
Mean attitude error: 0.123268 deg
Std attitude error:  0.056350 deg
Median attitude error: 0.100565 deg
Min/Max attitude error: 0.073809 / 0.218133 deg
```

Detailed per-scenario results are written to `data/batch_attitude_results.csv`.

---

## Current Performance (Monitor-Based HWITL) 05.06.26

| Metric | Value |
|--------|-------|
| Convergence rate | 4/4 (100%) |
| Mean attitude error | 0.123 deg (7.4 arcmin) |
| Std | 0.056 deg |
| Min | 0.074 deg |
| Max | 0.218 deg |

Note: Monitor-based testing has an inherent error floor due to finite display distance (perspective warp), monitor gamma/nonlinearity, and calibration mode mismatch. These results validate end-to-end pipeline functionality.

---

## Known Limitations & Next Steps

1. **Recalibrate in capture mode** — calibrate with same Mono/resolution/lens/gamma settings used during capture.
2. **Increase star detection count** — tune exposure and POI threshold for more matched stars per frame.
3. **Reduce perspective effects** — increase camera-to-monitor distance.
4. **Validate with perturbed initial quaternion** — confirm robustness to poor a priori estimates.
5. **Future: real-time capture loop** — automate display/capture/process cycle.

---

## References

- GIANT documentation: https://aliounis.github.io/giant_documentation/
- GIANT source: https://github.com/nasa/giant
- NASA software entry: https://software.nasa.gov/software/GSC-18758-1
- IDS uEye documentation: https://www.ids-imaging.us/manuals/ids-software-suite/ueye-cockpit/en/index.html
- IMU background: https://en.wikipedia.org/wiki/Inertial_measurement_unit
- Deep Space Network: https://www.nasa.gov/communicating-with-missions/dsn/

---

## Repository Hygiene

- Commit source code, docs, configs, and small essential datasets.
- Do not commit generated FITS bulk outputs or captured frames (add to `.gitignore`).
- Do not commit environment artifacts (`__pycache__`, `.pyc`, IDE caches).

