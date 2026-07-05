# Cool-Earth-NAV (develop_RT_Raspberry)

## Purpose

This sub-module now targets real-time Raspberry execution for optical attitude updates only.

The runtime path is intentionally simple:

1. A state machine loop calls `_capture_frame(frame_index)` every NAVIGATION iteration.
2. `_capture_frame(...)` contains two sections: a future real camera capture stub and a current CSV-based fallback that selects a frame from `data/scenarios.csv`/`data/captured/`.
3. The OpNav pipeline processes that frame through GIANT star identification.
4. Navigation quality is mapped to status values 1/2/3 for state transitions.

Legacy synthetic generation, monitor display, and offline batch utilities were removed from this branch path.

## Current Runtime Files

- `config.py`: central runtime and camera tuning configuration.
- `process_star_image.py`: single-image OpNav pipeline (`run_single_image_pipeline`).
- `../StateMachine.py`: mission state machine that calls OpNav in NAVIGATION state.
- `data/scenarios.csv`: ordered scenario list and reference quaternions.
- `data/captured/`: pre-saved FITS frames consumed by the state machine.

## Navigation Status Convention

The NAVIGATION loop uses fixed semantics:

- `1`: bad (pipeline failed).
- `2`: degraded (pipeline ran but not enough matched stars).
- `3`: good (pipeline solved successfully).

In `StateMachine.py`, NAVIGATION repeats until status becomes `3`, then transitions to ADCS.

## Data Contract

`data/scenarios.csv` is the frame source of truth. Required columns:

```csv
filename,qx,qy,qz,qw
Scenario_01.fits,-0.704416026,0.061628417,0.0,0.707106781
Scenario_02.fits,-0.704416026,0.061628417,0.005,0.707106781
```

- `filename` is resolved against `data/captured/`.
- Quaternion columns are used for scenario context and optional initialization behavior.

## Configuration Guide

All behavior is controlled in `config.py`.

Recommended real-time values:

```python
PROC_MODE = 'single'
PROC_INTERACTIVE_PROMPTS = False
PROC_ENABLE_PLOTS = False
PROC_MIN_MATCHES = 3
PROC_USE_SCENARIO_CSV_INITIAL_QUAT = False
PROC_REPORT_ATTITUDE_ERROR = False
```

### Boresight (Optional)

Enable boresight correction only if a validated alignment quaternion is available.

```python
PROC_USE_BORESIGHT = True
PROC_BORESIGHT_Q_CB = [qx, qy, qz, qw]
```

Correction model used by the pipeline:

$$
q^B_I = (q^C_B)^{-1} q^C_I
$$

## Setup

From repository root:

```bash
cd giant
pip install -e .
cd ..
pip install numpy scipy astropy matplotlib
```

## How To Run

### Full State Machine Path (recommended for Raspberry flow)

From repository root:

```bash
python StateMachine.py
```

During NAVIGATION, each cycle prints a placeholder capture line:

```text
Captured frame: <.../Cool-Earth-NAV/data/captured/Scenario_XX.fits>
```

Then one OpNav attempt is executed on that frame. If capture returns no frame, that iteration is skipped and NAVIGATION retries.

### Direct OpNav Single-Image Test

From `Cool-Earth-NAV`:

```bash
python process_star_image.py
```

If no explicit `image_path` argument is passed, the script uses `PROCESS_IMAGE_PATH` from `config.py`.

## Notes On Removed Utilities

The following scripts/files are not part of this real-time branch flow and were removed from active use:

- synthetic generator
- image display tool
- PNG to FITS batch converter
- batch attitude analyzer
- boresight estimation utility
- generated FITS output folder and batch result artifacts

If any local tooling still references them, update those calls to the state-machine + single-image pipeline shown above.

## Troubleshooting

- `ModuleNotFoundError: giant...`
  - Install GIANT in editable mode from the local `giant/` folder.
- NAVIGATION stuck in status `1` or `2`
  - Verify scenario file names match files in `data/captured/`.
  - Reduce POI threshold or adjust exposure for more matched stars.
  - Confirm calibrated camera intrinsics in `config.py`.
- Direct run fails on missing default image
  - Update `PROCESS_IMAGE_PATH` to an existing FITS under `data/captured/`.

## Next Planned Increment

Implement the real camera section inside `_capture_frame(...)` (currently a stub) and add timestamped frame metadata while keeping the same NAVIGATION status interface.

