"""
File: config.py
Description: Configuration constants.
"""
from datetime import datetime, timedelta
import os

# --- Camera Model ---
CAM_FOCAL_LENGTH = 14774.74
CAM_CENTER_X = 511.5
CAM_CENTER_Y = 511.5
IMG_RES = 1024

# Distortion [k1, k2, p1, p2, k3]
DISTORTION_COEFFS = [0.023, 0.002, 0.8, 0.05, 0]
MAX_MAGNITUDE = 9
SYNTH_PSF_SIGMA = 0.1  # For synthetic image generation (if needed)

# --- Time ---
# J2000 epoch and observation offset in seconds
J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
OBSERVATION_SECONDS = 757339269.184
OBSERVATION_DATE = J2000_EPOCH + timedelta(seconds=OBSERVATION_SECONDS)

# --- Paths ---
# Use relative paths so it works on any computer
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'generated')
TRUTH_FILE = os.path.join(DATA_DIR, 'scenarios.csv')