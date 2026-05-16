"""
File: Synthesize_Image.py
Description:
    Modular pipeline for Synthetic Star Image Generation.

    Structure:
    1. generate_raw_star_image(): Projects stars -> FITS file.
    2. apply_image_effects():     FITS -> NumPy Array (with optional noise).
    3. visualize_result():        NumPy Array + Quaternion -> Plot with Blue Circles.
    4. run_batch_from_csv():      Loads a CSV and runs the pipeline for multiple scenarios.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# GIANT Imports
from giant.camera import Camera
from giant.camera_models import BrownModel
from giant.image import OpNavImage
from giant.stellar_opnav.stellar_class import StellarOpNav, StellarOpNavOptions
from giant.catalogs.gaia import Gaia
from giant.rotations import Rotation
import config

# --- CONFIGURATION ---
SAVE_DIR = os.path.join(config.DATA_DIR, 'generated')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# =============================================================================
# PART 1: CREATE & SAVE (The Generator)
# =============================================================================
def generate_raw_star_image(target_quaternion, filename="synth_star_field.fits"):
    """
    Generates a star image from a quaternion and saves it as a FITS file.
    """
    print(f"--- Generating Raw FITS for Q={np.round(target_quaternion, 3)} ---")

    # A. Setup Camera (match processing model parameters)
    k1, k2, p1, p2, k3 = config.DISTORTION_COEFFS
    model = BrownModel(
        kx=config.CAM_FOCAL_LENGTH, ky=config.CAM_FOCAL_LENGTH,
        px=config.CAM_CENTER_X, py=config.CAM_CENTER_Y,
        k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
        n_rows=config.IMG_RES, n_cols=config.IMG_RES
    )
    camera_obj = Camera(model=model, name='Synth_Cam')

    # B. Dummy File Trick (Required for GIANT memory init)
    temp_dummy = "temp_dummy_black.fits"
    if not os.path.exists(temp_dummy):
        fits.writeto(temp_dummy, np.zeros((config.IMG_RES, config.IMG_RES), dtype=np.uint8), overwrite=True)

    # C. Initialize GIANT
    opnav_image = OpNavImage(temp_dummy, observation_date=config.OBSERVATION_DATE)
    opnav_image.rotation_inertial_to_camera = Rotation(target_quaternion)

    # D. Run Identification (Math only)
    opts = StellarOpNavOptions()
    opts.star_id_options.catalog = Gaia()
    opts.star_id_options.max_magnitude = config.MAX_MAGNITUDE

    sopnav = StellarOpNav(camera_obj, options=opts)
    sopnav.add_images([opnav_image])

    # See dimmer stars
    sopnav.star_id.max_magnitude = config.MAX_MAGNITUDE

    sopnav.id_stars()

    # E. Extract Data & Rasterize
    projected_points = sopnav.queried_catalog_image_points[0]
    star_records = sopnav.queried_catalog_star_records[0]

    synth_image = np.zeros((config.IMG_RES, config.IMG_RES), dtype=np.float32)

    if (
        projected_points is not None and
        star_records is not None and
        getattr(projected_points, 'ndim', 0) == 2 and
        projected_points.shape[1] > 0
    ):
        xs = projected_points[0, :]
        ys = projected_points[1, :]
        mags = star_records['mag'].values

        print(f"   Rendering {len(xs)} stars with subpixel PSF...")

        # PSF parameters
        psf_sigma = getattr(config, 'SYNTH_PSF_SIGMA', 0.1)  # px, fallback to 0.1 if not in config
        psf_half_size = int(np.ceil(3 * psf_sigma))  # truncate at 3 sigma

        for x, y, mag in zip(xs, ys, mags):
            if 0 <= x < config.IMG_RES and 0 <= y < config.IMG_RES:
                flux = float(10 ** (-0.4 * mag))
                # Subpixel-centered PSF
                x0 = x
                y0 = y
                x_min = max(0, int(np.floor(x0 - psf_half_size)))
                x_max = min(config.IMG_RES, int(np.ceil(x0 + psf_half_size + 1)))
                y_min = max(0, int(np.floor(y0 - psf_half_size)))
                y_max = min(config.IMG_RES, int(np.ceil(y0 + psf_half_size + 1)))

                x_grid, y_grid = np.meshgrid(
                    np.arange(x_min, x_max),
                    np.arange(y_min, y_max)
                )
                psf = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * psf_sigma ** 2))
                psf_sum = np.sum(psf)
                if psf_sum > 0:
                    psf *= flux / psf_sum  # Normalize PSF to star flux
                    synth_image[y_min:y_max, x_min:x_max] += psf
    else:
        print("   No stars projected into FOV - saving blank image.")

    # Scale deterministic synthetic image to uint16 sensor-like range.
    max_val = float(np.max(synth_image))
    if max_val > 0:
        synth_image = (synth_image / max_val * 65535.0).astype(np.uint16)
    else:
        synth_image = synth_image.astype(np.uint16)

    # G. Save FITS with Header
    save_path = os.path.join(SAVE_DIR, filename)
    hdu = fits.PrimaryHDU(synth_image)

    # Add metadata to the FITS header
    hdu.header['Q_X'] = target_quaternion[0]
    hdu.header['Q_Y'] = target_quaternion[1]
    hdu.header['Q_Z'] = target_quaternion[2]
    hdu.header['Q_W'] = target_quaternion[3]
    hdu.header['COMMENT'] = "Generated by GIANT Synthetic Pipeline"

    hdu.writeto(save_path, overwrite=True)
    print(f"   Saved FITS to: {save_path}")

    return save_path

# =============================================================================
# PART 2: VISUALIZE (The Viewer)
# =============================================================================
def visualize_result(image_data, quaternion):
    """
    Overlays 'Blue Circles' (Catalog Truth) on top of the image.
    """
    print("--- Visualizing Image (with noise) ---")

    # 1. Re-Calculate Star Positions from Quaternion
    model = BrownModel(
        kx=config.CAM_FOCAL_LENGTH, ky=config.CAM_FOCAL_LENGTH,
        px=config.CAM_CENTER_X, py=config.CAM_CENTER_Y,
        n_rows=config.IMG_RES, n_cols=config.IMG_RES
    )
    camera_obj = Camera(model=model, name='Vis_Cam')

    # Reuse dummy file for init
    temp_dummy = "temp_dummy_black.fits"
    if not os.path.exists(temp_dummy):
        fits.writeto(temp_dummy, np.zeros((config.IMG_RES, config.IMG_RES), dtype=np.uint8), overwrite=True)

    opnav_image = OpNavImage(temp_dummy, observation_date=config.OBSERVATION_DATE)
    opnav_image.rotation_inertial_to_camera = Rotation(quaternion)

    sopnav = StellarOpNav(camera_obj, options=StellarOpNavOptions())
    sopnav.star_id_options.catalog = Gaia()
    sopnav.add_images([opnav_image])
    sopnav.star_id.max_magnitude = config.MAX_MAGNITUDE
    sopnav.id_stars()

    cat_points = sopnav.queried_catalog_image_points[0]

    # 2. Plot
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # A. The Image (Background)
    plt.imshow(np.log10(image_data.astype(float) + 1), cmap='gray', origin='upper')

    # B. The Blue Circles (Truth)
    if cat_points is not None:
        in_fov = (
            (cat_points[0, :] >= 0) & (cat_points[0, :] <= config.IMG_RES) &
            (cat_points[1, :] >= 0) & (cat_points[1, :] <= config.IMG_RES)
        )
        points = cat_points[:, in_fov]

        plt.scatter(points[0], points[1],
                    s=80, edgecolors='cyan', facecolors='none',
                    linewidth=1.5, label='Catalog Truth')
        print(f"   Projected {points.shape[1]} catalog stars onto visualization.")

    plt.title(f"Synthetic Result\nQ: {np.round(quaternion, 3)}")
    plt.xlabel("X [pix]")
    plt.ylabel("Y [pix]")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# =============================================================================
# PART 4: BATCH PROCESSING (CSV Loader)
# =============================================================================
def run_batch_from_csv(csv_path):
    """
    Reads a CSV file and generates an image for every row.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found: {csv_path}")
        return

    print(f"\n=== Starting Batch Processing from: {csv_path} ===\n")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        count = 0
        for row in reader:
            filename = row['filename']
            try:
                q = [
                    float(row['qx']),
                    float(row['qy']),
                    float(row['qz']),
                    float(row['qw'])
                ]
            except ValueError as e:
                print(f"[SKIP] Invalid data in row for {filename}: {e}")
                continue

            print(f"Processing Scene {count+1}: {filename}")

            # Generate directly from configured synthetic parameters without post effects.
            generate_raw_star_image(q, filename=filename)

            count += 1

    print(f"\n=== Batch Complete. Generated {count} images in {SAVE_DIR} ===")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # --- MODE SELECTION ---
    # Set this to True to run the batch from CSV, False to run single test
    RUN_BATCH_MODE = True

    CSV_FILENAME = "scenarios.csv"
    CSV_PATH = os.path.join(config.DATA_DIR, CSV_FILENAME)

    if RUN_BATCH_MODE:
        # Create a sample CSV if it doesn't exist (for demonstration)
        if not os.path.exists(CSV_PATH):
            print("CSV not found, please create 'scenarios.csv' or ensure it exists.")
        else:
            run_batch_from_csv(CSV_PATH)

    else:
        # Single Run Mode (Legacy)
        q_orion = [-0.704416026, 0.061628417, 0.0, 0.707106781]
        fits_path = generate_raw_star_image(q_orion, filename="Single_Scenario.fits")
        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data
        visualize_result(image_data, q_orion)