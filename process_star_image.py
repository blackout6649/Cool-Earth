import numpy as np
from datetime import datetime, timedelta
import sys
import traceback
import matplotlib.pyplot as plt
import os
import config

# GIANT Imports
from giant.camera import Camera
from giant.camera_models import PinholeModel, BrownModel
from giant.image import OpNavImage
from giant.stellar_opnav.stellar_class import StellarOpNav, StellarOpNavOptions
from giant.catalogs.gaia import Gaia
from giant.rotations import Rotation
from scipy.spatial.distance import cdist


def safe_get_count(points_array):
    """
    Safely determine the number of points (columns) in a numpy array.
    """
    if points_array is None:
        return 0
    if isinstance(points_array, np.ndarray):
        if points_array.ndim == 2:
            return points_array.shape[1]
        elif points_array.ndim == 1:
            if points_array.size == 2:
                return 1
            return 0
    return 0


# =========================================================================
# NEW VISUALIZATION FUNCTION
# =========================================================================
def plot_results(opnav_image, sopnav, img_width=1024, img_height=1024):
    """
    Visualizes the Raw Detections vs. Catalog Projections.
    """
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # 1. Plot Background Image (if available)
    #    Note: Fits images often need contrast stretching, but this gives context.
    if hasattr(opnav_image, 'data') and opnav_image.data is not None:
        # Using log scale for visibility usually helps with star images
        plt.imshow(np.log10(np.clip(opnav_image.data, 1, None)), cmap='gray', origin='upper')
    else:
        # If no image data, black background
        ax.set_facecolor('black')
        plt.xlim(0, img_width)
        plt.ylim(img_height, 0)  # Invert Y to match image coordinates

    # 2. Get Data points
    raw_points = sopnav.extracted_image_points[0]  # Detected (2xN)
    cat_points = sopnav.queried_catalog_image_points[0]  # Catalog (2xN)

    # 3. Plot Catalog Stars (Expected) - BLUE CIRCLES
    if cat_points is not None and cat_points.ndim == 2:
        # Filter to FOV for cleaner plotting
        in_fov = (
                (cat_points[0, :] >= 0) & (cat_points[0, :] <= img_width) &
                (cat_points[1, :] >= 0) & (cat_points[1, :] <= img_height)
        )
        plt.scatter(cat_points[0, in_fov], cat_points[1, in_fov],
                    s=80, edgecolors='cyan', facecolors='none', label='Catalog (A Priori)', linewidth=1.5)

    # 4. Plot Detected Spots (Measured) - RED CROSSES
    if raw_points is not None and raw_points.ndim == 2:
        plt.scatter(raw_points[0, :], raw_points[1, :],
                    c='red', marker='x', s=60, label='Detected Spots')

    # 5. Plot Matches (if any) - GREEN CONNECTORS
    matched_extracted = sopnav.matched_extracted_image_points[0]
    matched_catalog = sopnav.matched_catalog_image_points[0]

    if matched_extracted is not None and matched_catalog is not None:
        if matched_extracted.ndim == 2 and matched_catalog.ndim == 2:
            plt.scatter(matched_extracted[0, :], matched_extracted[1, :],
                        c='lime', marker='+', s=100, label='Successfully Matched')

            # Draw lines connecting match pairs to visualize residual errors
            for i in range(matched_extracted.shape[1]):
                plt.plot([matched_extracted[0, i], matched_catalog[0, i]],
                         [matched_extracted[1, i], matched_catalog[1, i]],
                         c='lime', linestyle='--', alpha=0.5)

    plt.title(f"Star ID Results\n{opnav_image.observation_date} UTC")
    plt.xlabel("X Pixel")
    plt.ylabel("Y Pixel")
    plt.legend(loc='upper right')
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # =========================================================================
    # 1. Setup Camera Model
    # =========================================================================
    effective_focal_length_pixels = 14774.74

    # Note: Usually you only need one model. BrownModel overwrites PinholeModel here.
    model = BrownModel(
        kx=effective_focal_length_pixels,
        ky=effective_focal_length_pixels,
        px=511.5,
        py=511.5,
        n_rows=1024,
        n_cols=1024
    )

    camera_obj = Camera(
        model=model,
        name='Synthetic_L1_Camera'
    )

    # =========================================================================
    # 2. Load Image & Time
    # =========================================================================
    image_path = os.path.join(config.IMAGE_DIR, 'Scenario_02.fits')
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)
    obs_time = j2000_epoch + timedelta(seconds=757339269.184)

    print(f"Loading image: {image_path}")
    try:
        opnav_image = OpNavImage(image_path, observation_date=obs_time)
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find {image_path}. Run MATLAB script first.")
        return

    # --- A Priori Attitude Setup ---
    initial_quaternion = [-0.704416026, 0.061628417, 0.000000000, 0.707106781]
    print(f"A Priori Attitude: {initial_quaternion}")
    opnav_image.rotation_inertial_to_camera = Rotation(initial_quaternion)

    # =========================================================================
    # 3. Configure Navigation
    # =========================================================================
    sopnav_options = StellarOpNavOptions()
    try:
        sopnav_options.star_id_options.catalog = Gaia()
    except Exception as e:
        print(f"[ERROR] Catalog Init Failed: {e}")
        return

    sopnav = StellarOpNav(camera_obj, options=sopnav_options)
    sopnav.add_images([opnav_image])

    # --- TUNING ---
    sopnav.star_id.max_magnitude = 10.0
    # sopnav.star_id.tolerance = 200.0 # Uncomment if large offset expected
    sopnav.star_id.ransac_tolerance = 10.0
    sopnav.star_id.max_combos = 0

    print("\n--- Starting Processing ---")

    # A. Identify Stars
    print("1. Identifying stars...")
    try:
        sopnav.id_stars()

        # Data retrieval for debug printing
        raw_points = sopnav.extracted_image_points[0]
        projected_catalog = sopnav.queried_catalog_image_points[0]
        matched_points = sopnav.matched_extracted_image_points[0]
        num_raw = safe_get_count(raw_points)
        num_matched = safe_get_count(matched_points)

        # --- VISUALIZATION CALL ---
        print("Generating visual comparison...")
        plot_results(opnav_image, sopnav)
        # --------------------------

        num_cat_in_fov = 0
        if projected_catalog is not None and projected_catalog.ndim == 2:
            in_fov_mask = (
                    (projected_catalog[0, :] >= 0) & (projected_catalog[0, :] <= 1024) &
                    (projected_catalog[1, :] >= 0) & (projected_catalog[1, :] <= 1024)
            )
            num_cat_in_fov = np.sum(in_fov_mask)
            projected_catalog_in_fov = projected_catalog[:, in_fov_mask]

        print(f"\n--- DEBUG INFO ---")
        print(f"Raw spots detected: {num_raw}")
        print(f"Catalog stars in FOV: {num_cat_in_fov}")
        print(f"Total Matched Stars: {num_matched}")

        # Calculate distances for console output
        if num_raw > 0 and num_cat_in_fov > 0 and raw_points.ndim == 2:
            distances = cdist(raw_points.T, projected_catalog_in_fov.T, metric='euclidean')
            print(f"Overall min distance: {np.min(distances):.2f} pixels")

        if num_matched < 3:
            print("[WARNING] Not enough matched stars.")
            return

    except Exception as e:
        print(f"[ERROR] Star Identification failed: {e}")
        traceback.print_exc()
        return

    # B. Estimate Attitude
    print("\n2. Estimating Attitude...")
    try:
        sopnav.estimate_attitude()

        if opnav_image.pointing_post_fit:
            q = opnav_image.rotation_inertial_to_camera.quaternion
            print(f"Refined Quaternion: {q}")
        else:
            print("\n[ERROR] Attitude estimation failed.")

    except Exception as e:
        print(f"[ERROR] Attitude Estimation failed: {e}")


if __name__ == "__main__":
    main()