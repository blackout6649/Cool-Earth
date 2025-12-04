"""
File: giant_processor.py
Description:
    Wraps GIANT library workflow based on the user's working script.
    Accepts a filepath, loads it into OpNavImage, and solves for attitude.
"""
import numpy as np
from giant.camera import Camera
from giant.camera_models import BrownModel
from giant.image import OpNavImage
from giant.stellar_opnav.stellar_class import StellarOpNav, StellarOpNavOptions
from giant.catalogs.gaia import Gaia
from giant.rotations import Rotation
import config

def process_image_file(filepath, initial_guess_q):
    """
    Runs GIANT processing on a FITS file.

    Args:
        filepath (str): Path to the .fits file.
        initial_guess_q (list): [x, y, z, w] quaternion for initial seed.

    Returns:
        tuple: (opnav_image, sopnav_object, success_boolean)
    """

    # 1. Setup Camera Model (Matching your working script)
    # Note: We use the parameters from config, which should match your script's numbers
    k1, k2, p1, p2, k3 = config.DISTORTION_COEFFS

    model = BrownModel(
        kx=config.CAM_FOCAL_LENGTH,
        ky=config.CAM_FOCAL_LENGTH,
        px=config.CAM_CENTER_X,
        py=config.CAM_CENTER_Y,
        n_rows=config.IMG_RES,
        n_cols=config.IMG_RES
    )

    camera_obj = Camera(
        model=model,
        name='Synthetic_L1_Camera'
    )

    # 2. Load Image & Time
    # CRITICAL CHANGE: We pass the filepath string directly to OpNavImage.
    # GIANT handles the FITS loading internally.
    try:
        opnav_image = OpNavImage(filepath, observation_date=config.OBSERVATION_DATE)
    except Exception as e:
        print(f" [ERROR] Could not load image {filepath}: {e}")
        return None, None, False

    # 3. Apply A Priori Attitude
    # Your script uses Rotation(initial_quaternion)
    opnav_image.rotation_inertial_to_camera = Rotation(initial_guess_q)

    # 4. Configure Navigation Options
    sopnav_options = StellarOpNavOptions()
    try:
        sopnav_options.star_id_options.catalog = Gaia()
    except Exception as e:
        print(f" [ERROR] Catalog Init Failed: {e}")
        return opnav_image, None, False

    sopnav = StellarOpNav(camera_obj, options=sopnav_options)
    sopnav.add_images([opnav_image])

    # --- TUNING (Matching your working script) ---
    sopnav.star_id.max_magnitude = 9
    sopnav.star_id.ransac_tolerance = 10.0
    sopnav.star_id.max_combos = 0
    #sopnav.star_id.tolerance = 200.0 # Uncomment if needed later

    # 5. Run Processing
    try:
        # A. Identify Stars
        # print(" [GIANT] Identifying stars...")
        sopnav.id_stars()

        # Check if we actually found stars before trying to estimate
        matched_points = sopnav.matched_extracted_image_points[0]
        if matched_points is None or matched_points.shape[1] < 3:
             print(f" [WARNING] Not enough matches found ({0 if matched_points is None else matched_points.shape[1]}).")
             return opnav_image, sopnav, False

        # B. Estimate Attitude
        # print(" [GIANT] Estimating Attitude...")
        sopnav.estimate_attitude()

        if opnav_image.pointing_post_fit:
            return opnav_image, sopnav, True
        else:
            return opnav_image, sopnav, False

    except Exception as e:
        print(f" [ERROR] GIANT Processing Exception: {e}")
        return opnav_image, sopnav, False