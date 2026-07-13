import numpy as np
from datetime import datetime, timedelta
import csv
import sys
import traceback
import warnings
import matplotlib.pyplot as plt
import os
import config
from astropy.io import fits

# GIANT Imports
from giant.camera import Camera
from giant.camera_models import BrownModel
from giant.image import OpNavImage
from giant.stellar_opnav.stellar_class import StellarOpNav, StellarOpNavOptions
from giant.catalogs.gaia import Gaia
from giant.rotations import Rotation
from scipy.spatial.distance import cdist


def _configure_warning_filters():
    suppress_giant_runtime = bool(getattr(config, 'PROC_SUPPRESS_GIANT_RUNTIME_WARNINGS', True))
    if suppress_giant_runtime:
        warnings.filterwarnings(
            'ignore',
            message=r'overflow encountered.*',
            category=RuntimeWarning
        )
        warnings.filterwarnings(
            'ignore',
            message=r'invalid value encountered.*',
            category=RuntimeWarning
        )
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            module=r'.*giant\.point_spread_functions\.gaussians'
        )
        warnings.filterwarnings(
            'ignore',
            category=RuntimeWarning,
            module=r'.*giant\.point_spread_functions\..*'
        )

    suppress_all_runtime = bool(getattr(config, 'PROC_SUPPRESS_ALL_RUNTIME_WARNINGS', False))
    if suppress_all_runtime:
        warnings.filterwarnings('ignore', category=RuntimeWarning)


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


def _normalize_quaternion(q):
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.size != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {q_arr.size}")
    n = np.linalg.norm(q_arr)
    if n <= 0:
        raise ValueError("Quaternion norm must be > 0")
    return q_arr / n


def _attitude_error_deg(q_est, q_gt):
    """Return principal angle between two quaternions (deg), sign-invariant."""
    q_est_n = _normalize_quaternion(q_est)
    q_gt_n = _normalize_quaternion(q_gt)
    dot = float(np.clip(np.abs(np.dot(q_est_n, q_gt_n)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _quat_conjugate(q):
    qn = _normalize_quaternion(q)
    return np.array([-qn[0], -qn[1], -qn[2], qn[3]], dtype=float)


def _quat_inverse(q):
    return _quat_conjugate(q)


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = _normalize_quaternion(q1)
    x2, y2, z2, w2 = _normalize_quaternion(q2)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return _normalize_quaternion([x, y, z, w])


def _prompt_yes_no(prompt, default=False):
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        value = input(prompt + suffix).strip().lower()
        if value == "":
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print("Please enter y or n.")


def _prompt_quaternion(prompt):
    while True:
        raw = input(prompt).strip()
        cleaned = raw.replace(",", " ")
        parts = [p for p in cleaned.split() if p]
        if len(parts) != 4:
            print("Enter 4 values in x y z w order, separated by spaces or commas.")
            continue

        try:
            q = [float(p) for p in parts]
            return _normalize_quaternion(q)
        except Exception as e:
            print(f"Invalid quaternion: {e}")


def _prompt_boresight_quaternion():
    if not sys.stdin.isatty():
        return None

    use_boresight = _prompt_yes_no("Apply boresight alignment correction q^C_B?", default=False)
    if not use_boresight:
        return None

    q_cb = _prompt_quaternion("Enter q^C_B as x y z w: ")
    print(f"Using boresight q^C_B: {q_cb}")
    return q_cb


def _resolve_boresight_quaternion(boresight_q_cb=None):
    if boresight_q_cb is not None:
        return _normalize_quaternion(boresight_q_cb)

    if not bool(getattr(config, 'PROC_USE_BORESIGHT', False)):
        return None

    configured_q_cb = getattr(config, 'PROC_BORESIGHT_Q_CB', None)
    if configured_q_cb is None:
        return None

    return _normalize_quaternion(configured_q_cb)


def _read_gt_quaternion_from_fits_header(path):
    if not path:
        return None
    if not os.path.exists(path):
        return None

    with fits.open(path) as hdul:
        hdr = hdul[0].header

    keys = ("Q_X", "Q_Y", "Q_Z", "Q_W")
    if not all(k in hdr for k in keys):
        return None

    return [float(hdr["Q_X"]), float(hdr["Q_Y"]), float(hdr["Q_Z"]), float(hdr["Q_W"])]


def _resolve_gt_quaternion():
    configured_gt = getattr(config, 'PROC_GT_QUATERNION', None)
    if configured_gt is not None:
        return _normalize_quaternion(configured_gt), 'config.PROC_GT_QUATERNION'

    gt_path = getattr(config, 'PROC_GT_IMAGE_PATH', getattr(config, 'DISPLAY_IMAGE_PATH', None))
    gt_from_header = _read_gt_quaternion_from_fits_header(gt_path)
    if gt_from_header is not None:
        return _normalize_quaternion(gt_from_header), f'FITS header ({gt_path})'

    return None, None


def _load_scenario_quaternions(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return {}

    out = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fname = row['filename'].strip()
                q = _normalize_quaternion([
                    float(row['qx']),
                    float(row['qy']),
                    float(row['qz']),
                    float(row['qw']),
                ])
            except Exception:
                continue
            out[fname] = q
    return out


def _resolve_initial_quaternion(process_image_path):
    use_csv = bool(getattr(config, 'PROC_USE_SCENARIO_CSV_INITIAL_QUAT', False))
    if use_csv:
        scenario_csv = getattr(config, 'PROC_SCENARIO_CSV', getattr(config, 'TRUTH_FILE', None))
        scenario_quats = _load_scenario_quaternions(scenario_csv)
        if scenario_quats:
            explicit_name = getattr(config, 'PROC_SCENARIO_FILENAME', None)
            if explicit_name and explicit_name in scenario_quats:
                return scenario_quats[explicit_name], f'scenarios.csv explicit key: {explicit_name}'

            image_base = os.path.basename(process_image_path).lower()
            candidates = []
            for scenario_fname, q in scenario_quats.items():
                scenario_stem = os.path.splitext(scenario_fname)[0].lower()
                if scenario_stem and scenario_stem in image_base:
                    candidates.append((scenario_fname, q))

            if len(candidates) == 1:
                return candidates[0][1], f'scenarios.csv inferred from image name: {candidates[0][0]}'

    fallback = getattr(config, 'PROC_INITIAL_QUATERNION', [-0.704416026, 0.061628417, 0.0, 0.707106781])
    return _normalize_quaternion(fallback), 'config.PROC_INITIAL_QUATERNION'


# =========================================================================
# SUN-POINTING FRAME HELPERS
# =========================================================================

# Earth's mean angular velocity around the sun (rad/s)
_OMEGA_EARTH_RAD_PER_S = 2.0 * np.pi / (365.25 * 24.0 * 3600.0)
# Mean obliquity (deg) for J2000 <-> HCI frame transform.
_J2000_OBLIQUITY_DEG = 23.439291111


def _compute_j2000_to_hci_dcm(obliquity_deg=None):
    """
    Rotation DCM from J2000 to HCI frame.

    C_J2000^HCI(eps) = R_x(eps)

    where eps is the obliquity of the ecliptic.

    Args:
        obliquity_deg: Optional obliquity override in degrees.

    Returns:
        3x3 numpy array.
    """
    from scipy.spatial.transform import Rotation as _ScipyRot

    eps_deg = _J2000_OBLIQUITY_DEG if obliquity_deg is None else float(obliquity_deg)
    return _ScipyRot.from_euler('x', np.radians(eps_deg)).as_matrix()


def _compute_hci_to_sun_pointing_prime_dcm(lam):
    """
    Rotation DCM from HCI to intermediate Sun-Pointing prime (SP') frame.

    SP' is defined by rotating frame axes about HCI z by lambda, such that x' points to the sun.

    C_HCI^SP'(lambda) = [[ cos(l),  sin(l), 0 ],
                         [-sin(l),  cos(l), 0 ],
                         [ 0,       0,      1 ]]

    Args:
        lam: Ecliptic longitude lambda (radians).

    Returns:
        3x3 numpy array.
    """
    sl, cl = np.sin(lam), np.cos(lam)
    return np.array([
        [ cl,  sl, 0.0],
        [-sl,  cl, 0.0],
        [ 0.0, 0.0, 1.0],
    ], dtype=float)


def _compute_sun_pointing_prime_to_sun_pointing_dcm():
    """
    Rotation DCM from SP' to final Sun-Pointing (SP) frame.

    This is a +90 degree rotation about the SP' y' axis.

    C_SP'^SP = R_y(+90 deg)

    Returns:
        3x3 numpy array.
    """
    from scipy.spatial.transform import Rotation as _ScipyRot

    return _ScipyRot.from_euler('y', np.pi / 2.0).as_matrix()


def _compute_sun_pointing_dcm(lam):
    """
    Rotation DCM from HCI to final Sun-Pointing (SP) frame.

    This is composed as:
    C_HCI^SP(lambda) = C_SP'^SP @ C_HCI^SP'(lambda)

    Returns:
        3x3 numpy array.
    """
    dcm_hci_sp_prime = _compute_hci_to_sun_pointing_prime_dcm(lam)
    dcm_sp_prime_sp = _compute_sun_pointing_prime_to_sun_pointing_dcm()
    return dcm_sp_prime_sp @ dcm_hci_sp_prime


def _quat_to_dcm(q):
    """Convert a quaternion [x, y, z, w] to a 3x3 rotation DCM."""
    from scipy.spatial.transform import Rotation as _ScipyRot
    return _ScipyRot.from_quat(_normalize_quaternion(q)).as_matrix()


def _dcm_to_quaternion(R):
    """Convert a 3x3 rotation DCM to a normalized quaternion [x, y, z, w]."""
    from scipy.spatial.transform import Rotation as _ScipyRot
    return _normalize_quaternion(_ScipyRot.from_matrix(R).as_quat())


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


def run_single_image_pipeline(image_path=None, boresight_q_cb=None, show_plot=None, min_matches=None, verbose=None, t=0.0, lambda_0=0.0, show_matched_stars_popup=None):
    _configure_warning_filters()

    if verbose is None:
        verbose = bool(getattr(config, 'PROC_VERBOSE', False))
    else:
        verbose = bool(verbose)

    if show_plot is None:
        show_plot = bool(getattr(config, 'PROC_ENABLE_PLOTS', False))
    else:
        show_plot = bool(show_plot)

    if min_matches is None:
        min_matches = int(getattr(config, 'PROC_MIN_MATCHES', 3))
    else:
        min_matches = int(min_matches)

    try:
        resolved_boresight_q_cb = _resolve_boresight_quaternion(boresight_q_cb=boresight_q_cb)
    except Exception as e:
        return {
            'status': 'failed',
            'message': f'invalid_boresight_quaternion: {e}',
        }

    # Ecliptic longitude at time t: lambda(t) = lambda_0 + omega_earth * t
    # lambda_0 is accepted in degrees and converted to radians internally.
    lambda_val = np.radians(float(lambda_0)) + _OMEGA_EARTH_RAD_PER_S * float(t)

    if show_matched_stars_popup is None:
        show_matched_stars_popup = bool(getattr(config, 'PROC_SHOW_MATCHED_STARS_POPUP', False))
    else:
        show_matched_stars_popup = bool(show_matched_stars_popup)

    def _vprint(message):
        if verbose:
            print(message)

    result = {
        'status': 'failed',
        'message': 'unknown_failure',
        'image_path': '',
        'num_raw': 0,
        'num_catalog_in_fov': 0,
        'num_matched': 0,
        'q_ci': None,
        'q_ci_SunPointing': None,
        'q_bi_corrected': None,
        'error_deg': None,
        'corrected_error_deg': None,
    }

    # =========================================================================
    # 1. Setup Camera Model
    # =========================================================================
    k1, k2, p1, p2, k3 = getattr(config, 'PROC_DISTORTION_COEFFS', [0.0, 0.0, 0.0, 0.0, 0.0])
    model = BrownModel(
        kx=getattr(config, 'PROC_CAM_FOCAL_LENGTH_PX', 14774.74),
        ky=getattr(config, 'PROC_CAM_FOCAL_LENGTH_PX', 14774.74),
        px=getattr(config, 'PROC_CAM_CENTER_X', 511.5),
        py=getattr(config, 'PROC_CAM_CENTER_Y', 511.5),
        k1=k1,
        k2=k2,
        k3=k3,
        p1=p1,
        p2=p2,
        n_rows=getattr(config, 'PROC_CAM_N_ROWS', 1024),
        n_cols=getattr(config, 'PROC_CAM_N_COLS', 1024)
    )

    camera_obj = Camera(
        model=model,
        name='Synthetic_L1_Camera'
    )

    # =========================================================================
    # 2. Load Image & Time
    # =========================================================================
    image_path = image_path or getattr(config, 'PROCESS_IMAGE_PATH', os.path.join(config.IMAGE_DIR, 'Scenario_02.fits'))
    obs_time = getattr(config, 'OBSERVATION_DATE', datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=757339269.184))
    result['image_path'] = image_path

    print(f"Loading image: {image_path}")
    try:
        opnav_image = OpNavImage(image_path, observation_date=obs_time)
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find {image_path}. Run MATLAB script first.")
        result['message'] = 'image_not_found'
        return result

    # --- A Priori Attitude Setup ---
    initial_quaternion, init_source = _resolve_initial_quaternion(image_path)
    _vprint(f"A Priori Attitude Source: {init_source}")
    _vprint(f"A Priori Attitude: {initial_quaternion}")
    opnav_image.rotation_inertial_to_camera = Rotation(initial_quaternion)

    # =========================================================================
    # 3. Configure Navigation
    # =========================================================================
    sopnav_options = StellarOpNavOptions()
    try:
        sopnav_options.star_id_options.catalog = Gaia()
    except Exception as e:
        print(f"[ERROR] Catalog Init Failed: {e}")
        result['message'] = f'catalog_init_failed: {e}'
        return result

    sopnav = StellarOpNav(camera_obj, options=sopnav_options)
    sopnav.add_images([opnav_image])

    # --- POINT EXTRACTION TUNING (critical for noisy captured images) ---
    sopnav.point_of_interest_finder.threshold = float(getattr(config, 'PROC_POI_THRESHOLD', 8.0))
    sopnav.point_of_interest_finder.min_size = int(getattr(config, 'PROC_POI_MIN_SIZE', 2))
    sopnav.point_of_interest_finder.max_size = int(getattr(config, 'PROC_POI_MAX_SIZE', 50))
    sopnav.point_of_interest_finder.centroid_size = int(getattr(config, 'PROC_POI_CENTROID_SIZE', 1))
    sopnav.point_of_interest_finder.reject_saturation = bool(getattr(config, 'PROC_POI_REJECT_SATURATION', True))

    # --- TUNING ---
    sopnav.star_id.max_magnitude = float(getattr(config, 'PROC_MAX_MAGNITUDE', 10.0))
    sopnav.star_id.tolerance = float(getattr(config, 'PROC_STARID_TOLERANCE', 20.0))
    sopnav.star_id.ransac_tolerance = float(getattr(config, 'PROC_RANSAC_TOLERANCE', 10.0))
    sopnav.star_id.max_combos = int(getattr(config, 'PROC_MAX_COMBOS', 0))

    _vprint(
        "POI tuning: "
        f"threshold={sopnav.point_of_interest_finder.threshold}, "
        f"min_size={sopnav.point_of_interest_finder.min_size}, "
        f"max_size={sopnav.point_of_interest_finder.max_size}, "
        f"centroid_size={sopnav.point_of_interest_finder.centroid_size}, "
        f"reject_saturation={sopnav.point_of_interest_finder.reject_saturation}"
    )
    _vprint(
        "Star-ID tuning: "
        f"max_magnitude={sopnav.star_id.max_magnitude}, "
        f"tolerance={sopnav.star_id.tolerance}, "
        f"ransac_tolerance={sopnav.star_id.ransac_tolerance}, "
        f"max_combos={sopnav.star_id.max_combos}"
    )

    _vprint("\n--- Starting Processing ---")

    # A. Identify Stars
    _vprint("1. Identifying stars...")
    try:
        sopnav.id_stars()

        # Data retrieval for debug printing
        raw_points = sopnav.extracted_image_points[0]
        projected_catalog = sopnav.queried_catalog_image_points[0]
        matched_points = sopnav.matched_extracted_image_points[0]
        num_raw = safe_get_count(raw_points)
        num_matched = safe_get_count(matched_points)
        result['num_raw'] = int(num_raw)
        result['num_matched'] = int(num_matched)

        # --- VISUALIZATION CALL ---
        if show_plot:
            _vprint("Generating visual comparison...")
            plot_results(
                opnav_image,
                sopnav,
                img_width=int(getattr(config, 'PROC_CAM_N_COLS', 1024)),
                img_height=int(getattr(config, 'PROC_CAM_N_ROWS', 1024)),
            )
        # --------------------------

        num_cat_in_fov = 0
        if projected_catalog is not None and projected_catalog.ndim == 2:
            in_fov_mask = (
                    (projected_catalog[0, :] >= 0) & (projected_catalog[0, :] <= getattr(config, 'PROC_CAM_N_COLS', 1024)) &
                    (projected_catalog[1, :] >= 0) & (projected_catalog[1, :] <= getattr(config, 'PROC_CAM_N_ROWS', 1024))
            )
            num_cat_in_fov = np.sum(in_fov_mask)
            projected_catalog_in_fov = projected_catalog[:, in_fov_mask]
            result['num_catalog_in_fov'] = int(num_cat_in_fov)

        print(f"Detected spots: {num_raw} | Catalog in FOV: {num_cat_in_fov} | Matched stars: {num_matched}")

        # --- MATCHED STARS POPUP ---
        if show_matched_stars_popup:
            import plot_matched_stars as _pms
            _pms.show_matched_stars(
                opnav_image, sopnav,
                img_width=int(getattr(config, 'PROC_CAM_N_COLS', 1024)),
                img_height=int(getattr(config, 'PROC_CAM_N_ROWS', 1024)),
                title_suffix=os.path.basename(image_path),
            )
        # ---------------------------

        # Calculate distances for console output
        if num_raw > 0 and num_cat_in_fov > 0 and raw_points.ndim == 2:
            distances = cdist(raw_points.T, projected_catalog_in_fov.T, metric='euclidean')
            _vprint(f"Overall min distance: {np.min(distances):.2f} pixels")

        if num_matched < min_matches:
            print("[WARNING] Not enough matched stars.")
            result['message'] = 'not_enough_matches'
            return result

    except Exception as e:
        print(f"[ERROR] Star Identification failed: {e}")
        if verbose:
            traceback.print_exc()
        result['message'] = f'star_identification_failed: {e}'
        return result

    # B. Estimate Attitude
    _vprint("\n2. Estimating Attitude...")
    try:
        sopnav.estimate_attitude()

        if opnav_image.pointing_post_fit:
            q_ci = _normalize_quaternion(opnav_image.rotation_inertial_to_camera.quaternion)
            result['q_ci'] = q_ci.tolist()
            print(f"Refined Quaternion q^C_I: {q_ci}")

            # Sun-Pointing frame attitude with explicit chain:
            # J2000 -> HCI -> SP' -> SP
            # R^C_SP = R^C_J2000 @ (C_J2000^HCI)^T @ (C_HCI^SP)^T
            _dcm_j2000_hci = _compute_j2000_to_hci_dcm(
                obliquity_deg=getattr(config, 'PROC_J2000_TO_HCI_OBLIQUITY_DEG', _J2000_OBLIQUITY_DEG)
            )
            _dcm_hci_sp = _compute_sun_pointing_dcm(lambda_val)
            _R_c_sp = _quat_to_dcm(q_ci) @ _dcm_j2000_hci.T @ _dcm_hci_sp.T
            q_ci_sp = _dcm_to_quaternion(_R_c_sp)
            result['q_ci_SunPointing'] = q_ci_sp.tolist()
            print(f"Sun-Pointing Quaternion q^C_SP (lambda={np.degrees(lambda_val):.4f} deg): {q_ci_sp}")

            q_bi_corrected = None
            if resolved_boresight_q_cb is not None:
                q_bi_corrected = _quat_multiply(_quat_inverse(resolved_boresight_q_cb), q_ci)
                result['q_bi_corrected'] = q_bi_corrected.tolist()
                print(f"Boresight q^C_B: {resolved_boresight_q_cb}")
                print(f"Corrected Quaternion q^B_I = (q^C_B)^-1 q^C_I: {q_bi_corrected}")

            if bool(getattr(config, 'PROC_REPORT_ATTITUDE_ERROR', True)):
                q_gt, gt_source = _resolve_gt_quaternion()
                if q_gt is None:
                    print("[INFO] Ground truth quaternion not available; skipping attitude error report.")
                    print("       Set config.PROC_GT_QUATERNION or ensure Q_X/Q_Y/Q_Z/Q_W in config.PROC_GT_IMAGE_PATH FITS header.")
                else:
                    err_deg = _attitude_error_deg(q_ci, q_gt)
                    result['error_deg'] = float(err_deg)
                    err_arcmin = err_deg * 60.0
                    print(f"Ground Truth Quaternion ({gt_source}): {q_gt}")
                    print(f"Camera-vs-GT Error: {err_deg:.6f} deg ({err_arcmin:.3f} arcmin)")
                    if q_bi_corrected is not None:
                        corrected_err_deg = _attitude_error_deg(q_bi_corrected, q_gt)
                        result['corrected_error_deg'] = float(corrected_err_deg)
                        corrected_err_arcmin = corrected_err_deg * 60.0
                        print(
                            f"Corrected Body-vs-GT Error: {corrected_err_deg:.6f} deg "
                            f"({corrected_err_arcmin:.3f} arcmin)"
                        )
            result['status'] = 'ok'
            result['message'] = 'success'
        else:
            print("\n[ERROR] Attitude estimation failed.")
            result['message'] = 'no_post_fit_solution'

    except Exception as e:
        print(f"[ERROR] Attitude Estimation failed: {e}")
        result['message'] = f'attitude_estimation_failed: {e}'

    return result


def main(boresight_q_cb=None):
    return run_single_image_pipeline(boresight_q_cb=boresight_q_cb)


if __name__ == "__main__":
    _configure_warning_filters()
    boresight_q_cb = None
    if bool(getattr(config, 'PROC_INTERACTIVE_PROMPTS', False)):
        boresight_q_cb = _prompt_boresight_quaternion()
    else:
        try:
            boresight_q_cb = _resolve_boresight_quaternion()
        except Exception as e:
            print(f"[ERROR] Invalid boresight config: {e}")
            boresight_q_cb = None

    mode = str(getattr(config, 'PROC_MODE', 'single')).strip().lower()
    if mode == 'batch':
        import batch_attitude_analysis
        batch_attitude_analysis.main(boresight_q_cb=boresight_q_cb)
    else:
        main(boresight_q_cb=boresight_q_cb)