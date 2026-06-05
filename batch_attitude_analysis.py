import csv
import os
from datetime import datetime, timedelta

import numpy as np
from astropy.io import fits

import config

from giant.camera import Camera
from giant.camera_models import BrownModel
from giant.image import OpNavImage
from giant.stellar_opnav.stellar_class import StellarOpNav, StellarOpNavOptions
from giant.catalogs.gaia import Gaia
from giant.rotations import Rotation


def _normalize_quaternion(q):
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.size != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {q_arr.size}")
    n = np.linalg.norm(q_arr)
    if n <= 0:
        raise ValueError("Quaternion norm must be > 0")
    return q_arr / n


def _attitude_error_deg(q_est, q_gt):
    q_est_n = _normalize_quaternion(q_est)
    q_gt_n = _normalize_quaternion(q_gt)
    dot = float(np.clip(np.abs(np.dot(q_est_n, q_gt_n)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _safe_count(points_array):
    if points_array is None:
        return 0
    if not isinstance(points_array, np.ndarray):
        return 0
    if points_array.ndim != 2:
        return 0
    return int(points_array.shape[1])


def _build_camera():
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
        n_cols=getattr(config, 'PROC_CAM_N_COLS', 1024),
    )
    return Camera(model=model, name='Batch_Analysis_Camera')


def _configure_sopnav(camera_obj):
    options = StellarOpNavOptions()
    options.star_id_options.catalog = Gaia()
    sopnav = StellarOpNav(camera_obj, options=options)

    sopnav.point_of_interest_finder.threshold = float(getattr(config, 'PROC_POI_THRESHOLD', 8.0))
    sopnav.point_of_interest_finder.min_size = int(getattr(config, 'PROC_POI_MIN_SIZE', 2))
    sopnav.point_of_interest_finder.max_size = int(getattr(config, 'PROC_POI_MAX_SIZE', 50))
    sopnav.point_of_interest_finder.centroid_size = int(getattr(config, 'PROC_POI_CENTROID_SIZE', 1))
    sopnav.point_of_interest_finder.reject_saturation = bool(getattr(config, 'PROC_POI_REJECT_SATURATION', True))

    sopnav.star_id.max_magnitude = float(getattr(config, 'PROC_MAX_MAGNITUDE', 10.0))
    sopnav.star_id.tolerance = float(getattr(config, 'PROC_STARID_TOLERANCE', 20.0))
    sopnav.star_id.ransac_tolerance = float(getattr(config, 'PROC_RANSAC_TOLERANCE', 10.0))
    sopnav.star_id.max_combos = int(getattr(config, 'PROC_MAX_COMBOS', 0))

    return sopnav


def _resolve_capture_path(row):
    capture_dir = getattr(config, 'BATCH_CAPTURE_DIR', getattr(config, 'CAPTURED_DIR', ''))
    scenario_filename = (row.get('filename') or '').strip()
    if not scenario_filename:
        return None

    explicit_capture = (row.get('captured_filename') or '').strip()
    if explicit_capture:
        return explicit_capture if os.path.isabs(explicit_capture) else os.path.join(capture_dir, explicit_capture)

    scenario_stem = os.path.splitext(os.path.basename(scenario_filename))[0]
    template = getattr(config, 'BATCH_CAPTURE_FILENAME_TEMPLATE', '{stem}.fits')
    capture_name = template.format(stem=scenario_stem, scenario=scenario_filename)
    return capture_name if os.path.isabs(capture_name) else os.path.join(capture_dir, capture_name)


def _process_one(row):
    scenario_filename = (row.get('filename') or '').strip()
    q_gt = _normalize_quaternion([
        float(row['qx']),
        float(row['qy']),
        float(row['qz']),
        float(row['qw']),
    ])

    capture_path = _resolve_capture_path(row)
    if not capture_path or not os.path.exists(capture_path):
        return {
            'scenario_filename': scenario_filename,
            'captured_image': capture_path or '',
            'status': 'missing_capture',
        }

    obs_time = getattr(config, 'OBSERVATION_DATE', datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=757339269.184))
    opnav_image = OpNavImage(capture_path, observation_date=obs_time)
    opnav_image.rotation_inertial_to_camera = Rotation(q_gt)

    camera_obj = _build_camera()
    sopnav = _configure_sopnav(camera_obj)
    sopnav.add_images([opnav_image])

    try:
        sopnav.id_stars()
    except Exception as e:
        return {
            'scenario_filename': scenario_filename,
            'captured_image': capture_path,
            'status': f'id_stars_failed: {e}',
        }

    raw_points = sopnav.extracted_image_points[0]
    projected_catalog = sopnav.queried_catalog_image_points[0]
    matched_points = sopnav.matched_extracted_image_points[0]

    num_raw = _safe_count(raw_points)
    num_cat = _safe_count(projected_catalog)
    num_matched = _safe_count(matched_points)

    min_matches = int(getattr(config, 'BATCH_REQUIRE_MIN_MATCHES', 3))
    if num_matched < min_matches:
        return {
            'scenario_filename': scenario_filename,
            'captured_image': capture_path,
            'status': 'not_enough_matches',
            'num_raw': num_raw,
            'num_catalog': num_cat,
            'num_matched': num_matched,
        }

    try:
        sopnav.estimate_attitude()
    except Exception as e:
        return {
            'scenario_filename': scenario_filename,
            'captured_image': capture_path,
            'status': f'estimate_attitude_failed: {e}',
            'num_raw': num_raw,
            'num_catalog': num_cat,
            'num_matched': num_matched,
        }

    if not opnav_image.pointing_post_fit:
        return {
            'scenario_filename': scenario_filename,
            'captured_image': capture_path,
            'status': 'no_post_fit_solution',
            'num_raw': num_raw,
            'num_catalog': num_cat,
            'num_matched': num_matched,
        }

    q_est = _normalize_quaternion(opnav_image.rotation_inertial_to_camera.quaternion)
    err_deg = _attitude_error_deg(q_est, q_gt)

    return {
        'scenario_filename': scenario_filename,
        'captured_image': capture_path,
        'status': 'ok',
        'num_raw': num_raw,
        'num_catalog': num_cat,
        'num_matched': num_matched,
        'qx_est': q_est[0],
        'qy_est': q_est[1],
        'qz_est': q_est[2],
        'qw_est': q_est[3],
        'qx_gt': q_gt[0],
        'qy_gt': q_gt[1],
        'qz_gt': q_gt[2],
        'qw_gt': q_gt[3],
        'error_deg': err_deg,
        'error_arcmin': err_deg * 60.0,
    }


def main():
    scenario_csv = getattr(config, 'BATCH_SCENARIO_CSV', getattr(config, 'TRUTH_FILE', None))
    if not scenario_csv or not os.path.exists(scenario_csv):
        print(f"[ERROR] Scenario CSV not found: {scenario_csv}")
        return

    with open(scenario_csv, 'r', newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[ERROR] Scenario CSV is empty: {scenario_csv}")
        return

    print(f"Running batch analysis for {len(rows)} scenarios from {scenario_csv}")
    results = []

    for i, row in enumerate(rows, start=1):
        name = (row.get('filename') or f'row_{i}').strip()
        print(f"[{i}/{len(rows)}] Processing {name}...")
        try:
            result = _process_one(row)
        except Exception as e:
            result = {
                'scenario_filename': name,
                'captured_image': '',
                'status': f'unhandled_error: {e}',
            }
        results.append(result)

    output_csv = getattr(config, 'BATCH_OUTPUT_CSV', os.path.join(getattr(config, 'DATA_DIR', '.'), 'batch_attitude_results.csv'))
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    keys = sorted({k for r in results for k in r.keys()})
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    ok_errors = [r['error_deg'] for r in results if r.get('status') == 'ok' and isinstance(r.get('error_deg'), (float, int))]
    ok_count = len(ok_errors)

    print("\n=== Batch Summary ===")
    print(f"Total scenarios: {len(results)}")
    print(f"Successful solutions: {ok_count}")
    print(f"Failed/Skipped: {len(results) - ok_count}")

    if ok_count > 0:
        err = np.asarray(ok_errors, dtype=float)
        print(f"Mean attitude error: {np.mean(err):.6f} deg")
        print(f"Std attitude error:  {np.std(err):.6f} deg")
        print(f"Median attitude error:{np.median(err):.6f} deg")
        print(f"Min/Max attitude error: {np.min(err):.6f} / {np.max(err):.6f} deg")

    print(f"Detailed results written to: {output_csv}")


if __name__ == '__main__':
    main()
