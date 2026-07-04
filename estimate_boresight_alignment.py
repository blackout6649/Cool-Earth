import argparse
import csv
from pathlib import Path

import numpy as np

import config


def _normalize_quaternion(q):
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.size != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {q_arr.size}")
    norm = np.linalg.norm(q_arr)
    if norm <= 0:
        raise ValueError("Quaternion norm must be > 0")
    return q_arr / norm


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


def _attitude_error_deg(q1, q2):
    qa = _normalize_quaternion(q1)
    qb = _normalize_quaternion(q2)
    dot = float(np.clip(np.abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _markley_average(quaternions):
    if not quaternions:
        raise ValueError("No quaternions were provided for averaging")

    accum = np.zeros((4, 4), dtype=float)
    for q in quaternions:
        qn = _normalize_quaternion(q)
        accum += np.outer(qn, qn)

    eigvals, eigvecs = np.linalg.eigh(accum)
    q_avg = eigvecs[:, int(np.argmax(eigvals))]
    q_avg = _normalize_quaternion(q_avg)
    if q_avg[3] < 0:
        q_avg = -q_avg
    return q_avg


def _read_batch_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Batch results CSV not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")

    return rows


def estimate_boresight(csv_path: Path):
    rows = _read_batch_rows(csv_path)
    q_cb_samples = []
    used_rows = 0

    for row in rows:
        if (row.get("status") or "").strip().lower() != "ok":
            continue

        try:
            q_ci = _normalize_quaternion([
                float(row["qx_est"]),
                float(row["qy_est"]),
                float(row["qz_est"]),
                float(row["qw_est"]),
            ])
            q_bi = _normalize_quaternion([
                float(row["qx_gt"]),
                float(row["qy_gt"]),
                float(row["qz_gt"]),
                float(row["qw_gt"]),
            ])
        except Exception:
            continue

        # q^C_B = q^C_I * (q^B_I)^-1
        q_cb = _quat_multiply(q_ci, _quat_inverse(q_bi))
        q_cb_samples.append(q_cb)
        used_rows += 1

    if not q_cb_samples:
        raise ValueError("No valid 'ok' rows with quaternion columns found in batch CSV")

    q_cb_avg = _markley_average(q_cb_samples)
    q_bc_avg = _quat_inverse(q_cb_avg)

    residual_errors = []
    for row in rows:
        if (row.get("status") or "").strip().lower() != "ok":
            continue
        try:
            q_ci = _normalize_quaternion([
                float(row["qx_est"]),
                float(row["qy_est"]),
                float(row["qz_est"]),
                float(row["qw_est"]),
            ])
            q_bi = _normalize_quaternion([
                float(row["qx_gt"]),
                float(row["qy_gt"]),
                float(row["qz_gt"]),
                float(row["qw_gt"]),
            ])
        except Exception:
            continue

        q_bi_corrected = _quat_multiply(_quat_inverse(q_cb_avg), q_ci)
        residual_errors.append(_attitude_error_deg(q_bi_corrected, q_bi))

    return q_cb_avg, q_bc_avg, np.asarray(residual_errors, dtype=float), used_rows


def main():
    parser = argparse.ArgumentParser(description="Estimate boresight alignment quaternion from batch attitude CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(getattr(config, "BATCH_OUTPUT_CSV", Path(config.DATA_DIR) / "batch_attitude_results.csv")),
        help="Path to batch_attitude_results.csv",
    )
    args = parser.parse_args()

    q_cb_avg, q_bc_avg, residual_errors, used_rows = estimate_boresight(args.csv)
    print(f"Used rows: {used_rows}")
    print(f"Estimated boresight q^C_B (camera wrt body): {q_cb_avg}")
    print(f"Inverse boresight q^B_C (body wrt camera): {q_bc_avg}")

    if residual_errors.size > 0:
        print(f"Residual mean error after correction:   {np.mean(residual_errors):.6f} deg")
        print(f"Residual std error after correction:    {np.std(residual_errors):.6f} deg")
        print(f"Residual median error after correction: {np.median(residual_errors):.6f} deg")


if __name__ == "__main__":
    main()