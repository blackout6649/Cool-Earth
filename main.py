"""
File: main.py
Description:
    Orchestrates the pipeline using the robust file-based approach.
"""

import os
import csv
import numpy as np
import config
from giant_processor import process_image_file  # Import the new function
from visualization import plot_star_id_results

def main():
    print(f"--- Starting GIANT Pipeline ---")

    # 1. Check for Truth File
    if not os.path.exists(config.TRUTH_FILE):
        print("CRITICAL: truth_data.csv not found. Run MATLAB script first!")
        return

    # 2. Iterate through scenarios
    with open(config.TRUTH_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            filename = row['filename']
            print(f"\n==========================================")
            print(f"Processing: {filename}")

            # Parse Truth Quaternion
            truth_q = [
                float(row['qx']),
                float(row['qy']),
                float(row['qz']),
                float(row['qw'])
            ]

            # 3. Get Full Path (Do not load data here, just get the string)
            full_path = os.path.join(config.IMAGE_DIR, filename)

            if not os.path.exists(full_path):
                print(f" [ERROR] File missing: {full_path}")
                continue

            # 4. Process (Pass path string)
            opnav_img, sopnav, success = process_image_file(full_path, truth_q)

            # 5. Visuals & Metrics
            if sopnav is not None:
                # Always plot to debug what happened
                plot_star_id_results(opnav_img, sopnav, title_suffix=filename)

            if success:
                solved_q = opnav_img.rotation_inertial_to_camera.quaternion
                truth_q_arr = np.array(truth_q, dtype=float)
                solved_q_arr = np.array(solved_q, dtype=float)

                truth_q_arr = truth_q_arr / np.linalg.norm(truth_q_arr)
                solved_q_arr = solved_q_arr / np.linalg.norm(solved_q_arr)

                # Calculate the dot product
                dot_product = np.dot(truth_q_arr, solved_q_arr)
                
                # Clip to [-1.0, 1.0] to prevent NaN errors in arccos due to floating-point precision limits
                dot_product = np.clip(dot_product, -1.0, 1.0)
                
                # Calculate the principal rotation angle in radians
                # The absolute value handles the q and -q ambiguity
                error_rad = 2 * np.arccos(np.abs(dot_product))
                
                # Convert to degrees
                error_deg = np.degrees(error_rad)

                print(f"   [SUCCESS] Solved Q: {np.round(solved_q, 4)}")
                print(f"   [METRIC]  Error: {error_deg:.6f}°")
            else:
                print("   [FAILURE] Could not determine attitude.")

if __name__ == "__main__":
    main()