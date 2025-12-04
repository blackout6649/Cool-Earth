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
                dist = np.linalg.norm(np.array(truth_q) - np.array(solved_q))
                print(f"   [SUCCESS] Solved Q: {np.round(solved_q, 4)}")
                print(f"   [METRIC]  Error: {dist:.6f}")
            else:
                print("   [FAILURE] Could not determine attitude.")

if __name__ == "__main__":
    main()