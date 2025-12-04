"""
File: visualization.py
Description: Visualizes Star ID matches, Catalog projections, AND Raw Detections.
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_star_id_results(opnav_image, sopnav, title_suffix=""):
    """
    Plots the image with three layers of data:
    1. Blue Circles: Where stars SHOULD be (Catalog).
    2. Red Crosses: Where bright spots WERE found (Raw Detections).
    3. Green+: Which spots were successfully identified (Matches).
    """
    plt.figure(figsize=(10, 10))

    # 1. Plot Background Image
    if hasattr(opnav_image, 'data') and opnav_image.data is not None:
        # Log scale helps visualize faint stars
        clean_data = np.clip(opnav_image.data, 1, None)
        plt.imshow(np.log10(clean_data), cmap='gray', origin='upper')
    else:
        # Fallback if no image data
        plt.gca().set_facecolor('black')
        plt.xlim(0, 1024)
        plt.ylim(1024, 0)

    # 2. Extract Data
    # GIANT stores data in lists (one per image), so we take index [0]
    raw_points = sopnav.extracted_image_points[0]       # <--- NEW: The raw centroid detections
    cat_points = sopnav.queried_catalog_image_points[0] # The catalog stars projected onto camera
    matched_ex = sopnav.matched_extracted_image_points[0] # Successfully matched measured points
    matched_cat = sopnav.matched_catalog_image_points[0]  # Successfully matched catalog points

    # 3. Plot Catalog Stars (Cyan Circles)
    # These represent "Truth" / "Expectation"
    if cat_points is not None and cat_points.size > 0:
        # Filter to FOV (assuming 1024x1024)
        mask = (cat_points[0,:] >= 0) & (cat_points[0,:] < 1024) & \
               (cat_points[1,:] >= 0) & (cat_points[1,:] < 1024)
        plt.scatter(cat_points[0, mask], cat_points[1, mask],
                    s=100, edgecolors='cyan', facecolors='none',
                    label='Catalog (Expected)', linewidth=1.5, alpha=0.8)

    # 4. Plot ALL Detected Spots (Red Crosses)
    # These represent what the image processing actually "saw"
    if raw_points is not None and raw_points.size > 0:
        plt.scatter(raw_points[0, :], raw_points[1, :],
                    c='red', marker='x', s=60, label='Raw Detections', alpha=0.7)
    else:
        print(" [VISUALIZER] Warning: No raw points to plot (extracted_image_points is empty).")

    # 5. Plot Matches (Green Plus & Connectors)
    # These are the detections that successfully paired with a catalog star
    if matched_ex is not None and matched_ex.size > 0:
        plt.scatter(matched_ex[0, :], matched_ex[1, :],
                    c='lime', marker='+', s=120, label='Matched', linewidth=2)

        # Draw residual lines (connect the Red 'x' to the Cyan 'o')
        for i in range(matched_ex.shape[1]):
            plt.plot([matched_ex[0, i], matched_cat[0, i]],
                     [matched_ex[1, i], matched_cat[1, i]],
                     c='lime', linestyle='--', alpha=0.6)

    plt.title(f"Star ID Results: {title_suffix}")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()