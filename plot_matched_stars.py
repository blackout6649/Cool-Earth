"""
plot_matched_stars.py
---------------------
Standalone visualization utility for the Cool-Earth-NAV OpNav pipeline.

Displays a blocking popup window after each star-ID run, showing:
  - Background: raw FITS image (log-scaled for contrast)
  - Red   X  : all detected spot locations
  - Cyan  O  : all catalog star projections inside the FOV
  - Green *  : successfully matched stars (individually numbered)
  - Green -- : residual lines connecting matched extracted → catalog positions

Usage (called automatically from process_star_image.py when enabled):
    import plot_matched_stars
    plot_matched_stars.show_matched_stars(opnav_image, sopnav,
                                          img_width=1280, img_height=1024)

Controlled by config.PROC_SHOW_MATCHED_STARS_POPUP (default: False).
Can be overridden per-call from StateMachine.py via OPNAV_SHOW_MATCHED_STARS.
"""

import numpy as np
import matplotlib.pyplot as plt


def show_matched_stars(opnav_image, sopnav,
                       img_width=1024, img_height=1024,
                       title_suffix=""):
    """
    Opens a blocking popup window showing matched stars on the captured image.

    Args:
        opnav_image : giant OpNavImage object (provides .data and .observation_date).
        sopnav      : giant StellarOpNav object (provides extracted / queried /
                      matched image points after id_stars() has been called).
        img_width   : sensor width in pixels (used for FOV clipping).
        img_height  : sensor height in pixels.
        title_suffix: optional extra text appended to the window title (e.g. filename).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')

    # -------------------------------------------------------------------------
    # 1. Background image (log-scaled for star visibility)
    # -------------------------------------------------------------------------
    if hasattr(opnav_image, 'data') and opnav_image.data is not None:
        ax.imshow(
            np.log10(np.clip(opnav_image.data, 1, None)),
            cmap='gray', origin='upper',
            extent=[0, img_width, img_height, 0]
        )
    else:
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)

    # -------------------------------------------------------------------------
    # 2. All detected spots — red X
    # -------------------------------------------------------------------------
    raw_points = sopnav.extracted_image_points[0]
    n_raw = 0
    if raw_points is not None and raw_points.ndim == 2:
        n_raw = raw_points.shape[1]
        ax.scatter(
            raw_points[0, :], raw_points[1, :],
            c='red', marker='x', s=60, linewidths=1.5, zorder=3,
            label=f'Detected ({n_raw})'
        )

    # -------------------------------------------------------------------------
    # 3. Catalog star projections inside FOV — cyan open circle
    # -------------------------------------------------------------------------
    cat_points = sopnav.queried_catalog_image_points[0]
    n_cat = 0
    if cat_points is not None and cat_points.ndim == 2:
        in_fov = (
            (cat_points[0, :] >= 0) & (cat_points[0, :] <= img_width) &
            (cat_points[1, :] >= 0) & (cat_points[1, :] <= img_height)
        )
        n_cat = int(np.sum(in_fov))
        ax.scatter(
            cat_points[0, in_fov], cat_points[1, in_fov],
            s=140, edgecolors='cyan', facecolors='none',
            linewidths=1.5, zorder=2,
            label=f'Catalog in FOV ({n_cat})'
        )

    # -------------------------------------------------------------------------
    # 4. Matched stars — green star marker + residual line + index label
    # -------------------------------------------------------------------------
    matched_extracted = sopnav.matched_extracted_image_points[0]
    matched_catalog   = sopnav.matched_catalog_image_points[0]
    n_matched = 0

    if (matched_extracted is not None and matched_catalog is not None and
            matched_extracted.ndim == 2 and matched_catalog.ndim == 2):

        n_matched = matched_extracted.shape[1]

        # Residual lines (extracted → catalog)
        for i in range(n_matched):
            ax.plot(
                [matched_extracted[0, i], matched_catalog[0, i]],
                [matched_extracted[1, i], matched_catalog[1, i]],
                color='lime', linestyle='--', linewidth=0.8, alpha=0.7, zorder=4
            )

        # Matched positions
        ax.scatter(
            matched_extracted[0, :], matched_extracted[1, :],
            c='lime', marker='*', s=200, zorder=5,
            label=f'Matched ({n_matched})'
        )

        # Index labels
        for i in range(n_matched):
            ax.annotate(
                str(i + 1),
                xy=(matched_extracted[0, i], matched_extracted[1, i]),
                xytext=(5, 5), textcoords='offset points',
                color='yellow', fontsize=8, fontweight='bold', zorder=6
            )

    # -------------------------------------------------------------------------
    # 5. Title, labels, legend
    # -------------------------------------------------------------------------
    title = f"Matched Stars — {n_matched} / {n_raw} detected  |  {n_cat} catalog in FOV"
    if title_suffix:
        title += f"  |  {title_suffix}"
    obs_date = getattr(opnav_image, 'observation_date', None)
    if obs_date:
        title += f"\n{obs_date} UTC"

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X Pixel")
    ax.set_ylabel("Y Pixel")
    ax.legend(loc='upper right', fontsize=9, framealpha=0.6)
    ax.grid(color='gray', linestyle='--', linewidth=0.4, alpha=0.25)
    plt.tight_layout()
    plt.show()
