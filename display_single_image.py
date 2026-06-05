"""
Display one synthetic star image with 1:1 pixel mapping and black padding.

Usage example (monitor 2 starts at x=1920):
    python display_single_image.py \
        --input data/generated/Single_Scenario.fits \
        --monitor-width 1920 --monitor-height 1080 \
        --offset-x 1920 --offset-y 0

Interactive monitor selection:
    python display_single_image.py --choose-monitor

Notes:
- Keep Windows display scaling at 100% for true pixel mapping.
- This script performs nearest-neighbor rendering (no smoothing/interpolation).
- Press Q or Esc to close.
"""

import argparse
import csv
import ctypes
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import config

# Remove Matplotlib's navigation toolbar/status strip to avoid a bottom white bar.
plt.rcParams["toolbar"] = "None"


def _detect_monitors_windows():
    """Return monitor rectangles as dicts with x, y, width, height (Windows only)."""
    if not sys.platform.startswith("win"):
        return []

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    monitors = []

    MONITORENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.POINTER(RECT),
        ctypes.c_double,
    )

    def _callback(_hmonitor, _hdc, lprc_monitor, _lparam):
        r = lprc_monitor.contents
        monitors.append(
            {
                "x": int(r.left),
                "y": int(r.top),
                "width": int(r.right - r.left),
                "height": int(r.bottom - r.top),
            }
        )
        return 1

    user32 = ctypes.windll.user32
    user32.EnumDisplayMonitors(0, 0, MONITORENUMPROC(_callback), 0)

    # Stable ordering for easier index usage.
    monitors.sort(key=lambda m: (m["x"], m["y"]))
    return monitors


def _print_monitors(monitors) -> None:
    if not monitors:
        print("No monitor metadata found from Windows API.")
        return

    print("Detected monitors (index: widthxheight at x,y):")
    for i, m in enumerate(monitors):
        print(f"  {i}: {m['width']}x{m['height']} at ({m['x']}, {m['y']})")


def _prompt_monitor_index(monitors) -> int:
    while True:
        raw = input(f"Choose monitor index [0..{len(monitors)-1}]: ").strip()
        if raw == "":
            print("Please enter a monitor index.")
            continue

        try:
            idx = int(raw)
        except ValueError:
            print("Invalid input. Enter an integer index.")
            continue

        if 0 <= idx < len(monitors):
            return idx

        print(f"Out of range. Valid range is 0..{len(monitors)-1}.")


def _resolve_target_monitor(args):
    monitors = _detect_monitors_windows()

    if args.list_monitors:
        _print_monitors(monitors)
        return None

    if args.monitor_index is not None:
        if not monitors:
            raise RuntimeError("Monitor index requested, but monitor detection failed on this platform.")
        if args.monitor_index < 0 or args.monitor_index >= len(monitors):
            raise ValueError(f"Invalid monitor index {args.monitor_index}; valid range is 0..{len(monitors)-1}")

        m = monitors[args.monitor_index]
        return m["width"], m["height"], m["x"], m["y"]

    if args.choose_monitor:
        if not monitors:
            raise RuntimeError("Interactive monitor selection requested, but monitor detection failed.")
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive monitor selection requires a terminal (TTY).")

        _print_monitors(monitors)
        chosen_index = _prompt_monitor_index(monitors)
        m = monitors[chosen_index]
        print(f"Using monitor {chosen_index}: {m['width']}x{m['height']} at ({m['x']}, {m['y']})")
        return m["width"], m["height"], m["x"], m["y"]

    # If multiple monitors are present and no explicit target is provided,
    # default to interactive selection when possible.
    prompt_if_multi = bool(getattr(config, "DISPLAY_PROMPT_IF_MULTI_MONITOR", True))
    if prompt_if_multi and len(monitors) > 1 and sys.stdin.isatty():
        _print_monitors(monitors)
        chosen_index = _prompt_monitor_index(monitors)
        m = monitors[chosen_index]
        print(f"Using monitor {chosen_index}: {m['width']}x{m['height']} at ({m['x']}, {m['y']})")
        return m["width"], m["height"], m["x"], m["y"]

    return args.monitor_width, args.monitor_height, args.offset_x, args.offset_y


def _read_fits(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input FITS not found: {path}")

    with fits.open(path) as hdul:
        data = hdul[0].data

    if data is None:
        raise ValueError("FITS primary HDU contains no image data")

    image = np.asarray(data)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    return image.astype(np.float64)


def _load_series_paths_from_csv(csv_path: str, image_dir: str) -> list[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Scenario CSV not found: {csv_path}")

    out = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = (row.get('filename') or '').strip()
            if not filename:
                continue

            path = filename if os.path.isabs(filename) else os.path.join(image_dir, filename)
            if os.path.exists(path):
                out.append(path)
            else:
                print(f"[WARN] Skipping missing image from CSV: {path}")

    if not out:
        raise ValueError(f"No valid scenario images found in CSV: {csv_path}")

    return out


def _to_display_u8(image: np.ndarray, gamma: float, apply_gamma: bool) -> np.ndarray:
    max_val = float(np.max(image))
    if max_val <= 0:
        norm = np.zeros_like(image, dtype=np.float64)
    else:
        norm = np.clip(image / max_val, 0.0, 1.0)

    if apply_gamma:
        if gamma <= 0:
            raise ValueError("Gamma must be > 0")
        # Convert linear intensities to monitor driving values.
        norm = np.power(norm, 1.0 / gamma)

    return np.round(norm * 255.0).astype(np.uint8)


def _pad_center(image_u8: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = image_u8.shape
    if in_h > out_h or in_w > out_w:
        raise ValueError(
            f"Input image ({in_w}x{in_h}) is larger than monitor ({out_w}x{out_h})."
        )

    canvas = np.zeros((out_h, out_w), dtype=np.uint8)
    y0 = (out_h - in_h) // 2
    x0 = (out_w - in_w) // 2
    canvas[y0:y0 + in_h, x0:x0 + in_w] = image_u8
    return canvas


def _set_window_geometry(fig, width: int, height: int, x: int, y: int, borderless: bool) -> None:
    manager = plt.get_current_fig_manager()

    if hasattr(manager, "toolbar") and manager.toolbar is not None:
        if hasattr(manager.toolbar, "pack_forget"):
            manager.toolbar.pack_forget()

    # TkAgg on Windows supports direct geometry and borderless mode.
    if hasattr(manager, "window"):
        win = manager.window
        if hasattr(win, "wm_geometry"):
            win.wm_geometry(f"{width}x{height}+{x}+{y}")
        if borderless and hasattr(win, "overrideredirect"):
            win.overrideredirect(True)
        if hasattr(win, "attributes"):
            win.attributes("-topmost", True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display one FITS image or scenario series with 1:1 mapping.")
    parser.add_argument(
        "--input",
        default=getattr(config, "DISPLAY_IMAGE_PATH", os.path.join(config.IMAGE_DIR, "Single_Scenario.fits")),
        help="Path to input FITS image",
    )
    parser.add_argument(
        "--series-csv",
        default=getattr(config, "DISPLAY_SCENARIO_CSV", getattr(config, "TRUTH_FILE", None)),
        help="Scenario CSV for slideshow mode",
    )
    _series_default = str(getattr(config, "PROC_MODE", "single")).strip().lower() == "batch"
    parser.add_argument(
        "--series",
        action="store_true",
        default=_series_default,
        help="Display image series from scenario CSV (keys: n/p or arrow keys)",
    )
    parser.add_argument(
        "--no-series",
        dest="series",
        action="store_false",
        help="Force single-image display regardless of PROC_MODE",
    )
    parser.add_argument("--monitor-width", type=int, default=getattr(config, "DISPLAY_MONITOR_WIDTH", 1920))
    parser.add_argument("--monitor-height", type=int, default=getattr(config, "DISPLAY_MONITOR_HEIGHT", 1080))
    parser.add_argument("--offset-x", type=int, default=getattr(config, "DISPLAY_OFFSET_X", 0))
    parser.add_argument("--offset-y", type=int, default=getattr(config, "DISPLAY_OFFSET_Y", 0))
    parser.add_argument(
        "--monitor-index",
        type=int,
        default=getattr(config, "DISPLAY_MONITOR_INDEX", None),
        help="Use detected monitor by index (run with --list-monitors to inspect)",
    )
    parser.add_argument(
        "--list-monitors",
        action="store_true",
        help="Print detected monitor rectangles and exit",
    )
    parser.add_argument(
        "--choose-monitor",
        action="store_true",
        help="Prompt to choose a monitor from detected displays",
    )
    parser.add_argument("--dpi", type=int, default=getattr(config, "DISPLAY_DPI", 100))
    parser.add_argument("--gamma", type=float, default=getattr(config, "DISPLAY_GAMMA", 2.2))
    parser.add_argument(
        "--apply-gamma",
        dest="apply_gamma",
        action="store_true",
        help="Enable gamma pre-correction",
    )
    parser.add_argument(
        "--no-gamma",
        dest="apply_gamma",
        action="store_false",
        help="Disable gamma pre-correction",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Keep window border/title bar (default is borderless)",
    )
    parser.set_defaults(apply_gamma=bool(getattr(config, "DISPLAY_APPLY_GAMMA", True)))
    args = parser.parse_args()

    resolved = _resolve_target_monitor(args)
    if resolved is None:
        return

    target_w, target_h, target_x, target_y = resolved

    if args.series:
        paths = _load_series_paths_from_csv(args.series_csv, config.IMAGE_DIR)
        idx = int(getattr(config, "DISPLAY_SERIES_START_INDEX", 0))
        if idx < 0 or idx >= len(paths):
            idx = 0
    else:
        paths = [args.input]
        idx = 0

    image = _read_fits(paths[idx])
    display_u8 = _to_display_u8(image, gamma=args.gamma, apply_gamma=args.apply_gamma)
    canvas = _pad_center(display_u8, out_h=target_h, out_w=target_w)

    if args.series:
        print(f"[{idx+1}/{len(paths)}] {os.path.basename(paths[idx])}")
    else:
        print(f"Displaying: {paths[idx]}")
    print(f"Image size: {display_u8.shape[1]}x{display_u8.shape[0]}")
    print(f"Monitor window: {target_w}x{target_h} at ({target_x}, {target_y})")
    print(f"Gamma correction: {'ON' if args.apply_gamma else 'OFF'}")
    if args.apply_gamma:
        print(f"Gamma value: {args.gamma:.3f}")
    if args.series:
        print("Series mode keys: N/Right/Space=next, P/Left=previous, Q/Esc=quit")
    else:
        print("Close with Q or Esc.")

    fig = plt.figure(figsize=(target_w / args.dpi, target_h / args.dpi), dpi=args.dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    im_artist = ax.imshow(canvas, cmap="gray", vmin=0, vmax=255, origin="upper", interpolation="nearest")

    _set_window_geometry(
        fig,
        width=target_w,
        height=target_h,
        x=target_x,
        y=target_y,
        borderless=not args.windowed,
    )

    def _set_frame(new_idx):
        nonlocal idx
        idx = new_idx % len(paths)
        current = paths[idx]
        img = _read_fits(current)
        u8 = _to_display_u8(img, gamma=args.gamma, apply_gamma=args.apply_gamma)
        can = _pad_center(u8, out_h=target_h, out_w=target_w)
        im_artist.set_data(can)
        fig.canvas.draw_idle()
        if args.series:
            print(f"[{idx+1}/{len(paths)}] {os.path.basename(current)}")

    def _on_key(event):
        if event.key in {"q", "Q", "escape"}:
            plt.close(fig)
        elif args.series and event.key in {"right", "n", "N", " "}:
            _set_frame(idx + 1)
        elif args.series and event.key in {"left", "p", "P", "backspace"}:
            _set_frame(idx - 1)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show()


if __name__ == "__main__":
    main()
