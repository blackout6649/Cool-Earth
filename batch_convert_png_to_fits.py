import argparse
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

import config


DEFAULT_INPUT_DIR = Path(config.CAPTURED_DIR) / "raw"
DEFAULT_OUTPUT_DIR = Path(config.CAPTURED_DIR)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    channels = image.shape[2]
    if channels == 4:
        image = image[:, :, :3]
        channels = 3

    if channels != 3:
        raise ValueError(f"Unsupported channel count: {channels}")

    # OpenCV loads PNG color data as BGR.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def convert_png_to_fits(png_path: Path, fits_path: Path, overwrite: bool) -> None:
    image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read PNG file: {png_path}")

    gray_image = _to_grayscale(image)

    header = fits.Header()
    header["SRCFILE"] = png_path.name

    fits.PrimaryHDU(data=gray_image, header=header).writeto(fits_path, overwrite=overwrite)


def batch_convert(input_dir: Path, output_dir: Path, overwrite: bool) -> int:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    png_paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")
    for png_path in png_paths:
        fits_path = output_dir / f"{png_path.stem}.fits"
        if fits_path.exists() and not overwrite:
            print(f"[SKIP] {fits_path.name} already exists")
            continue

        convert_png_to_fits(png_path, fits_path, overwrite=overwrite)
        converted_count += 1
        print(f"[OK] {png_path.name} -> {fits_path.name}")

    return converted_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert PNG images to grayscale FITS files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing PNG images to convert",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write FITS files into",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing FITS files in the output directory",
    )
    args = parser.parse_args()

    converted_count = batch_convert(args.input_dir, args.output_dir, overwrite=args.overwrite)
    print(f"Converted {converted_count} file(s).")


if __name__ == "__main__":
    main()