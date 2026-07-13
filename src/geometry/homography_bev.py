# src/geometry/homography_bev.py

import cv2
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> np.ndarray:
    """
    Computes the 3x3 homography matrix H mapping src_points to dst_points.

    Uses RANSAC method for numerical stability even with manually selected
    points that may have slight pixel-level imprecision.

    Args:
        src_points: (4, 2) array of points in the source image (trapezoid on road).
        dst_points: (4, 2) array of corresponding points in the destination
                    top-down view (rectangle).

    Returns:
        H: (3, 3) homography matrix.
    """
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

    if H is None:
        raise ValueError(
            "Homography computation failed — check that your 4 source points "
            "are not collinear (3 or more points on the same line makes the "
            "system unsolvable)."
        )

    logger.debug(f"Computed homography matrix H:\n{H}")
    return H


def apply_ipm(
    image: np.ndarray,
    H: np.ndarray,
    output_size: tuple[int, int] = (500, 500)
) -> np.ndarray:
    """
    Applies Inverse Perspective Mapping to produce a top-down BEV image.

    cv2.warpPerspective applies H to every pixel simultaneously.
    INTER_LINEAR interpolation produces smoother output than nearest-neighbour
    when pixels are stretched during the warp — negligible computational cost.

    Args:
        image:       Input front-facing camera frame (H, W, 3).
        H:           (3, 3) homography matrix from compute_homography().
        output_size: (width, height) of the BEV output canvas in pixels.

    Returns:
        bev: Warped top-down image of shape (height, width, 3).
    """
    width, height = output_size
    bev = cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_LINEAR)
    logger.debug(f"IPM output shape: {bev.shape}")
    return bev


if __name__ == "__main__":
    # Load a KITTI front-camera frame.
    # These source points form a trapezoid on the road plane — wider at the
    # bottom (close range) and narrower at the top (far range), matching how
    # a flat road surface appears in a perspective image.
    # Tuned for KITTI image resolution 1242x375.
    image_path = Path("src/geometry/data/2011_09_26_drive_0002_sync/image_02/data/0000000000.png")

    if not image_path.exists():
        logger.error(f"KITTI frame not found at {image_path}. Check your data path.")
        raise SystemExit(1)

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("cv2.imread returned None — file may be corrupted.")
        raise SystemExit(1)

    logger.info(f"Loaded KITTI frame: {image.shape}")  # expect (375, 1242, 3)

    # Source points: trapezoid on the road plane in the perspective image.
    # These were selected by inspecting the KITTI frame visually.
    # Bottom edge is wide (close road), top edge is narrow (far road).
    src_points = np.float32([
        [200,  350],   # bottom-left
        [1100, 350],   # bottom-right
        [730,  250],   # top-right
        [550,  250],   # top-left
    ])

    # Destination points: rectangle in a 500x500 top-down canvas.
    # The road region maps to a rectangle occupying the centre of the canvas,
    # leaving margins for context.
    dst_points = np.float32([
        [100, 450],    # bottom-left
        [400, 450],    # bottom-right
        [400, 100],    # top-right
        [100, 100],    # top-left
    ])

    H = compute_homography(src_points, dst_points)
    logger.info(f"Homography matrix H:\n{H}")

    bev = apply_ipm(image, H, output_size=(500, 500))

    # Save BEV output to assets/ for README display.
    output_path = Path("assets/bev_ipm_kitti.png")
    cv2.imwrite(str(output_path), bev)
    logger.info(f"BEV output saved to {output_path}")

    # Also save a side-by-side comparison for visual inspection.
    # Resize original to same height as BEV for clean concatenation.
    original_resized = cv2.resize(image, (500, 500))
    comparison = np.hstack([original_resized, bev])
    comparison_path = Path("assets/bev_ipm_comparison.png")
    cv2.imwrite(str(comparison_path), comparison)
    logger.info(f"Side-by-side comparison saved to {comparison_path}")