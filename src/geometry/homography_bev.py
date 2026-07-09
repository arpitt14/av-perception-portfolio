# src/geometry/homography_bev.py
import cv2
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
    if H is None:
        raise ValueError("Homography computation failed — check source points are not collinear.")
    logger.debug(f"Computed H:\n{H}")
    return H

def apply_ipm(image: np.ndarray, H: np.ndarray, output_size: tuple = (500, 500)) -> np.ndarray:
    width, height = output_size
    bev = cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_LINEAR)
    logger.debug(f"IPM output shape: {bev.shape}")
    return bev

if __name__ == "__main__":
    image_path = Path("src/geometry/data/2011_09_26_drive_0002_sync/image_02/data/0000000000.png")
    if not image_path.exists():
        logger.error(f"KITTI frame not found at {image_path}.")
        raise SystemExit(1)
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error("cv2.imread returned None.")
        raise SystemExit(1)
    logger.info(f"Loaded KITTI frame: {image.shape}")

    # Source points selected manually on frame 0000000000.png by clicking
    # the road surface — bottom-left, bottom-right, top-right, top-left.
    # Tuned to the tram-track road geometry visible in this KITTI sequence.
    src_points = np.float32([
        [120, 348],   # bottom-left
        [712, 336],   # bottom-right
        [458, 191],   # top-right
        [389, 195],   # top-left
    ])

    # Destination: rectangle in a 500x500 canvas.
    # Margins kept at 50px on each side to preserve context around the road.
    dst_points = np.float32([
        [ 50, 450],   # bottom-left
        [450, 450],   # bottom-right
        [450,  50],   # top-right
        [ 50,  50],   # top-left
    ])

    H = compute_homography(src_points, dst_points)
    logger.info(f"Homography matrix H:\n{H}")

    bev = apply_ipm(image, H, output_size=(500, 500))
    cv2.imwrite("assets/bev_ipm_kitti.png", bev)
    logger.info("BEV output saved to assets/bev_ipm_kitti.png")

    original_resized = cv2.resize(image, (500, 500))
    comparison = np.hstack([original_resized, bev])
    cv2.imwrite("assets/bev_ipm_comparison.png", comparison)
    logger.info("Comparison saved to assets/bev_ipm_comparison.png")
