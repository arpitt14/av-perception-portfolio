# src/geometry/run_pipeline.py

from pathlib import Path
import cv2
import numpy as np

from src.geometry.projection import build_intrinsic_matrix, build_extrinsic_matrix, project_points
from src.geometry.lidar_projection import (
    load_velo_to_cam, load_cam_to_cam, load_lidar_points,
    project_lidar_to_image, colorize_by_depth
)
from src.geometry.homography_bev import compute_homography, apply_ipm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_geometry_pipeline(
    calib_dir: Path,
    sequence_dir: Path,
    frame_id: str = "0000000005"
) -> None:
    """
    End-to-end geometry pipeline for one KITTI frame.

    Stage 1: LiDAR-to-camera projection with depth colormap overlay.
    Stage 2: Inverse Perspective Mapping to produce top-down BEV view.

    All outputs saved to assets/ for README display.
    """
    logger.info("=== Week 1 Geometry Pipeline ===")

    # ── Stage 1: LiDAR projection ─────────────────────────────────────────────
    logger.info("Stage 1: LiDAR-to-camera projection")

    image_path = sequence_dir / "image_02" / "data" / f"{frame_id}.png"
    lidar_path = sequence_dir / "velodyne_points" / "data" / f"{frame_id}.bin"

    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        raise SystemExit(1)
    logger.info(f"Loaded camera frame: {image.shape} from {image_path.name}")

    T_velo_to_cam = load_velo_to_cam(str(calib_dir / "calib_velo_to_cam.txt"))
    R_rect, P_rect = load_cam_to_cam(str(calib_dir / "calib_cam_to_cam.txt"), cam_id=2)

    points_xyz = load_lidar_points(str(lidar_path))

    pixels, depths = project_lidar_to_image(
        points_xyz, T_velo_to_cam, R_rect, P_rect,
        image_shape=image.shape[:2]
    )

    colors = colorize_by_depth(depths, max_depth=50.0)
    overlay = image.copy()
    for i, (u, v) in enumerate(pixels):
        cv2.circle(overlay, (u, v), radius=2, color=colors[i].tolist(), thickness=-1)

    lidar_output = Path("assets/lidar_overlay_kitti.png")
    cv2.imwrite(str(lidar_output), overlay)
    logger.info(f"LiDAR overlay saved to {lidar_output}")

    # ── Stage 2: IPM BEV warp ─────────────────────────────────────────────────
    logger.info("Stage 2: Inverse Perspective Mapping")

    # Source points manually selected on frame 0000000000.png road surface.
    src_points = np.float32([
        [120, 348],
        [712, 336],
        [458, 191],
        [389, 195],
    ])
    dst_points = np.float32([
        [ 50, 450],
        [450, 450],
        [450,  50],
        [ 50,  50],
    ])

    H = compute_homography(src_points, dst_points)
    bev = apply_ipm(image, H, output_size=(500, 500))

    bev_output = Path("assets/bev_ipm_kitti.png")
    cv2.imwrite(str(bev_output), bev)
    logger.info(f"BEV output saved to {bev_output}")

    # ── Stage 3: verify projection module independently ───────────────────────
    logger.info("Stage 3: Verifying synthetic projection unit test")

    K = build_intrinsic_matrix(fx=718.856, fy=718.856, cx=607.193, cy=185.216)
    R = np.eye(3)
    t = np.zeros(3)
    build_extrinsic_matrix(R, t)

    test_point = np.array([[0.0, 0.0, 10.0]])
    result = project_points(test_point, K, R, t)
    assert np.allclose(result[0], [607.193, 185.216], atol=1e-3), \
        f"Projection unit test failed: got {result[0]}"
    logger.info("Synthetic projection unit test passed")

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    run_geometry_pipeline(
        calib_dir=Path("src/geometry/data/2011_09_26"),
        sequence_dir=Path("src/geometry/data/2011_09_26_drive_0002_sync"),
        frame_id="0000000005"
    )
