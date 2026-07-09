# src/geometry/lidar_projection.py

import numpy as np
import cv2
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_velo_to_cam(calib_path: str) -> np.ndarray:
    """
    Parses calib_velo_to_cam.txt and returns the 3x4 extrinsic matrix [R|t].

    R (3x3) and T (3,) are stored as flat space-separated values.
    We reshape and concatenate them into the standard [R|t] form.
    """
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                try:
                    data[key.strip()] = np.array([float(x) for x in val.strip().split()])
                except ValueError:
                    continue  # skip non-numeric lines like calib_time

    R = data["R"].reshape(3, 3)
    t = data["T"].reshape(3, 1)
    T_velo_to_cam = np.hstack([R, t])   # shape: (3, 4)
    logger.debug(f"T_velo_to_cam:\n{T_velo_to_cam}")
    return T_velo_to_cam


def load_cam_to_cam(calib_path: str, cam_id: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses calib_cam_to_cam.txt and returns R_rect and P_rect for the given camera.

    cam_id=2 selects the left colour camera (image_02/), which is the primary
    camera in KITTI used for most perception tasks.

    Returns:
        R_rect: (4, 4) rectification rotation matrix, padded to 4x4 so it can
                be chained with the 3x4 velo_to_cam matrix via matrix multiply.
        P_rect: (3, 4) projection matrix combining intrinsics and stereo baseline.
    """
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                try:
                    data[key.strip()] = np.array([float(x) for x in val.strip().split()])
                except ValueError:
                    continue  # skip non-numeric lines like calib_time

    cam_str = f"{cam_id:02d}"   # formats cam_id=2 as "02"

    R_rect_3x3 = data[f"R_rect_{cam_str}"].reshape(3, 3)

    # Pad R_rect to 4x4 so it can multiply a (4,N) homogeneous point matrix.
    # The extra row/column are [0,0,0,1] — identity for the homogeneous coordinate.
    R_rect = np.eye(4)
    R_rect[:3, :3] = R_rect_3x3

    P_rect = data[f"P_rect_{cam_str}"].reshape(3, 4)

    logger.debug(f"R_rect_{cam_str}:\n{R_rect}")
    logger.debug(f"P_rect_{cam_str}:\n{P_rect}")
    return R_rect, P_rect


def load_lidar_points(bin_path: str) -> np.ndarray:
    """
    Loads a KITTI LiDAR scan from a .bin file.

    Each point is stored as 4 float32 values: (X, Y, Z, reflectance).
    We load all points and return only the XYZ coordinates as (N, 3).
    Reflectance is discarded here — it's used in Week 5 snow injection.
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    xyz = points[:, :3]   # drop reflectance column
    logger.info(f"Loaded {xyz.shape[0]} LiDAR points from {Path(bin_path).name}")
    return xyz


def project_lidar_to_image(
    points_xyz: np.ndarray,
    T_velo_to_cam: np.ndarray,
    R_rect: np.ndarray,
    P_rect: np.ndarray,
    image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects LiDAR points onto the camera image plane.

    Full coordinate chain:
        LiDAR frame → camera frame (T_velo_to_cam)
                    → rectified camera frame (R_rect)
                    → homogeneous pixel (P_rect)
                    → pixel coordinates (perspective division)

    Points are filtered at two stages:
        1. Z > 0 after velo_to_cam transform — removes points behind the camera.
        2. Pixel coords within image bounds — removes points outside the frame.

    Args:
        points_xyz:    (N, 3) LiDAR points in LiDAR frame.
        T_velo_to_cam: (3, 4) LiDAR-to-camera extrinsic matrix.
        R_rect:        (4, 4) rectification rotation matrix.
        P_rect:        (3, 4) camera projection matrix.
        image_shape:   (height, width) of the target camera image.

    Returns:
        pixels:  (M, 2) array of (u, v) pixel coordinates for valid points.
        depths:  (M,)   array of Z depths for valid points, used for colouring.
    """
    height, width = image_shape

    # Convert to homogeneous coordinates: (N, 3) → (N, 4) by appending ones
    N = points_xyz.shape[0]
    ones = np.ones((N, 1))
    points_hom = np.hstack([points_xyz, ones])   # (N, 4)

    # Step 1: LiDAR → camera frame.  Result shape: (3, N)
    cam_points = T_velo_to_cam @ points_hom.T    # (3, 4) @ (4, N) = (3, N)

    # Step 2: filter points behind the camera (Z <= 0)
    valid_z = cam_points[2, :] > 0
    cam_points = cam_points[:, valid_z]

    # Step 3: pad to homogeneous 4D for R_rect multiply: (3,N) → (4,N)
    cam_points_hom = np.vstack([cam_points, np.ones((1, cam_points.shape[1]))])

    # Step 4: apply rectification.  (4,4) @ (4,N) = (4,N)
    rect_points = R_rect @ cam_points_hom        # (4, N)

    # Step 5: project to image plane.  (3,4) @ (4,N) = (3,N)
    img_points = P_rect @ rect_points            # (3, N)

    # Step 6: perspective division to get pixel coords
    depths = img_points[2, :]                    # Z values before division
    u = img_points[0, :] / depths
    v = img_points[1, :] / depths

    # Step 7: filter to image bounds
    valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_uv].astype(np.int32)
    v = v[valid_uv].astype(np.int32)
    depths = depths[valid_uv]

    pixels = np.stack([u, v], axis=1)   # (M, 2)
    logger.info(f"Projected {N} points → {pixels.shape[0]} visible on image")
    return pixels, depths


def colorize_by_depth(depths: np.ndarray, max_depth: float = 50.0) -> np.ndarray:
    """
    Maps depth values to BGR colours using JET colormap.

    Close points (small depth) → red.
    Far points (large depth)   → blue.
    Depths are clipped at max_depth metres before normalisation — points
    beyond this range are all blue and add no useful visual information.
    """
    depths_clipped = np.clip(depths, 0, max_depth)
    normalized = (depths_clipped / max_depth * 255).astype(np.uint8)
    # Invert so close = red (high value in JET) and far = blue (low value)
    normalized = 255 - normalized
    colors = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET)
    return colors.reshape(-1, 3)   # (M, 3) BGR


if __name__ == "__main__":
    # Paths
    calib_dir   = Path("src/geometry/data/2011_09_26")
    sequence    = Path("src/geometry/data/2011_09_26_drive_0002_sync")
    frame_id    = "0000000005"

    image_path  = sequence / "image_02" / "data" / f"{frame_id}.png"
    lidar_path  = sequence / "velodyne_points" / "data" / f"{frame_id}.bin"

    # Load calibration
    T_velo_to_cam = load_velo_to_cam(str(calib_dir / "calib_velo_to_cam.txt"))
    R_rect, P_rect = load_cam_to_cam(str(calib_dir / "calib_cam_to_cam.txt"), cam_id=2)

    # Load image and LiDAR
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        raise SystemExit(1)

    points_xyz = load_lidar_points(str(lidar_path))

    # Project
    pixels, depths = project_lidar_to_image(
        points_xyz, T_velo_to_cam, R_rect, P_rect,
        image_shape=image.shape[:2]
    )

    # Overlay coloured points on image
    colors = colorize_by_depth(depths, max_depth=50.0)
    overlay = image.copy()
    for i, (u, v) in enumerate(pixels):
        cv2.circle(overlay, (u, v), radius=2, color=colors[i].tolist(), thickness=-1)

    # Save
    output_path = Path("assets/lidar_overlay_kitti.png")
    cv2.imwrite(str(output_path), overlay)
    logger.info(f"LiDAR overlay saved to {output_path}")
