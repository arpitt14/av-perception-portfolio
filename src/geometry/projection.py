# src/geometry/projection.py

import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Constructs the 3x3 camera intrinsic matrix K.

    K encodes how the physical lens maps 3D camera-frame points onto the 2D sensor.
    fx/fy: focal lengths in pixels. cx/cy: principal point (optical axis on sensor).
    """
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    logger.debug(f"Built intrinsic matrix K:\n{K}")
    return K


def build_extrinsic_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Constructs the 3x4 extrinsic matrix [R|t].

    R (3x3): rotation — camera orientation in the world.
    t (3x1): translation — camera position in the world.
    Together they transform a point from world coordinates to camera coordinates.
    """
    t = t.reshape(3, 1)
    E = np.hstack([R, t])   # shape: (3, 4)
    logger.debug(f"Built extrinsic matrix [R|t]:\n{E}")
    return E


def project_points(
    points_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Projects 3D world points onto the image plane.

    Full coordinate chain: world → camera frame → image plane → pixel coordinates.
    Perspective division (divide by Z) is applied after the linear projection —
    this is what makes distant objects appear smaller.

    Args:
        points_3d: (N, 3) array of 3D world points.
        K:         (3, 3) intrinsic matrix.
        R:         (3, 3) rotation matrix.
        t:         (3,)   translation vector.

    Returns:
        (N, 2) array of (u, v) pixel coordinates for points with Z > 0.
        Points behind the camera (Z <= 0) are filtered out before division
        to prevent invalid projections.
    """
    t = t.reshape(3, 1)

    # Step 1: world → camera frame.  P_c = R @ P_w + t
    # points_3d.T is (3, N); result P_c is (3, N)
    P_c = R @ points_3d.T + t

    # Step 2: filter points behind the camera.
    # Z <= 0 means the point is behind or at the camera origin.
    # Perspective division by a non-positive Z produces nonsensical pixel coords.
    valid = P_c[2, :] > 0
    P_c = P_c[:, valid]

    if P_c.shape[1] == 0:
        logger.warning("No points with Z > 0 after filtering — check your coordinate setup.")
        return np.empty((0, 2))

    # Step 3: apply K.  p = K @ P_c, result is (3, N) in homogeneous coords
    p_hom = K @ P_c

    # Step 4: perspective division.  u = X/Z, v = Y/Z
    u = p_hom[0, :] / p_hom[2, :]
    v = p_hom[1, :] / p_hom[2, :]

    pixels = np.stack([u, v], axis=1)   # (N, 2)
    logger.debug(f"Projected {points_3d.shape[0]} points → {pixels.shape[0]} valid pixels.")
    return pixels


if __name__ == "__main__":
    # Synthetic unit test: a point 10m ahead of a forward-facing camera
    # placed at the world origin should project near the principal point.

    K = build_intrinsic_matrix(fx=718.856, fy=718.856, cx=607.193, cy=185.216)

    # Camera at world origin, no rotation — identity R, zero t
    R = np.eye(3)
    t = np.zeros(3)
    E = build_extrinsic_matrix(R, t)

    # A point directly in front of the camera at (0, 0, 10) in world coords
    points = np.array([[0.0, 0.0, 10.0]])
    pixels = project_points(points, K, R, t)

    logger.info(f"Intrinsic matrix K:\n{K}")
    logger.info(f"Extrinsic matrix [R|t]:\n{E}")
    logger.info(f"3D point: {points[0]}")
    logger.info(f"Projected pixel: {pixels[0]}")
    logger.info(f"Expected: close to principal point (cx={607.193}, cy={185.216})")

    # Verify: a point at (0,0,10) with identity R and zero t should project
    # exactly to (cx, cy) — the principal point.
    assert np.allclose(pixels[0], [607.193, 185.216], atol=1e-3), \
        f"Projection test failed: got {pixels[0]}"
    logger.info("Assertion passed — projection is mathematically correct.")