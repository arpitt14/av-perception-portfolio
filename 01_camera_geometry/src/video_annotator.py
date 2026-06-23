from __future__ import annotations
import cv2
import time
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.file_utils import ensure_output_dir

logger = get_logger(__name__)


def annotate_image_sequence(
    frames_dir: str,
    output_dir: str,
    box: tuple[int, int, int, int] = (200, 100, 500, 400),
    label: str = "test box",
) -> None:
    """
    Reads a folder of sequential image frames (KITTI-style),
    annotates each one, and saves the result as a video.
    """
    # ── 1. Find all frames ────────────────────────────────────────────────────
    from src.utils.file_utils import get_image_paths, ensure_output_dir

    frame_paths = get_image_paths(frames_dir)
    if not frame_paths:
        logger.error(f"No images found in: {frames_dir}")
        return
    logger.info(f"Found {len(frame_paths)} frames in {frames_dir}")

    # ── 2. Read the first frame to get dimensions ─────────────────────────────
    first = cv2.imread(str(frame_paths[0]))
    height, width = first.shape[:2]
    logger.info(f"Frame size: {width}×{height}")

    # ── 3. Set up video writer ────────────────────────────────────────────────
    out_path = ensure_output_dir(output_dir, "kitti_annotated") / "output.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,          # KITTI sequences are ~10 fps
        (width, height),
    )

    # ── 4. Read → process → write loop ───────────────────────────────────────
    t_start = time.perf_counter()

    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            logger.warning(f"Could not read frame: {frame_path.name}, skipping")
            continue

        # Draw frame number so you can see it's actually playing sequentially
        cv2.putText(frame, f"Frame: {i:04d}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        # Draw bounding box
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)

    elapsed = time.perf_counter() - t_start
    logger.info(f"Done: {len(frame_paths)} frames in {elapsed:.2f}s → {out_path}")
    writer.release()


if __name__ == "__main__":
    annotate_image_sequence(
        frames_dir="01_camera_geometry/data/2011_09_26_drive_0002_sync/image_02/data",
        output_dir="01_camera_geometry/outputs/",
        box=(200, 100, 500, 400),
        label="Car: 0.94",
    )