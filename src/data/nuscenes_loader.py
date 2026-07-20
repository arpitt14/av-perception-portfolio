"""
src/data/nuscenes_loader.py

Wraps nuscenes-devkit access patterns needed for Week 3: scene iteration,
sample access, and calibrated-sensor + ego-pose lookups feeding the
LiDAR-to-camera projection chain (Day 2-3) and GeoJSON export (Day 5).

Run convention: python -m src.data.nuscenes_loader (from repo root).
"""

from pathlib import Path

from nuscenes.nuscenes import NuScenes

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NuScenesLoader:
    """Thin wrapper around the NuScenes devkit for scene/sample/sensor access.

    nuScenes is relational, not a flat file tree: a `scene` is a ~20s clip
    made of ordered `sample`s (keyframes, ~2Hz), stored as a linked list
    (`first_sample_token` -> `next` chain) rather than an indexable array.
    Each `sample` links out to per-sensor `sample_data` records, each
    carrying its own `calibrated_sensor` (fixed sensor->ego transform) and
    `ego_pose` (time-varying ego->global transform) tokens.
    """

    def __init__(self, dataroot: str, version: str = "v1.0-mini"):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        logger.info(
            f"Loaded nuScenes {version} from {dataroot} "
            f"({len(self.nusc.scene)} scenes)"
        )

    def list_scenes(self) -> list[str]:
        """Return all scene names available in this version."""
        return [s["name"] for s in self.nusc.scene]

    def get_scene_samples(self, scene_name: str) -> list[dict]:
        """Return ordered sample records for a given scene name.

        Walks the scene's sample linked list (first_sample_token -> next)
        since nuScenes does not store sample ordering as an indexable array.
        """
        scene = next(
            (s for s in self.nusc.scene if s["name"] == scene_name), None
        )
        if scene is None:
            raise ValueError(
                f"Scene '{scene_name}' not found. "
                f"Available: {self.list_scenes()}"
            )

        samples = []
        token = scene["first_sample_token"]
        while token:
            sample = self.nusc.get("sample", token)
            samples.append(sample)
            token = sample["next"]
        return samples

    def get_sensor_data(self, sample_token: str, sensor_channel: str) -> dict:
        """Return file path + calibration + ego pose for one sensor reading.

        sensor_channel examples: 'LIDAR_TOP', 'CAM_FRONT'.

        The returned ego_pose is the pose AT THIS SENSOR'S TIMESTAMP, not
        the sample's nominal timestamp -- LiDAR and camera keyframes are
        grouped into the same `sample` but are not identically stamped, so
        each sensor's own ego_pose must be looked up individually. This
        matters directly in Day 2/3: compensating for ego motion between
        the LiDAR read and the camera read is part of the projection chain.
        """
        sample = self.nusc.get("sample", sample_token)
        if sensor_channel not in sample["data"]:
            raise KeyError(
                f"'{sensor_channel}' not in sample data. "
                f"Available: {list(sample['data'].keys())}"
            )

        sd_token = sample["data"][sensor_channel]
        sample_data = self.nusc.get("sample_data", sd_token)
        calibrated_sensor = self.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

        return {
            "file_path": str(Path(self.nusc.dataroot) / sample_data["filename"]),
            "calibrated_sensor": calibrated_sensor,
            "ego_pose": ego_pose,
            "timestamp": sample_data["timestamp"],
            "sample_data_token": sd_token,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes mini root")
    args = parser.parse_args()

    loader = NuScenesLoader(dataroot=args.dataroot, version="v1.0-mini")
    scenes = loader.list_scenes()
    logger.info(f"First scene: {scenes[0]}")

    samples = loader.get_scene_samples(scenes[0])
    logger.info(f"Scene '{scenes[0]}' has {len(samples)} keyframe samples")

    lidar = loader.get_sensor_data(samples[0]["token"], "LIDAR_TOP")
    cam = loader.get_sensor_data(samples[0]["token"], "CAM_FRONT")
    logger.info(f"LIDAR_TOP file: {lidar['file_path']}")
    logger.info(f"CAM_FRONT file: {cam['file_path']}")
    logger.info(f"LIDAR timestamp: {lidar['timestamp']}, CAM timestamp: {cam['timestamp']}")
