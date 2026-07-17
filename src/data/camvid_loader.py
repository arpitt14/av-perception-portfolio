import pathlib
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from src.utils.logger import get_logger
from src.utils.file_utils import load_yaml

logger = get_logger(__name__)


def build_color_to_index_map(class_dict_path: str) -> dict:
    """
    Reads class_dict.csv and returns a dict mapping (R, G, B) → class_index.
    Used only as fallback when pre-indexed masks are not available.
    """
    import csv
    color_map = {}
    with open(class_dict_path, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            r, g, b = int(row["r"]), int(row["g"]), int(row["b"])
            color_map[(r, g, b)] = idx
    return color_map


def rgb_mask_to_index(mask_rgb: np.ndarray, color_map: dict) -> np.ndarray:
    """Converts RGB mask (H, W, 3) to integer index mask (H, W). Slow fallback path."""
    h, w = mask_rgb.shape[:2]
    index_mask = np.zeros((h, w), dtype=np.int64)
    for color, idx in color_map.items():
        match = np.all(mask_rgb == color, axis=-1)
        index_mask[match] = idx
    return index_mask


class CamVidDataset(Dataset):
    """
    Loads CamVid images and segmentation masks.

    Fast path: loads pre-indexed masks from {split}_labels_indexed/ — single
    PNG read, no color loop. Falls back to RGB conversion if indexed masks
    are not present (first-time setup or local dev without pre-conversion).
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        image_size: tuple,
        color_map: dict,
    ):
        root = pathlib.Path(root_dir)

        self.image_dir  = root / split
        self.image_size = image_size
        self.color_map  = color_map

        # Fast path: pre-indexed masks saved by the one-time conversion script
        indexed_dir = root / f"{split}_labels_indexed"
        if indexed_dir.exists():
            self.mask_dir    = indexed_dir
            self.use_indexed = True
            logger.info(f"CamVidDataset [{split}]: using pre-indexed masks (fast path)")
        else:
            self.mask_dir    = root / f"{split}_labels"
            self.use_indexed = False
            logger.info(f"CamVidDataset [{split}]: using RGB masks (slow path — run conversion)")

        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.mask_paths  = sorted(self.mask_dir.glob("*.png"))

        assert len(self.image_paths) > 0, f"No images found in {self.image_dir}"
        assert len(self.image_paths) == len(self.mask_paths), (
            f"Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks"
        )
        logger.info(f"CamVidDataset [{split}]: {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        h, w  = self.image_size

        image = TF.resize(image, [h, w], interpolation=InterpolationMode.BILINEAR)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        if self.use_indexed:
            # Fast path: mask is already a single-channel integer index image
            mask_raw = np.array(Image.open(self.mask_paths[idx]))
            mask_pil = Image.fromarray(mask_raw)
        else:
            # Slow fallback: convert RGB → index at load time
            mask_rgb = np.array(Image.open(self.mask_paths[idx]).convert("RGB"))
            index    = rgb_mask_to_index(mask_rgb, self.color_map)
            mask_pil = Image.fromarray(index.astype(np.uint8))

        mask_pil = TF.resize(mask_pil, [h, w], interpolation=InterpolationMode.NEAREST)
        mask     = torch.from_numpy(np.array(mask_pil)).long()

        return image, mask


def get_dataloaders(config: dict):
    data_cfg     = config["data"]
    training_cfg = config["training"]
    image_size   = (data_cfg["image_height"], data_cfg["image_width"])

    class_dict_path = pathlib.Path(data_cfg["root_dir"]) / "class_dict.csv"
    color_map = build_color_to_index_map(str(class_dict_path))
    logger.info(f"Loaded color map: {len(color_map)} classes from class_dict.csv")

    train_dataset = CamVidDataset(data_cfg["root_dir"], "train", image_size, color_map)
    val_dataset   = CamVidDataset(data_cfg["root_dir"], "val",   image_size, color_map)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    logger.info(
        f"Dataloaders ready — "
        f"train batches: {len(train_loader)}, val batches: {len(val_loader)}"
    )
    return train_loader, val_loader


if __name__ == "__main__":
    config = load_yaml("configs/default.yaml")
    train_loader, val_loader = get_dataloaders(config)

    images, masks = next(iter(train_loader))
    logger.info(f"Train batch — images: {images.shape}  masks: {masks.shape}  dtype: {masks.dtype}")
    logger.info(f"Mask value range: [{masks.min().item()}, {masks.max().item()}]")
