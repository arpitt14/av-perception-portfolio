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

    CamVid masks from Kaggle are RGB-encoded — each pixel's color identifies
    its class, not a raw integer index. We build this lookup table once at
    startup and reuse it for every mask in the dataset.
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
    """
    Converts an RGB mask (H, W, 3) to a class-index mask (H, W).

    For each class color in the map, finds all pixels matching that color
    and writes the class index into those positions. Pixels with no match
    (void/unlabeled) remain 0.
    """
    h, w = mask_rgb.shape[:2]
    index_mask = np.zeros((h, w), dtype=np.int64)
    for color, idx in color_map.items():
        match = np.all(mask_rgb == color, axis=-1)
        index_mask[match] = idx
    return index_mask


class CamVidDataset(Dataset):
    """
    Loads CamVid images and segmentation masks (Kaggle folder layout).

    Kaggle CamVid structure:
        train/          RGB camera images (.png)
        train_labels/   RGB-encoded class masks (.png)
        val/            RGB camera images (.png)
        val_labels/     RGB-encoded class masks (.png)
        class_dict.csv  color → class name mapping
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
        self.mask_dir   = root / f"{split}_labels"
        self.image_size = image_size
        self.color_map  = color_map

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
        image    = Image.open(self.image_paths[idx]).convert("RGB")
        mask_rgb = np.array(Image.open(self.mask_paths[idx]).convert("RGB"))

        h, w = self.image_size

        # Resize image with bilinear — correct for continuous RGB pixel values
        image = TF.resize(image, [h, w], interpolation=InterpolationMode.BILINEAR)

        # Convert RGB mask → integer indices BEFORE resizing.
        # Resizing an RGB mask with bilinear would blend class colors, producing
        # colors not in the lookup table and silently breaking the conversion.
        index_mask     = rgb_mask_to_index(mask_rgb, self.color_map)
        index_mask_pil = Image.fromarray(index_mask.astype(np.uint8))
        index_mask_pil = TF.resize(
            index_mask_pil, [h, w], interpolation=InterpolationMode.NEAREST
        )

        # to_tensor: PIL → float32 (3, H, W) in [0, 1]
        # normalize: ImageNet stats — compatible with pretrained encoders later
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # Mask must be torch.long (int64) — required by CrossEntropyLoss
        mask = torch.from_numpy(np.array(index_mask_pil)).long()

        return image, mask


def get_dataloaders(config: dict):
    """
    Constructs train and val DataLoaders from config dict.
    Builds the color map once and shares it across both datasets.
    """
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

    images, masks = next(iter(val_loader))
    logger.info(f"Val batch   — images: {images.shape}  masks: {masks.shape}  dtype: {masks.dtype}")
    logger.info(f"Mask value range: [{masks.min().item()}, {masks.max().item()}]")
