import pathlib
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from src.utils.logger import get_logger
from src.utils.file_utils import load_yaml
from src.models.unet import UNet
from src.data.camvid_loader import get_dataloaders, build_color_to_index_map

logger = get_logger(__name__)


def build_index_to_color_map(class_dict_path: str) -> np.ndarray:
    """
    Builds a (num_classes, 3) uint8 array mapping class index → RGB color.
    Used to colorize integer prediction masks for visualization.
    """
    import csv
    colors = []
    with open(class_dict_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            colors.append([int(row["r"]), int(row["g"]), int(row["b"])])
    return np.array(colors, dtype=np.uint8)


def colorize_mask(mask: np.ndarray, index_to_color: np.ndarray) -> np.ndarray:
    """
    Converts an integer index mask (H, W) to an RGB image (H, W, 3).
    Each pixel's class index is mapped to its corresponding color.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(index_to_color):
        rgb[mask == cls_idx] = color
    return rgb


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverses ImageNet normalization and converts tensor to uint8 numpy array.
    tensor: (3, H, W) float32
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def run_visualization(config_path: str, num_samples: int = 4) -> None:
    config = load_yaml(config_path)
    data_cfg = config["data"]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = UNet(in_channels=3, num_classes=data_cfg["num_classes"])
    checkpoint_path = pathlib.Path(config["training"]["checkpoint_dir"]) / "best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Build color map for visualization
    class_dict_path = pathlib.Path(data_cfg["root_dir"]) / "class_dict.csv"
    index_to_color  = build_index_to_color_map(str(class_dict_path))

    # Get val loader — shuffle=False so we get the first N samples deterministically
    _, val_loader = get_dataloaders(config)

    images_batch, masks_batch = next(iter(val_loader))
    images_batch = images_batch[:num_samples]
    masks_batch  = masks_batch[:num_samples]

    with torch.no_grad():
        logits = model(images_batch.to(device))
        preds  = logits.argmax(dim=1).cpu().numpy()  # (N, H, W)

    masks_np = masks_batch.numpy()  # (N, H, W)

    # Build grid: num_samples rows × 3 columns (image | gt mask | pred mask)
    h, w    = images_batch.shape[2], images_batch.shape[3]
    padding = 4
    grid_h  = num_samples * (h + padding) + padding
    grid_w  = 3 * (w + padding) + padding
    grid    = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 40  # dark grey background

    col_labels = ["Input Image", "Ground Truth", "Prediction"]

    for i in range(num_samples):
        row_start = padding + i * (h + padding)

        # Column 0: original image (denormalized)
        img_np = denormalize(images_batch[i])
        grid[row_start:row_start+h, padding:padding+w] = img_np

        # Column 1: ground truth mask colorized
        gt_color = colorize_mask(masks_np[i], index_to_color)
        col1_start = padding + w + padding
        grid[row_start:row_start+h, col1_start:col1_start+w] = gt_color

        # Column 2: predicted mask colorized
        pred_color = colorize_mask(preds[i], index_to_color)
        col2_start = col1_start + w + padding
        grid[row_start:row_start+h, col2_start:col2_start+w] = pred_color

    output_path = pathlib.Path("assets/camvid_prediction_grid.png")
    Image.fromarray(grid).save(output_path)
    logger.info(f"Saved prediction grid to {output_path}")


if __name__ == "__main__":
    run_visualization("configs/default.yaml", num_samples=4)
