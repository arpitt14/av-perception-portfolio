import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.logger import get_logger
from src.utils.file_utils import load_yaml
from src.data.camvid_loader import get_dataloaders
from src.models.unet import UNet
from src.models.loss import CombinedLoss

logger = get_logger(__name__)


def compute_miou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """
    Computes mean Intersection over Union across all classes present in the batch.

    IoU per class = |pred ∩ target| / |pred ∪ target|
    Classes absent from both prediction and target are excluded from the mean
    rather than contributing a 0/0 term.
    """
    iou_list = []
    pred_classes = preds.argmax(dim=1)  # (B, H, W)

    for cls in range(num_classes):
        pred_mask   = (pred_classes == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().item()
        union        = (pred_mask | target_mask).sum().item()

        if union == 0:
            continue
        iou_list.append(intersection / union)

    return sum(iou_list) / len(iou_list) if iou_list else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Runs one full pass over the training set. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            logger.info(f"  batch {batch_idx}/{len(loader)} — loss: {loss.item():.4f}")

    return total_loss / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float]:
    """Runs inference over the val set. Returns (mean_loss, mean_iou)."""
    model.eval()
    total_loss = 0.0
    total_miou = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)
            loss   = loss_fn(logits, masks)

            total_loss += loss.item()
            total_miou += compute_miou(logits, masks, num_classes)

    return total_loss / len(loader), total_miou / len(loader)


def run_training(config_path: str) -> None:
    config       = load_yaml(config_path)
    data_cfg     = config["data"]
    training_cfg = config["training"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(config)

    model     = UNet(in_channels=3, num_classes=data_cfg["num_classes"]).to(device)
    loss_fn   = CombinedLoss(num_classes=data_cfg["num_classes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["learning_rate"])

    checkpoint_dir = pathlib.Path(training_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, training_cfg["epochs"] + 1):
        logger.info(f"Epoch {epoch}/{training_cfg['epochs']}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_miou = validate(model, val_loader, loss_fn, device, data_cfg["num_classes"])

        logger.info(
            f"Epoch {epoch} — train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | val_mIoU: {val_miou:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            logger.info(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")

    logger.info("Training complete.")
    logger.info(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    run_training("configs/default.yaml")
