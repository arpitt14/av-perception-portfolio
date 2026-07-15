import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiceLoss(nn.Module):
    """
    Soft Dice loss for multiclass segmentation.

    Dice = (2 * |P ∩ G|) / (|P| + |G|)

    We compute per-class Dice scores and return 1 - mean_dice so the value
    is minimized during training (higher overlap = lower loss).

    smooth prevents division by zero on empty classes — without it, a class
    absent from both prediction and ground truth produces 0/0 = NaN.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) — raw model output, no softmax applied yet
            targets: (B, H, W)    — integer class indices, dtype torch.long
        """
        num_classes = logits.shape[1]

        # Convert logits → probabilities along class dimension
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, H, W) → (B, H, W, C) → (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Flatten spatial dimensions for dot-product intersection computation
        # probs and targets_one_hot both become (B, C, H*W)
        probs_flat   = probs.view(probs.shape[0], num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)

        # Per-class intersection and union, summed over batch and spatial dims
        intersection = (probs_flat * targets_flat).sum(dim=(0, 2))
        cardinality  = probs_flat.sum(dim=(0, 2)) + targets_flat.sum(dim=(0, 2))

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of CrossEntropyLoss and DiceLoss.

    CrossEntropy provides stable per-pixel gradient signals.
    Dice provides class-balanced region overlap.
    dice_weight=0.5 gives equal contribution from both terms.
    """

    def __init__(self, num_classes: int, dice_weight: float = 0.5):
        super().__init__()
        # CrossEntropyLoss expects raw logits (B, C, H, W) and integer targets (B, H, W)
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        logger.info(f"CombinedLoss — CE + {dice_weight} * Dice, num_classes: {num_classes}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + self.dice_weight * dice_loss


if __name__ == "__main__":
    from src.utils.file_utils import load_yaml
    config = load_yaml("configs/default.yaml")
    num_classes = config["data"]["num_classes"]

    loss_fn = CombinedLoss(num_classes=num_classes)

    # Dummy logits and targets — verify loss computes without error
    dummy_logits  = torch.randn(4, num_classes, 360, 480)
    dummy_targets = torch.randint(0, num_classes, (4, 360, 480))

    loss = loss_fn(dummy_logits, dummy_targets)
    logger.info(f"Loss value on random inputs: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN — check smooth value or input shapes"
    logger.info("Loss sanity check passed.")
