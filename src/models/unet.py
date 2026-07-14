import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm → ReLU blocks.

    padding=1 on each 3×3 conv preserves spatial dimensions so resolution
    only changes at explicit pool/upsample steps. BatchNorm stabilizes
    gradient flow and typically halves the epochs needed to converge.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    DoubleConv followed by MaxPool2d downsampling.

    Returns both the pre-pool feature map (skip connection) and the pooled
    output (input to the next encoder stage).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """
    ConvTranspose2d upsampling followed by skip connection concatenation
    and DoubleConv fusion.

    When input spatial dimensions are not perfectly divisible by 2^4, the
    upsampled tensor can be 1 pixel shorter than the skip tensor in H or W.
    F.pad corrects this before concatenation — without it, torch.cat raises
    a size mismatch error at the offending decoder stage.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Pad x to match skip's spatial dimensions if they differ by 1 pixel.
        # This arises when H or W is not divisible by 2 at every pooling stage.
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        if diff_h > 0 or diff_w > 0:
            x = F.pad(x, [0, diff_w, 0, diff_h])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net encoder-decoder with skip connections for semantic segmentation.

    Follows Ronneberger et al. 2015 with BatchNorm added for training stability.
    No pretrained weights — trained from scratch on CamVid.

    Input:  (B, in_channels, H, W)
    Output: (B, num_classes, H, W) — raw logits, softmax applied at inference.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 32):
        super().__init__()

        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        logger.info(
            f"UNet initialised — in_channels: {in_channels}, num_classes: {num_classes}, "
            f"params: {sum(p.numel() for p in self.parameters()):,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return self.output_conv(x)


if __name__ == "__main__":
    from src.utils.file_utils import load_yaml
    config = load_yaml("configs/default.yaml")

    num_classes = config["data"]["num_classes"]
    model = UNet(in_channels=3, num_classes=num_classes)

    dummy = torch.zeros(1, 3, 360, 480)
    with torch.no_grad():
        out = model(dummy)

    logger.info(f"Input:  {dummy.shape}")
    logger.info(f"Output: {out.shape}")
    assert out.shape == (1, num_classes, 360, 480), f"Unexpected output shape: {out.shape}"
    logger.info("Shape assertion passed.")
