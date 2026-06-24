# src/week2/day1_read_write.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def inspect_image(img: np.ndarray, label: str) -> None:
    """Prints the key properties of an image array."""
    logger.info(f"--- {label} ---")
    logger.info(f"  Shape : {img.shape}")   # (Height, Width, Channels)
    logger.info(f"  Dtype : {img.dtype}")   # uint8 means pixels are 0-255
    logger.info(f"  Min   : {img.min()}")
    logger.info(f"  Max   : {img.max()}")


def main():
    # ── 1. Paths ──────────────────────────────────────────────────────────────
    input_path  = Path("assets/sample_frame.jpg")
    output_path = Path("outputs/day1_bgr_vs_rgb.png")
    output_path.parent.mkdir(exist_ok=True)

    # ── 2. Load the image ─────────────────────────────────────────────────────
    # cv2.imread returns a NumPy array. Shape: (H, W, 3).
    # The 3 channels are ordered Blue, Green, Red — NOT Red, Green, Blue.
    img_bgr = cv2.imread(str(input_path))

    if img_bgr is None:
        logger.error(f"Could not load image at {input_path}. Check the path.")
        return

    inspect_image(img_bgr, "Loaded image (BGR)")

    # ── 3. Convert to RGB ─────────────────────────────────────────────────────
    # Matplotlib's imshow() expects Red in channel 0, not Blue.
    # Without this conversion, reds look blue and blues look red.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── 4. Plot both side by side so you can SEE the difference ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(img_bgr)
    axes[0].set_title("Wrong — BGR passed to Matplotlib (notice the colour shift)")
    axes[0].axis("off")

    axes[1].imshow(img_rgb)
    axes[1].set_title("Correct — converted to RGB first")
    axes[1].axis("off")

    plt.suptitle("The BGR Trap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    logger.info(f"Comparison saved to {output_path}")

    # ── 5. Write back to disk ─────────────────────────────────────────────────
    # cv2.imwrite expects BGR — so no conversion needed here.
    # This round-trip (read → write) is the foundation of every pipeline.
    saved_path = Path("outputs/day1_saved.jpg")
    cv2.imwrite(str(saved_path), img_bgr)
    logger.info(f"Image saved to {saved_path}")

    # ── 6. One bonus thing: slice a single channel and look at it ─────────────
    # This makes the "array of numbers" concept concrete.
    blue_channel  = img_bgr[:, :, 0]   # all rows, all cols, channel index 0
    green_channel = img_bgr[:, :, 1]
    red_channel   = img_bgr[:, :, 2]

    logger.info(f"Blue channel shape : {blue_channel.shape}")   # (H, W) — no channel dim
    logger.info(f"A small patch of the blue channel:\n{blue_channel[100:105, 100:105]}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    axes2[0].imshow(blue_channel,  cmap="Blues_r")
    axes2[0].set_title("Blue channel")
    axes2[1].imshow(green_channel, cmap="Greens_r")
    axes2[1].set_title("Green channel")
    axes2[2].imshow(red_channel,   cmap="Reds_r")
    axes2[2].set_title("Red channel")
    for ax in axes2:
        ax.axis("off")
    plt.savefig("outputs/day1_channels.png", dpi=150, bbox_inches="tight")
    logger.info("Channel breakdown saved to outputs/day1_channels.png")


if __name__ == "__main__":
    main()