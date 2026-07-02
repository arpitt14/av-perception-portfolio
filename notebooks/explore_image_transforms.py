# src/week2/day2_transforms.py

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


def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    img = cv2.imread("assets/sample_frame.jpg")
    if img is None:
        logger.error("Could not load image. Check assets/sample_frame.jpg")
        return

    h, w = img.shape[:2]
    logger.info(f"Original size: width={w}, height={h}")

    # ── 2. Resize ─────────────────────────────────────────────────────────────
    # Neural networks need fixed-size inputs — you can't feed variable
    # sized images into a network with fixed weight matrices.
    #
    # IMPORTANT: cv2.resize takes (width, height) — the OPPOSITE of img.shape
    # img.shape → (height, width, channels)
    # cv2.resize → (width, height)
    # This axis reversal is one of the most common bugs in CV code.
    resized_640 = cv2.resize(img, (640, 640))
    resized_224 = cv2.resize(img, (224, 224))

    logger.info(f"Resized to 640x640: {resized_640.shape}")
    logger.info(f"Resized to 224x224: {resized_224.shape}")

    # ── 3. Grayscale ──────────────────────────────────────────────────────────
    # Converts (H, W, 3) → (H, W)
    # The channel dimension disappears entirely.
    # Used when colour information isn't needed — reduces computation.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    logger.info(f"Grayscale shape: {gray.shape}")  # no channel dim

    # ── 4. Normalization ──────────────────────────────────────────────────────
    # Raw pixels are integers in [0, 255].
    # Neural networks train better with small float values in [0.0, 1.0].
    # Two steps: change dtype first, then divide.
    #
    # Why change dtype first?
    # uint8 can only hold integers 0-255. If you divide 128 (uint8) by 255,
    # Python truncates to 0 — you lose all your data.
    # float32 can hold decimals, so 128.0 / 255.0 = 0.502 correctly.
    normalized = img.astype("float32") / 255.0

    logger.info(f"Original dtype : {img.dtype}        range: [{img.min()}, {img.max()}]")
    logger.info(f"Normalized dtype: {normalized.dtype}  range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # ── 5. Visualize everything ───────────────────────────────────────────────
    Path("outputs").mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original ({w}x{h})")

    # Resized
    axes[0, 1].imshow(cv2.cvtColor(resized_640, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Resized (640x640)")

    # Grayscale — note: cmap="gray" needed so matplotlib
    # doesn't try to colour a 2D array with a default colormap
    axes[1, 0].imshow(gray, cmap="gray")
    axes[1, 0].set_title(f"Grayscale {gray.shape}")

    # Normalized — values are floats now, matplotlib handles [0,1] floats fine
    axes[1, 1].imshow(cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Normalized [0.0, 1.0]")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("Image Transforms", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/day2_transforms.png", dpi=150, bbox_inches="tight")
    logger.info("Saved outputs/day2_transforms.png")

    # ── 6. One thing to verify manually ──────────────────────────────────────
    # Confirm the dtype trap with your own eyes:
    bad  = img[0, 0, 0]          # a uint8 pixel value, e.g. 180
    good = img.astype("float32")[0, 0, 0] / 255.0
    logger.info(f"Top-left blue pixel — raw: {bad}, normalized: {good:.4f}")


if __name__ == "__main__":
    main()