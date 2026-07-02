# 01_camera_geometry/src/image_basics.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── 1. Read and inspect ───────────────────────────────────────────────────────
img_bgr = cv2.imread("data/sample_frame.jpg")

print(f"Shape:  {img_bgr.shape}")    # (height, width, 3)
print(f"Dtype:  {img_bgr.dtype}")    # uint8
print(f"Min px: {img_bgr.min()}  Max px: {img_bgr.max()}")

# ── 2. Display correctly in matplotlib ───────────────────────────────────────
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

Path("outputs").mkdir(exist_ok=True)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_bgr)          # wrong colours — intentional, see the difference
plt.title("BGR (wrong)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_rgb)          # correct colours
plt.title("RGB (correct)")
plt.axis("off")

plt.savefig("outputs/bgr_vs_rgb.png", bbox_inches="tight")
print("Saved bgr_vs_rgb.png")

# ── 3. Basic transforms ───────────────────────────────────────────────────────
resized = cv2.resize(img_bgr, (640, 480))
gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

print(f"Resized shape: {resized.shape}")   # (480, 640, 3)
print(f"Gray shape:    {gray.shape}")      # (480, width) — no channel dim

cv2.imwrite("outputs/resized.jpg", resized)
cv2.imwrite("outputs/gray.jpg", gray)

# ── 4. Draw a bounding box and label ─────────────────────────────────────────
annotated = img_bgr.copy()    # always work on a copy, never modify the original

cv2.rectangle(
    annotated,
    pt1=(100, 80),            # top-left corner (x, y)
    pt2=(400, 300),           # bottom-right corner (x, y)
    color=(0, 255, 0),        # green in BGR
    thickness=2
)
cv2.putText(
    annotated,
    text="Car: 0.94",
    org=(100, 74),            # position of text bottom-left corner
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.7,
    color=(0, 255, 0),
    thickness=2
)
cv2.imwrite("outputs/annotated.jpg", annotated)
print("Saved annotated.jpg")

# ── 5. Read a video frame by frame ───────────────────────────────────────────
# If you don't have a video file yet, skip this block for now
# and come back to it on Day 6 when you build the video annotator
cap = cv2.VideoCapture("data/sample_frame.jpg")  # placeholder
# We'll do the real video loop on Day 6