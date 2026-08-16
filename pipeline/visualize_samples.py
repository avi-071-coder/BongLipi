"""Create a visual grid showing samples from each class."""
import cv2
import numpy as np
from pathlib import Path
import json

DATASET = Path("dataset")

# Load label map
with open(DATASET / "label_map.json", "r", encoding="utf-8") as f:
    lm = json.load(f)

# Show sample from each class
grid_imgs = []
for idx in range(int(lm["num_classes"])):
    cls_dir = DATASET / f"class_{idx:03d}"
    files = sorted(cls_dir.glob("*.png"))
    if files:
        img = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=128)
            grid_imgs.append(img)

print(f"Loaded {len(grid_imgs)} class samples")

cols = 9
while len(grid_imgs) % cols != 0:
    grid_imgs.append(np.full_like(grid_imgs[0], 200))

rows = len(grid_imgs) // cols
grid = np.vstack([np.hstack(grid_imgs[r*cols:(r+1)*cols]) for r in range(rows)])
cv2.imwrite("class_overview.jpg", grid)
print(f"Saved class_overview.jpg ({grid.shape})")

# Also make a strip of 10 samples per class for a few classes
for cls_id in [0, 14, 19, 27, 43, 46, 62]:
    cls_dir = DATASET / f"class_{cls_id:03d}"
    char = lm["id_to_char"][str(cls_id)]
    files = sorted(cls_dir.glob("*.png"))[:10]
    row = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=128)
            row.append(img)
    if row:
        strip = np.hstack(row)
        cv2.imwrite(f"samples_class_{cls_id:03d}.jpg", strip)
        print(f"  class_{cls_id:03d} ({char}): {len(row)} samples")
