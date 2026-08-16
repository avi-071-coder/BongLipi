"""
Generate visual grids. Handles Windows Unicode path issues.
"""
import cv2
import numpy as np
from pathlib import Path
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

DATASET = Path("dataset_raw")

# List all class directories
all_classes = sorted([d.name for d in DATASET.iterdir() if d.is_dir()])
print(f"Total classes: {len(all_classes)}")

# Test reading a single file
test_cls = all_classes[0]
test_dir = DATASET / test_cls
test_files = list(test_dir.glob("*.png"))
print(f"Testing class '{test_cls}': {len(test_files)} files")
if test_files:
    print(f"  Test file: {test_files[0]}")
    # Try different read methods
    
    # Method 1: cv2 with string
    img1 = cv2.imread(str(test_files[0]), cv2.IMREAD_GRAYSCALE)
    print(f"  cv2.imread str: {img1 is not None}")
    
    # Method 2: numpy fromfile + cv2 decode
    with open(test_files[0], 'rb') as f:
        data = f.read()
    nparr = np.frombuffer(data, np.uint8)
    img2 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    print(f"  cv2.imdecode: {img2 is not None}")
    if img2 is not None:
        print(f"  Shape: {img2.shape}")
