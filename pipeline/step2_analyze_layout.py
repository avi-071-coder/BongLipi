"""
Analyze page layouts to understand the grid structure for segmentation.
Detect bounding boxes of the handwritten character cells.
"""
import cv2
import numpy as np
from pathlib import Path
import json

def analyze_page(img_path, debug_dir=None):
    """Detect rectangular cells on a page and return their bounding boxes."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to get binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 25, 10)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter for square-ish boxes of appropriate size
    # The handwritten boxes are small squares, roughly 1/15 to 1/8 of page width
    min_side = w * 0.03
    max_side = w * 0.15
    
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / max(bh, 1)
        area = bw * bh
        
        # Square-ish boxes (aspect ratio close to 1)
        if 0.6 < aspect_ratio < 1.6 and min_side < bw < max_side and min_side < bh < max_side:
            # Check if it's roughly rectangular by comparing contour area vs bounding box area
            contour_area = cv2.contourArea(cnt)
            if contour_area > 0.5 * area:
                boxes.append((x, y, bw, bh))
    
    # Remove duplicate/overlapping boxes
    boxes = remove_overlaps(boxes)
    
    if debug_dir:
        debug_img = img.copy()
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(debug_img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        cv2.putText(debug_img, f"Boxes: {len(boxes)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out_path = Path(debug_dir) / img_path.name
        cv2.imwrite(str(out_path), debug_img)
    
    return boxes

def remove_overlaps(boxes, iou_threshold=0.3):
    """Remove overlapping boxes, keeping larger ones."""
    if not boxes:
        return []
    
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep = []
    
    for box in boxes:
        x1, y1, w1, h1 = box
        overlap = False
        for kept in keep:
            x2, y2, w2, h2 = kept
            # Calculate overlap
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            intersection = ix * iy
            union = w1*h1 + w2*h2 - intersection
            if union > 0 and intersection / union > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(box)
    
    return keep


# Test on sample pages from different layouts
EXTRACTED = Path("extracted_pages")
DEBUG_DIR = Path("debug_segmentation")
DEBUG_DIR.mkdir(exist_ok=True)

# Pick sample pages from different PDFs to test
test_pages = []
for pdf_dir in sorted(EXTRACTED.iterdir()):
    if pdf_dir.is_dir():
        pages = sorted(pdf_dir.glob("*.jp*"))
        if not pages:
            pages = sorted(pdf_dir.glob("*.png"))
        if pages:
            # Take first 3 pages (different layouts)
            for p in pages[:3]:
                test_pages.append(p)

print(f"Testing on {len(test_pages)} sample pages...")
results = {}
for page_path in test_pages:
    boxes = analyze_page(page_path, DEBUG_DIR)
    key = f"{page_path.parent.name}/{page_path.name}"
    results[key] = len(boxes)
    print(f"  {key}: {len(boxes)} boxes detected")

# Summary
box_counts = list(results.values())
print(f"\nBox detection summary:")
print(f"  Min: {min(box_counts)}, Max: {max(box_counts)}, Mean: {np.mean(box_counts):.1f}")
print(f"  Distribution: {sorted(set(box_counts))}")
