"""
=============================================================================
BONGLIPI - Step 2 v3: Fixed for Windows Unicode
=============================================================================
Uses numeric class IDs as folder names to avoid Windows Unicode path issues.
Maintains a label_map.json that maps class_id → Bengali character.
=============================================================================
"""

import cv2
import numpy as np
from pathlib import Path
import json
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# LABEL DEFINITIONS
# ============================================================================

# Page 1: Left column → Right column (each 12 chars)
PAGE1_LEFT  = ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ", "ক"]
PAGE1_RIGHT = ["খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড"]
PAGE1_ALL   = PAGE1_LEFT + PAGE1_RIGHT

PAGE2_LEFT  = ["ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন", "প", "ফ", "ব", "ভ", "ম"]
PAGE2_RIGHT = ["য", "র", "ল", "শ", "ষ", "স", "হ", "ড়", "ঢ়", "য়", "ক্ষ", "ঁ"]
PAGE2_ALL   = PAGE2_LEFT + PAGE2_RIGHT

PAGE3_LEFT  = ["ং", "ঃ", "ে", "ৈ", "ো", "ৌ", "০", "১", "২", "৩", "৪", "৫"]
PAGE3_RIGHT = ["৬", "৭", "৮", "৯", "া", "ি", "ী", "ু", "ূ", "ৃ", "কা", "কি"]
PAGE3_ALL   = PAGE3_LEFT + PAGE3_RIGHT

# Build global label map: character → numeric ID
ALL_CHARS = sorted(list(set(PAGE1_ALL + PAGE2_ALL + PAGE3_ALL)))
LABEL_MAP = {char: idx for idx, char in enumerate(ALL_CHARS)}
INV_LABEL_MAP = {idx: char for char, idx in LABEL_MAP.items()}

print(f"Total unique classes: {len(ALL_CHARS)}")

# ============================================================================
# BOX DETECTION
# ============================================================================

def detect_boxes(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 10)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    min_s = w * 0.03
    max_s = w * 0.15
    
    cands = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        asp = bw / max(bh, 1)
        if 0.6 < asp < 1.6 and min_s < bw < max_s and min_s < bh < max_s:
            ca = cv2.contourArea(cnt)
            if ca > 0.4 * bw * bh:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) >= 4:
                    cands.append((x, y, bw, bh))
    
    return _remove_overlaps(cands)


def _remove_overlaps(boxes, thresh=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    keep = []
    for b in boxes:
        x1, y1, w1, h1 = b
        dup = False
        for kx, ky, kw, kh in keep:
            ix = max(0, min(x1+w1, kx+kw) - max(x1, kx))
            iy = max(0, min(y1+h1, ky+kh) - max(y1, ky))
            inter = ix * iy
            union = w1*h1 + kw*kh - inter
            if union > 0 and inter/union > thresh:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


def sort_two_cols(boxes, img_w):
    mid = img_w / 2
    left = sorted([b for b in boxes if b[0]+b[2]/2 < mid], key=lambda b: b[1])
    right = sorted([b for b in boxes if b[0]+b[2]/2 >= mid], key=lambda b: b[1])
    return left + right


def extract_char(img, box, size=64, pad=5):
    x, y, w, h = box
    x1, y1 = max(0, x+pad), max(0, y+pad)
    x2, y2 = min(img.shape[1], x+w-pad), min(img.shape[0], y+h-pad)
    
    roi = img[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return None
    
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Center on ink
    coords = cv2.findNonZero(binary)
    if coords is not None and len(coords) > 10:
        bx, by, bw, bh = cv2.boundingRect(coords)
        m = 3
        bx, by = max(0, bx-m), max(0, by-m)
        bw = min(binary.shape[1]-bx, bw+2*m)
        bh = min(binary.shape[0]-by, bh+2*m)
        binary = binary[by:by+bh, bx:bx+bw]
    
    if binary.size == 0:
        return None
    
    rh, rw = binary.shape
    scale = min((size-8)/max(rw,1), (size-8)/max(rh,1))
    nw, nh = max(1, int(rw*scale)), max(1, int(rh*scale))
    resized = cv2.resize(binary, (nw, nh), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((size, size), dtype=np.uint8)
    yo, xo = (size-nh)//2, (size-nw)//2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return canvas


def detect_cycle_length(n_pages):
    for c in [2, 3, 4, 5, 10]:
        if n_pages % c == 0:
            p = n_pages // c
            if 2 <= p <= 50:
                return c
    for c in [3, 2, 4, 5]:
        if n_pages % c == 0:
            return c
    return 3


# ============================================================================
# MAIN
# ============================================================================

def run():
    EXTRACTED = Path("extracted_pages")
    DATASET = Path("dataset")
    
    if DATASET.exists():
        shutil.rmtree(DATASET)
    DATASET.mkdir()
    
    # Create all class directories with numeric IDs
    for idx in range(len(ALL_CHARS)):
        (DATASET / f"class_{idx:03d}").mkdir()
    
    # Save label map
    label_map_data = {
        "id_to_char": {str(k): v for k, v in INV_LABEL_MAP.items()},
        "char_to_id": {k: v for k, v in LABEL_MAP.items()},
        "num_classes": len(ALL_CHARS),
        "class_list": ALL_CHARS,
    }
    with open(DATASET / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map_data, f, ensure_ascii=False, indent=2)
    
    stats = {"processed": 0, "skipped": 0, "total_chars": 0, "class_counts": {}}
    
    for pdf_dir in sorted(EXTRACTED.iterdir()):
        if not pdf_dir.is_dir():
            continue
        
        pages = sorted(
            list(pdf_dir.glob("*.jpeg")) +
            list(pdf_dir.glob("*.jpg")) +
            list(pdf_dir.glob("*.png"))
        )
        if not pages:
            continue
        
        n = len(pages)
        cycle = detect_cycle_length(n)
        print(f"\n{pdf_dir.name}: {n} pages, cycle={cycle}")
        
        for i, page_path in enumerate(pages):
            pos = i % cycle
            
            # Determine page type and labels
            if pos == 0:
                labels = PAGE1_ALL
            elif pos == 1:
                labels = PAGE2_ALL
            elif pos == 2:
                labels = PAGE3_ALL
            else:
                stats["skipped"] += 1
                continue
            
            img = cv2.imread(str(page_path))
            if img is None:
                stats["skipped"] += 1
                continue
            
            h, w = img.shape[:2]
            boxes = detect_boxes(img)
            sorted_b = sort_two_cols(boxes, w)
            
            if len(sorted_b) < 20:
                stats["skipped"] += 1
                continue
            
            stats["processed"] += 1
            n_ext = 0
            
            for j in range(min(len(sorted_b), len(labels))):
                char_img = extract_char(img, sorted_b[j])
                if char_img is None:
                    continue
                
                label = labels[j]
                if label not in LABEL_MAP:
                    continue
                
                cls_id = LABEL_MAP[label]
                src = f"{pdf_dir.name}_p{i+1:03d}_b{j:02d}"
                out = DATASET / f"class_{cls_id:03d}" / f"{src}.png"
                cv2.imwrite(str(out), char_img)
                
                n_ext += 1
                stats["total_chars"] += 1
                stats["class_counts"][label] = stats["class_counts"].get(label, 0) + 1
            
            if (i+1) % 30 == 0 or i == n-1:
                print(f"  Page {i+1}/{n}: {n_ext} chars (pos_in_cycle={pos})")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"DONE: {stats['processed']} pages, {stats['total_chars']} chars, {len(stats['class_counts'])} classes")
    print(f"Skipped: {stats['skipped']} pages")
    
    counts = stats["class_counts"]
    print(f"\nPer-class counts:")
    for char in sorted(counts.keys()):
        c = counts[char]
        bar = '█' * min(c // 5, 40)
        print(f"  {char:>4s} (class_{LABEL_MAP[char]:03d}): {c:4d} {bar}")
    
    # Verify files actually exist
    total_on_disk = 0
    empty_classes = []
    for idx in range(len(ALL_CHARS)):
        cls_dir = DATASET / f"class_{idx:03d}"
        n_files = len(list(cls_dir.glob("*.png")))
        total_on_disk += n_files
        if n_files == 0:
            empty_classes.append(f"class_{idx:03d} = {INV_LABEL_MAP[idx]}")
    
    print(f"\nFiles on disk: {total_on_disk}")
    if empty_classes:
        print(f"EMPTY classes: {empty_classes}")
    
    stats["total_on_disk"] = total_on_disk
    with open(DATASET / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run()
