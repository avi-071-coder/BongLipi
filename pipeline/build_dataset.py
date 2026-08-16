import os
import cv2
import numpy as np
import json
import shutil
import time
from pathlib import Path
from PIL import Image

# Definitive 222 Bengali character schema corresponding to slots 1..222 across Pages 1..10
LABELS_222 = [
    # Page 1 (Slots 1..24)
    'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক',
    'খ', 'গ', 'ঘ', 'ঙ', 'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড',
    
    # Page 2 (Slots 25..48)
    'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প', 'ফ', 'ব', 'ভ', 'ম',
    'য', 'র', 'ল', 'ব_2', 'শ', 'ষ', 'স', 'হ', 'ড়', 'ঢ়', 'য়', 'ৎ',
    
    # Page 3 (Slots 49..72)
    'ং', 'ঃ', 'ঁ', '০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮',
    '৯', 'া', 'ি', 'ী', 'ু', 'ূ', 'ৃ', 'ে', 'ৈ', 'ো', 'ৌ', '্‌',
    
    # Page 4 (Slots 73..96)
    'ক্ক', 'ক্ত', 'ক্ত্ৰ', 'ক্ন', 'ক্ষ', 'ক্ষ্ম', 'ক্ষ্ৰ', 'ক্ৰ', 'ক্ল', 'ক্স', 'গ্ধ', 'গ্ন',
    'গ্র', 'গ্ম', 'গ্ল', 'ঘ্ন', 'ঘ্ৰ', 'ঙ্ক', 'ঙ্খ', 'ঙ্গ', 'ঙ্ক্ষ', 'ঙ্ঘ', 'চ্চ', 'চ্ছ',
    
    # Page 5 (Slots 97..120)
    'জ্জ', 'জ্জ্ব', 'জ্ঞ', 'জ্ৰ', 'ঞ্চ', 'ঞ্ছ', 'ঞ্জ', 'ঞ্ঝ', 'ট্ট', 'ট্ৰ', 'ড্ড', 'ড্ৰ',
    'ণ্ট', 'ণ্ঠ', 'ণ্ড', 'ণ্ঢ', 'ণ্ণ', 'ণ্ব', 'ণ্ম', 'ত্ত', 'ত্থ', 'ত্ন', 'ত্ম', 'ত্র',
    
    # Page 6 (Slots 121..144)
    'ত্ব', 'থ্ৰ', 'দ্গ', 'দ্ঘ', 'দ্দ', 'দ্ধ', 'দ্ব', 'দ্ম', 'দ্র', 'ধ্ন', 'ধ্ম', 'ধ্ৰ',
    'ধ্বনি', 'ন্ট', 'ন্ঠ', 'ন্ড', 'ন্ত', 'ন্ত্ৰ', 'ন্থ', 'ন্দ', 'ন্ধ', 'ন্ন', 'ম্প', 'ন্ফ',
    
    # Page 7 (Slots 145..168)
    'ন্ব', 'ন্ম', 'ন্স', 'প্ত', 'প্ন', 'প্প', 'প্র', 'প্ল', 'প্স', 'ফ্ৰ', 'ফ্ল', 'ব্জ',
    'ব্দ', 'ব্ধ', 'ব্ব', 'ব্র', 'ব্ল', 'ভ্ৰ', 'ম্ন', 'ম্প_2', 'ম্ফ', 'ম্ব', 'ম্ভ', 'ম্ম',
    
    # Page 8 (Slots 169..192)
    'ম্ৰ', 'ম্ল', 'ল্ক', 'ল্গ', 'ল্ট', 'ল্ড', 'ল্প', 'ল্ব', 'ল্ম', 'ল্ল', 'শ্চ', 'শ্থ',
    'শ্ন', 'শ্ম', 'শ্র', 'শ্ল', 'শ্ব', 'ষ্ক', 'ষ্ট', 'ষ্ঠ', 'ষ্ণ', 'ষ্প', 'ষ্প্ৰ', 'ষ্ম',
    
    # Page 9 (Slots 193..216)
    'স্ক', 'স্ক্ৰ', 'স্খ', 'স্ত', 'স্ত্ৰ', 'স্থ', 'স্ন', 'স্প', 'স্প্ৰ', 'স্প্ল', 'স্ম', 'স্র',
    'স্ল', 'স্ব', 'স্ফ', 'হ্ণ', 'হ্ন', 'হ্ম', 'হ্ল', 'হ্ব', 'হ্ৰ', 'র্ক', 'র্খ', 'র্গ',
    
    # Page 10 (Slots 217..222)
    'র্ঘ', 'র্ঙ', 'র্চ', 'র্ছ', 'র্জ', 'র্ঝ'
]

# OS-safe numeric folder names: class_001 .. class_222
SLOT_TO_FOLDER = {
    idx + 1: f"class_{idx+1:03d}"
    for idx, char in enumerate(LABELS_222)
}

SLOT_TO_CHAR = {
    idx + 1: char
    for idx, char in enumerate(LABELS_222)
}

def get_rotation_score(gray):
    h, w = gray.shape[:2]
    if w > h:
        return -1
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_s, max_s = w * 0.035, w * 0.12
    score = 0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if 0.7 < aspect < 1.4 and min_s < bw < max_s and min_s < bh < max_s:
            if cv2.contourArea(cnt) > 0.45 * bw * bh:
                xc = (x + bw/2) / w
                if (0.35 < xc < 0.50) or (0.70 < xc < 0.85):
                    score += 1
    return score

def auto_orient_perfect(img):
    """
    Bulletproof 4-Way Rotatory Scoring Engine:
    Evaluates all 4 potential scan rotations (0°, 90° CW, 180°, 270° CW)
    and selects the exact orientation maximizing column-box contour grid alignment.
    """
    best_img = img
    best_score = -1
    
    rotations = [
        None,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE
    ]
    
    for rot_func in rotations:
        curr = img if rot_func is None else cv2.rotate(img, rot_func)
        gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        score = get_rotation_score(gray)
        if score > best_score:
            best_score = score
            best_img = curr
            
    return best_img

def extract_boxes_bulletproof(img, page_in_cycle):
    """
    Hybrid Box Extraction Engine with NMS Duplicate Removal:
    Locates printed square handwritten boxes via contour analysis with precise grid fallback.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_s, max_s = w * 0.035, w * 0.12
    
    raw_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if 0.7 < aspect < 1.4 and min_s < bw < max_s and min_s < bh < max_s:
            if cv2.contourArea(cnt) > 0.45 * bw * bh:
                raw_boxes.append((x, y, bw, bh))
                
    # NMS: Keep largest box per grid location to eliminate outer/inner double contour duplicates
    clean_boxes = []
    for b in sorted(raw_boxes, key=lambda item: item[2]*item[3], reverse=True):
        bx, by, bw, bh = b
        b_xc, b_yc = bx + bw/2, by + bh/2
        is_dup = False
        for (cx, cy, cw, ch) in clean_boxes:
            c_xc, c_yc = cx + cw/2, cy + ch/2
            if np.hypot(b_xc - c_xc, b_yc - c_yc) < 0.03 * w:
                is_dup = True
                break
        if not is_dup:
            clean_boxes.append(b)
                
    num_rows = 12 if page_in_cycle < 10 else 3
    start_slot = (page_in_cycle - 1) * 24 + 1
    box_s = int(0.065 * w)
    crops = []
    
    # Left column slots
    for r in range(num_rows):
        slot_idx = start_slot + r
        if page_in_cycle < 10:
            exp_y = int((0.1023 + r * 0.0685) * h)
            exp_x = int(0.435 * w)
        else:
            exp_y = int((0.115 + r * 0.066) * h)
            exp_x = int(0.398 * w)
            
        best_box = None
        best_dist = 99999
        for (bx, by, bw, bh) in clean_boxes:
            b_xc, b_yc = bx + bw/2, by + bh/2
            dist = np.hypot(b_xc - exp_x, b_yc - exp_y)
            if dist < 0.04 * w and dist < best_dist:
                best_dist = dist
                best_box = (bx, by, bw, bh)
                
        if best_box is not None:
            bx, by, bw, bh = best_box
            pad = int(0.08 * min(bw, bh))
            y1, y2 = max(0, by + pad), min(h, by + bh - pad)
            x1, x2 = max(0, bx + pad), min(w, bx + bw - pad)
        else:
            y1, y2 = max(0, exp_y - box_s//2), min(h, exp_y + box_s//2)
            x1, x2 = max(0, exp_x - box_s//2), min(w, exp_x + box_s//2)
            
        crop = gray[y1:y2, x1:x2]
        crops.append((slot_idx, crop))
        
    # Right column slots
    for r in range(num_rows):
        slot_idx = start_slot + num_rows + r
        if page_in_cycle < 10:
            exp_y = int((0.1023 + r * 0.0685) * h)
            exp_x = int(0.770 * w)
        else:
            exp_y = int((0.115 + r * 0.066) * h)
            exp_x = int(0.763 * w)
            
        best_box = None
        best_dist = 99999
        for (bx, by, bw, bh) in clean_boxes:
            b_xc, b_yc = bx + bw/2, by + bh/2
            dist = np.hypot(b_xc - exp_x, b_yc - exp_y)
            if dist < 0.04 * w and dist < best_dist:
                best_dist = dist
                best_box = (bx, by, bw, bh)
                
        if best_box is not None:
            bx, by, bw, bh = best_box
            pad = int(0.08 * min(bw, bh))
            y1, y2 = max(0, by + pad), min(h, by + bh - pad)
            x1, x2 = max(0, bx + pad), min(w, bx + bw - pad)
        else:
            y1, y2 = max(0, exp_y - box_s//2), min(h, exp_y + box_s//2)
            x1, x2 = max(0, exp_x - box_s//2), min(w, exp_x + box_s//2)
            
        crop = gray[y1:y2, x1:x2]
        crops.append((slot_idx, crop))
        
    return crops

def preprocess_crop(crop, target_size=64):
    """
    Normalizes character crop:
    - Adaptive binarization
    - Outer border line trimming
    - Noise filtering
    - Empty box / stray line validation
    - Centered square padding
    - Resized to 64x64 grayscale
    Returns (is_valid, processed_image)
    """
    if crop is None or crop.size == 0:
        return False, None
        
    binary = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 8)
    
    # Outer border trimming (eliminate square box outline edges)
    pad = 6
    binary[:pad, :] = 0
    binary[-pad:, :] = 0
    binary[:, :pad] = 0
    binary[:, -pad:] = 0
    
    # Remove tiny noise specs (< 10 px)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 10:
            binary[labels == i] = 0
            
    stroke_pixel_count = np.sum(binary > 0)
    
    # Empty box check: if less than 35 ink pixels, box was left blank!
    if stroke_pixel_count < 35:
        return False, None
        
    coords = cv2.findNonZero(binary)
    if coords is None:
        return False, None
        
    x, y, bw, bh = cv2.boundingRect(coords)
    if bw < 5 or bh < 5:  # Stray line segment check
        return False, None
        
    cropped = binary[y:y+bh, x:x+bw]
    ch, cw = cropped.shape[:2]
    max_dim = max(ch, cw, 1)
    padded = np.zeros((max_dim + 8, max_dim + 8), dtype=np.uint8)
    y_off = (max_dim + 8 - ch) // 2
    x_off = (max_dim + 8 - cw) // 2
    padded[y_off:y_off+ch, x_off:x_off+cw] = cropped
    
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    final_img = 255 - resized
    return True, final_img

def build_dataset_222(src_dir="extracted_pages", output_dir="DATASET"):
    """
    Builds the complete 222-class Bengali HCR dataset with OS-safe numeric folder names (class_001 .. class_222).
    Automatically rejects empty white boxes and stray lines.
    """
    out_path = Path(output_dir)
    
    if out_path.exists():
        shutil.rmtree(out_path, ignore_errors=True)
        time.sleep(0.5)
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Create 222 character folders with numeric names (class_001 .. class_222)
    for folder_name in SLOT_TO_FOLDER.values():
        (out_path / folder_name).mkdir(parents=True, exist_ok=True)
        
    print(f"Created all 222 character folders (class_001 .. class_222) in {output_dir}/")
    
    src_path = Path(src_dir)
    pdf_folders = sorted([f for f in src_path.iterdir() if f.is_dir()])
    
    total_pages = 0
    processed_pages = 0
    total_chars = 0
    skipped_empty = 0
    folder_counts = {folder_name: 0 for folder_name in SLOT_TO_FOLDER.values()}
    
    for pdf_folder in pdf_folders:
        pages = sorted(list(pdf_folder.glob('*.jpeg')) + list(pdf_folder.glob('*.jpg')) + list(pdf_folder.glob('*.png')))
        n_pages = len(pages)
        total_pages += n_pages
        print(f"Processing {pdf_folder.name}: {n_pages} pages")
        
        for idx, page_path in enumerate(pages):
            form_page_num = (idx % 10) + 1  # Standard 10-page booklet cycle across all PDFs
            img = cv2.imread(str(page_path))
            if img is None:
                continue
                
            img_oriented = auto_orient_perfect(img)
            crops = extract_boxes_bulletproof(img_oriented, form_page_num)
            
            for slot_idx, crop_raw in crops:
                if slot_idx not in SLOT_TO_FOLDER:
                    continue
                    
                folder_name = SLOT_TO_FOLDER[slot_idx]
                is_valid, prep_img = preprocess_crop(crop_raw)
                
                if not is_valid:
                    skipped_empty += 1
                    continue
                    
                sample_name = f"{pdf_folder.name}_p{idx+1:03d}_formP{form_page_num:02d}_slot{slot_idx:03d}.png"
                save_file = out_path / folder_name / sample_name
                Image.fromarray(prep_img).save(save_file)
                
                folder_counts[folder_name] += 1
                total_chars += 1
                
            processed_pages += 1
            
    metadata = {
        "total_pages": total_pages,
        "processed_pages": processed_pages,
        "total_chars": total_chars,
        "skipped_empty_or_line": skipped_empty,
        "num_classes": len(SLOT_TO_FOLDER),
        "folder_counts": folder_counts,
        "class_to_char": {f"class_{slot:03d}": char for slot, char in SLOT_TO_CHAR.items()},
        "slot_to_folder": {slot: f"class_{slot:03d}" for slot, char in SLOT_TO_CHAR.items()}
    }
    
    with open(out_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"\nDataset build complete!")
    print(f"Total pages processed: {processed_pages} / {total_pages}")
    print(f"Total valid character images saved: {total_chars}")
    print(f"Total empty white boxes / stray lines skipped: {skipped_empty}")
    print(f"Total character classes (folders): {len(SLOT_TO_FOLDER)}")

if __name__ == "__main__":
    build_dataset_222()
