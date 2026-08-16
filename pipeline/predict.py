import sys
import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Import BengaliNet222 definition
sys.path.append(str(Path(__file__).parent.parent))
from models.train import BengaliNet222

# Load 222 label mappings
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

INDEX_TO_CHAR = {idx: char for idx, char in enumerate(LABELS_222)}

def preprocess_user_image(image_path, target_size=64):
    """
    Robust Camera & Photo Preprocessing Engine:
    - Binarizes photo using Otsu + Gaussian Blur (handles camera lighting & paper shadows)
    - Removes noise components
    - Intelligently trims long horizontal trailing lines (e.g. Matra extensions)
    - Centers glyph in square frame without aspect ratio distortion
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Check if image is already 64x64 pre-cropped grayscale (like dataset samples)
    if gray.shape == (target_size, target_size) and np.mean(gray) > 200:
        return Image.fromarray(gray)
        
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    cleaned = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100:
            cleaned[labels == i] = 255
            
    coords = cv2.findNonZero(cleaned)
    if coords is None:
        return Image.fromarray(255 - np.zeros((target_size, target_size), dtype=np.uint8))
        
    x, y, w, h = cv2.boundingRect(coords)
    char_region = cleaned[y:y+h, x:x+w]
    
    # Intelligently trim horizontal trailing line extensions (Matra tails)
    col_heights = []
    for c in range(w):
        col = char_region[:, c]
        nz = np.where(col > 0)[0]
        col_heights.append(len(nz))
        
    col_heights = np.array(col_heights)
    threshold_h = 0.22 * h
    
    valid_indices = np.where(col_heights >= threshold_h)[0]
    if len(valid_indices) > 0:
        w_trim = min(w, valid_indices[-1] + 6)
    else:
        w_trim = w
        
    cropped = char_region[:, :w_trim]
    ch, cw = cropped.shape[:2]
    max_dim = max(ch, cw, 1)
    padded = np.zeros((max_dim + 12, max_dim + 12), dtype=np.uint8)
    y_off = (max_dim + 12 - ch) // 2
    x_off = (max_dim + 12 - cw) // 2
    padded[y_off:y_off+ch, x_off:x_off+cw] = cropped
    
    resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
    final_img = 255 - resized
    return Image.fromarray(final_img)

def predict_image(image_path, model_path="models/bengali_char_net_best.pth", top_k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BengaliNet222(num_classes=222).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    pil_img = preprocess_user_image(image_path)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_prob, top_class = torch.topk(probabilities, top_k)
        
    results = []
    for prob, idx in zip(top_prob[0], top_class[0]):
        char_symbol = INDEX_TO_CHAR.get(idx.item(), f"class_{idx.item()+1:03d}")
        folder_class = f"class_{idx.item()+1:03d}"
        results.append((char_symbol, folder_class, prob.item() * 100))
        
    return results

def main():
    upload_dir = Path("UPLOAD")
    upload_dir.mkdir(exist_ok=True)
    
    if len(sys.argv) > 1:
        target_images = [Path(sys.argv[1])]
    else:
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        target_images = sorted([
            f for f in upload_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in valid_exts
        ])
            
    if not target_images:
        print("\n=======================================================")
        print("         BONGLIPI - BENGALI HCR PREDICTION ENGINE       ")
        print("=======================================================")
        print(" [!] No images found in UPLOAD/ folder!")
        print(" [-->] Please drop your Bengali handwritten image(s) into: ")
        print(f"       {upload_dir.resolve()}")
        print("\n [-->] Then run this command again:")
        print("       python pipeline/predict.py")
        print("=======================================================\n")
        return

    print("\n=======================================================")
    print("         BONGLIPI - BENGALI HCR PREDICTION RESULTS      ")
    print("=======================================================")

    for img_p in target_images:
        if not img_p.exists():
            print(f"\n [!] File not found: {img_p}")
            continue
            
        print(f"\n Image File: {img_p.name}")
        print("-" * 55)
        
        try:
            preds = predict_image(img_p, top_k=5)
            top_char, top_class, top_conf = preds[0]
            
            print(f" PREDICTED CHARACTER : '{top_char}'  ({top_class})")
            print(f" CONFIDENCE SCORE    : {top_conf:.2f}%\n")
            print(" Top-5 Predictions:")
            for rank, (char, cls, conf) in enumerate(preds, 1):
                bar = "█" * int(conf / 5)
                print(f"   #{rank} | '{char}' ({cls}) : {conf:6.2f}%  {bar}")
        except Exception as e:
            print(f" [!] Error processing image: {e}")
            
    print("=======================================================\n")

if __name__ == "__main__":
    main()
