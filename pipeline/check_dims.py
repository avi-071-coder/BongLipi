"""
Quick check: For the horizontal-layout pages, what's the actual w vs h?
And for ALL page types, what are the actual dimensions?
"""
import cv2
from pathlib import Path

EXTRACTED = Path("extracted_pages")

for pdf_dir in sorted(EXTRACTED.iterdir()):
    if not pdf_dir.is_dir():
        continue
    pages = sorted(list(pdf_dir.glob("*.jpeg")) + list(pdf_dir.glob("*.jpg")) + list(pdf_dir.glob("*.png")))
    
    if not pages:
        continue
    
    dims = []
    for p in pages[:5]:
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            dims.append((w, h, "LANDSCAPE" if w > h * 1.2 else "PORTRAIT"))
    
    print(f"{pdf_dir.name}: {len(pages)} pages")
    for i, (w, h, orient) in enumerate(dims):
        print(f"  Page {i+1}: {w}x{h} ({orient})")
