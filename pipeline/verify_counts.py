"""
Deep check: count ALL images across ALL pages in ALL PDFs,
not just first image per page. Also check for multi-image pages.
"""
import fitz
from pathlib import Path

OG_DIR = Path("OG")
total_pages = 0
total_images = 0
multi_image_pages = 0

for pdf_file in sorted(OG_DIR.glob("*.pdf")):
    doc = fitz.open(str(pdf_file))
    pdf_pages = len(doc)
    pdf_images = 0
    
    for pg_idx in range(pdf_pages):
        page = doc[pg_idx]
        images = page.get_images(full=True)
        pdf_images += len(images)
        if len(images) > 1:
            multi_image_pages += 1
            print(f"  MULTI-IMAGE: {pdf_file.name} page {pg_idx+1}: {len(images)} images")
    
    total_pages += pdf_pages
    total_images += pdf_images
    print(f"{pdf_file.name}: {pdf_pages} pages, {pdf_images} images")
    doc.close()

print(f"\nTotal pages: {total_pages}")
print(f"Total images: {total_images}")
print(f"Multi-image pages: {multi_image_pages}")

# Now check: each page is a form with ~22 handwritten boxes each
# So total handwritten samples is pages * ~22
print(f"\nEstimated handwritten character samples: {total_pages} pages × ~22 chars = ~{total_pages * 22}")

# Day 2: Count helper notes
