"""
Full re-extraction with the updated OG folder (17 PDFs now).
"""
import fitz
import os
from pathlib import Path
import shutil

OG_DIR = Path("OG")
OUT_DIR = Path("extracted_pages")

# Clean previous extraction
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(exist_ok=True)

total_pages = 0
total_images = 0

for pdf_file in sorted(OG_DIR.glob("*.pdf")):
    doc = fitz.open(str(pdf_file))
    clean_name = pdf_file.stem.replace(" ", "_").replace("(", "").replace(")", "")
    pdf_out = OUT_DIR / clean_name
    pdf_out.mkdir(exist_ok=True)

    pages_in_pdf = len(doc)
    total_pages += pages_in_pdf

    for pg_idx in range(pages_in_pdf):
        page = doc[pg_idx]
        images = page.get_images(full=True)

        if images:
            xref = images[0][0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            ext = base_image["ext"]
            out_path = pdf_out / f"page_{pg_idx+1:03d}.{ext}"
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            total_images += 1
        else:
            pix = page.get_pixmap(dpi=200)
            out_path = pdf_out / f"page_{pg_idx+1:03d}.png"
            pix.save(str(out_path))
            total_images += 1

    print(f"{pdf_file.name}: {pages_in_pdf} pages extracted")
    doc.close()

print(f"\n{'='*50}")
print(f"TOTAL PAGES ACROSS ALL PDFs: {total_pages}")
print(f"TOTAL IMAGES EXTRACTED: {total_images}")
