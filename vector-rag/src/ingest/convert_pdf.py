import os
from pdf2image import convert_from_path
import json

def convert_pdf_to_images(pdf_path, output_dir, dpi):
    os.makedirs(output_dir, exist_ok=True)

    pages = convert_from_path(pdf_path, dpi=dpi)

    img_paths = []
    for i, page in enumerate(pages, start=1):
        out = os.path.join(output_dir, f"page_{i}.png")
        page.save(out, "PNG")
        img_paths.append(out)

    return img_paths
