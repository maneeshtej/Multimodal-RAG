import os, json, gc
from pathlib import Path
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from doclayout_yolo import YOLO
import pdfplumber


def pdf_to_images(pdf_path, pages_dir, dpi=450):
    pages_dir = Path(pages_dir)
    pages_dir.mkdir(parents=True, exist_ok=True)

    # First load ONE page to ensure PDF loads
    convert_from_path(pdf_path, first_page=1, last_page=1, dpi=dpi)

    # Count pages using pdfplumber (accurate)
  
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)

    print(f"📄 Total pages: {page_count}")

    page_paths = []
    for i in range(1, page_count + 1):
        print(f"➡️  Rendering page {i}/{page_count}")
        page_img = convert_from_path(pdf_path, dpi=dpi, first_page=i, last_page=i)[0]
        out_path = pages_dir / f"page_{i}.png"
        page_img.save(out_path, "PNG")
        page_paths.append(str(out_path))

    return page_paths


def detect_layout_and_text(pdf_path, pages_dir, crops_dir, model_path, out_json, dpi=450):
    pages_dir = Path(pages_dir)
    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)

    page_paths = pdf_to_images(pdf_path, pages_dir, dpi=dpi)
    model = YOLO(model_path)

    doc = {"document": os.path.basename(pdf_path), "pages": []}

    for page_index, page_img_path in enumerate(page_paths, start=1):
        print(f"\n🧠 Processing page {page_index}/{len(page_paths)}...")

        img = cv2.imread(page_img_path)
        np_img = np.array(img)

        # OCR TEXT
        ocr_text = pytesseract.image_to_string(page_img_path)

        # YOLO inference
        det = model.predict(page_img_path, conf=0.4, imgsz=1024)
        boxes = det[0].boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        coords = boxes.xyxy.cpu().numpy().astype(int)

        blocks = []

        # TEXT block
        blocks.append({
            "block_id": f"{page_index}-T",
            "type": "text",
            "text": ocr_text,
            "extra": {}
        })

        # CROPS
        for i, (box, c) in enumerate(zip(coords, classes)):
            x1, y1, x2, y2 = map(int, box)
            class_name = det[0].names[c].lower()

            if not any(k in class_name for k in ["table", "image", "picture"]):
                continue

            crop = np_img[y1:y2, x1:x2]
            crop_path = crops_dir / f"page_{page_index}_b{i}_{class_name}.png"
            cv2.imwrite(str(crop_path), crop)

            blocks.append({
                "block_id": f"{page_index}-{i}",
                "type": "table" if "table" in class_name else "image",
                "image_path": str(crop_path),
                "bbox": [x1, y1, x2, y2],
                "extra": {}
            })

        doc["pages"].append({"page_number": page_index, "blocks": blocks})
        gc.collect()

    with open(out_json, "w") as f:
        json.dump(doc, f, indent=2)

    print(f"\n✅ Saved structured doc to {out_json}")
    return out_json


if __name__ == "__main__":
    # Load config.json in same folder
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    pdf = config["pdf_input_dir"]
    output_dir = Path(config["output_dir"])
    pages = output_dir / "pages"
    crops = output_dir / "crops"
    model = config["yolo_model_path"]

    global_out = Path(config["global_output_dir"])
    global_out.mkdir(parents=True, exist_ok=True)
    out_json = global_out / "doc.json"

    detect_layout_and_text(
        pdf_path=pdf,
        pages_dir=str(pages),
        crops_dir=str(crops),
        model_path=model,
        out_json=str(out_json),
        dpi=config.get("dpi", 450)
    )

