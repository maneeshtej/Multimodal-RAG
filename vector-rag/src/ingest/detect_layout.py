import cv2, os, json
from doclayout_yolo import YOLO

def detect_layout(img_paths, config):
    model = YOLO(config["yolo_model_path"])
    crops_dir = os.path.join(config["output_dir"], "crops")
    os.makedirs(crops_dir, exist_ok=True)

    doc = { "pages": [] }

    for idx, img_path in enumerate(img_paths, start=1):
        det = model.predict(img_path, conf=config["min_confidence"], iou=config["iou_threshold"])
        boxes = det[0].boxes
        cls = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy().astype(int)

        img = cv2.imread(img_path)
        page_key = f"page_{idx}"
        blocks = []

        for i, (box, c) in enumerate(zip(xyxy, cls)):
            x1, y1, x2, y2 = box
            label = det[0].names[c].lower()

            if "table" not in label and "image" not in label and "picture" not in label:
                continue

            crop = img[y1:y2, x1:x2]
            crop_path = os.path.join(crops_dir, f"{page_key}_b{i}_{label}.png")
            cv2.imwrite(crop_path, crop)

            blocks.append({
                "block_id": f"{idx}-{i}",
                "type": "table" if "table" in label else "image",
                "image_path": crop_path,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "extra": {}
            })

        # append full page text later
        blocks.insert(0, {
            "block_id": f"{idx}-T",
            "type": "text",
            "text": "",
            "extra": {}
        })

        doc["pages"].append({
            "page_number": idx,
            "blocks": blocks
        })

    return doc
