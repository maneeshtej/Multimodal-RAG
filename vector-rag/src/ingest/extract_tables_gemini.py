from google import genai
from google.genai import types
import pandas as pd, json, os, time
import concurrent.futures

def md_table(df):
    try:
        return df.to_markdown(index=False)
    except:
        return df.to_csv(index=False)


def call_gemini_with_timeout(client, model, img_bytes, prompt, timeout):
    print("[LOG] Sending request to Gemini with timeout =", timeout)
    def task():
        return client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                prompt
            ]
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(task)
        try:
            result = future.result(timeout=timeout)
            print("[LOG] Gemini response received successfully.")
            return result
        except Exception as e:
            print(f"[ERROR] Gemini timeout or failure: {e}")
            raise TimeoutError("Gemini request timed out")


def extract_tables(doc, config):
    print("[LOG] Initializing Gemini client")

    client = genai.Client(api_key=config["gemini_api_key"])
    model = config["gemini_model"]

    timeout_sec = config.get("gemini_timeout", 45)
    retries = config.get("gemini_retries", 3)

    print("\n===== Extracting Tables With Gemini =====\n")
    print(f"[LOG] Gemini Model: {model}")
    print(f"[LOG] Timeout: {timeout_sec}  Retries: {retries}")

    for page in doc["pages"]:
        print(f"\n========== PAGE {page['page_number']} ==========")

        for blk in page["blocks"]:
            if blk["type"] != "table":
                continue

            img_path = blk.get("image_path")
            blk.setdefault("extra", {})

            print(f"\n[INFO] Table block found: {img_path}")

            if not img_path or not os.path.exists(img_path):
                print("[WARN] No image found for table, skipping")
                blk["extra"]["ocr_status"] = "no_image"
                continue

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            print("[LOG] Loaded image bytes. Calling Gemini...")

            prompt = """
You are a precise OCR table extractor.
Return ONLY JSON in this schema:
{
 "columns":[...],
 "rows":[ [...], ... ]
}
Use "" for empty cells.
"""

            success = False

            for attempt in range(1, retries + 1):
                print(f"[LOG] Gemini OCR Attempt {attempt}/{retries}")

                try:
                    resp = call_gemini_with_timeout(client, model, img_bytes, prompt, timeout_sec)

                    raw = resp.text.strip()
                    print("[LOG] Raw Gemini response:", raw[:200].replace("\n"," "), "...")

                    json_text = raw[raw.find("{"): raw.rfind("}") + 1]

                    tbl = json.loads(json_text)
                    df = pd.DataFrame(tbl["rows"], columns=tbl["columns"])

                    csv_path = img_path.replace(".png", ".csv")
                    df.to_csv(csv_path, index=False)

                    blk["extra"]["ocr_status"] = "success"
                    blk["extra"]["csv_path"] = csv_path
                    blk["extra"]["table_json"] = tbl

                    print(f"[SUCCESS] Table extracted successfully -> {csv_path}")
                    success = True
                    break

                except TimeoutError:
                    print("[WARN] Gemini Timeout. Retrying...")
                except Exception as e:
                    print("[ERROR] JSON parsing or Gemini output error:", e)
                    if 'raw' in locals():
                        print("[DEBUG] Raw response snippet:", raw[:200])

                time.sleep(2)

            if not success:
                blk["extra"]["ocr_status"] = "failed"
                print("[FAIL] Failed to extract table after retries")


if __name__ == "__main__":
    from pathlib import Path

    print("[LOG] Loading config.json...")
    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    doc_path = Path(config["global_output_dir"]) / "doc.json"
    print("[LOG] Loading doc file from:", doc_path)

    if not doc_path.exists():
        raise FileNotFoundError(f"[ERROR] doc.json not found at: {doc_path}")

    with open(doc_path, "r") as f:
        doc = json.load(f)

    print(f"[INFO] Successfully loaded doc.json")
    print("🔍 Starting Gemini table OCR...\n")

    extract_tables(doc, config)

    with open(doc_path, "w") as f:
        json.dump(doc, f, indent=2)

    print(f"\n✅ Table OCR complete. Updated doc saved to: {doc_path}\n")
