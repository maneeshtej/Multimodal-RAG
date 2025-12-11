import json, uuid, jsonlines, tiktoken
from pathlib import Path

MAX_TOKENS = 400
OVERLAP = 80

enc = tiktoken.get_encoding("cl100k_base")

def split_with_overlap(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    tokens = enc.encode(text)
    start = 0
    n = len(tokens)

    while start < n:
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        yield enc.decode(chunk_tokens)
        start = end - overlap
        if start < 0:
            start = 0


def run_chunker(input_json: str, output_jsonl: str):
    print(f"[CHUNKER] Reading {input_json}")
    doc = json.load(open(input_json))

    final_chunks = []

    for page_obj in doc["pages"]:
        page = page_obj["page_number"]

        for block in page_obj["blocks"]:
            typ = block["type"]

            # ----- TEXT BLOCK -----
            if typ == "text":
                txt = block.get("text", "").strip()
                if txt:
                    for part in split_with_overlap(txt):
                        final_chunks.append({
                            "id": str(uuid.uuid4()),
                            "type": "text",
                            "page": page,
                            "content": part,
                            "bbox": block.get("bbox")
                        })

            # ----- TABLE BLOCK -----
           # ----- TABLE BLOCK -----
           # ----- TABLE BLOCK -----
            elif typ == "table":
                extra = block.get("extra", {})
                table_struct = extra.get("table_json")

                # If table OCR failed or missing structure, skip
                if not table_struct:
                    print(f"[WARN] Missing table_json for table on page {page}, skipping.")
                    continue

                cols = " | ".join(table_struct["columns"])
                lines = [f"Table page {page}", f"Columns: {cols}"]

                for row in table_struct["rows"]:
                    lines.append("Row: " + " | ".join(str(x) for x in row))

                final_chunks.append({
                    "id": str(uuid.uuid4()),
                    "type": "table",
                    "page": page,
                    "content": "\n".join(lines),
                    "bbox": block.get("bbox")
                })



            # ----- IMAGE BLOCK -----
            elif typ == "image":
                final_chunks.append({
                    "id": str(uuid.uuid4()),
                    "type": "image",
                    "page": page,
                    "content": "[IMAGE]",
                    "bbox": block.get("bbox")
                })

    # Save chunks
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_jsonl, "w") as writer:
        for c in final_chunks:
            writer.write(c)

    print(f"[CHUNKER] Created {len(final_chunks)} chunks → {output_jsonl}")
