import json

def index_jsonl(jsonl_path, embedder, store):
    ids, docs, metas = [], [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            c = json.loads(line)

            if c.get("type") != "text":
                continue

            text = c.get("text") or c.get("content") or ""
            if not text.strip():
                continue

            cid = str(c.get("id", f"auto_{len(ids)}"))
            meta = c.get("metadata", {})
            meta = {**(meta or {}), "type": c.get("type"), "source": c.get("source"), "bbox": c.get("bbox")}

            ids.append(cid)
            docs.append(text)
            metas.append(meta)

    embs = embedder.encode(docs)
    store.add(ids, embs, docs, metas)

    return len(ids)
