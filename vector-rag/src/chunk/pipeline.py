# pipeline.py
from chunker import run_chunker
from pathlib import Path

def main():
    # Hard-coded paths
    input_json = "../output/ingest/doc.json"       # where your ingest writes doc.json
    output_jsonl = "../output/chunks/chunks.jsonl" # where you want chunks

    # Create output directory if needed
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    print("=== Chunking Pipeline Start ===")
    print(f"Input:  {input_json}")
    print(f"Output: {output_jsonl}")

    run_chunker(input_json, output_jsonl)

    print("\n✅ Chunking pipeline complete")

if __name__ == "__main__":
    main()
