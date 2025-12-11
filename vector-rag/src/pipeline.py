import subprocess
import sys
from pathlib import Path
import argparse

def run(cmd, cwd=None):
    print(f"\n==> Running: {' '.join(cmd)} (cwd={cwd})")
    p = subprocess.Popen(cmd, cwd=cwd)
    p.communicate()
    if p.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}")
        sys.exit(1)

def ingest(ingest_dir, ingest_script):
    print("\n--- Stage: Ingest PDF ---")
    run(["python3", ingest_script], cwd=str(ingest_dir))

def chunk(chunk_dir, chunk_script):
    print("\n--- Stage: Chunk JSON ---")
    run(["python3", chunk_script], cwd=str(chunk_dir))

def extract_tables(chunk_dir):
    print("\n--- Stage: Extract Tables via Gemini ---")
    run(["python3", "extract_tables_gemini.py"], cwd=str(chunk_dir))

def embed(rag_dir, rag_script):
    print("\n--- Stage: Embed + Index into Vector DB ---")
    run(["python3", rag_script, "index"], cwd=str(rag_dir))

def chat(rag_dir, rag_script):
    print("\n--- Stage: Retrieval Chat ---")
    run(["python3", rag_script, "chat"], cwd=str(rag_dir))

def main():
    project_root = Path(__file__).resolve().parent
    ingest_dir = project_root / "ingest"
    chunk_dir  = project_root / "chunk"
    rag_dir    = project_root / "embedd"

    ingest_script = "pipeline.py"
    chunk_script  = "pipeline.py"
    rag_script    = "pipeline.py"

    parser = argparse.ArgumentParser(description="Master Pipeline Controller")
    parser.add_argument("--ingest", action="store_true", help="Run only ingestion")
    parser.add_argument("--chunk", action="store_true", help="Run only chunking")
    parser.add_argument("--tables", action="store_true", help="Extract tables using Gemini")
    parser.add_argument("--embed", action="store_true", help="Embeddings + index DB")
    parser.add_argument("--chat", action="store_true", help="Chat / retrieve only")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    if args.all:
        ingest(ingest_dir, ingest_script)
        chunk(chunk_dir, chunk_script)
        # extract_tables(chunk_dir)
        embed(rag_dir, rag_script)
        # chat(rag_dir, rag_script)
        print("\n✅ FULL PIPELINE COMPLETE")
        return

    if args.ingest: ingest(ingest_dir, ingest_script)
    if args.chunk: chunk(chunk_dir, chunk_script)
    if args.tables: extract_tables(ingest_dir)
    if args.embed: embed(rag_dir, rag_script)
    if args.chat: chat(rag_dir, rag_script)

    if not any(vars(args).values()):
        print("No arguments provided. Use --help for options.")

if __name__ == "__main__":
    main()
