import argparse
import json
import os

from embeddings import Embedder
from vector_store import VectorStore
from indexer import index_jsonl
from retriever import Retriever
from gemini import ask_gemini, load_gemini_config

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Convert jsonl_path and db_path to absolute paths relative to this file
    base_dir = os.path.dirname(config_path)
    if "jsonl_path" in config:
        config["jsonl_path"] = os.path.abspath(os.path.join(base_dir, config["jsonl_path"]))
    if "db_path" in config:
        config["db_path"] = os.path.abspath(os.path.join(base_dir, config["db_path"]))

    # print(f"[CONFIG] Using DB at: {config['db_path']}")
    # print(f"[CONFIG] Using Chunks at: {config['jsonl_path']}")

    return config


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="RAG Vector Pipeline (With Gemini)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # index
    p_index = sub.add_parser("index")
    p_index.add_argument("--jsonl", default=config["jsonl_path"])
    p_index.add_argument("--db", default=config["db_path"])
    p_index.add_argument("--collection", default=config["collection_name"])

    # search
    p_search = sub.add_parser("search")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--db", default=config["db_path"])
    p_search.add_argument("--collection", default=config["collection_name"])

    # reset
    p_reset = sub.add_parser("reset")
    p_reset.add_argument("--db", default=config["db_path"])
    p_reset.add_argument("--collection", default=config["collection_name"])

    # delete
    p_delete = sub.add_parser("delete")
    p_delete.add_argument("--ids", nargs="+", required=True)
    p_delete.add_argument("--db", default=config["db_path"])
    p_delete.add_argument("--collection", default=config["collection_name"])

    # chat (retrieval only)
    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--db", default=config["db_path"])
    p_chat.add_argument("--collection", default=config["collection_name"])
    p_chat.add_argument("--top_k", type=int, default=config["top_k"])

    # gemini RAG
    p_gemini = sub.add_parser("gemini")
    p_gemini.add_argument("--query", required=True)
    p_gemini.add_argument("--db", default=config["db_path"])
    p_gemini.add_argument("--collection", default=config["collection_name"])
    p_gemini.add_argument("--top_k", type=int, default=config["top_k"])
    p_chat.add_argument("--query", help="Run a single question in chat mode")


    args = parser.parse_args()

    # INDEX
    if args.cmd == "index":
        embedder = Embedder()
        store = VectorStore(db_path=args.db, collection_name=args.collection)
        count = index_jsonl(args.jsonl, embedder, store)
        print(f"Indexed {count} chunks.")

    # SEARCH (retrieval only)
    elif args.cmd == "search":
        embedder = Embedder()
        store = VectorStore(db_path=args.db, collection_name=args.collection)
        retriever = Retriever(embedder, store, top_k=5)
        res = retriever.search(args.query)
        print(res)

    # RESET DB
    elif args.cmd == "reset":
        store = VectorStore(db_path=args.db, collection_name=args.collection)
        store.reset()
        print("✅ Vector DB reset.")

    # DELETE BY ID
    elif args.cmd == "delete":
        store = VectorStore(db_path=args.db, collection_name=args.collection)
        store.delete(args.ids)
        print(f"✅ Deleted IDs: {args.ids}")

    # CHAT (terminal retrieval explorer)
    elif args.cmd == "chat":
        embedder = Embedder()
        store = VectorStore(db_path=args.db, collection_name=args.collection)
        retriever = Retriever(embedder, store, top_k=args.top_k)

        from gemini import ask_gemini

      
        if getattr(args, "query", None):
            q = args.query
            run_single_chat_query(q, retriever)
            return
        
        print("\n🤖 Gemini RAG Chat Mode (old working API style)")
        print("Type 'exit' to quit.\n")

        # ---- NEW: allow one-shot query mode ----

        while True:
            q = input("You: ").strip()
            if q.lower() in ("exit", "quit", "bye"):
                print("👋 Bye.")
                break

            # Retrieve context
            res = retriever.search(q)
            docs = res.get("documents", [[]])[0]

            if not docs:
                print("⚠️ No relevant chunks found")
                continue

            context = "\n\n".join(docs)

            # Call Gemini with your working style
            answer = ask_gemini(q, context)

           


    # GEMINI RAG MODE
        # GEMINI RAG MODE
    elif args.cmd == "gemini":
        embedder = Embedder()
        store = VectorStore(db_path=args.db, collection_name=args.collection)
        retriever = Retriever(embedder, store, top_k=args.top_k)

        # --- NEW: auto-index if DB is empty ---
        try:
            res = retriever.search(args.query)
            docs = res.get("documents", [[]])[0]
        except Exception as e:
            docs = []
            print(f"⚠️ Retrieval failed: {e}")

        if not docs:
            print("🧠 Vector DB empty — running auto-index...")
            jsonl_path = config.get("jsonl_path", None)
            if jsonl_path and os.path.exists(jsonl_path):
                count = index_jsonl(jsonl_path, embedder, store)
                print(f"✅ Auto-indexed {count} chunks.")
                # retry retrieval after indexing
                res = retriever.search(args.query)
                docs = res.get("documents", [[]])[0]
            else:
                print(json.dumps({
                    "status": "error",
                    "message": "❌ No chunks found and jsonl_path missing."
                }, ensure_ascii=False))
                return


def run_single_chat_query(q, retriever):
    res = retriever.search(q)
    docs = res.get("documents", [[]])[0]
    if not docs:
        print("⚠️ No relevant chunks found")
        return
    context = "\n\n".join(docs)
    answer = ask_gemini(q, context)
    print(f"\n🧠 Gemini: {answer}\n")
    print("-------------------------------------------\n")



if __name__ == "__main__":
    main()
