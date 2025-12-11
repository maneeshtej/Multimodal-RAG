import chromadb

class VectorStore:
    def __init__(self, db_path="../output/", collection_name="rag_store"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _clean_meta(meta):
        out = {}
        for k, v in (meta or {}).items():
            if v is None: v = ""
            if isinstance(v, (str, int, float, bool, dict)):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    def add(self, ids, embeddings, documents, metadatas):
        metadatas = [self._clean_meta(m) for m in metadatas]
        self.collection.add(
            ids=ids, embeddings=embeddings,
            documents=documents, metadatas=metadatas
        )

    def query(self, query_embedding, top_k=5):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    def delete(self, ids):
        self.collection.delete(ids)

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
