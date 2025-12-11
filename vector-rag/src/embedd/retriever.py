class Retriever:
    def __init__(self, embedder, store, top_k=5):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def search(self, query):
        emb = self.embedder.encode([query])[0]
        return self.store.query(emb, top_k=self.top_k)
