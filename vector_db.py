from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    # CHANGED: dim=768 (Google's embedding size) and collection name changed to avoid conflict
    def __init__(self, url="http://localhost:6333", collection="docs_gemini", dim=768):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # Create collection if it doesn't exist
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def search(self, query_vector, top_k=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )

        contexts = []
        sources = set()

        for r in results.points:
            payload = r.payload or {}
            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                if source:
                    sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources),
        }