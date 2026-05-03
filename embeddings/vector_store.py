# embeddings/vector_store.py
# ─────────────────────────────────────────────────────────────────────
#  FAISS Vector Store — Stores resume embeddings, retrieves top-K matches
#  FAISS = Facebook AI Similarity Search (local, no server needed)
# ─────────────────────────────────────────────────────────────────────

import os
import pickle
import numpy as np
import faiss

STORE_PATH = "embeddings/faiss_index.bin"
META_PATH  = "embeddings/faiss_meta.pkl"


class FAISSVectorStore:
    """
    Builds and queries a FAISS flat index over resume embeddings.

    Flow:
      1. build_index(embeddings, metadata) → stores all resumes
      2. search(query_embedding, top_k)    → returns top-K similar resumes
      3. save() / load()                   → persist index to disk
    """

    def __init__(self, dim: int = 384):
        self.dim   = dim
        self.index = faiss.IndexFlatIP(dim)   # Inner Product (= cosine when L2-normalized)
        self.metadata: list[dict] = []         # stores {candidate, text} per entry

    # ── Build ────────────────────────────────────────────────────────
    def build_index(self, embeddings: np.ndarray, metadata: list[dict]):
        """
        Add all resume embeddings + their metadata to the FAISS index.
        embeddings: np.ndarray of shape (N, dim), float32, L2-normalized
        metadata  : list of dicts with at least 'candidate' and 'text' keys
        """
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.metadata = metadata
        print(f"  📦  FAISS index built → {self.index.ntotal} vectors stored")

    # ── Search ───────────────────────────────────────────────────────
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search for top-K most similar resumes to a query embedding.
        Returns list of dicts: {candidate, text, score}
        """
        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["similarity_score"] = float(score)  # cosine similarity (0–1)
            results.append(result)

        return results

    # ── Persist ──────────────────────────────────────────────────────
    def save(self, index_path: str = STORE_PATH, meta_path: str = META_PATH):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"  💾  FAISS index saved → {index_path}")

    def load(self, index_path: str = STORE_PATH, meta_path: str = META_PATH):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No saved index at {index_path}. Run indexing first.")
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"  📂  FAISS index loaded → {self.index.ntotal} vectors")

    def is_saved(self, index_path: str = STORE_PATH) -> bool:
        return os.path.exists(index_path)
