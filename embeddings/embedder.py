# embeddings/embedder.py
# ─────────────────────────────────────────────────────────────────────
#  Embedding Engine — HuggingFace Sentence Transformers (FREE, local)
#  Model: all-MiniLM-L6-v2  (~80MB, fast, great for semantic search)
# ─────────────────────────────────────────────────────────────────────

import numpy as np
from sentence_transformers import SentenceTransformer

# ── Model (downloaded once, cached locally) ───────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"

class Embedder:
    """
    Wraps HuggingFace SentenceTransformer for generating dense embeddings.
    Used to embed both resumes and job descriptions for semantic similarity.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"  🤖  Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"  ✅  Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts → numpy array of shape (N, dim).
        Normalize vectors so cosine similarity = dot product.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalize → cosine = dot product
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings  # shape: (N, 384) for MiniLM

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text → 1D numpy array."""
        return self.embed([text])[0]
