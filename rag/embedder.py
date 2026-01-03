# rag/embedder.py

import os
import numpy as np
from sentence_transformers import SentenceTransformer


class ReviewEmbedder:
    """
    Creates embeddings for IMDB reviews using SentenceTransformers.
    Provides methods for:
    - embedding corpus reviews
    - embedding user queries
    - saving/loading embedding arrays
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize the embedding model.
        """

        self.model_name = model_name
        self.device = device or "cpu"  # Good default
        print(f"[ReviewEmbedder] Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

        # Stored embeddings + metadata
        self.embeddings = None          # numpy array [N, D]
        self.reviews = None             # list of review strings
        self.movie_titles = None        # parallel list mapping index → movie title

    # ----------------------------------------------------------------------
    # Build embeddings for a list of reviews
    # ----------------------------------------------------------------------
    def embed_reviews(self, reviews: list, titles: list):
        """
        Given lists:
        - reviews: list of text strings
        - titles:  list of movie titles (parallel array)

        Creates embeddings and stores everything internally.
        """

        print(f"[ReviewEmbedder] Encoding {len(reviews)} reviews...")

        embeddings = self.model.encode(
            reviews,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        self.embeddings = embeddings
        self.reviews = reviews
        self.movie_titles = titles

        print("[ReviewEmbedder] Review embeddings created.")

        return embeddings

    # ----------------------------------------------------------------------
    # Embed a single user query
    # ----------------------------------------------------------------------
    def embed_query(self, text: str):
        """
        Encode a single piece of text (user review input).
        Returns a numpy embedding vector.
        """
        return self.model.encode(text, convert_to_numpy=True)

    # ----------------------------------------------------------------------
    # Save embeddings to disk for later loading
    # ----------------------------------------------------------------------
    def save(self, directory: str = "embeddings"):
        os.makedirs(directory, exist_ok=True)

        np.save(os.path.join(directory, "embeddings.npy"), self.embeddings)
        np.save(os.path.join(directory, "titles.npy"), np.array(self.movie_titles, dtype=object))
        np.save(os.path.join(directory, "reviews.npy"), np.array(self.reviews, dtype=object))

        print(f"[ReviewEmbedder] Saved embeddings to folder: {directory}")

    # ----------------------------------------------------------------------
    # Load embeddings from disk
    # ----------------------------------------------------------------------
    def load(self, directory: str = "embeddings"):
        self.embeddings = np.load(os.path.join(directory, "embeddings.npy"))
        self.movie_titles = np.load(os.path.join(directory, "titles.npy"), allow_pickle=True).tolist()
        self.reviews = np.load(os.path.join(directory, "reviews.npy"), allow_pickle=True).tolist()

        print(f"[ReviewEmbedder] Loaded embeddings from folder: {directory}")

        return self.embeddings
