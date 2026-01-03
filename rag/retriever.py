# rag/retriever.py

import numpy as np
import faiss


class ReviewRetriever:
    """
    Retrieves top-k most similar reviews for a given movie title
    using a global FAISS nearest neighbor index and then filtering
    results by movie.
    """

    def __init__(self, embedder, use_gpu: bool = False):
        """
        embedder: instance of ReviewEmbedder
        use_gpu: if True, FAISS GPU acceleration will be used (if available)
        """

        self.embedder = embedder
        self.use_gpu = use_gpu

        self.embeddings = embedder.embeddings      # np.array [N, D]
        self.reviews = embedder.reviews            # list[str] length N
        self.movie_titles = embedder.movie_titles  # list[str] length N

        self.index = None
        self._build_global_index()

    # ------------------------------------------------------------------
    # Build ONE global FAISS index over all reviews
    # ------------------------------------------------------------------
    def _build_global_index(self):
        if self.embeddings is None:
            raise ValueError(
                "ReviewRetriever: embeddings not found. "
                "Run embedder.embed_reviews(...) first."
            )

        dim = self.embeddings.shape[1]
        print(f"[ReviewRetriever] Building global FAISS index (dim={dim})...")

        cpu_index = faiss.IndexFlatL2(dim)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("[ReviewRetriever] Using GPU FAISS index.")
            self.index = faiss.index_cpu_to_all_gpus(cpu_index)
        else:
            self.index = cpu_index

        self.index.add(self.embeddings.astype(np.float32))
        print(f"[ReviewRetriever] Index built with {self.embeddings.shape[0]} vectors.")

    # ------------------------------------------------------------------
    # Retrieve top-k reviews for a specific movie
    # ------------------------------------------------------------------
    def retrieve(self, movie_title: str, query_embedding: np.ndarray, top_k: int = 5):
        """
        1. Search the global FAISS index for nearest neighbors.
        2. Filter results to only those whose movie title == movie_title.
        3. Return top_k matching reviews (or fewer if not enough).
        """

        if self.index is None:
            raise ValueError("FAISS index not built.")

        # How many neighbors to ask from FAISS before filtering.
        # We overshoot a bit so that after filtering by movie we still
        # have enough. You can tune this factor.
        n_total = self.embeddings.shape[0]
        search_k = min(top_k * 10, n_total)

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)

        distances, indices = self.index.search(query_vec, search_k)
        neighbors = indices[0]

        results = []
        for idx in neighbors:
            title = self.movie_titles[int(idx)]
            if title == movie_title:
                results.append(self.reviews[int(idx)])
                if len(results) >= top_k:
                    break

        # If not enough results for that movie, you just get fewer.
        if not results:
            print(f"[ReviewRetriever] No matching reviews for movie '{movie_title}'.")
        return results
