# configs/rag_config.py

RAG_CONFIG = {
    # Embedding model for SentenceTransformers
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",

    # Retrieval parameters
    "top_k": 5,
    "faiss_overshoot_factor": 10,  # request top_k * 10 neighbors before filtering

    # Generator model parameters
    "generator_model": "microsoft/phi-2",
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9
}
