# rag/rag_pipeline.py

from models.classifier import SentimentClassifier
from models.generator import ResponseGenerator
from rag.embedder import ReviewEmbedder
from rag.prompt_builder import PromptBuilder
from rag.retriever import ReviewRetriever


class RAGPipeline:
    """
    Full RAG pipeline:
    1. Predict sentiment
    2. Embed user review
    3. Retrieve similar reviews for that movie
    4. Build prompt
    5. Generate final response
    """

    def __init__(
        self,
        classifier: SentimentClassifier,
        embedder: ReviewEmbedder,
        retriever: ReviewRetriever,
        generator: ResponseGenerator,
        prompt_builder: PromptBuilder,
    ):
        self.classifier = classifier
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder

    # -------------------------------------------------------------
    # Main user-facing method
    # -------------------------------------------------------------
    def run(self, movie_title: str, user_review: str, top_k: int = 5):
        """
        Full pipeline execution.
        Returns (sentiment_text, retrieved_reviews, final_response)
        """

        # 1. Predict sentiment
        sentiment_label = self.classifier.predict(user_review)
        sentiment_text = "positive" if sentiment_label == 1 else "negative"

        # 2. Embed user text
        query_emb = self.embedder.embed_query(user_review)

        # 3. Retrieve context
        retrieved_reviews = self.retriever.retrieve(
            movie_title,
            query_emb,
            top_k=top_k
        )

        # 4. Build prompt for generator
        prompt = self.prompt_builder.build(
            movie_title=movie_title,
            user_review=user_review,
            sentiment_label=sentiment_label,
            retrieved_reviews=retrieved_reviews,
        )

        # 5. Generate response
        response = self.generator.generate(prompt)

        return sentiment_text, retrieved_reviews, response
