# rag/prompt_builder.py

class PromptBuilder:
    """
    Builds a clean prompt for the generative model using:
    - the movie title
    - user sentiment
    - the user review text
    - retrieved contextual reviews
    """

    def __init__(self):
        pass

    # -------------------------------------------------------------
    # Build prompt for generator
    # -------------------------------------------------------------
    def build(
        self,
        movie_title: str,
        user_review: str,
        sentiment_label: int,
        retrieved_reviews: list,
    ):
        """
        Constructs the final prompt string.

        sentiment_label: 0 = negative, 1 = positive
        retrieved_reviews: list[str] of similar reviews
        """

        sentiment_text = "positive" if sentiment_label == 1 else "negative"

        # Format the retrieved reviews
        context_block = ""
        if retrieved_reviews:
            context_block = "\n".join(
                f"- {rev[:300].strip()}" for rev in retrieved_reviews
            )
        else:
            context_block = "No relevant reviews found."

        prompt = f"""
You are a movie discussion assistant.
The user has selected the movie: "{movie_title}".

User sentiment: {sentiment_text}
User review: "{user_review}"

Here are some related reviews from other users about this movie:
{context_block}

Using the above context, generate a clear, friendly, and helpful response
that addresses the user's opinion and provides useful commentary about the movie.
Your response should be 3–5 sentences, coherent, and relevant.
Avoid repeating user text verbatim.

Response:
"""

        return prompt.strip()
