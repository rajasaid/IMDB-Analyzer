# ui/app.py

import streamlit as st
import os
from data import IMDBDatasetLoader
from models import SentimentClassifier, ResponseGenerator
from rag import ReviewEmbedder, ReviewRetriever, PromptBuilder, RAGPipeline
from configs.rag_config import RAG_CONFIG
from configs.training_config import TRAINING_CONFIG
import sys

# Add project root directory to Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# =====================================================================
# INITIALIZATION SECTION (Runs once thanks to Streamlit caching)
# =====================================================================

@st.cache_resource
def load_data_and_embeddings():
    """Load IMDB, assign titles, and build title index + embeddings."""

    # Load dataset
    loader = IMDBDatasetLoader(model_name=TRAINING_CONFIG["model_name"])
    loader.load_dataset()
    loader.assign_movie_titles()
    loader.build_title_index()

    # Flatten reviews into lists
    reviews = []
    titles = []
    for title, revs in loader.title_to_reviews.items():
        for r in revs:
            reviews.append(r)
            titles.append(title)

    # Create embedder + encode all reviews
    embedder = ReviewEmbedder(
        model_name=RAG_CONFIG["embedding_model"],
        device="cpu"
    )
    embedder.embed_reviews(reviews, titles)

    # Create retriever using these embeddings
    retriever = ReviewRetriever(embedder)

    return loader, embedder, retriever


@st.cache_resource
def load_models():
    """Load classifier, generator, and prompt builder once."""
    model_dir = TRAINING_CONFIG["output_dir"]

    if not os.path.exists(model_dir):
        st.warning(
            f"Trained sentiment model not found at '{model_dir}'. "
            "The classifier will be untrained. "
            "Run your training script first to get good results."
        )

    classifier = SentimentClassifier(
        model_name=TRAINING_CONFIG["model_name"],
        model_dir=model_dir,
        lora_r=TRAINING_CONFIG["lora_r"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
    )

    generator = ResponseGenerator(
        model_name=RAG_CONFIG["generator_model"],
        max_new_tokens=RAG_CONFIG["max_new_tokens"],
        temperature=RAG_CONFIG["temperature"],
        top_p=RAG_CONFIG["top_p"],
    )
    prompt_builder = PromptBuilder()

    return classifier, generator, prompt_builder


# Load everything
loader, embedder, retriever = load_data_and_embeddings()
classifier, generator, prompt_builder = load_models()

# Build the RAG pipeline
pipeline = RAGPipeline(
    classifier=classifier,
    embedder=embedder,
    retriever=retriever,
    generator=generator,
    prompt_builder=prompt_builder
)


# =====================================================================
# STREAMLIT UI LOGIC — 3 SCREEN APPLICATION
# =====================================================================

st.set_page_config(page_title="Movie Review Assistant (RAG + Sentiment)", layout="centered")
st.title("🎬 Movie Review Assistant — Sentiment + RAG")


# ----------------------------------------
# STAGE 1 — ENTER MOVIE TITLE
# ----------------------------------------

if "stage" not in st.session_state:
    st.session_state.stage = 1
if "selected_title" not in st.session_state:
    st.session_state.selected_title = None
if "user_review" not in st.session_state:
    st.session_state.user_review = None


if st.session_state.stage == 1:
    st.subheader("Step 1 — Choose a Movie Title")

    movie_title = st.text_input("Enter a movie title:").strip()

    if st.button("Submit Title"):
        if loader.title_exists(movie_title):
            st.session_state.selected_title = movie_title
            st.session_state.stage = 2
            st.rerun()
        else:
            st.error("❌ Movie not found in dataset. Please try another title.")


# ----------------------------------------
# STAGE 2 — ENTER USER REVIEW
# ----------------------------------------

elif st.session_state.stage == 2:
    st.subheader(f"Step 2 — Write Your Review about “{st.session_state.selected_title}”")

    user_review = st.text_area("Enter your review:", height=150)

    if st.button("Analyze Review"):
        if len(user_review.strip()) < 5:
            st.error("Please enter a valid review (at least 5 characters).")
        else:
            st.session_state.user_review = user_review
            st.session_state.stage = 3
            st.rerun()

# ----------------------------------------
# STAGE 3 — SHOW RESULTS FROM RAG PIPELINE
# ----------------------------------------

elif st.session_state.stage == 3:
    st.subheader("Step 3 — Analysis Results")

    movie_title = st.session_state.selected_title
    user_review = st.session_state.user_review

    with st.spinner("Running sentiment analysis + retrieval + generation..."):
        sentiment, retrieved_reviews, response = pipeline.run(
            movie_title=movie_title,
            user_review=user_review,
            top_k=RAG_CONFIG["top_k"]
        )

    # Display sentiment
    st.markdown(f"### 🎭 Sentiment Detected: **{sentiment.capitalize()}**")

    # Display retrieved context
    st.markdown("### 🔍 Retrieved Context from Similar Reviews:")
    if retrieved_reviews:
        for r in retrieved_reviews:
            st.markdown(f"- {r[:300]}...")
    else:
        st.write("No similar reviews found.")

    # Display final generated response
    st.markdown("### 🤖 Assistant Response:")
    st.write(response)

    # Option to restart
    if st.button("Analyze Another Review"):
        st.session_state.stage = 1
        st.session_state.selected_title = None
        st.session_state.user_review = None
        st.rerun()
