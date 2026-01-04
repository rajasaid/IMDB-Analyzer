# main.py

from configs.training_config import TRAINING_CONFIG
from configs.rag_config import RAG_CONFIG

from data import IMDBDatasetLoader
from models import SentimentClassifier, ResponseGenerator, ImageGenerator
from rag import ReviewEmbedder, ReviewRetriever, PromptBuilder, RAGPipeline


def prepare_datasets():
    """
    Load IMDB, assign synthetic movie titles, build title index,
    and return the loader + flattened review/title lists.
    """
    print("[main] Loading IMDB dataset...")
    loader = IMDBDatasetLoader(model_name=TRAINING_CONFIG["model_name"])
    loader.load_dataset()
    loader.assign_movie_titles()
    loader.build_title_index()

    # Flatten title_to_reviews into parallel lists
    reviews = []
    titles = []
    for title, revs in loader.title_to_reviews.items():
        for r in revs:
            reviews.append(r)
            titles.append(title)

    print(f"[main] Collected {len(reviews)} reviews with synthetic titles.")
    return loader, reviews, titles


def prepare_classifier(train_dataset, eval_dataset):
    """
    Initialize SentimentClassifier, train once if no saved model,
    then return it ready for inference.
    """
    clf = SentimentClassifier(
        model_name=TRAINING_CONFIG["model_name"],
        lora_r=TRAINING_CONFIG["lora_r"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        lora_dropout=TRAINING_CONFIG["lora_dropout"],
        model_dir=TRAINING_CONFIG["output_dir"],
    )

    # Will only actually train if model_dir does not already exist
    clf.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        lr=TRAINING_CONFIG["learning_rate"],
        num_epochs=TRAINING_CONFIG["num_epochs"],
    )

    return clf


def prepare_rag_components(reviews, titles):
    """
    Create the embedder, build embeddings, retriever, generator,
    prompt builder, and full RAG pipeline.
    """
    print("[main] Building embeddings for RAG...")
    embedder = ReviewEmbedder(
        model_name=RAG_CONFIG["embedding_model"],
        device="cpu"
    )
    embedder.embed_reviews(reviews, titles)

    print("[main] Building retriever...")
    retriever = ReviewRetriever(embedder)

    print("[main] Loading generator and prompt builder...")
    generator = ResponseGenerator(
        model_name=RAG_CONFIG["generator_model"],
        max_new_tokens=RAG_CONFIG["max_new_tokens"],
        temperature=RAG_CONFIG["temperature"],
        top_p=RAG_CONFIG["top_p"],
    )
    image_generator = ImageGenerator()
    prompt_builder = PromptBuilder()

    print("[main] Assembling RAG pipeline...")
    # Classifier will be passed in separately from caller
    return embedder, retriever, generator, image_generator, prompt_builder


def cli_demo(pipeline, loader):
    """
    Simple command-line demo of the full pipeline:
    - ask for movie title
    - ask for review
    - print sentiment + generated answer
    """
    print("\n================= CLI DEMO =================")
    print("You can test the RAG + sentiment system here.")
    print("Known synthetic titles include e.g.:")
    for t in loader.get_all_titles()[:10]:
        print("  -", t)
    print("============================================\n")

    movie_title = input("Enter a movie title: ").strip()
    if not loader.title_exists(movie_title):
        print(f"[CLI] Title '{movie_title}' not found in dataset.")
        return

    user_review = input(f"Enter your review for '{movie_title}':\n> ").strip()
    if len(user_review) < 5:
        print("[CLI] Review too short, aborting.")
        return

    print("\n[CLI] Running pipeline...")
    sentiment, retrieved_reviews, response, image = pipeline.run(
        movie_title=movie_title,
        user_review=user_review,
        top_k=RAG_CONFIG["top_k"],
    )

    print("\n=== RESULTS ===")
    print("Sentiment detected:", sentiment)
    print("\nRetrieved similar reviews (snippets):")
    for r in retrieved_reviews[:3]:
        print("-", r[:200].replace("\n", " "), "...")
    print("\nAssistant response:\n")
    print(response)
    print("\n=============================\n")
    
    image.show()

def main():
    
    # 1. Prepare data
    loader, reviews, titles = prepare_datasets()

    # 2. Tokenize dataset for classifier training
    print("[main] Tokenizing dataset for classifier training...")
    tokenized = loader.tokenize_for_training()
    train_ds = tokenized["train"]
    eval_ds = tokenized["test"]

    # 3. Prepare classifier (train-once)
    classifier = prepare_classifier(train_ds, eval_ds)

    # 4. Prepare RAG components
    embedder, retriever, generator, image_generator, prompt_builder = prepare_rag_components(
        reviews, titles
    )

    # 5. Build RAG pipeline
    pipeline = RAGPipeline(
        classifier=classifier,
        embedder=embedder,
        retriever=retriever,
        generator=generator,
        image_generator=image_generator,
        prompt_builder=prompt_builder,
    )

    # 6. Optional CLI test
    cli_demo(pipeline, loader)


if __name__ == "__main__":
    main()
