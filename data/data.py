# data/data.py

from datasets import load_dataset
from transformers import AutoTokenizer
import random

class IMDBDatasetLoader:
    """
    Loads the IMDB dataset, assigns synthetic movie titles,
    tokenizes for RoBERTa-small, and exposes helper methods
    for the UI and RAG pipeline.
    """

    def __init__(self, model_name: str, max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        self.raw = None
        self.tokenized = None
        self.title_to_reviews = {}

        # Curated list of known movie titles for synthetic mapping
        self.movie_titles = [
            "The Dark Knight", "Inception", "Interstellar", "The Matrix",
            "Titanic", "Jurassic Park", "The Godfather", "The Shawshank Redemption",
            "Pulp Fiction", "Fight Club", "Forrest Gump", "The Lion King",
            "Avatar", "Gladiator", "Toy Story", "The Avengers",
            "The Social Network", "Whiplash", "La La Land", "Mad Max Fury Road"
        ]

    # ------------------------------------------------------
    # Load IMDB dataset
    # ------------------------------------------------------
    def load_dataset(self):
        self.raw = load_dataset("imdb")
        return self.raw

    # ------------------------------------------------------
    # Assign movie titles round-robin to IMDB reviews
    # ------------------------------------------------------
    def assign_movie_titles(self):
        titles_count = len(self.movie_titles)

        def add_title(example, idx):
            title = self.movie_titles[idx % titles_count]
            example["movie_title"] = title
            return example

        # Apply mapping to both train and test sets
        self.raw["train"] = self.raw["train"].map(
            lambda x, i: add_title(x, i),
            with_indices=True
        )
        self.raw["test"] = self.raw["test"].map(
            lambda x, i: add_title(x, i),
            with_indices=True
        )

    # ------------------------------------------------------
    # Build mapping: title → list of reviews (strings)
    # ------------------------------------------------------
    def build_title_index(self):
        index = {}

        for split in ["train", "test"]:
            for item in self.raw[split]:
                title = item["movie_title"]
                review = item["text"]

                if title not in index:
                    index[title] = []
                index[title].append(review)

        self.title_to_reviews = index
        return index

    # ------------------------------------------------------
    # Tokenize IMDB for training sentiment classifier
    # ------------------------------------------------------
    def tokenize_for_training(self):
        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )

        self.tokenized = self.raw.map(tokenize, batched=True)
        return self.tokenized

    # ------------------------------------------------------
    # Helper methods for UI
    # ------------------------------------------------------
    def get_all_titles(self):
        return list(self.title_to_reviews.keys())

    def title_exists(self, title: str):
        return title in self.title_to_reviews

    def get_reviews_for_title(self, title: str):
        if title not in self.title_to_reviews:
            return []
        return self.title_to_reviews[title]
