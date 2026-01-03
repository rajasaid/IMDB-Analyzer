# models/classifier.py

import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, PeftModel
from configs.training_config import TRAINING_CONFIG

class SentimentClassifier:
    """
    Handles sentiment classification using RoBERTa-small + LoRA.
    - Trains only if model directory does NOT already exist.
    - Otherwise loads the trained LoRA adapter.
    """

    def __init__(
        self,
        model_name: str = TRAINING_CONFIG["model_name"],
        num_labels: int = 2,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        model_dir: str = "sentiment_lora_model",
        device: str = None
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.num_labels = num_labels

        # 🔴 FORCE CPU (to avoid MPS issues)
        self.device = "cpu"
        print(f"[SentimentClassifier] Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        # If model_dir exists → load trained model instead of initializing
        if os.path.exists(model_dir):
            print(f"[SentimentClassifier] Loading trained model from: {model_dir}")
            self.model = PeftModel.from_pretrained(
                AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_labels
                ),
                model_dir
            ).to(self.device)

        else:
            print("[SentimentClassifier] No trained model found. Creating new model...")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels
            )

            # LoRA configuration
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_CLS"
            )

            # Add LoRA layers
            self.model = get_peft_model(base_model, self.lora_config).to(self.device)

    # ----------------------------------------------------------------------
    # TRAIN MODEL (only run if model does not already exist)
    # ----------------------------------------------------------------------
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        batch_size=8,
        lr=1e-4,
        num_epochs=3,
    ):
        """
        Train only if model_dir does NOT exist.
        """
        if os.path.exists(self.model_dir):
            print(f"[SentimentClassifier] Model already exists at {self.model_dir}. Skipping training.")
            return

        print("[SentimentClassifier] Starting training...")

        training_args = TrainingArguments(
            output_dir=self.model_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model(self.model_dir)

        print(f"[SentimentClassifier] Training finished. Model saved to {self.model_dir}.")

    # ----------------------------------------------------------------------
    # MANUAL LOAD AFTER TRAINING
    # ----------------------------------------------------------------------
    def load(self):
        """
        Explicitly reload trained LoRA weights.
        """
        if not os.path.exists(self.model_dir):
            raise ValueError(f"No saved model found at {self.model_dir}")

        print(f"[SentimentClassifier] Loading trained model from {self.model_dir}...")

        base = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.model = PeftModel.from_pretrained(base, self.model_dir).to(self.device)

    # ----------------------------------------------------------------------
    # INFERENCE
    # ----------------------------------------------------------------------
    def predict(self, text: str):
        self.model.eval()

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        ).to(self.device)

        # 🔴 Ensure tensors are on CPU (same as model)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        
        with torch.no_grad():
            logits = self.model(**enc).logits

        return torch.argmax(logits, dim=1).item()

   
