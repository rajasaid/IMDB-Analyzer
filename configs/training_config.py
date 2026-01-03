# configs/training_config.py

TRAINING_CONFIG = {
    "model_name": "distilroberta-base",

    # LoRA parameters
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,

    # Training hyperparameters
    "batch_size": 8,
    "learning_rate": 1e-4,
    "num_epochs": 3,

    # Tokenization
    "max_length": 256,

    "output_dir": "sentiment_lora_model"
}
