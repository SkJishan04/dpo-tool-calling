"""Train SFT model."""

import argparse
import yaml
from transformers import AutoTokenizer
from data.preference_dataset import PreferenceDataset
from models.base_model import BaseToolCallingModel
from models.sft_trainer import SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="Train SFT model")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model")
    parser.add_argument("--dataset", type=str, default="data/raw/preference_dataset.jsonl", help="Dataset path")
    parser.add_argument("--output", type=str, default="models/sft_checkpoint", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Load dataset
    dataset = PreferenceDataset(args.dataset, tokenizer)
    
    # Initialize model with LoRA
    lora_config = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj"]
    }
    model = BaseToolCallingModel(args.model, lora_config=lora_config)
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output,
        use_wandb=args.use_wandb
    )
    
    # Train
    trainer.train()
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()