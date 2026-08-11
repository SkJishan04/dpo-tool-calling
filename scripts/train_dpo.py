"""Train DPO model."""

import argparse
from transformers import AutoTokenizer
from data.preference_dataset import PreferenceDataset
from models.base_model import BaseToolCallingModel
from models.dpo_trainer import DPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train DPO model")
    parser.add_argument("--sft-checkpoint", type=str, required=True, help="SFT model checkpoint")
    parser.add_argument("--dataset", type=str, default="data/raw/preference_dataset.jsonl", help="Dataset path")
    parser.add_argument("--output", type=str, default="models/dpo_checkpoint", help="Output directory")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint)
    
    # Load dataset
    dataset = PreferenceDataset(args.dataset, tokenizer)
    
    # Initialize model from SFT checkpoint
    model = BaseToolCallingModel(args.sft_checkpoint)
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        train_dataset=dataset,
        beta=args.beta,
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