# models/sft_trainer.py
"""Supervised Fine-Tuning (SFT) trainer for tool calling."""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from typing import Dict
import wandb

from models.base_model import BaseToolCallingModel
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SFTTrainer:
    """SFT trainer for tool calling models."""

    def __init__(
        self,
        model: BaseToolCallingModel,
        train_dataset,
        val_dataset=None,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        output_dir: str = "./models/sft_checkpoint",
        use_wandb: bool = False,
    ):
        """Initialize SFT trainer."""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.device = model.device

        # Setup optimizer and scheduler
        self.optimizer = AdamW(model.model.parameters(), lr=learning_rate)
        self.total_steps = len(train_dataset) * num_epochs // batch_size
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )

    def train(self) -> Dict[str, float]:
        """Run SFT training."""
        logger.info("Starting SFT training...")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            epoch_loss = self._train_epoch(train_loader, epoch)
            
            if self.val_dataset:
                val_loss = self._validate()
                logger.info(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}")
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.model.save_pretrained(self.output_dir)
                    logger.info(f"Saved best model to {self.output_dir}")
            else:
                logger.info(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}")
                self.model.save_pretrained(self.output_dir)

        logger.info("SFT training completed!")
        return {"best_loss": best_loss}

    def _train_epoch(self, train_loader, epoch: int) -> float:
        """Train one epoch."""
        self.model.model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["chosen_input_ids"].to(self.device)
            attention_mask = batch["chosen_attention_mask"].to(self.device)

            # Forward pass
            outputs = self.model.forward(input_ids, attention_mask)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item()})

            if self.use_wandb:
                wandb.log({"train_loss": loss.item(), "epoch": epoch})

        return total_loss / len(train_loader)

    def _validate(self) -> float:
        """Validate on validation set."""
        self.model.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["chosen_input_ids"].to(self.device)
                attention_mask = batch["chosen_attention_mask"].to(self.device)
                outputs = self.model.forward(input_ids, attention_mask)
                total_loss += outputs.loss.item()

        return total_loss / len(val_loader)