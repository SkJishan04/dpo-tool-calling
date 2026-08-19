# models/dpo_trainer.py
"""Direct Preference Optimization (DPO) trainer for tool calling."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from typing import Dict
import wandb

from models.base_model import BaseToolCallingModel
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DPOTrainer:
    """DPO trainer for preference-based optimization."""

    def __init__(
        self,
        model: BaseToolCallingModel,
        train_dataset,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        output_dir: str = "./models/dpo_checkpoint",
        use_wandb: bool = False,
    ):
        """Initialize DPO trainer."""
        self.model = model
        self.train_dataset = train_dataset
        self.beta = beta
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.device = model.device

        self.optimizer = AdamW(model.model.parameters(), lr=learning_rate)
        # FIX: Ensure total_steps is at least 1
        self.total_steps = max(1, len(train_dataset) * num_epochs // batch_size)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=min(warmup_steps, self.total_steps // 10),
            num_training_steps=self.total_steps
        )

    def compute_dpo_loss(self, chosen_logits, rejected_logits) -> torch.Tensor:
        """Compute DPO loss."""
        if self.loss_type == "sigmoid":
            # Sigmoid loss: log(sigmoid(beta * (y_chosen - y_rejected)))
            preference_logits = self.beta * (chosen_logits - rejected_logits)
            loss = -F.logsigmoid(preference_logits).mean()
        elif self.loss_type == "hinge":
            # Hinge loss: max(0, 1 - (y_chosen - y_rejected))
            margin = chosen_logits - rejected_logits
            loss = torch.clamp(1 - margin, min=0).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss

    def train(self) -> Dict[str, float]:
        """Run DPO training."""
        logger.info("Starting DPO training...")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # FIX: Required for Colab compatibility
            pin_memory=True
        )

        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            epoch_loss = self._train_epoch(train_loader, epoch)
            
            logger.info(f"Epoch {epoch+1}: DPO Loss={epoch_loss:.4f}")
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.model.save_pretrained(self.output_dir)
                logger.info(f"Saved best model to {self.output_dir}")

        logger.info("DPO training completed!")
        return {"best_loss": best_loss}

    def _train_epoch(self, train_loader, epoch: int) -> float:
        """Train one epoch with DPO."""
        self.model.model.train()
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f"DPO Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            chosen_input_ids = batch["chosen_input_ids"].to(self.device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
            rejected_input_ids = batch["rejected_input_ids"].to(self.device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)

            # Forward pass for chosen
            chosen_outputs = self.model.forward(chosen_input_ids, chosen_attention_mask)
            chosen_logits = chosen_outputs.logits.sum(dim=1)  # Aggregate logits
            
            # Forward pass for rejected
            rejected_outputs = self.model.forward(rejected_input_ids, rejected_attention_mask)
            rejected_logits = rejected_outputs.logits.sum(dim=1)  # Aggregate logits

            # Compute DPO loss
            loss = self.compute_dpo_loss(chosen_logits, rejected_logits)
            
            # Backward pass
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            progress_bar.set_postfix({"dpo_loss": loss.item()})

            if self.use_wandb:
                wandb.log({"dpo_loss": loss.item(), "epoch": epoch})

        return total_loss / len(train_loader)