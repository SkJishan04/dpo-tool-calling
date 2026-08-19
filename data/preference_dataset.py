"""Preference dataset loader for DPO training."""

import json
from typing import Dict, List, Any
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    """PyTorch Dataset for preference pairs."""

    def __init__(self, filepath: str, tokenizer, max_length: int = 512):
        """
        Load preference dataset from file.
        
        Args:
            filepath: Path to JSONL or JSON file
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []

        self._load_data(filepath)

    def _load_data(self, filepath: str) -> None:
        """Load data from JSONL or JSON."""
        if filepath.endswith('.jsonl'):
            with open(filepath, 'r') as f:
                for line in f:
                    self.pairs.append(json.loads(line))
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.pairs = data if isinstance(data, list) else [data]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a preference pair and tokenize."""
        pair = self.pairs[idx]
        
        # FIX: Safe access with validation
        prompt = pair.get('prompt', '')
        chosen = pair.get('chosen', '')
        rejected = pair.get('rejected', '')
        
        # Validate required fields
        if not all([prompt, chosen, rejected]):
            raise ValueError(
                f"Pair {idx} missing required fields. "
                f"Got keys: {list(pair.keys())}"
            )

        # Tokenize prompt + chosen
        chosen_text = f"{prompt}\n{chosen}"
        chosen_encodings = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize prompt + rejected
        rejected_text = f"{prompt}\n{rejected}"
        rejected_encodings = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "chosen_input_ids": chosen_encodings["input_ids"].squeeze(),
            "chosen_attention_mask": chosen_encodings["attention_mask"].squeeze(),
            "rejected_input_ids": rejected_encodings["input_ids"].squeeze(),
            "rejected_attention_mask": rejected_encodings["attention_mask"].squeeze(),
            "metadata": pair.get('metadata', {})
        }