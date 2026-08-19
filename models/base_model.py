"""Base model wrapper for tool calling."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class BaseToolCallingModel:
    """Base model for tool calling with LoRA."""

    def __init__(self, model_name: str, lora_config: dict = None, device: str = "cuda"):
        """
        Initialize base model with optional LoRA.
        
        Args:
            model_name: HuggingFace model ID
            lora_config: LoRA configuration dict
            device: Device to load model on
        """
        self.model_name = model_name
        self.device = device
        
        # Load tokenizer and base model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # FIX: Set padding token (CRITICAL for batch processing)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=device
            )
        except RuntimeError as e:
            # Fallback to float32 if bfloat16 not supported
            if "bfloat16" in str(e):
                print(f"bfloat16 not supported, using float32")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=device
                )
            else:
                raise
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise
        
        # Apply LoRA if config provided
        if lora_config:
            self._apply_lora(lora_config)

    def _apply_lora(self, lora_config: dict) -> None:
        """Apply LoRA to model."""
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('lora_alpha', 32),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', 'none'),
            target_modules=lora_config.get('target_modules', ['q_proj', 'v_proj']),
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def forward(self, input_ids, attention_mask=None):
        """Forward pass."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def save_pretrained(self, save_path: str) -> None:
        """Save model."""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_pretrained(self, load_path: str) -> None:
        """Load model weights."""
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )