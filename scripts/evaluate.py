"""Evaluate model on tool calling metrics."""

import argparse
from transformers import AutoTokenizer
from models.base_model import BaseToolCallingModel
from evaluation.metrics import ToolCallingMetrics
from evaluation.schema_validator import SchemaValidator
from data.dataset_generator import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Evaluate model")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples")
    
    args = parser.parse_args()

    # Load model
    model = BaseToolCallingModel(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Generate test dataset
    generator = DatasetGenerator()
    test_pairs = generator.generate_full_dataset(
        num_tool_calls=args.num_samples // 2,
        num_direct_answers=args.num_samples // 2
    )

    # Make predictions
    predictions = []
    references = []
    
    for pair in test_pairs:
        output = model.generate(pair.prompt, max_length=512)
        validation = SchemaValidator.full_validation(output)
        
        predictions.append(validation["parsed_data"] or {})
        references.append({
            "should_call_tool": True if pair.metadata.get("type") == "tool_call" else False,
            "tool_name": pair.metadata.get("tool")
        })

    # Compute metrics
    metrics = ToolCallingMetrics.compute_all_metrics(predictions, references)
    
    print("\n" + "="*50)
    print(f"Evaluation Results ({args.num_samples} samples)")
    print("="*50)
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()