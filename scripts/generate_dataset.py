"""Generate preference dataset for DPO training."""

import argparse
import json
from data.dataset_generator import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate preference dataset")
    parser.add_argument("--num-tool-calls", type=int, default=500, help="Number of tool call samples")
    parser.add_argument("--num-direct-answers", type=int, default=500, help="Number of direct answer samples")
    parser.add_argument("--output", type=str, default="data/raw/preference_dataset.jsonl", help="Output file path")
    parser.add_argument("--format", type=str, choices=["jsonl", "json"], default="jsonl", help="Output format")
    
    args = parser.parse_args()

    # Generate dataset
    generator = DatasetGenerator()
    dataset = generator.generate_full_dataset(
        num_tool_calls=args.num_tool_calls,
        num_direct_answers=args.num_direct_answers
    )

    # Save dataset
    if args.format == "jsonl":
        generator.save_to_jsonl(args.output)
    else:
        generator.save_to_json(args.output)

    print(f"Generated {len(dataset)} preference pairs")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()