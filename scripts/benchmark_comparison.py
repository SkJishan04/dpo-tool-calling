"""Benchmark comparison across model versions."""

import argparse
from transformers import AutoTokenizer
from models.base_model import BaseToolCallingModel
from evaluation.benchmark import BenchmarkRunner
from data.dataset_generator import DatasetGenerator


def main():
    parser = argparse.ArgumentParser(description="Benchmark model comparison")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline model")
    parser.add_argument("--sft", type=str, help="SFT model checkpoint")
    parser.add_argument("--dpo", type=str, help="DPO model checkpoint")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of test samples")
    
    args = parser.parse_args()

    # Generate test dataset
    generator = DatasetGenerator()
    test_pairs = generator.generate_full_dataset(
        num_tool_calls=args.num_samples // 2,
        num_direct_answers=args.num_samples // 2
    )
    
    test_prompts = [pair.prompt for pair in test_pairs]
    references = [
        {"tool_name": pair.metadata.get("tool")} for pair in test_pairs
    ]

    # Initialize benchmark runner
    benchmark = BenchmarkRunner(test_prompts, references)

    # Load models
    models = {}
    if args.baseline:
        models["Baseline"] = BaseToolCallingModel(args.baseline)
    if args.sft:
        models["SFT Only"] = BaseToolCallingModel(args.sft)
    if args.dpo:
        models["SFT + DPO"] = BaseToolCallingModel(args.dpo)

    # Run benchmarks
    results = benchmark.compare_models(models, num_runs=args.num_samples)
    
    # Print results
    BenchmarkRunner.print_benchmark_table(results)


if __name__ == "__main__":
    main()