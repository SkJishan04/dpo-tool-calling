"""Benchmark comparison for different model versions."""

import time
from typing import Dict, List, Any
import torch
from models.base_model import BaseToolCallingModel
from evaluation.metrics import ToolCallingMetrics
from evaluation.schema_validator import SchemaValidator
from utils.logger import setup_logger

logger = setup_logger(__name__)


class BenchmarkRunner:
    """Run benchmarks across model versions."""

    def __init__(self, test_prompts: List[str], references: List[Dict]):
        """Initialize with test prompts and reference outputs."""
        self.test_prompts = test_prompts
        self.references = references

    def benchmark_model(
        self,
        model: BaseToolCallingModel,
        model_name: str,
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark a single model.
        
        Returns:
            Dictionary with metrics and timing
        """
        logger.info(f"Benchmarking {model_name}...")
        
        predictions = []
        latencies = []

        # Generate predictions
        for prompt in self.test_prompts[:num_runs]:
            start_time = time.time()
            output = model.generate(prompt, max_length=512)
            latency = time.time() - start_time
            latencies.append(latency)

            # Validate output
            validation = SchemaValidator.full_validation(output)
            predictions.append(validation["parsed_data"] or {})

        # Compute metrics
        metrics = ToolCallingMetrics.compute_all_metrics(predictions, self.references[:num_runs])
        metrics["avg_latency_ms"] = sum(latencies) / len(latencies) * 1000
        metrics["model_name"] = model_name
        metrics["num_samples"] = num_runs

        return metrics

    def compare_models(
        self,
        models: Dict[str, BaseToolCallingModel],
        num_runs: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple models.
        
        Returns:
            Dictionary of metrics for each model
        """
        results = {}
        for name, model in models.items():
            results[name] = self.benchmark_model(model, name, num_runs)

        return results

    @staticmethod
    def print_benchmark_table(results: Dict[str, Dict[str, Any]]) -> None:
        """Print benchmark results as formatted table."""
        print("\n" + "="*100)
        print(f"{'Model':<20} {'Schema Acc':<15} {'Precision':<15} {'Recall':<15} {'Latency (ms)':<15}")
        print("="*100)

        for model_name, metrics in results.items():
            print(
                f"{model_name:<20} "
                f"{metrics['schema_accuracy']:<14.1f}% "
                f"{metrics['tool_precision']:<14.1f}% "
                f"{metrics['tool_recall']:<14.1f}% "
                f"{metrics['avg_latency_ms']:<14.2f}"
            )

        print("="*100 + "\n")