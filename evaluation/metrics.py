"""Evaluation metrics for tool calling."""

from typing import Dict, List, Any
from sklearn.metrics import precision_score, recall_score, accuracy_score
import json


class ToolCallingMetrics:
    """Compute metrics for tool calling evaluation."""

    @staticmethod
    def schema_accuracy(predictions: List[Dict], references: List[Dict]) -> float:
        """
        Compute schema accuracy - percentage of valid JSON schemas.
        """
        valid_count = 0
        for pred in predictions:
            try:
                if isinstance(pred, str):
                    json.loads(pred)
                valid_count += 1
            except (json.JSONDecodeError, TypeError):
                pass
        
        return (valid_count / len(predictions)) * 100 if predictions else 0.0

    @staticmethod
    def tool_precision(predictions: List[Dict], references: List[Dict]) -> float:
        """
        Tool precision - correct tool calls / total predicted tool calls.
        """
        correct = 0
        predicted_tools = 0

        for pred, ref in zip(predictions, references):
            pred_tool = pred.get('tool_name')
            ref_tool = ref.get('tool_name')
            
            if pred_tool:
                predicted_tools += 1
                if pred_tool == ref_tool:
                    correct += 1

        return (correct / predicted_tools * 100) if predicted_tools > 0 else 0.0

    @staticmethod
    def tool_recall(predictions: List[Dict], references: List[Dict]) -> float:
        """
        Tool recall - correct tool calls / total actual tool calls.
        """
        correct = 0
        actual_tools = 0

        for pred, ref in zip(predictions, references):
            ref_tool = ref.get('tool_name')
            pred_tool = pred.get('tool_name')
            
            if ref_tool:
                actual_tools += 1
                if pred_tool == ref_tool:
                    correct += 1

        return (correct / actual_tools * 100) if actual_tools > 0 else 0.0

    @staticmethod
    def parameter_accuracy(predictions: List[Dict], references: List[Dict]) -> float:
        """
        Parameter accuracy - correct parameters for tool calls.
        """
        correct = 0
        total = 0

        for pred, ref in zip(predictions, references):
            if pred.get('tool_name') == ref.get('tool_name'):
                total += 1
                pred_params = pred.get('parameters', {})
                ref_params = ref.get('parameters', {})
                
                if pred_params == ref_params:
                    correct += 1

        return (correct / total * 100) if total > 0 else 0.0

    @staticmethod
    def compute_all_metrics(predictions: List[Dict], references: List[Dict]) -> Dict[str, float]:
        """Compute all metrics at once."""
        return {
            "schema_accuracy": ToolCallingMetrics.schema_accuracy(predictions, references),
            "tool_precision": ToolCallingMetrics.tool_precision(predictions, references),
            "tool_recall": ToolCallingMetrics.tool_recall(predictions, references),
            "parameter_accuracy": ToolCallingMetrics.parameter_accuracy(predictions, references),
        }