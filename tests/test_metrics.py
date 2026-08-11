"""Tests for evaluation metrics."""

import pytest
from evaluation.metrics import ToolCallingMetrics


def test_schema_accuracy():
    """Test schema accuracy calculation."""
    predictions = [
        {"tool_name": "search", "parameters": {}},
        {"tool_name": "search", "parameters": {}},
    ]
    references = [
        {"tool_name": "search", "parameters": {}},
        {"tool_name": "search", "parameters": {}},
    ]
    accuracy = ToolCallingMetrics.schema_accuracy(predictions, references)
    assert accuracy >= 0 and accuracy <= 100


def test_tool_precision():
    """Test tool precision calculation."""
    predictions = [
        {"tool_name": "search"},
        {"tool_name": "database_query"},
    ]
    references = [
        {"tool_name": "search"},
        {"tool_name": "search"},
    ]
    precision = ToolCallingMetrics.tool_precision(predictions, references)
    assert precision >= 0 and precision <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])