"""Tests for schema validator."""

import pytest
from evaluation.schema_validator import SchemaValidator


def test_validate_json_format_valid():
    """Test valid JSON format."""
    text = '{"should_call_tool": true, "tool_name": "search"}'
    is_valid, data, error = SchemaValidator.validate_json_format(text)
    assert is_valid
    assert data["should_call_tool"] is True


def test_validate_json_format_invalid():
    """Test invalid JSON format."""
    text = "This is not JSON"
    is_valid, data, error = SchemaValidator.validate_json_format(text)
    assert not is_valid


def test_validate_tool_call_schema():
    """Test tool call schema validation."""
    data = {
        "should_call_tool": True,
        "tool_name": "search",
        "parameters": {"query": "test"},
        "reasoning": "Test reasoning"
    }
    is_valid, error = SchemaValidator.validate_tool_call_schema(data)
    assert is_valid


def test_full_validation():
    """Test full validation pipeline."""
    text = '{"should_call_tool": true, "tool_name": "search", "parameters": {"query": "test"}, "reasoning": "Test"}'
    result = SchemaValidator.full_validation(text)
    assert result["json_valid"]
    assert result["schema_valid"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])