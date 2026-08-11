"""Tests for JSON parser."""

import pytest
from utils.json_parser import JSONParser


def test_extract_json_direct():
    """Test direct JSON extraction."""
    text = '{"key": "value"}'
    success, data, error = JSONParser.extract_json(text)
    assert success
    assert data["key"] == "value"


def test_extract_json_from_text():
    """Test JSON extraction from mixed text."""
    text = 'Here is some JSON: {"key": "value"} and more text'
    success, data, error = JSONParser.extract_json(text)
    assert success
    assert data["key"] == "value"


def test_clean_json_string():
    """Test JSON string cleaning."""
    json_str = "```json\n{\"key\": \"value\"}\n```"
    cleaned = JSONParser.clean_json_string(json_str)
    assert "```" not in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])