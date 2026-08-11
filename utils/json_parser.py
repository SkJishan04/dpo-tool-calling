"""JSON parsing utilities."""

import json
import re
from typing import Dict, Any, Tuple


class JSONParser:
    """Parse and extract JSON from text."""

    @staticmethod
    def extract_json(text: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Extract JSON object from text.
        
        Returns:
            (success, parsed_dict, error_message)
        """
        try:
            # Try direct parsing first
            return True, json.loads(text), ""
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in text
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start == -1 or end == 0:
                return False, {}, "No JSON object found in text"
            
            json_str = text[start:end]
            parsed = json.loads(json_str)
            return True, parsed, ""
        except json.JSONDecodeError as e:
            return False, {}, f"Failed to parse JSON: {str(e)}"

    @staticmethod
    def extract_json_array(text: str) -> Tuple[bool, list, str]:
        """Extract JSON array from text."""
        try:
            return True, json.loads(text), ""
        except json.JSONDecodeError:
            pass

        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            
            if start == -1 or end == 0:
                return False, [], "No JSON array found in text"
            
            json_str = text[start:end]
            parsed = json.loads(json_str)
            return True, parsed, ""
        except json.JSONDecodeError as e:
            return False, [], f"Failed to parse JSON array: {str(e)}"

    @staticmethod
    def clean_json_string(json_str: str) -> str:
        """Clean JSON string (remove markdown code blocks, etc)."""
        # Remove markdown code blocks
        json_str = re.sub(r'```json\s*', '', json_str)
        json_str = re.sub(r'```\s*', '', json_str)
        return json_str.strip()