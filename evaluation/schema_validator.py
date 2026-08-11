"""JSON schema validation for tool calls."""

import json
from typing import Dict, Any, Tuple
from pydantic import BaseModel, ValidationError
from config.schema_definitions import ToolCallJson, DirectAnswerJson, TOOL_REGISTRY


class SchemaValidator:
    """Validate tool call schemas."""

    @staticmethod
    def validate_json_format(text: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Validate if text contains valid JSON.
        
        Returns:
            (is_valid, parsed_dict, error_message)
        """
        try:
            # Try to extract JSON from text
            start = text.find('{')
            end = text.rfind('}') + 1
            
            if start == -1 or end == 0:
                return False, {}, "No JSON object found"
            
            json_str = text[start:end]
            parsed = json.loads(json_str)
            return True, parsed, ""
        except json.JSONDecodeError as e:
            return False, {}, str(e)

    @staticmethod
    def validate_tool_call_schema(data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if data conforms to ToolCallJson schema.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            ToolCallJson(**data)
            return True, ""
        except ValidationError as e:
            return False, str(e)

    @staticmethod
    def validate_direct_answer_schema(data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if data conforms to DirectAnswerJson schema.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            DirectAnswerJson(**data)
            return True, ""
        except ValidationError as e:
            return False, str(e)

    @staticmethod
    def validate_tool_parameters(tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate if parameters match tool schema.
        
        Returns:
            (is_valid, error_message)
        """
        if tool_name not in TOOL_REGISTRY:
            return False, f"Tool '{tool_name}' not found in registry"
        
        tool_schema = TOOL_REGISTRY[tool_name]
        required_params = set(tool_schema.required_params)
        provided_params = set(parameters.keys())
        
        # Check required parameters
        missing = required_params - provided_params
        if missing:
            return False, f"Missing required parameters: {missing}"
        
        # Check parameter types (basic check)
        for param_name, param_value in parameters.items():
            param_schema = next(
                (p for p in tool_schema.parameters if p.name == param_name),
                None
            )
            if param_schema is None:
                return False, f"Unexpected parameter: {param_name}"
        
        return True, ""

    @staticmethod
    def full_validation(text: str) -> Dict[str, Any]:
        """
        Run full validation pipeline.
        
        Returns:
            Validation result dictionary
        """
        result = {
            "raw_text": text,
            "json_valid": False,
            "schema_valid": False,
            "tool_params_valid": False,
            "errors": [],
            "parsed_data": None,
        }

        # Step 1: JSON format validation
        json_valid, parsed_data, json_error = SchemaValidator.validate_json_format(text)
        result["json_valid"] = json_valid
        result["parsed_data"] = parsed_data

        if not json_valid:
            result["errors"].append(f"JSON Format: {json_error}")
            return result

        # Step 2: Schema validation
        if parsed_data.get("should_call_tool"):
            schema_valid, schema_error = SchemaValidator.validate_tool_call_schema(parsed_data)
        else:
            schema_valid, schema_error = SchemaValidator.validate_direct_answer_schema(parsed_data)
        
        result["schema_valid"] = schema_valid
        if not schema_valid:
            result["errors"].append(f"Schema: {schema_error}")
            return result

        # Step 3: Tool parameters validation (if applicable)
        if parsed_data.get("should_call_tool"):
            tool_name = parsed_data.get("tool_name")
            parameters = parsed_data.get("parameters", {})
            params_valid, params_error = SchemaValidator.validate_tool_parameters(tool_name, parameters)
            result["tool_params_valid"] = params_valid
            if not params_valid:
                result["errors"].append(f"Tool Parameters: {params_error}")

        return result