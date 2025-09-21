import json
import logging
import pydantic
from dataclasses import is_dataclass
from typing import Any, Type, TypeVar, Dict

logger = logging.getLogger(__name__)

T = TypeVar("T")


def extract_from_text(text: str, schema: Type[T]) -> T:
    """Extract structured data from text based on schema.
    
    Args:
        text: Text to extract from.
        schema: Target schema (dataclass, Pydantic model, or JSON schema dict).
        
    Returns:
        Instance of the schema type with extracted data.
    """
    try:
        # Try to parse as JSON first
        data = None
        
        # Find JSON in the text (looking for {} or [])
        import re
        json_pattern = r'\{[^{}]*\}|\[[^\[\]]*\]'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                data = json.loads(match)
                break
            except json.JSONDecodeError:
                continue
        
        # If no JSON found, try to parse the entire text
        if data is None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract key-value pairs
                data = _extract_key_values(text)
        
        # Convert to target schema
        if is_dataclass(schema):
            return schema(**data) if isinstance(data, dict) else schema()
        elif issubclass(schema, pydantic.BaseModel):
            return schema(**data) if isinstance(data, dict) else schema()
        elif isinstance(schema, dict):  # JSON schema
            return data
        else:
            return schema(data) if data else schema()
            
    except Exception as e:
        logger.warning(f"Extraction failed: {e}, returning empty instance")
        # Return empty instance on failure
        if is_dataclass(schema):
            return schema()
        elif issubclass(schema, pydantic.BaseModel):
            return schema()
        else:
            return {}


def _extract_key_values(text: str) -> Dict[str, Any]:
    """Extract key-value pairs from text."""
    import re
    result = {}
    
    # Look for patterns like "name: John" or "age: 25"
    pattern = r'(\w+):\s*([^\n,]+)'
    matches = re.findall(pattern, text)
    
    for key, value in matches:
        # Try to parse value as number
        value = value.strip()
        try:
            if '.' in value:
                result[key] = float(value)
            else:
                result[key] = int(value)
        except ValueError:
            result[key] = value
    
    return result