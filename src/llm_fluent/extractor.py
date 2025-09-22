"""Define a utility class for extracting structured data from text."""
import json
from dataclasses import fields, is_dataclass
from typing import (
    Any, Type, TypeVar, Union, Dict, List
)
try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None
    ValidationError = None


T = TypeVar("T")


class SchemaExtractor:
    """Utility class for extracting structured data from text."""
    
    @staticmethod
    def extract_all_json_from_text(text: str) -> List[Dict[str, Any]]:
        """Extract ALL JSON objects/arrays from text.
        
        Returns a list of extracted JSON objects. If the extracted item is
        an array, it unpacks the array elements as individual items.
        """
        import re
        results = []
        
        # Try direct JSON parsing first (might be a single object or array)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                results.extend(parsed)
            else:
                results.append(parsed)
            return results
        except json.JSONDecodeError:
            pass
        
        # Find JSON in markdown code blocks
        json_pattern = r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    results.extend(parsed)
                else:
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
        
        # If we found JSON in code blocks, return those
        if results:
            return results
        
        # Try to find all JSON objects in text
        # More sophisticated regex to match balanced braces
        def find_json_objects(text):
            objects = []
            i = 0
            while i < len(text):
                if text[i] in '{[':
                    # Try to parse from this position
                    bracket_count = 0
                    start = i
                    in_string = False
                    escape_next = False
                    
                    while i < len(text):
                        if escape_next:
                            escape_next = False
                        elif text[i] == '\\':
                            escape_next = True
                        elif text[i] == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if text[i] in '{[':
                                bracket_count += 1
                            elif text[i] in '}]':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    # Found complete JSON object/array
                                    try:
                                        parsed = json.loads(text[start:i+1])
                                        if isinstance(parsed, list):
                                            objects.extend(parsed)
                                        else:
                                            objects.append(parsed)
                                    except json.JSONDecodeError:
                                        pass
                                    break
                        i += 1
                i += 1
            return objects
        
        results = find_json_objects(text)
        return results if results else []
    
    @staticmethod
    def parse_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
        """Parse dictionary to dataclass instance."""
        if not is_dataclass(cls):
            raise ValueError(f"{cls} is not a dataclass")
        
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)
    
    @staticmethod
    def parse_to_pydantic(data: Dict[str, Any], cls: Type[BaseModel]) -> BaseModel:
        """Parse dictionary to Pydantic model instance."""
        if not HAS_PYDANTIC or not issubclass(cls, BaseModel):
            raise ValueError(f"{cls} is not a Pydantic model")
        return cls(**data)
    
    @staticmethod
    def create_extraction_prompt(
        original_prompt: str,
        schema: Union[Type, Dict[str, Any]]
    ) -> str:
        """Create a prompt that includes extraction instructions."""
        if isinstance(schema, dict):
            schema_str = json.dumps(schema, indent=2)
        elif is_dataclass(schema):
            schema_dict = {
                f.name: str(f.type.__name__ if hasattr(f.type, '__name__') else f.type)
                for f in fields(schema)
            }
            schema_str = json.dumps(schema_dict, indent=2)
        elif HAS_PYDANTIC and issubclass(schema, BaseModel):
            schema_str = schema.model_json_schema()
            schema_str = json.dumps(schema_str, indent=2)
        else:
            schema_str = str(schema)
        
        return f"""{original_prompt}

Please provide your response as a JSON array containing ALL matching items. 
Each item should match this schema:
{schema_str}

Return ONLY the JSON array, no additional text. Example format:
[
  {{"field1": "value1", "field2": value2}},
  {{"field1": "value3", "field2": value4}}
]"""