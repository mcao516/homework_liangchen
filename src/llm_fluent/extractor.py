"""SchemaExtractor methods with field matching validation."""

import json
import logging
from dataclasses import fields, is_dataclass
from typing import Any, Dict, Union, Type, TypeVar, Set

logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = None
    ValidationError = None

T = TypeVar('T')


class FieldMismatchError(Exception):
    """Exception raised when fields completely mismatch between data and schema."""
    pass


class SchemaExtractor:
    """Utility class for extracting structured data from text."""
    
    @staticmethod
    def calculate_field_match(
        data_fields: Set[str],
        schema_fields: Set[str]
    ) -> Dict[str, Any]:
        """Calculate field matching statistics.
        
        Args:
            data_fields: Fields present in the data
            schema_fields: Fields expected by the schema
            
        Returns:
            Dictionary with matching statistics
        """
        matched_fields = data_fields & schema_fields
        extra_fields = data_fields - schema_fields
        missing_fields = schema_fields - data_fields
        
        total_unique_fields = len(data_fields | schema_fields)
        match_percentage = (len(matched_fields) / total_unique_fields * 100) if total_unique_fields > 0 else 0
        
        # Calculate coverage (how many schema fields are present in data)
        coverage_percentage = (len(matched_fields) / len(schema_fields) * 100) if schema_fields else 100
        
        return {
            'matched_fields': matched_fields,
            'extra_fields': extra_fields,
            'missing_fields': missing_fields,
            'match_percentage': match_percentage,
            'coverage_percentage': coverage_percentage,
            'total_matched': len(matched_fields),
            'total_extra': len(extra_fields),
            'total_missing': len(missing_fields)
        }
    
    @staticmethod
    def parse_to_dataclass(
        data: Dict[str, Any], 
        cls: Type[T],
        strict: bool = False,
        min_match_percentage: float = 0.0
    ) -> T:
        """Parse dictionary to dataclass instance with field validation.
        
        Args:
            data: Dictionary containing the data
            cls: Target dataclass type
            strict: If True, raise error on any extra fields
            min_match_percentage: Minimum required field match percentage (0-100)
            
        Returns:
            Instance of the dataclass
            
        Raises:
            ValueError: If cls is not a dataclass
            FieldMismatchError: If fields completely mismatch or below threshold
        """
        if not is_dataclass(cls):
            raise ValueError(f"{cls} is not a dataclass")
        
        # Get field names from dataclass
        dataclass_fields = fields(cls)
        schema_field_names = {f.name for f in dataclass_fields}
        
        # Get field names from data
        data_field_names = set(data.keys())
        
        # Calculate matching statistics
        match_stats = SchemaExtractor.calculate_field_match(
            data_field_names, 
            schema_field_names
        )
        
        # Check for complete mismatch
        if match_stats['total_matched'] == 0 and len(schema_field_names) > 0:
            error_msg = (
                f"Complete field mismatch for dataclass '{cls.__name__}'. "
                f"Expected fields: {schema_field_names}, "
                f"Got fields: {data_field_names}"
            )
            logger.error(error_msg)
            raise FieldMismatchError(error_msg)
        
        # Check minimum match percentage
        if match_stats['match_percentage'] < min_match_percentage:
            error_msg = (
                f"Field match percentage {match_stats['match_percentage']:.1f}% "
                f"is below minimum threshold {min_match_percentage}% for dataclass '{cls.__name__}'"
            )
            logger.error(error_msg)
            raise FieldMismatchError(error_msg)
        
        # Check strict mode
        if strict and match_stats['extra_fields']:
            error_msg = (
                f"Strict mode: Extra fields not allowed for dataclass '{cls.__name__}'. "
                f"Extra fields: {match_stats['extra_fields']}"
            )
            logger.error(error_msg)
            raise FieldMismatchError(error_msg)
        
        # Filter data to only include matching fields
        filtered_data = {k: v for k, v in data.items() if k in schema_field_names}
        
        try:
            instance = cls(**filtered_data)
            logger.info(f"  ✓ Successfully created {cls.__name__} instance")
            return instance
        except TypeError as e:
            # This happens when required fields are missing
            error_msg = f"Failed to create dataclass '{cls.__name__}': {str(e)}"
            logger.error(error_msg)
            raise FieldMismatchError(error_msg) from e
    
    @staticmethod
    def parse_to_pydantic(
        data: Dict[str, Any], 
        cls: Type[BaseModel],
        strict: bool = False,
        min_match_percentage: float = 0.0
    ) -> BaseModel:
        """Parse dictionary to Pydantic model instance with field validation.
        
        Args:
            data: Dictionary containing the data
            cls: Target Pydantic model type
            strict: If True, raise error on any extra fields
            min_match_percentage: Minimum required field match percentage (0-100)
            
        Returns:
            Instance of the Pydantic model
            
        Raises:
            ValueError: If cls is not a Pydantic model or pydantic not installed
            FieldMismatchError: If fields completely mismatch or below threshold
            ValidationError: If Pydantic validation fails
        """
        if not HAS_PYDANTIC:
            raise ValueError("Pydantic is not installed")
        
        if not issubclass(cls, BaseModel):
            raise ValueError(f"{cls} is not a Pydantic model")
        
        # Get field names from Pydantic model
        schema_field_names = set(cls.model_fields.keys())
        
        # Get field names from data
        data_field_names = set(data.keys())
        
        # Calculate matching statistics
        match_stats = SchemaExtractor.calculate_field_match(
            data_field_names,
            schema_field_names
        )
        
        # Check for complete mismatch
        if match_stats['total_matched'] == 0 and len(schema_field_names) > 0:
            error_msg = (
                f"Complete field mismatch for Pydantic model '{cls.__name__}'. "
                f"Expected fields: {schema_field_names}, "
                f"Got fields: {data_field_names}"
            )
            logger.error(error_msg)
            raise FieldMismatchError(error_msg)
        
        # Check minimum match percentage
        if match_stats['match_percentage'] < min_match_percentage:
            error_msg = (
                f"Field match percentage {match_stats['match_percentage']:.1f}% "
                f"is below minimum threshold {min_match_percentage}% for Pydantic model '{cls.__name__}'"
            )
            logger.error(error_msg)
            raise FieldMismatchError(error_msg)
        
        # Check strict mode
        if strict and match_stats['extra_fields']:
            error_msg = (
                f"Strict mode: Extra fields not allowed for Pydantic model '{cls.__name__}'. "
                f"Extra fields: {match_stats['extra_fields']}"
            )
            logger.error(error_msg)
            raise FieldMismatchError(error_msg)
        
        try:
            # Pydantic will handle validation and missing fields with defaults
            instance = cls(**data)
            logger.info(f"  ✓ Successfully created {cls.__name__} instance")
            return instance
        except ValidationError as e:
            logger.error(f"Pydantic validation failed for '{cls.__name__}': {str(e)}")
            raise
    
    @staticmethod
    def extract_all_json_from_text(text: str) -> list[Dict[str, Any]]:
        """Extract ALL JSON objects/arrays from text."""
        import re
        results = []
        
        # Try direct JSON parsing first
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
        
        if results:
            return results
        
        # Try to find all JSON objects in text
        def find_json_objects(text):
            objects = []
            i = 0
            while i < len(text):
                if text[i] in '{[':
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