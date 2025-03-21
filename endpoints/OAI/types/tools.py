from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Literal, Optional
from pydantic.json_schema import SkipJsonSchema
from loguru import logger

import json
# tool_call_schema = {
#     "$schema": "http://json-schema.org/draft-07/schema#",
#     "type": "array",
#     "items": {
#         "type": "object",
#         "properties": {
#             "id": {"type": "string"},
#             "function": {
#                 "type": "object",
#                 "properties": {
#                     "name": {"type": "string"},
#                     "arguments": {
#                         # Converted to OAI's string in post process
#                         "type": "object"
#                     },
#                 },
#                 "required": ["name", "arguments"],
#             },
#             "type": {"type": "string", "enum": ["function"]},
#         },
#         "required": ["id", "function", "type"],
#     },
# }

tool_call_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {
                # Converted to OAI's string in post process
                "type": "object"
            },
        },
        "required": ["name", "arguments"],
    },
}



def generate_json_schema(tools):
    # Extract function names for the 'enum' of 'name'
    function_names = [tool['function']['name'] for tool in tools]

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": function_names  # Set 'enum' to the list of function names
                },
                "arguments": {
                    "type": "object"
                }
            },
            "required": ["name", "arguments"],
            "allOf": []
        }
    }

    # Add conditional schemas
    for tool in tools:
        function_name = tool['function']['name']
        function_schema = tool['function']['parameters']

        # Modify schema directly in the 'then' block
        schema['items']['allOf'].append({
            "if": {
                "properties": {
                    "name": {"const": function_name}
                }
            },
            "then": {
                "properties": {
                    "arguments": function_schema  # Directly set the schema instead of referencing it
                }
            }
        })

    return schema


class Function(BaseModel):
    """Represents a description of a tool function."""

    name: str
    description: str
    parameters: Dict[str, object]


class ToolSpec(BaseModel):
    """Wrapper for an inner tool function."""

    function: Function
    type: Literal["function"]

class Tool(BaseModel):
    """Represents an OAI tool description."""

    name: str

    # Makes more sense to be a dict, but OAI knows best
    arguments: str

    # We can also store the full dictionary version here
    arguments_dict: SkipJsonSchema[Optional[Dict[Any, Any]]] = None

    @model_validator(mode="before")
    @classmethod
    def sync_arguments(cls, values):
        """
        Ensure `arguments` and `arguments_dict` are in sync.
        We prioritize arguments_dict over arguments, since its easier to
        work with in Python. So when we load from an OAI request, arguments is set, and we set arguments_dict to json.loads of arguments, and visa versa.
        """
        arguments = values.get("arguments")
        if arguments is not None:
            if isinstance(arguments, str):
                # If arguments is provided, deserialize it to arguments_dict
                try:
                    values["arguments_dict"] = json.loads(arguments)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON string in `arguments` field")
            elif isinstance(arguments, dict):
                values["arguments_dict"] = arguments
                values["arguments"] = json.dumps(arguments)
            else:
                raise Exception(f"Tool.arguments encountered unexpected type, expected str or dict, but got: {type(arguments)}")

        return values

    class Config:
        json_schema_extra = {"exclude": {"arguments_dict"}}  # Optional schema exclusion

class ToolCall(BaseModel):
    """Represents an OAI tool description."""

    id: str
    function: Tool
    type: Literal["function"]
