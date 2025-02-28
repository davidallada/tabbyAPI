from typing import Dict, List
import datetime
import json

class FunctionCallingBaseClass:
    """
    The base class for defining the custom tool/function handling
    so that you can use the native function calling format for a given model.
    The minimum required implementation is:
    1. unique_identifier class variable: This should be a unique string
    2. convert_tool_call_str: A function to convert the
        string output of the function call to an OpenAI compliant list of dicts.
    """
    _registry: dict = {}

    # Called when a subclass of Vehicle is created
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def convert_tool_call_str(cls, tool_call_input: str) -> List[Dict]:
        """
        Take the raw string representing the tool/function call and convert it to an
        OpenAI compatible format
        Args:
            tool_call_input (str): The raw string representing the tool call input
        Returns:
            List[Dict]: A list of OpenAI Tool Call compliant dictionaries.
        """
        tool_calls = json.loads(tool_call_input)
        for tool_call in tool_calls:
            tool_call["function"]["arguments"] = json.dumps(
                tool_call["function"]["arguments"]
            )

        return tool_calls

    @classmethod
    def get_timestamp(cls):
        return datetime.datetime.now().timestamp()