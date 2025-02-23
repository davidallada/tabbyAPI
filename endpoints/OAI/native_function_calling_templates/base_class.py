from abc import ABC, abstractmethod
from typing import Dict, List
import datetime

class FunctionCallingHandlerBaseClass(ABC):
    """
    The base class for defining the custom tool/function handling
    so that you can use the native function calling format for a given model.

    The minimum required implementation is:
    1. unique_identifier class variable: This should be a unique string
    2. convert_tool_call_to_openai_tool_list: A function to convert the
        string output of the function call to an OpenAI compliant list of dicts.
    """

    @property
    @abstractmethod
    def unique_identifier(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def convert_tool_call_to_openai_tool_list(cls, tool_call_input: str) -> List[Dict]:
        """
        Take the raw string representing the tool/function call and convert it to an
        OpenAI compatible format

        Args:
            tool_call_input (str): The raw string representing the tool call input

        Returns:
            List[Dict]: A list of OpenAI Tool Call compliant dictionaries.
        """
        raise NotImplementedError("You must implement this class function")

    @classmethod
    def get_timestamp(cls):
        return datetime.datetime.now().timestamp()