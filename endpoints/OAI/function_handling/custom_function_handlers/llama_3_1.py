import json
import re
from typing import Dict, List
from endpoints.OAI.function_handling.default_function_handler import FunctionCallingBaseClass
from endpoints.OAI.types.tools import ToolCall


class Llama3_1_FunctionHanlder(FunctionCallingBaseClass):
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
        if not tool_call_input or not isinstance(tool_call_input, str):
            return []

        # At the moment, LLama3.1 only supports 1 tool/function call at a time :(
        regex = r"^(?P<TOOL_START><function=)?(?P<function_name>[a-z_][a-z0-9_]*)\>(?P<arguments>(.*))(?P<TOOL_END></function>)?$"

        if re_match := re.match(regex, tool_call_input):
            func_name = re_match.group("function_name")
            arguments_str = re_match.group("arguments")

            assert func_name is not None and arguments_str is not None, f"Unable to extract func_name and arguments_str from {tool_call_input}"
            tool_call_list = [
                ToolCall(**{
                    "id": f"{func_name}_{cls.get_timestamp()}",
                    "function": {
                        "name": func_name,
                        "arguments": json.loads(arguments_str)
                    },
                    "type": "function"
                })
            ]

            return tool_call_list