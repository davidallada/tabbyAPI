import json
import re
from typing import Dict, List
from endpoints.OAI.tool_calling.base_tool_calling_class import ToolCallingBaseClass
from endpoints.OAI.types.tools import ToolCall


class Llama3_1_ToolCalling(ToolCallingBaseClass):
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
        
        tool_call_list = []

        # At the moment, LLama3.1 only supports 1 tool/function call at a time :(
        regex = r"^(?P<TOOL_START><function=)?(?P<function_name>[a-z_][a-z0-9_]*)\>(?P<arguments>(.*))(?P<TOOL_END></function>)?$"

        if re_match := re.match(regex, tool_call_input.strip()):
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
        else:
            # We can default back to our original function parsing of a list of dicts
            tool_call_list = cls.convert_tool_call_str(tool_call_input)

        return tool_call_list