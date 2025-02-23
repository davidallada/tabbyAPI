# builtin imports
import json
import re

from loguru import logger
from typing import Dict, List

from endpoints.OAI.native_function_calling_templates.base_class import (
    FunctionCallingHandlerBaseClass
)


class LLama3_1_FunctionCallingTemplate(FunctionCallingHandlerBaseClass):
    """
    Citation: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/

    LLama3.1 formats function calls between <function=....</function>
    The format is:
    <function=example_function_name>{"example_name": "example_value"}</function>

    So we have to clean this up to make it OpenAI compatible.
    """

    unique_identifier = "llama_3_1_native_func_calling_template"

    @classmethod
    def convert_tool_call_to_openai_tool_list(cls, tool_call_input: str) -> List[Dict]:
        """
        Take the raw string representing the tool/function call and convert it to an
        OpenAI compatible format

        Args:
            tool_call_input (str): The raw string representing the tool call input

        Returns:
            List[Dict]: A list of OpenAI Tool Call compliant dictionaries.

        <function=example_function_name>{"example_name": "example_value"}</function>
        """
        # For LLama3.1 (and 3.2 and 3.3) we need to basically just use regex to parse
        # out the function that should be called and the params
        # NOTE: The TOOL_START and TOOL_END are NOT included in tool_call_input

        # TODO: Remove before PR
        logger.info(tool_call_input)
        if not tool_call_input or not isinstance(tool_call_input, str):
            return []

        # At the moment, LLama3.1 only supports 1 tool/function call at a time :(
        regex = r"^(?P<TOOL_START><function=)?(?P<function_name>[a-z_][a-z0-9_]*)\>(?P<arguments>(.*))(?P<TOOL_END></function>)?$"

        if re_match := re.match(regex, tool_call_input):
            func_name = re_match.group("function_name")
            arguments_str = re_match.group("arguments")

            assert func_name is not None and arguments_str is not None, f"Unable to extract func_name and arguments_str from {tool_call_input}"
            tool_call_list = [
                {
                    "id": f"{func_name}_{cls.get_timestamp()}",
                    "function": {
                        "name": func_name,
                        "arguments": json.loads(arguments_str)
                    },
                    "type": "function"
                }
            ]
            logger.info(f"Tool Call List: {tool_call_list}")