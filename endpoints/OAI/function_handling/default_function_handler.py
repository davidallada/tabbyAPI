from typing import Any, Dict, List
import datetime
import json
from loguru import logger

from endpoints.OAI.types.tools import ToolCall

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
        FunctionCallingBaseClass._registry[cls.__name__] = cls

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
    def format_template_vars(cls, data: Any):
        data.template_vars.update(
            {
                "add_generation_prompt": data.add_generation_prompt,
                "tools_json": json.dumps(data.model_dump()["tools"], indent=2),
                "functions_json": json.dumps(data.functions, indent=2),
                "skip_bos_token": data.template_vars.get("skip_bos_token", data.skip_bos_token),
                "tool_class_name": data.template_vars.get("tool_class_name", data.tool_class_name),
            }
        )

    @classmethod
    def format_tool_call_for_prompt(cls, message: Any) -> str:
        """
        message.tool_calls is of type List[ToolCall], so we cannot simply json.dumps it.
        Im just going to do this quick and dirty, feel free to improve.
        Args:
            message (ChatCompletionMessage): The chat completion message to convert the tool
                calls to json.
        Returns:
            str: JSON representation of the tool calls
        """
        if not message.tool_calls:
            return ""

        list_of_tool_call_dicts: List[Dict] = []

        for tool_call_obj in message.tool_calls:
            # ToolCall stores arguments as a json dumped string, so we need to json.loads it
            # back to a dict
            func_dict = json.loads(tool_call_obj.model_dump_json())
            func_dict["function"]["arguments"] = json.loads(
                func_dict.get("function", {}
            ).get("arguments", "{}"))
            list_of_tool_call_dicts.append(func_dict)

        json_str = json.dumps(list_of_tool_call_dicts, indent=2)
        return json_str
    
    @classmethod
    def postprocess_tool_call(cls, call_str: str) -> List[ToolCall]:
        tool_calls = json.loads(call_str)
        tool_call_list = [ToolCall(**tool_call) for tool_call in tool_calls]
        return tool_call_list

    @classmethod
    def get_timestamp(cls):
        return datetime.datetime.now().timestamp()
    
DEFAULT_FUNCTION_HANDLER_CLASS = FunctionCallingBaseClass

def get_function_calling_class(tool_class_name: str):
    if not tool_class_name:
        return DEFAULT_FUNCTION_HANDLER_CLASS
    logger.info(f"FUNC_CALLING_CLASS: {FunctionCallingBaseClass._registry}")
    if tool_class_name == DEFAULT_FUNCTION_HANDLER_CLASS.__name__:
        return DEFAULT_FUNCTION_HANDLER_CLASS
    return FunctionCallingBaseClass._registry[tool_class_name]

from endpoints.OAI.function_handling.custom_function_handlers import *
