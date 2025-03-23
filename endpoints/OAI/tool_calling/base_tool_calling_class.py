from typing import Any, Dict, List
import datetime
import json
from loguru import logger

from endpoints.OAI.types.tools import ToolCall

class ToolCallingBaseClass:
    """
    The base class for defining the custom tool handling
    so that you can use the native tool calling format for a given model.
    The minimum required implementation is:
    1. unique_identifier class variable: This should be a unique string
    2. convert_tool_call_str: A tool to convert the
        string output of the tool call to an OpenAI compliant list of dicts.
    """
    _registry: dict = {}

    # Called when a subclass of Vehicle is created
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        ToolCallingBaseClass._registry[cls.__name__] = cls
    
    @classmethod
    def format_template_vars(cls, data: Any):
        """
        Format the variables that will be passed into the Jinja propmt template.
        This allows for a wide variety of chat templates and prevents us from having
        to fully customize each one, rather we can easily swap out var names or formats
        in a single spot.
        """
        data.template_vars.update(
            {
                "add_generation_prompt": data.add_generation_prompt,
                "tools_json": json.dumps(data.model_dump()["tools"], indent=2),
                "functions_json": json.dumps(data.functions, indent=2),
                "skip_bos_token": data.template_vars.get(
                    "skip_bos_token",
                    data.skip_bos_token
                ),
                "tabby_tool_class_name": data.template_vars.get(
                    "tabby_tool_class_name",
                    data.tabby_tool_class_name
                ),
            }
        )

    @classmethod
    def message_tool_calls_to_json(cls, message: Any) -> str:
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
    def tool_call_dict_to_class(cls, tool_call_dict: dict):
        return ToolCall(
            **{
                "function": tool_call_dict,
                "id": f"{tool_call_dict['name']}_{cls.get_timestamp()}",
                "type": "function"
            }
        )

    @classmethod
    def postprocess_tool_call(cls, call_str: str) -> List[ToolCall]:
        tool_calls = json.loads(call_str)

        # We want a list of dicts to turn into tool calls
        # If we just return a single dict, make it a list
        if not isinstance(tool_calls, (list, dict)):
            raise Exception(f"Expected Tool call string to be a JSON list of dicts or a dict: {call_str}")

        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]

        return [
            cls.tool_call_dict_to_class(tool_call_dict) for tool_call_dict in tool_calls
        ]

    @classmethod
    def get_timestamp(cls):
        return int(datetime.datetime.now(datetime.UTC).timestamp() * 1000)
    
DEFAULT_TOOL_HANDLER_CLASS = ToolCallingBaseClass

def get_tool_calling_class(tabby_tool_class_name: str):
    if not tabby_tool_class_name:
        return DEFAULT_TOOL_HANDLER_CLASS
    logger.info(f"FUNC_CALLING_CLASS: {ToolCallingBaseClass._registry}")
    if tabby_tool_class_name == DEFAULT_TOOL_HANDLER_CLASS.__name__:
        return DEFAULT_TOOL_HANDLER_CLASS
    return ToolCallingBaseClass._registry[tabby_tool_class_name]

from endpoints.OAI.tool_calling.custom_tool_calling_classes import *