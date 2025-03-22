import json
import re
from typing import Any, Dict, List
from endpoints.OAI.function_handling.default_function_handler import FunctionCallingBaseClass
from loguru import logger

from endpoints.OAI.types.tools import ToolCall


class WattFunctionHandler(FunctionCallingBaseClass):
    @classmethod
    def format_template_vars(cls, data: Any):
        data.template_vars.update(
            {
                "add_generation_prompt": data.add_generation_prompt,
                "tools_json": json.dumps(data.model_dump()["tools"], indent=2),
                "functions_json": json.dumps(data.functions, indent=2),
                "skip_bos_token": data.template_vars.get("skip_bos_token", data.skip_bos_token),
                "tool_class_name": data.template_vars.get("tool_class_name", data.tool_class_name),
                # Qwen 2.5 Expects "tools" to be a list of tool dicts that it will call | tojson on each dict
                "custom_tools": [t.model_dump() for t in data.tools],
                "tools_in_user_message": False
            }
        )

    @classmethod
    def postprocess_tool_call(cls, call_str: str) -> List[ToolCall]:
        tool_calls = json.loads(call_str)
        # logger.debug(f"_LOG: in postprocess_tool_call: {call_str=}")  # DEBUG:REMOVE
        if isinstance(tool_calls, list):
            tool_call_list = []
            for tool_call in tool_calls:
                tool_call_list.append(
                    ToolCall(
                        **{
                            "function": tool_call,
                            "id": f"{tool_call['name']}_{cls.get_timestamp()}",
                            "type": "function"
                        }
                    )
                )
            return tool_call_list
        else:
            return [
                ToolCall(
                    **{
                        "function": tool_calls,
                        "id": f"{tool_calls['name']}_{cls.get_timestamp()}",
                        "type": "function"
                    }
                )
            ]
        
