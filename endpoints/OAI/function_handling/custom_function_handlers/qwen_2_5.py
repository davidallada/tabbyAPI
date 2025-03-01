import json
import re
from typing import Any, Dict, List
from endpoints.OAI.function_handling.default_function_handler import FunctionCallingBaseClass
from loguru import logger

from endpoints.OAI.types.tools import ToolCall


class Qwen2_5_FunctionHandler(FunctionCallingBaseClass):
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
                "tools": [t.model_dump() for t in data.tools],
            }
        )
        logger.debug(f"Default_Tool: Format Template Vars: {data.template_vars}\n\n{data}\n\n\n")

    @classmethod
    def postprocess_tool_call(cls, call_str: str) -> List[ToolCall]:
        logger.debug(f"Qwen postprocess_tool_call: {call_str}")
        tool_calls = json.loads(call_str)
        tool_call_list = [ToolCall(**tool_call) for tool_call in tool_calls]
        logger.debug(f"Default_Tool: Post Process Tool Call {tool_call_list}")
        return tool_call_list