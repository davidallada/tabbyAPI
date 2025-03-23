import json
import re
from typing import Any, Dict, List
from endpoints.OAI.tool_calling.base_tool_calling_class import ToolCallingBaseClass
from loguru import logger

from endpoints.OAI.types.tools import ToolCall


class Qwen2_5_FunctionHandler(ToolCallingBaseClass):
    @classmethod
    def format_template_vars(cls, data: Any):
        data.template_vars.update(
            {
                "add_generation_prompt": data.add_generation_prompt,
                "tools_json": json.dumps(data.model_dump()["tools"], indent=2),
                "functions_json": json.dumps(data.functions, indent=2),
                "skip_bos_token": data.template_vars.get("skip_bos_token", data.skip_bos_token),
                "tabby_tool_class_name": data.template_vars.get("tabby_tool_class_name", data.tabby_tool_class_name),
                # Qwen 2.5 Expects "tools" to be a list of tool dicts that it will call | tojson on each dict
                "tools": [t.model_dump() for t in data.tools],
            }
        )
        