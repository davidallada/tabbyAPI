from endpoints.OAI.native_function_calling_templates.llama_3_1_function_calling_template import (
    LLama3_1_FunctionCallingTemplate
)


FUNCTION_CALLING_HANDLER_CLASSES = {
    LLama3_1_FunctionCallingTemplate.unique_identifier: LLama3_1_FunctionCallingTemplate.convert_tool_call_to_openai_tool_list
}


