"""Chat completion utilities for OAI server."""

import asyncio
import json
import pathlib
from asyncio import CancelledError
from typing import Any, Callable, Dict, List, Optional
from fastapi import HTTPException, Request
from jinja2 import TemplateError
from loguru import logger

from common import model
from common.multimodal import MultimodalEmbeddingWrapper
from common.networking import (
    get_generator_error,
    handle_request_disconnect,
    handle_request_error,
    request_disconnect_loop,
)
from common.utils import unwrap
from endpoints.OAI.function_handling.default_function_handler import DEFAULT_FUNCTION_HANDLER_CLASS, get_function_calling_class
from endpoints.OAI.types.chat_completion import (
    ChatCompletionLogprobs,
    ChatCompletionLogprob,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionRespChoice,
    ChatCompletionStreamChunk,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
)
from endpoints.OAI.types.common import UsageStats
from endpoints.OAI.utils.completion import _stream_collector
from endpoints.OAI.types.tools import ToolCall


def postprocess_tool_call(call_str: str) -> List[ToolCall]:
    tool_calls = json.loads(call_str)
    for tool_call in tool_calls:
        tool_call["function"]["arguments"] = json.dumps(
            tool_call["function"]["arguments"]
        )
    return [ToolCall(**tool_call) for tool_call in tool_calls]


def _create_response(
    request_id: str, generations: List[dict], model_name: Optional[str], postprocess_tool_calls_func: Optional[Callable] = None
):
    """Create a chat completion response from the provided text."""

    if not postprocess_tool_calls_func:
        postprocess_tool_calls_func = postprocess_tool_call

    prompt_tokens = unwrap(generations[-1].get("prompt_tokens"), 0)
    completion_tokens = unwrap(generations[-1].get("generated_tokens"), 0)

    choices = []
    for index, generation in enumerate(generations):
        message = ChatCompletionMessage(
            role="assistant", content=unwrap(generation.get("text"), "")
        )

        tool_calls = generation["tool_calls"]
        if tool_calls:
            message.tool_calls = postprocess_tool_calls_func(tool_calls)
        logger.debug(f"_LOG: create_reponse: {message.tool_calls=}")  # DEBUG:REMOVE

        

        logprob_response = None

        token_probs = unwrap(generation.get("token_probs"), {})
        if token_probs:
            logprobs = unwrap(generation.get("logprobs"), [])

            collected_token_probs = []
            for index, token in enumerate(token_probs.keys()):
                top_logprobs = [
                    ChatCompletionLogprob(token=token, logprob=logprob)
                    for token, logprob in logprobs[index].items()
                ]

                collected_token_probs.append(
                    ChatCompletionLogprob(
                        token=token,
                        logprob=token_probs[token],
                        top_logprobs=top_logprobs,
                    )
                )

            logprob_response = ChatCompletionLogprobs(content=collected_token_probs)

        # Grab the original finish_reason
        finish_reason = generation.get("finish_reason")

        # In the case of tool call use, we mark the finish_reason for the choice as
        # "tool_calls". I check for truthy finish_reason because for some reason I
        # remember something about streaming chunks having no finish reason until
        # the chunk is done or something like that. Feel free to correct me.
        # I am purposly leaving stop_str because I don't know the ramifications
        # of removing, and I don't have time.
        if message.tool_calls and finish_reason:
            finish_reason = "tool_calls"

        choice = ChatCompletionRespChoice(
            index=index,
            finish_reason=finish_reason,
            stop_str=generation.get("stop_str"),
            message=message,
            logprobs=logprob_response,
        )

        choices.append(choice)

    response = ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        choices=choices,
        model=unwrap(model_name, ""),
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


def _create_stream_chunk(
    request_id: str,
    generation: Optional[dict] = None,
    model_name: Optional[str] = None,
    is_usage_chunk: bool = False,
    postprocess_tool_calls_func: Optional[Callable] = None
):
    """Create a chat completion stream chunk from the provided text."""

    if not postprocess_tool_calls_func:
        postprocess_tool_calls_func = postprocess_tool_call
    
    index = generation.get("index")
    choices = []
    usage_stats = None

    if is_usage_chunk:
        prompt_tokens = unwrap(generation.get("prompt_tokens"), 0)
        completion_tokens = unwrap(generation.get("generated_tokens"), 0)

        usage_stats = UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
    elif "finish_reason" in generation:
        # Get the finish reason from the generation
        finish_reason = generation.get("finish_reason")
        choice = ChatCompletionStreamChoice(index=index, finish_reason=finish_reason)

        # lets check if we have tool calls since we are at the end of the generation
        if "tool_calls" in generation:
            tool_calls = generation["tool_calls"]
            message = ChatCompletionMessage(
                tool_calls=postprocess_tool_calls_func(tool_calls)
            )
            choice.delta = message

            # In the case of tool call use, we mark the finish_reason for the choice as
            # "tool_calls". I check for truthy finish_reason because for some reason I
            # remember something about streaming chunks having no finish reason until
            # the chunk is done or something like that. Feel free to correct me.
            if finish_reason:
                choice.finish_reason = "tool_calls"

        choices.append(choice)

    else:
        message = ChatCompletionMessage(
            role="assistant", content=unwrap(generation.get("text"), "")
        )

        logprob_response = None

        token_probs = unwrap(generation.get("token_probs"), {})
        if token_probs:
            logprobs = unwrap(generation.get("logprobs"), {})
            top_logprobs = [
                ChatCompletionLogprob(token=token, logprob=logprob)
                for token, logprob in logprobs.items()
            ]

            generated_token = next(iter(token_probs))
            token_prob_response = ChatCompletionLogprob(
                token=generated_token,
                logprob=token_probs[generated_token],
                top_logprobs=top_logprobs,
            )

            logprob_response = ChatCompletionLogprobs(content=[token_prob_response])

        choice = ChatCompletionStreamChoice(
            index=index,
            delta=message,
            logprobs=logprob_response,
        )

        choices.append(choice)

    chunk = ChatCompletionStreamChunk(
        id=f"chatcmpl-{request_id}",
        choices=choices,
        model=unwrap(model_name, ""),
        usage=usage_stats,
    )

    return chunk

def tool_calls_to_tool_calls_json(message: ChatCompletionMessage) -> str:
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

    return json.dumps(list_of_tool_call_dicts, indent=2)

async def _append_template_metadata(data: ChatCompletionRequest, template_vars: dict):
    """Adding metadata is a one-time process."""

    template_metadata = await model.container.prompt_template.extract_metadata(
        template_vars
    )

    # Stop strings
    if isinstance(data.stop, str):
        data.stop = [data.stop] + template_metadata.stop_strings
    else:
        data.stop += template_metadata.stop_strings

    # Tool call start strings
    if template_metadata.tool_starts:
        if data.tool_call_start is None:
            data.tool_call_start = template_metadata.tool_starts

        # Append to stop strings to halt for a tool call generation
        data.stop.extend(template_metadata.tool_starts)
    
    if template_metadata.tool_class_name:
        data.tool_class_name = template_metadata.tool_class_name

    data.skip_bos_token = template_metadata.skip_bos_token
    
    data.tool_class = template_metadata.tool_class = get_function_calling_class(data.tool_class_name)


async def format_messages_with_template(
    messages: List[ChatCompletionMessage],
    existing_template_vars: Optional[dict] = None,
    add_bos_token: bool = True,
    ban_eos_token: bool = False,
    format_tool_call_for_prompt: Callable = tool_calls_to_tool_calls_json
):
    """Barebones function to format chat completion messages into a prompt."""

    if format_tool_call_for_prompt is None:
        format_tool_call_for_prompt = tool_calls_to_tool_calls_json

    template_vars = unwrap(existing_template_vars, {})
    mm_embeddings = MultimodalEmbeddingWrapper() if model.container.use_vision else None

    for message in messages:
        if isinstance(message.content, list):
            concatenated_content = ""
            for content in message.content:
                if content.type == "text":
                    concatenated_content += content.text
                elif content.type == "image_url" and mm_embeddings:
                    await mm_embeddings.add(content.image_url.url)
                    concatenated_content += mm_embeddings.text_alias[-1]

            # Convert the message content into a concatenated string
            message.content = concatenated_content

        if message.tool_calls:
            message.tool_calls_json = format_tool_call_for_prompt(message)

    special_tokens_dict = model.container.get_special_tokens(
        add_bos_token, ban_eos_token
    )
    template_vars.update({"messages": messages, **special_tokens_dict})

    prompt = await model.container.prompt_template.render(template_vars)
    return prompt, mm_embeddings, template_vars


async def apply_chat_template(
    data: ChatCompletionRequest, tool_precursor: Optional[str] = None
):
    """
    Compile the prompt and get any additional stop strings from the template.
    Template stop strings can be overriden by sampler overrides if force is true.
    """
    try:
        data.template_vars["tool_precursor"] = tool_precursor
        data.add_bos_token = False if data.skip_bos_token else data.add_bos_token

        if not data.tool_class:
            data.tool_class = get_function_calling_class(data.tool_class_name)

        _ = data.tool_class.format_template_vars(data)

        tool_call_format_func = data.tool_class.format_tool_call_for_prompt if data.tool_class else None

        prompt, mm_embeddings, template_vars = await format_messages_with_template(
            data.messages, data.template_vars, data.add_bos_token, data.ban_eos_token, tool_call_format_func
        )

        # Append response prefix if present
        if data.response_prefix:
            if data.add_generation_prompt:
                if not prompt.endswith(data.response_prefix):
                    prompt += data.response_prefix
            else:
                logger.warning(
                    "Could not add response prefix because "
                    "add_generation_prompt is False"
                )

        if data.skip_bos_token:
            template_vars["bos_token"] = ""
    
        # Removes the starting BOS token if present
        # This is to prevent add_bos_token from adding multiple bos tokens
        bos_token = template_vars.get("bos_token")
        if bos_token and prompt.startswith(bos_token):
            prompt = prompt.removeprefix(bos_token)

        # Add template metadata
        await _append_template_metadata(data, template_vars)

        return prompt, mm_embeddings

    except KeyError as exc:
        error_message = handle_request_error(
            "Could not find a Conversation from prompt template "
            f"'{model.container.prompt_template.name}'. "
            "Check your spelling?",
        ).error.message

        raise HTTPException(400, error_message) from exc
    except TemplateError as exc:
        error_message = handle_request_error(f"TemplateError: {str(exc)}").error.message

        raise HTTPException(400, error_message) from exc


async def stream_generate_chat_completion(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
):
    """Generator for the generation process."""
    abort_event = asyncio.Event()
    gen_queue = asyncio.Queue()
    gen_tasks: List[asyncio.Task] = []
    disconnect_task = asyncio.create_task(request_disconnect_loop(request))

    try:
        logger.info(f"Received chat completion streaming request {request.state.id}")

        for n in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)

            gen_task = asyncio.create_task(
                _stream_collector(
                    n,
                    gen_queue,
                    prompt,
                    request.state.id,
                    abort_event,
                    embeddings=embeddings,
                    **task_gen_params.model_dump(exclude={"prompt"}),
                )
            )

            gen_tasks.append(gen_task)

        # We need to keep track of the text generated so we can resume the tool calls
        current_generation_text = ""

        # Consumer loop
        while True:
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect(
                    f"Chat completion generation {request.state.id} cancelled by user."
                )

            generation = await gen_queue.get()
            # lets only append the text if we need it for tool calls later
            if data.tool_call_start and "text" in generation:
                current_generation_text += generation["text"]

            # check if we are running a tool model, and that we are at stop
            if data.tool_call_start and "stop_str" in generation:
                generations = await generate_tool_calls(
                    data,
                    [generation],
                    request,
                    current_generations=current_generation_text,
                )
                generation = generations[0]  # We only have one generation in this case

            # Stream collector will push an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            response = _create_stream_chunk(
                request.state.id, generation, model_path.name, postprocess_tool_calls_func=data.tool_class.postprocess_tool_call if data.tool_class else None
            )
            yield response.model_dump_json()

            # Check if all tasks are completed
            if all(task.done() for task in gen_tasks) and gen_queue.empty():
                # Send a usage chunk
                if data.stream_options and data.stream_options.include_usage:
                    usage_chunk = _create_stream_chunk(
                        request.state.id,
                        generation,
                        model_path.name,
                        is_usage_chunk=True,
                        postprocess_tool_calls_func=data.tool_class.postprocess_tool_call if data.tool_class else None
                    )
                    yield usage_chunk.model_dump_json()

                logger.info(
                    f"Finished chat completion streaming request {request.state.id}"
                )

                yield "[DONE]"
                break
    except CancelledError:
        # Get out if the request gets disconnected

        if not disconnect_task.done():
            abort_event.set()
            handle_request_disconnect("Chat completion generation cancelled by user.")
    except Exception:
        yield get_generator_error(
            "Chat completion aborted. Please check the server console."
        )


async def generate_chat_completion(
    prompt: str,
    embeddings: MultimodalEmbeddingWrapper,
    data: ChatCompletionRequest,
    request: Request,
    model_path: pathlib.Path,
):
    gen_tasks: List[asyncio.Task] = []

    try:
        for _ in range(0, data.n):
            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        prompt,
                        request.state.id,
                        embeddings=embeddings,
                        **data.model_dump(exclude={"prompt"}),
                    )
                )
            )

        generations = await asyncio.gather(*gen_tasks)

        logger.debug(f"_LOG: generate_chat_completion__generations: {generations}")  # DEBUG:REMOVE
        # Let's not waste our time if we arn't running a tool model
        if data.tool_call_start:
            logger.debug(f"_LOG: In data.tool_call_start")  # DEBUG:REMOVE
            generations = await generate_tool_calls(data, generations, request)
            logger.debug(f"generate_chat_completion__tool__generations: {generations}")  # DEBUG:REMOVE

        response = _create_response(request.state.id, generations, model_path.name, data.tool_class.postprocess_tool_call if data.tool_class else None)

        logger.info(f"Finished chat completion request {request.state.id}")

        return response
    except Exception as exc:
        error_message = handle_request_error(
            f"Chat completion {request.state.id} aborted. "
            "Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc


async def generate_tool_calls(
    data: ChatCompletionRequest,
    generations: List[str],
    request: Request,
    current_generations: str = None,
):
    gen_tasks: List[asyncio.Task] = []
    tool_idx: List[int] = []

    # Copy to make sure the parent JSON schema doesn't get modified
    # FIXME: May not be necessary depending on how the codebase evolves
    tool_data = data.model_copy(deep=True)
    tool_data.json_schema = tool_data.tool_call_schema
    gen_params = tool_data.model_dump()

    logger.debug(f"_LOG: generate_tool_calls:\n{tool_data=}\n\n{tool_data.json_schema=}\n\n{gen_params=}\n\n")  # DEBUG:REMOVE
    logger.debug(f"_LOG: generate_tool_calls:\n{generations=}\n\n{current_generations=}\n\n{tool_idx=}\n\n")  # DEBUG:REMOVE
    for idx, gen in enumerate(generations):
        logger.debug(f"_LOG: {idx=}\n\n{gen=}\n\n")
        if gen["stop_str"] in tool_data.tool_call_start:
            logger.debug(f"_LOG: generate_tool_calls__gen[stop_str] in tool_data.tool_call_start: {gen=}")  # DEBUG:REMOVE
            if "text" in gen:
                # non streaming, all generations will have the text they generated
                pre_tool_prompt, mm_embeddings = await apply_chat_template(
                    data, gen["text"]
                )
                logger.debug(f"_LOG: generate_tool_calls__text in gen: {pre_tool_prompt=}")  # DEBUG:REMOVE
            elif current_generations is not None:
                # streaming, we wont have text in the generation,
                # we'll have to use the current_generations
                pre_tool_prompt, mm_embeddings = await apply_chat_template(
                    data, current_generations
                )
                logger.debug(f"_LOG: generate_tool_calls__text in gen, current_gen is not nonbe: {pre_tool_prompt=}")  # DEBUG:REMOVE

            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        pre_tool_prompt,
                        request.state.id,
                        embeddings=mm_embeddings,
                        **gen_params,
                    )
                )
            )
            tool_idx.append(idx)

    tool_calls = await asyncio.gather(*gen_tasks)
    logger.debug(f"_LOG: gather_too_calls: {tool_calls=}")  # DEBUG:REMOVE
    for outer_idx in range(0, len(tool_idx)):
        gen_idx = tool_idx[outer_idx]
        generations[gen_idx]["tool_calls"] = tool_calls[outer_idx]["text"]
        logger.debug(f"_LOG: for_loop_gen_idx: {tool_idx=}\n\n{gen_idx=}\n\n{generations[gen_idx]["tool_calls"]=}\n\n")  # DEBUG:REMOVE

    logger.debug(f"_LOG: final_generations_tool_call: {generations=}")  # DEBUG:REMOVE
    return generations
