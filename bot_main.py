#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys
from contextlib import asynccontextmanager
import asyncio

from fastapi import WebSocket
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner, PipelineTask
from pipecat.pipeline.task import PipelineParams
from pipecat_flows import FlowManager

from bot.config import BotConfig, flow_config
from bot.flow_handlers import (
    store_name,
    choose_time_handler,
    choose_vin_handler,
    collect_vin_handler,
    store_vin_in_state_callback,
    confirm_vin_handler,
    reject_vin_handler
)
from bot.mcp import register_mcp_clients
from bot.services import (
    create_llm,
    create_tts,
    create_stt,
    create_transport,
    create_audio_buffer_processor,
    create_stt_mute_filter,
    create_pipeline_context,
    create_context_aggregator,
    build_pipeline,
)
from bot.event_handlers import (
    on_client_connected_handler,
    on_client_disconnected_handler,
    on_audio_data_handler,
)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Mapping of placeholder strings to actual handler functions
PYTHON_HANDLERS_MAP = {
    "store_name_handler_placeholder": store_name,
    "choose_time_handler_placeholder": choose_time_handler,
    "choose_vin_handler_placeholder": choose_vin_handler,
    "collect_vin_handler_placeholder": collect_vin_handler,
    "store_vin_in_state_callback_placeholder": store_vin_in_state_callback,
    "confirm_vin_handler_placeholder": confirm_vin_handler,
    "reject_vin_handler_placeholder": reject_vin_handler,
}

def wire_python_handlers(config_to_wire):
    """Iterates through the flow_config and replaces string placeholders with actual functions."""
    for node_id, node_cfg in config_to_wire.get("nodes", {}).items():
        if "functions" in node_cfg:
            for fn_def_item in node_cfg["functions"]:
                # The function definition might be directly under 'function' or nested
                # This check ensures we are working with the actual function definition dictionary
                actual_fn_def = fn_def_item.get("function") if isinstance(fn_def_item.get("function"), dict) else fn_def_item

                if isinstance(actual_fn_def, dict):
                    handler_placeholder = actual_fn_def.get("handler")
                    if isinstance(handler_placeholder, str) and handler_placeholder in PYTHON_HANDLERS_MAP:
                        actual_fn_def["handler"] = PYTHON_HANDLERS_MAP[handler_placeholder]
                        logger.debug(f"Wired Python handler for '{actual_fn_def.get('name')}' in node '{node_id}'.")
                    
                    callback_placeholder = actual_fn_def.get("transition_callback")
                    if isinstance(callback_placeholder, str) and callback_placeholder in PYTHON_HANDLERS_MAP:
                        actual_fn_def["transition_callback"] = PYTHON_HANDLERS_MAP[callback_placeholder]
                        logger.debug(f"Wired Python transition_callback for '{actual_fn_def.get('name')}' in node '{node_id}'.")
    return config_to_wire

@asynccontextmanager
async def pipeline_runner_context(pipeline: Pipeline, task_params: PipelineParams):
    """Async context manager for PipelineRunner."""
    task = PipelineTask(pipeline, params=task_params)
    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    try:
        yield task 
        await runner.run(task)
    finally:
        logger.info("Pipeline stopping. Cancelling task...")
        await task.cancel()
        logger.info("Pipeline task cancelled.")

async def run_bot(websocket_client: WebSocket, stream_sid: str, testing: bool):
    try:
        cfg = BotConfig() 
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    # 1. Wire Python handlers into the imported flow_config
    # This operates on the flow_config imported from bot.config
    wired_flow_config = wire_python_handlers(flow_config) # flow_config is now modified in-place
    logger.info("Python handlers wired into flow_config.")

    # 2. Create services (LLM is needed for MCP registration and adapter)
    transport = create_transport(websocket_client, stream_sid, cfg)
    llm = create_llm(cfg)
    tts = create_tts(cfg)
    stt = create_stt(cfg)
    audio_buffer_processor = create_audio_buffer_processor(testing, cfg)
    stt_mute_filter = create_stt_mute_filter()

    # 3. Register MCP tools and then run the MCP adapter logic
    # This must happen AFTER Python handlers are wired if both might target the same function name,
    # though typically MCP tools have unique names not used by Python handlers.
    # The adapter logic for MCP tools should run on the `wired_flow_config`.
    try:
        tools_schema = await register_mcp_clients(llm, cfg)
        logger.info(f"Successfully registered MCP clients. Tools: {tools_schema}")

        # Adapt MCP wrappers for Flows - This uses the `wired_flow_config`
        if tools_schema and tools_schema.standard_tools:
            mcp_wrappers = {
                name: entry.handler
                for name, entry in llm._functions.items()
                if name in [tool.name for tool in tools_schema.standard_tools]
            }
            logger.debug(f"Raw MCP wrappers identified: {mcp_wrappers.keys()}")

            def make_flow_handler(wrapper_fn, tool_name, llm_service):
                async def handler(args: dict) -> dict: # Using dict for args and result for simplicity here
                    loop = asyncio.get_running_loop()
                    fut = loop.create_future()
                    async def result_cb(result, *, properties=None):
                        if not fut.done():
                            fut.set_result(result)
                        else:
                            logger.warning(f"Future for {tool_name} was already done. MCP result: {result}")
                    
                    call_id = f"flow_{tool_name}_{int(loop.time()*1000)}"
                    try:
                        await wrapper_fn(
                            tool_name, 
                            call_id, 
                            args, 
                            llm_service, 
                            None, 
                            result_cb,
                        )
                    except Exception as e:
                        logger.error(f"Error invoking MCP wrapper for {tool_name}: {e}")
                        if not fut.done():
                            fut.set_exception(e)
                    
                    try:
                        mcp_result = await asyncio.wait_for(fut, timeout=30.0)
                        if isinstance(mcp_result, str):
                             return {"status": "success", "message": mcp_result}
                        elif isinstance(mcp_result, dict):
                             return mcp_result
                        else:
                             logger.warning(f"Unexpected MCP result type for {tool_name}: {type(mcp_result)}. Wrapping.")
                             return {"status": "success", "data": mcp_result}
                    except asyncio.TimeoutError:
                        logger.error(f"Timeout waiting for MCP result for {tool_name}")
                        return {"status": "error", "message": f"Timeout for {tool_name}"}
                    except Exception as e:
                        logger.error(f"Error getting MCP result for {tool_name}: {e}")
                        return {"status": "error", "message": str(e)}
                return handler

            flow_wrappers_for_mcp = {
                name: make_flow_handler(wrapper_fn, name, llm)
                for name, wrapper_fn in mcp_wrappers.items()
            }
            logger.debug(f"Prepared Flow handlers for MCP tools: {flow_wrappers_for_mcp.keys()}")

            # Inject MCP handlers into the `wired_flow_config`
            # This will only add handlers where one isn't already present from Python wiring.
            for node_id, node_cfg in wired_flow_config["nodes"].items(): # Use wired_flow_config
                if "functions" in node_cfg:
                    for fn_def_item in node_cfg["functions"]:
                        actual_fn_def = fn_def_item.get("function") if isinstance(fn_def_item.get("function"), dict) else fn_def_item
                        if isinstance(actual_fn_def, dict) and (tool_name := actual_fn_def.get("name")):
                            # Only inject if there isn't already a Python handler AND it's an MCP tool
                            if tool_name in flow_wrappers_for_mcp and not callable(actual_fn_def.get("handler")):
                                actual_fn_def["handler"] = flow_wrappers_for_mcp[tool_name]
                                logger.info(f"Injected adapted MCP Flow handler for tool '{tool_name}' in node '{node_id}'.")
    except Exception as e:
        logger.error(f"Failed during MCP client registration or handler injection: {e}")
        return
    
    pipeline_context = create_pipeline_context(tools_schema if 'tools_schema' in locals() else None)
    context_aggregator = create_context_aggregator(llm, pipeline_context)

    # 4. Build pipeline
    pipeline = build_pipeline(
        transport=transport,
        stt=stt,
        llm=llm,
        tts=tts,
        audio_buffer_processor=audio_buffer_processor,
        stt_mute_filter=stt_mute_filter,
        context_aggregator=context_aggregator
    )

    # 5. Define PipelineParams
    pipeline_params = PipelineParams(
        audio_in_sample_rate=cfg.audio_in_sample_rate,
        audio_out_sample_rate=cfg.audio_out_sample_rate,
        allow_interruptions=True,
    )
    
    _task_ref_for_flow_manager = None 

    async with pipeline_runner_context(pipeline, pipeline_params) as task_for_handlers:
        _task_ref_for_flow_manager = task_for_handlers
        
        flow_manager = FlowManager(
            task=_task_ref_for_flow_manager,
            llm=llm,
            context_aggregator=context_aggregator,
            flow_config=wired_flow_config # Use the fully wired config
        )
        
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport_param, client_param):
            await on_client_connected_handler(transport_param, client_param, flow_manager, audio_buffer_processor)

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport_param, client_param):
            await on_client_disconnected_handler(transport_param, client_param, task_for_handlers)

        @audio_buffer_processor.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            await on_audio_data_handler(
                buffer, audio, sample_rate, num_channels, websocket_client.client.port
            )
        
        logger.info("Starting pipeline run...")
        # runner.run(task) is handled by the context manager

    logger.info("run_bot completed.")

