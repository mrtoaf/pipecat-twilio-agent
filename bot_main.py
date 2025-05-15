#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import sys
from contextlib import asynccontextmanager
import asyncio # Added import

from fastapi import WebSocket
from loguru import logger
import time

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner, PipelineTask
from pipecat.pipeline.task import PipelineParams
from pipecat_flows import FlowManager, FlowArgs, FlowResult

from bot.config import BotConfig, flow_config  # Changed to absolute import
from bot.mcp import register_mcp_clients      # Changed to absolute import
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

# Removed load_dotenv, it's in config.py
# Removed unused imports like os, shutil, datetime, io, wave, aiofiles (moved or not needed here)
# Removed MCP StdioServerParameters, MCPClient - they are now in bot.mcp
# Removed OpenAILLMContext, AudioBufferProcessor etc. direct imports - using factories from bot.services

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# store_name function remains here for now as it's tied to the flow_config structure.
# Ideally, flow handlers would also be part of a separate module or structured differently.
async def store_name(args: FlowArgs) -> FlowResult:
    """Store the user's name in the flow state."""
    return {"status": "success", "name": args["name"]}



@asynccontextmanager
async def pipeline_runner_context(pipeline: Pipeline, task_params: PipelineParams):
    """Async context manager for PipelineRunner."""
    task = PipelineTask(pipeline, params=task_params)
    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    try:
        # Yield the task so it can be used by event handlers if necessary
        # (e.g., for cancellation on disconnect)
        yield task 
        await runner.run(task) # This was missing, runner.run should be awaited
    finally:
        logger.info("Pipeline stopping. Cancelling task...")
        await task.cancel() # Ensure task is cancelled
        logger.info("Pipeline task cancelled.")


async def run_bot(websocket_client: WebSocket, stream_sid: str, testing: bool):
    try:
        cfg = BotConfig()  # Initialize config, loads .env and validates
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    # 0. Update flow_config with the actual handler function
    # This is a bit of a workaround. Ideally, handlers could be specified by name
    # and resolved by the FlowManager, or the flow_config could be built more dynamically.
    flow_config["nodes"]["greeting"]["functions"][0]["function"]["handler"] = store_name

    # 1. Create services
    transport = create_transport(websocket_client, stream_sid, cfg)
    llm = create_llm(cfg)
    tts = create_tts(cfg)
    stt = create_stt(cfg)
    audio_buffer_processor = create_audio_buffer_processor(testing, cfg)
    stt_mute_filter = create_stt_mute_filter()

    # 2. Register MCP tools and create context
    try:
        tools_schema = await register_mcp_clients(llm, cfg)
        logger.info(f"Successfully registered MCP clients. Tools: {tools_schema}")

        # Adapt MCP wrappers for Flows
        if tools_schema and tools_schema.standard_tools:
            # 1. Grab the raw MCP wrapper functions from the LLM registry
            mcp_wrappers = {
                name: entry.handler
                for name, entry in llm._functions.items()
                if name in [tool.name for tool in tools_schema.standard_tools]
            }
            logger.debug(f"Raw MCP wrappers identified: {mcp_wrappers.keys()}")

            # 2. Adapter: turn the 6-arg MCP wrapper into a 1-arg Flows handler
            def make_flow_handler(wrapper_fn, tool_name, llm_service):
                async def handler(args: FlowArgs) -> FlowResult: # Added type hints
                    loop = asyncio.get_running_loop()
                    fut = loop.create_future()
                    async def result_cb(result, *, properties=None):
                        # Deliver the MCP answer into our Future
                        if not fut.done(): # Ensure future is not already set
                            fut.set_result(result)
                        else:
                            logger.warning(f"Future for {tool_name} was already done. MCP result: {result}")

                    # You can synthesize any string as call_id
                    call_id = f"flow_{tool_name}_{int(loop.time()*1000)}"

                    # Invoke the real MCP wrapper with all six parameters
                    try:
                        await wrapper_fn(
                            tool_name,    # function_name
                            call_id,      # tool_call_id
                            args,         # arguments: FlowArgs is a dict-like object
                            llm_service,  # the OpenAILLMService instance
                            None,         # context: usually an OpenAIContext, but most MCP tools don't need it
                            result_cb,    # the callback the wrapper will await when it's done
                        )
                    except Exception as e:
                        logger.error(f"Error invoking MCP wrapper for {tool_name}: {e}")
                        if not fut.done():
                            fut.set_exception(e) # Propagate exception if wrapper fails

                    # Wait for the MCP wrapper to call us back
                    try:
                        mcp_result = await asyncio.wait_for(fut, timeout=30.0) # Added timeout
                        # Ensure a dictionary is returned, as FlowResult is a type alias for Dict[str, Any]
                        if isinstance(mcp_result, str):
                             return {"status": "success", "message": mcp_result}
                        elif isinstance(mcp_result, dict):
                             return mcp_result # Assume it's already a valid FlowResult
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

            # 3. Build a Flows handler for each MCP tool
            flow_wrappers = {
                name: make_flow_handler(wrapper_fn, name, llm)
                for name, wrapper_fn in mcp_wrappers.items()
            }
            logger.debug(f"Prepared Flow handlers for MCP tools: {flow_wrappers.keys()}")

            # 4. Inject those handlers into your flow_config
            for node_id, node_cfg in flow_config["nodes"].items():
                if "functions" in node_cfg: # Check if the node has functions defined
                    for fn_def in node_cfg["functions"]:
                        # The function definition might be directly under 'function' or nested
                        actual_fn_def = fn_def.get("function") if isinstance(fn_def.get("function"), dict) else fn_def
                        
                        if isinstance(actual_fn_def, dict) and (tool_name := actual_fn_def.get("name")): # Use walrus operator
                            # If we have an adapted Flow handler for this tool, inject it
                            if tool_name in flow_wrappers:
                                actual_fn_def["handler"] = flow_wrappers[tool_name]
                                logger.info(f"Injected adapted Flow handler for MCP tool '{tool_name}' in node '{node_id}'.")
                        else:
                            logger.warning(f"Skipping function definition in node '{node_id}' due to unexpected structure or missing name: {fn_def}")
                else:
                    logger.debug(f"Node '{node_id}' has no functions defined, skipping wrapper injection.")

    except Exception as e:
        logger.error(f"Failed to register MCP clients or inject handlers: {e}")
        # Depending on requirements, you might want to return or use a default empty schema
        return
    
    pipeline_context = create_pipeline_context(tools_schema)
    context_aggregator = create_context_aggregator(llm, pipeline_context)

    # 3. Build pipeline
    pipeline = build_pipeline(
        transport=transport,
        stt=stt,
        llm=llm,
        tts=tts,
        audio_buffer_processor=audio_buffer_processor,
        stt_mute_filter=stt_mute_filter,
        context_aggregator=context_aggregator
    )

    # 4. Define PipelineParams
    pipeline_params = PipelineParams(
        audio_in_sample_rate=cfg.audio_in_sample_rate,
        audio_out_sample_rate=cfg.audio_out_sample_rate,
        allow_interruptions=True,
    )

    # 5. Initialize FlowManager
    # The PipelineTask is now created inside the context manager
    # We'll pass the flow_manager to the connect handler after it's created.
    
    # This is a temporary solution for passing task to flow_manager
    # In a real scenario, flow_manager might not need direct task access like this
    # or it would be obtained differently.
    _task_ref_for_flow_manager = None 

    async with pipeline_runner_context(pipeline, pipeline_params) as task_for_handlers:
        _task_ref_for_flow_manager = task_for_handlers # Store the task for FlowManager
        
        flow_manager = FlowManager(
            task=_task_ref_for_flow_manager, # Use the task from context manager
            llm=llm,
            context_aggregator=context_aggregator,
            flow_config=flow_config
        )

        # 6. Register event handlers
        # We use functools.partial or lambdas to pass necessary context to handlers
        
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport_param, client_param):
            # Now flow_manager is defined and can be passed
            await on_client_connected_handler(transport_param, client_param, flow_manager, audio_buffer_processor)

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport_param, client_param):
            # task_for_handlers is from the context manager's scope
            await on_client_disconnected_handler(transport_param, client_param, task_for_handlers)

        @audio_buffer_processor.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            await on_audio_data_handler(
                buffer, audio, sample_rate, num_channels, websocket_client.client.port
            )
        
        logger.info("Starting pipeline run...")
        # The runner.run(task) is now handled by the context manager's exit
        # The `initialize` call for flow_manager is in `on_client_connected_handler`
        # We need a way to keep the `run_bot` alive until the pipeline is done.
        # The context manager's `await runner.run(task)` handles this.
        # If on_client_connected is not called, initialize won't run.
        # This implies that the connection must be established for the bot to start its flow.
        # If a connection is already established when run_bot is called, this setup is fine.
        # If run_bot sets up listeners and waits for a connection, initialize should be there.
        
        # For FastAPI, the connection is usually established before this `run_bot` is called as a dependency.
        # So, `on_client_connected` should fire shortly after `runner.run` begins (or is set up).
        
        # The `await runner.run(task)` in the context manager will block here until the task completes or is cancelled.
        logger.info("Pipeline runner_context entered. Waiting for pipeline to complete...")

    logger.info("run_bot completed.")

# Removed save_audio, it's in bot.utils
# Removed original PipelineRunner and task.cancel() calls, now handled by context manager

