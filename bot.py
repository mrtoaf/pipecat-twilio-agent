#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import datetime
import io
import os
import sys
import wave
import shutil

import aiofiles
from dotenv import load_dotenv
from fastapi import WebSocket
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

from mcp import StdioServerParameters
from pipecat.services.mcp_service import MCPClient
from pipecat.adapters.schemas.tools_schema import ToolsSchema

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(websocket_client: WebSocket, stream_sid: str, testing: bool):
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            serializer=TwilioFrameSerializer(stream_sid),
        ),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    # --- MCP Integration Start ---

    # 1. Define the MCP server parameters
    # Ensure 'uvx' is available in your PATH or provide the full path
    # Also ensure 'mcp-server-time' is installed via uv/pip
    uvx_path = shutil.which("uvx")
    if not uvx_path:
        logger.error("uvx command not found in PATH. Please install uv.")
        # Handle error appropriately, maybe raise an exception or exit
        return

    time_server_params = StdioServerParameters(
        command=uvx_path,
        args=["mcp-server-time"]
    )

    # 2. Create the MCPClient instance
    try:
        time_mcp = MCPClient(server_params=time_server_params)
    except Exception as e:
        logger.error(f"Failed to initialize MCP time server: {e}")
        # Handle error appropriately
        return

    # 3. Register the tools with the LLM service
    # This modifies the 'llm' object in place to include the tool specs
    # and sets up the internal hooks to execute the tool when called.
    try:
        time_tools_schema = await time_mcp.register_tools(llm)
        logger.info(f"Registered tools from time server: {time_tools_schema}")
    except Exception as e:
        logger.error(f"Failed to register tools from MCP time server: {e}")
        # Handle error appropriately
        return

    # --- MCP Integration End ---

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True)

    tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVEN_API_KEY"),
            voice_id=os.getenv("ELEVEN_VOICE_ID"),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant named Ghedion Samson Beyen. "
                "You have access to a tool that can provide the current time. Use it when asked about the time. "
                "Your output will be converted to audio so don't include special characters in your answers. "
                "Respond with a short short sentence."
            ),
        },
    ]

    # 4. Use the registered tools schema when creating the context
    # If you add more MCP servers, you'll merge their schemas here.
    # For now, we just use the one from the time server.
    context = OpenAILLMContext(messages, tools=time_tools_schema)
    context_aggregator = llm.create_context_aggregator(context)

    # NOTE: Watch out! This will save all the conversation in memory. You can
    # pass `buffer_size` to get periodic callbacks.
    audiobuffer = AudioBufferProcessor(user_continuous_stream=not testing)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM (handles tool calls internally now)
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            audiobuffer,  # Used to buffer the audio in the pipeline
            context_aggregator.assistant(), # Assistant spoken responses and tool context
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            allow_interruptions=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start recording.
        await audiobuffer.start_recording()
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        server_name = f"server_{websocket_client.client.port}"
        await save_audio(server_name, audio, sample_rate, num_channels)

    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)