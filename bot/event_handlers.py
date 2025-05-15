from fastapi import WebSocket
from pipecat.pipeline.runner import PipelineTask
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport
from pipecat_flows import FlowManager

from bot.utils import save_audio

# Note: We need to find a way to pass `task`, `flow_manager`, `audio_buffer_processor` 
# and `websocket_client` to these handlers, or refactor how they are used.
# For now, they are defined here and will be attached in bot.py

async def on_client_connected_handler(
    transport: FastAPIWebsocketTransport, 
    client: WebSocket, 
    flow_manager: FlowManager, 
    audio_buffer_processor: AudioBufferProcessor
):
    await audio_buffer_processor.start_recording()
    await flow_manager.initialize()

async def on_client_disconnected_handler(
    transport: FastAPIWebsocketTransport, 
    client: WebSocket, 
    task: PipelineTask
):
    await task.cancel()

async def on_audio_data_handler(
    buffer: AudioBufferProcessor, 
    audio: bytes, 
    sample_rate: int, 
    num_channels: int, 
    websocket_client_port: int # Passing port instead of the whole client object
):
    server_name = f"server_{websocket_client_port}"
    await save_audio(server_name, audio, sample_rate, num_channels) 