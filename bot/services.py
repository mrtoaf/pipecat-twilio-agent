from fastapi import WebSocket

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
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
from pipecat.processors.filters.stt_mute_filter import STTMuteConfig, STTMuteFilter, STTMuteStrategy
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai.llm import OpenAIContextAggregatorPair

from bot.config import BotConfig

def create_llm(cfg: BotConfig) -> OpenAILLMService:
    return OpenAILLMService(api_key=cfg.openai_api_key, model=cfg.openai_model)

def create_tts(cfg: BotConfig) -> ElevenLabsTTSService:
    return ElevenLabsTTSService(
        api_key=cfg.elevenlabs_api_key,
        voice_id=cfg.elevenlabs_voice_id,
    )

def create_stt(cfg: BotConfig) -> DeepgramSTTService:
    return DeepgramSTTService(
        api_key=cfg.deepgram_api_key,
        audio_passthrough=True, # Assuming this should be configurable if needed
        numerals=True,          # Assuming this should be configurable if needed
        smart_format=True       # Assuming this should be configurable if needed
    )

def create_transport(
    websocket_client: WebSocket, stream_sid: str, cfg: BotConfig
) -> FastAPIWebsocketTransport:
    return FastAPIWebsocketTransport(
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

def create_audio_buffer_processor(testing: bool, cfg: BotConfig) -> AudioBufferProcessor:
    return AudioBufferProcessor(user_continuous_stream=not testing)

def create_stt_mute_filter() -> STTMuteFilter:
    return STTMuteFilter(
        config=STTMuteConfig(
            strategies={
                STTMuteStrategy.FIRST_SPEECH,
                STTMuteStrategy.FUNCTION_CALL,
            }
        ),
    )

def create_pipeline_context(tools_schema: ToolsSchema) -> OpenAILLMContext:
    return OpenAILLMContext(tools=tools_schema)

def create_context_aggregator(llm: OpenAILLMService, context: OpenAILLMContext) -> OpenAIContextAggregatorPair:
    return llm.create_context_aggregator(context)

def build_pipeline(
    transport: FastAPIWebsocketTransport,
    stt: DeepgramSTTService,
    llm: OpenAILLMService, 
    tts: ElevenLabsTTSService,
    audio_buffer_processor: AudioBufferProcessor,
    stt_mute_filter: STTMuteFilter,
    context_aggregator: OpenAIContextAggregatorPair
) -> Pipeline:
    return Pipeline(
        [
            transport.input(),
            stt_mute_filter,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            audio_buffer_processor,
            context_aggregator.assistant(),
        ]
    ) 