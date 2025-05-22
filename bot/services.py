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
from deepgram import LiveOptions

from bot.config import BotConfig

def create_llm(cfg: BotConfig) -> OpenAILLMService:
    return OpenAILLMService(api_key=cfg.openai_api_key, model=cfg.openai_model)

def create_tts(cfg: BotConfig) -> ElevenLabsTTSService:
    return ElevenLabsTTSService(
        api_key=cfg.elevenlabs_api_key,
        voice_id=cfg.elevenlabs_voice_id,
    )

def create_stt(cfg: BotConfig) -> DeepgramSTTService:
    # Keywords to boost for better recognition
    # Includes common terms, VIN-related words, and individual characters often found in VINs
    deepgram_keywords = [
        "VIN:5", "decode:3", "time:3", "vehicle:3", "car:3", "automobile:3", "NHTSA:5",
        "alpha:3", "bravo:3", "charlie:3", "delta:3", "echo:3", "foxtrot:3", "golf:3", "hotel:3",
        "india:3", "juliett:3", "kilo:3", "lima:3", "mike:3", "november:3", "oscar:3", "papa:3",
        "quebec:3", "romeo:3", "sierra:3", "tango:3", "uniform:3", "victor:3", "whiskey:3",
        "x-ray:3", "yankee:3", "zulu:3",
        "zero:3", "one:3", "two:3", "three:3", "four:3", "five:3", "six:3", "seven:3", "eight:3", "nine:3",
        "A:3", "B:3", "C:3", "D:3", "E:3", "F:3", "G:3", "H:3", "J:3", "K:3", "L:3", "M:3", "N:3",
        "P:3", "R:3", "S:3", "T:3", "U:3", "V:3", "W:3", "X:3", "Y:3", "Z:3",
        "0:3", "1:3", "2:3", "3:3", "4:3", "5:3", "6:3", "7:3", "8:3", "9:3"
    ]
    
    return DeepgramSTTService(
        api_key=cfg.deepgram_api_key,
        # audio_passthrough=True, # This was in your original code, uncomment if needed
        # numerals=True,          # This was in your original code, uncomment if needed
        # smart_format=True       # This was in your original code, uncomment if needed
        live_options=LiveOptions(
            model="nova-2-general", # Or your preferred model
            language="en-US",       # Or your preferred language
            keywords=deepgram_keywords,
            smart_format=True,      # Good for readability
            numerals=True,          # To convert numbers from words to digits
            vad_events=True         # If you use VAD events from Deepgram
            # You might have had audio_passthrough=True here in LiveOptions or at the service level.
            # If you need audio passthrough, ensure it's set according to Pipecat/Deepgram docs.
            # For example, if it's a direct parameter of DeepgramSTTService:
            # audio_passthrough=True 
        )
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