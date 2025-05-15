import os
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()

@dataclass
class BotConfig:
    """
    Configuration for the bot, initialized from environment variables.
    """
    def __init__(self):
        # API Keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVEN_API_KEY")
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
        
        # Model configs
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.elevenlabs_voice_id = os.getenv("ELEVEN_VOICE_ID")
        
        # Audio settings
        self.audio_in_sample_rate = int(os.getenv("AUDIO_IN_SAMPLE_RATE", "16000"))
        self.audio_out_sample_rate = int(os.getenv("AUDIO_OUT_SAMPLE_RATE", "16000"))
        
        # Validate required settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all required configuration values are set."""
        required_vars = [
            ("OpenAI API Key", self.openai_api_key),
            ("ElevenLabs API Key", self.elevenlabs_api_key),
            ("Deepgram API Key", self.deepgram_api_key),
            ("ElevenLabs Voice ID", self.elevenlabs_voice_id),
        ]
        
        missing = [name for name, value in required_vars if not value]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# Define flow configuration
# Note: The handler for store_name will be set dynamically by run_bot
flow_config = {
    "initial_node": "greeting",
    "nodes": {
        "greeting": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are a friendly assistant name Ghedion Beyen-Chang who works for Blue Gem Motors in Flint, Michigan. All of your responses will be converted to speech, so don't use special characters or markdown."
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": " Introduce yourself and ask the caller for their name."
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "store_name",
                        "description": "Store the user's name",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"]
                        },
                        "transition_to": "time_info"
                    }
                }
            ]
        },

        # ──────────────────────────────────────
        # NEW node: time info
        # ──────────────────────────────────────
        "time_info": {
            "role_messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that answers questions about the current time "
                        "and converts times between time-zones. "
                        "You may ONLY call the functions provided below."
                    )
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "Ask the user what time information they need."
                }
            ],
            "functions": [
                # ------- Time MCP: get_current_time -------
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_time",
                        "description": "Get the current time in a given IANA timezone",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "timezone": {
                                    "type": "string",
                                    "description": "IANA timezone, e.g. 'America/New_York'"
                                }
                            },
                            "required": ["timezone"]
                        }
                    }
                },
                # ------- Time MCP: convert_time -------
                {
                    "type": "function",
                    "function": {
                        "name": "convert_time",
                        "description": "Convert a time between two IANA timezones",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "source_timezone": {"type": "string"},
                                "time":            {"type": "string", "description": "24-hour HH:MM"},
                                "target_timezone": {"type": "string"}
                            },
                            "required": ["source_timezone", "time", "target_timezone"]
                        }
                    }
                }
            ]
        }
    }
} 