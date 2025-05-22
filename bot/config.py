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
                {"role": "system", "content": "You are Ghedion Beyen, a friendly and helpful car expert from Blue Gem Motors in Flint, Michigan. Speak naturally, as if you are talking to someone on the phone. Do not use any special formatting like bullet points or numbered lists in your responses."}
            ],
            "task_messages": [
                {"role": "system", "content": "Introduce yourself and where you work. Then, politely ask for the user\'s name."}
            ],
            "functions": [{
                "type": "function",
                "function": {
                    "name": "store_name",
                    "handler": "store_name_handler_placeholder",
                    "description": "Store the user's name",
                    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
                    "transition_to": "service_selection"
                }
            }]
        },
        "service_selection": {
            "task_messages": [
                {"role": "system", "content": "Thanks {name}! Would you like to get the current time or decode a VIN?"}
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "choose_time",
                        "handler": "choose_time_handler_placeholder",
                        "description": "User chose to get the current time",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "transition_to": "time_node"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "choose_vin",
                        "handler": "choose_vin_handler_placeholder",
                        "description": "User chose to decode a VIN",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "transition_to": "vin_collection"
                    }
                }
            ]
        },
        "time_node": {
            "task_messages": [
                {"role": "system", "content": "Which city or timezone would you like the time for?"}
            ],
            "functions": [{
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time in a given timezone",
                    "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}
                }
            }]
        },
        "vin_collection": {
            "task_messages": [
                {"role": "system", "content": "Please say or enter the 17-digit VIN."}
            ],
            "functions": [{
                "type": "function",
                "function": {
                    "name": "collect_vin",
                    "handler": "collect_vin_handler_placeholder",
                    "description": "Capture the user's VIN",
                    "parameters": {"type": "object", "properties": {"vin": {"type": "string"}}, "required": ["vin"]},
                    "transition_to": "vin_confirmation"
                }
            }]
        },
        "vin_confirmation": {
            "task_messages": [
                {"role": "system", "content": "Okay, I have the VIN as {vin}. Is that correct?"}
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "confirm_vin",
                        "handler": "confirm_vin_handler_placeholder",
                        "description": "User confirmed the VIN is correct",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "transition_to": "vin_lookup"
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "reject_vin",
                        "handler": "reject_vin_handler_placeholder",
                        "description": "User said the VIN is incorrect",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "transition_to": "vin_collection"
                    }
                }
            ]
        },
        "vin_lookup": {
            "task_messages": [
                {"role": "system", "content": "Alright, looking up that VIN for you now..."}
            ],
            "functions": [{
                "type": "function",
                "function": {
                    "name": "decode_vin",
                    "description": "Decode a Vehicle Identification Number (VIN) using NHTSA's API.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vin": {"type": "string"},
                            "modelyear": {"type": "string"}
                        },
                        "required": ["vin"]
                    }
                }
            }]
        }
    }
} 