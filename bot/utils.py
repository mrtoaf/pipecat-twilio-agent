import datetime
import io
import wave
import aiofiles
from loguru import logger

async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int, path: str | None = None):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        if path:
            filename = f"{path}/{filename}"

        # Save the audio to a file
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
                wf.setnchannels(num_channels) # Mono (1 channel)
                wf.setframerate(sample_rate) # 8000 Hz
                wf.writeframes(audio) # Write the audio frames
            
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save") 