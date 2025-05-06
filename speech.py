# speech.py
import logging
import io
import time
import re
import traceback
from typing import Optional

from config import ConfigManager

logger = logging.getLogger("penelope.speech")

def clean_text_for_speech(text: str) -> str:
    """Remove special tokens and format text for better speech synthesis."""
    # Remove markdown code blocks
    text = re.sub(r'```[\s\S]*?```', ' code block omitted for speech ', text)
    
    # Remove markdown inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', ' URL omitted for speech ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s.,;:!?\'"\-()]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Limit length to avoid timeouts
    max_chars = 4000  # Adjust based on testing
    if len(text) > max_chars:
        text = text[:max_chars] + "... The rest of the message has been truncated for speech synthesis."
    
    logger.info(f"Cleaned text for speech ({len(text)} chars)")
    return text

def generate_speech(text: str, config_manager: Optional[ConfigManager] = None) -> Optional[str]:
    """Generate speech from text using ElevenLabs."""
    try:
        # Get API key
        elevenlabs_api_key = ""
        if config_manager:
            elevenlabs_api_key = config_manager.get_api_key("elevenlabs")
        
        if not elevenlabs_api_key:
            logger.error("ElevenLabs API key not set")
            return None
        
        # Clean and prepare text for speech
        text = clean_text_for_speech(text)
        
        logger.info(f"Generating speech for text ({len(text)} chars)...")
        
        # Try with newer API first, fall back to older API if needed
        try:
            # Try importing the newer ElevenLabs API
            from elevenlabs import ElevenLabs
            logger.info("Using newer ElevenLabs API...")
            
            # Initialize client
            client = ElevenLabs(api_key=elevenlabs_api_key)
            
            # Voice ID
            voice_id = "ZF6FPAbjXT4488VcRRnw"  # Rachel
            
            # Convert text to speech
            audio_stream = client.text_to_speech.convert_as_stream(
                text=text,
                voice_id=voice_id,
                model_id="eleven_turbo_v2_5"
            )
            
            # Process the audio stream
            audio_data = io.BytesIO()
            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    audio_data.write(chunk)
            
            audio_data.seek(0)
            
            # Save to file
            timestamp = int(time.time())
            filename = f"bitcoin_audio_{timestamp}.mp3"
            with open(filename, "wb") as f:
                f.write(audio_data.getbuffer())
            
            logger.info(f"Audio generated and saved as {filename}")
            return filename
                
        except (ImportError, AttributeError) as e:
            logger.warning(f"Error with newer API: {e}. Falling back to older API...")
            
            # Fall back to older ElevenLabs API
            from elevenlabs import generate, save, set_api_key
            logger.info("Using older ElevenLabs API...")
            
            # Set API key
            set_api_key(elevenlabs_api_key)
            
            # Generate audio with Rachel voice
            audio = generate(
                text=text,
                voice="Rachel",
                model="eleven_multilingual_v2"
            )
            
            # Save to file
            timestamp = int(time.time())
            filename = f"bitcoin_audio_{timestamp}.mp3"
            save(audio, filename)
            
            logger.info(f"Audio generated and saved as {filename}")
            return filename
            
    except Exception as e:
        error_msg = f"Speech generation error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return None