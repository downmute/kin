"""
TTS Service - Text-to-Speech using ElevenLabs WebSocket API
Extracted and simplified from the original LiveKit agent code.
Uses WebSocket for real-time streaming audio generation.
"""
import os
import re
import json
import logging
import asyncio
import uuid
import base64
from typing import Optional, AsyncIterator
import websockets

# Emoji pattern for removal
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "\U00002700-\U000027BF"  # dingbats
    "]+",
    flags=re.UNICODE
)


def strip_emojis(text: str) -> str:
    """Remove all emojis from text for TTS processing."""
    return EMOJI_PATTERN.sub('', text)


def strip_markdown(text: str) -> str:
    """
    Aggressively strip all special symbols for TTS processing.
    Only keeps: letters, numbers, spaces, and basic punctuation.
    Converts common symbols to words (& → and, @ → at)
    """
    if not text:
        return text
    
    # First, convert common symbols to words for TTS
    text = text.replace('&', ' and ')
    text = text.replace('@', ' at ')
    
    # Handle common markdown patterns
    # Remove code blocks (```code```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove inline code (`code`)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove links [text](url) - keep only the text part
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)       # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_
    
    # Remove headers (# ## ###)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove list markers (- * +)
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    
    # Remove blockquotes (>)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Remove horizontal rules (---, ***)
    text = re.sub(r'^[\-\*]{3,}$', '', text, flags=re.MULTILINE)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # AGGRESSIVE: Remove ALL special symbols except:
    # - Letters (a-z, A-Z)
    # - Numbers (0-9)
    # - Basic punctuation: . , ! ? : ; ' " - ( ) [ ] { }
    # - Math/common symbols: $ % = +
    # - Spaces and newlines
    allowed_chars = r"[a-zA-Z0-9\s\.\,\!\?\:\;\'\"\-\_\(\)\[\]\{\}\$\%\=\+]"
    
    # Keep only allowed characters
    text = ''.join(re.findall(allowed_chars, text))
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up multiple punctuation
    text = re.sub(r'([\.\,\!\?])\1+', r'\1', text)
    
    return text.strip()


class TTSService:
    """Text-to-Speech service using ElevenLabs WebSocket API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        voice_settings: Optional[dict] = None
    ):
        """
        Initialize TTS service.
        
        Args:
            api_key: ElevenLabs API key (defaults to ELEVENLABS_API_KEY env var)
            voice_id: Voice ID (defaults to ELEVENLABS_VOICE_ID env var)
            model: Model ID (defaults to "eleven_turbo_v2_5" or ELEVENLABS_MODEL env var)
            voice_settings: Voice settings dict with stability, similarity_boost, style, use_speaker_boost (optional)
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID")
        self.model = model or os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5")
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not provided")
        if not self.voice_id:
            raise ValueError("ELEVENLABS_VOICE_ID not provided")
        
        # Default voice settings if not provided
        if voice_settings is None:
            voice_settings = {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True
            }
        
        self.voice_settings = voice_settings
        
        # WebSocket URI for multi-context TTS
        self.websocket_uri = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"
            f"?model_id={self.model}"
        )
        
        logging.info(f"TTS Service initialized with voice ID: {self.voice_id}, model: {self.model}")
    
    async def generate_audio_stream(
        self,
        text: str,
        model: Optional[str] = None,
        voice_settings: Optional[dict] = None
    ) -> AsyncIterator[bytes]:
        """
        Generate audio from text using ElevenLabs WebSocket API (streaming).
        
        Args:
            text: Text to convert to speech
            model: Model ID (optional, uses instance default if not provided)
            voice_settings: Voice settings dict (optional, uses instance default if not provided)
        
        Yields:
            Audio chunks as bytes
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean text: strip emojis and markdown
        cleaned_text = strip_emojis(text)
        cleaned_text = strip_markdown(cleaned_text)
        
        if not cleaned_text.strip():
            raise ValueError("Text is empty after cleaning")
        
        logging.info(f"Generating TTS stream for text: '{cleaned_text[:50]}...'")
        
        # Use provided model or instance default
        model_id = model or self.model
        websocket_uri = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/multi-stream-input"
            f"?model_id={model_id}"
        )
        
        # Use provided voice settings or instance default
        settings = voice_settings or self.voice_settings
        
        # Generate unique context ID for this request
        context_id = str(uuid.uuid4())
        
        try:
            async with websockets.connect(
                websocket_uri,
                max_size=16 * 1024 * 1024,  # 16MB max message size
                additional_headers={"xi-api-key": self.api_key}
            ) as websocket:
                # Send initial message with text and voice settings
                initial_message = {
                    "text": cleaned_text,
                    "context_id": context_id,
                    "voice_settings": {
                        "stability": settings.get("stability", 0.5),
                        "similarity_boost": settings.get("similarity_boost", 0.75),
                        "style": settings.get("style", 0.0),
                        "use_speaker_boost": settings.get("use_speaker_boost", True)
                    }
                }
                
                await websocket.send(json.dumps(initial_message))
                
                # Flush to ensure generation
                await websocket.send(json.dumps({
                    "context_id": context_id,
                    "flush": True
                }))
                
                # Receive audio chunks
                async for message in websocket:
                    try:
                        # Messages from ElevenLabs are JSON strings
                        if isinstance(message, str):
                            data = json.loads(message)
                            
                            # Check if this is audio data (base64 encoded)
                            if "audio" in data:
                                try:
                                    audio_bytes = base64.b64decode(data["audio"])
                                    yield audio_bytes
                                except Exception as e:
                                    logging.error(f"Error decoding base64 audio: {e}")
                                    continue
                            
                            # Check if generation is complete
                            if data.get("is_final"):
                                logging.info(f"TTS generation complete for context '{context_id}'")
                                break
                            
                            # Check for errors
                            if "error" in data:
                                error_msg = data.get("error", {}).get("message", "Unknown error")
                                logging.error(f"TTS WebSocket error: {error_msg}")
                                raise ValueError(f"TTS generation error: {error_msg}")
                        else:
                            # Binary message (shouldn't happen but handle it)
                            logging.warning(f"Received binary message: {len(message)} bytes")
                            if isinstance(message, bytes):
                                yield message
                                
                    except json.JSONDecodeError as e:
                        logging.warning(f"Error parsing JSON message: {e}, message type: {type(message)}")
                        continue
                    except Exception as e:
                        logging.error(f"Error processing WebSocket message: {e}")
                        raise
                
                # Close the context
                await websocket.send(json.dumps({
                    "context_id": context_id,
                    "close_context": True
                }))
                
        except Exception as e:
            logging.error(f"Error generating TTS stream: {e}")
            raise
    
    async def generate_audio(self, text: str, model: Optional[str] = None, voice_settings: Optional[dict] = None) -> bytes:
        """
        Generate audio from text using ElevenLabs WebSocket API (non-streaming).
        Collects all audio chunks and returns as complete bytes.
        
        Args:
            text: Text to convert to speech
            model: Model ID (optional, uses instance default if not provided)
            voice_settings: Voice settings dict (optional, uses instance default if not provided)
        
        Returns:
            Complete audio data as bytes
        """
        audio_chunks = []
        async for chunk in self.generate_audio_stream(text, model, voice_settings):
            audio_chunks.append(chunk)
        
        audio_data = b''.join(audio_chunks)
        logging.info(f"Generated {len(audio_data)} bytes of audio data")
        return audio_data
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for TTS processing (remove emojis and markdown).
        
        Args:
            text: Raw text to clean
        
        Returns:
            Cleaned text
        """
        cleaned = strip_emojis(text)
        cleaned = strip_markdown(cleaned)
        return cleaned.strip()

