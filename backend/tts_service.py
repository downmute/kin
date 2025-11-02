"""
TTS Service - Text-to-Speech using Fish Audio API
Extracted and simplified from the original LiveKit agent code.
"""
import os
import re
import logging
import asyncio
from typing import Optional, AsyncIterator
from fish_audio_sdk import Session, TTSRequest

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
    """Text-to-Speech service using Fish Audio API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        format: str = "mp3"
    ):
        """
        Initialize TTS service.
        
        Args:
            api_key: Fish Audio secret key (defaults to FISH_AUDIO_SECRET_KEY env var)
            voice_id: Voice reference ID (defaults to ENGLISH_FEMALE_SOFT env var)
            format: Audio format (default: "mp3")
        """
        self.api_key = api_key or os.getenv("FISH_AUDIO_SECRET_KEY")
        self.voice_id = voice_id or os.getenv("MODEL_ID")
        self.format = format
        
        if not self.api_key:
            raise ValueError("FISH_AUDIO_SECRET_KEY not provided")
        if not self.voice_id:
            raise ValueError("MODEL_ID not provided")
        
        logging.info(f"TTS Service initialized with voice reference ID: {self.voice_id}, format: {self.format}")
    
    async def generate_audio_stream(
        self,
        text: str,
        format: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """
        Generate audio from text using Fish Audio API (streaming).
        Yields audio chunks as they are generated.
        
        Args:
            text: Text to convert to speech
            format: Audio format (optional, uses instance default if not provided)
        
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
        
        # Use provided format or instance default
        audio_format = format or self.format
        
        try:
            # Use a queue to bridge between sync iterator and async generator
            # Very large queue size to prevent blocking, allows heavy buffering
            chunk_queue = asyncio.Queue(maxsize=200)  # Very large buffer to prevent blocking
            exception_holder = [None]
            
            # Start streaming in background thread
            loop = asyncio.get_event_loop()
            import threading
            
            # Store loop reference for thread
            def _stream_chunks_to_queue_with_loop():
                """Wrapper that captures the event loop."""
                try:
                    session = Session(self.api_key)
                    
                    tts_request = TTSRequest(
                        text=cleaned_text,
                        reference_id=self.voice_id,
                        format=audio_format
                    )
                    
                    # Stream chunks to queue using thread-safe put
                    # Put chunks immediately without waiting to minimize latency
                    for chunk in session.tts(tts_request):
                        if chunk:
                            # Schedule put operation immediately without waiting
                            # This allows chunks to queue up and be consumed asynchronously
                            try:
                                asyncio.run_coroutine_threadsafe(
                                    chunk_queue.put(chunk),
                                    loop
                                )
                            except Exception as e:
                                logging.warning(f"Failed to queue chunk: {e}")
                                # If we can't queue, try put_nowait as fallback
                                try:
                                    # This won't work from another thread, but try anyway
                                    pass
                                except:
                                    pass  # Skip chunk if queue operations fail
                    
                    # Signal completion
                    asyncio.run_coroutine_threadsafe(chunk_queue.put(None), loop)
                        
                except Exception as e:
                    exception_holder[0] = e
                    try:
                        asyncio.run_coroutine_threadsafe(chunk_queue.put(None), loop)
                    except:
                        pass
            
            thread = threading.Thread(target=_stream_chunks_to_queue_with_loop, daemon=True)
            thread.start()
            
            # Yield chunks with heavy buffering to prevent audio stops
            # Large buffer ensures very smooth playback
            chunk_buffer = []
            buffer_size = 10  # Buffer 10 chunks for very smooth playback
            
            while True:
                chunk = await chunk_queue.get()
                
                if exception_holder[0]:
                    raise exception_holder[0]
                
                if chunk is None:  # Signal to stop
                    # Yield any remaining buffered chunks
                    if chunk_buffer:
                        combined_chunk = b''.join(chunk_buffer)
                        chunk_buffer = []
                        yield combined_chunk
                    break
                
                # Add chunk to buffer
                chunk_buffer.append(chunk)
                
                # Yield if buffer is full
                if len(chunk_buffer) >= buffer_size:
                    # Combine buffered chunks and yield
                    combined_chunk = b''.join(chunk_buffer)
                    chunk_buffer = []
                    yield combined_chunk
                    
        except Exception as e:
            logging.error(f"Error generating TTS stream: {e}")
            raise
    
    async def generate_audio(self, text: str, format: Optional[str] = None) -> bytes:
        """
        Generate audio from text using Fish Audio API (non-streaming).
        Collects all audio chunks and returns as complete bytes.
        
        Args:
            text: Text to convert to speech
            format: Audio format (optional, uses instance default if not provided)
        
        Returns:
            Complete audio data as bytes
        """
        audio_chunks = []
        async for chunk in self.generate_audio_stream(text, format):
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

