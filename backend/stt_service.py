"""
STT Service - Speech-to-Text using ElevenLabs API
"""
import os
import logging
import httpx
from typing import Optional
import json


class STTService:
    """Speech-to-Text service using ElevenLabs API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize STT service.
        
        Args:
            api_key: ElevenLabs API key (defaults to ELEVENLABS_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.api_url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY not provided")
        
        logging.info("STT Service initialized")
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_filename: str = "audio.mp3",
        model_id: str = "scribe_v1"
    ) -> dict:
        """
        Transcribe audio to text using ElevenLabs API.
        
        Args:
            audio_data: Audio file data as bytes
            audio_filename: Name of the audio file (used to determine format)
            model_id: Model ID to use (default: "scribe_v1")
        
        Returns:
            Transcription result dict with:
            - text: Full transcribed text
            - language_code: Detected language code
            - language_probability: Confidence in language detection
            - words: List of word-level timestamps and metadata (optional)
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")
        
        logging.info(f"Transcribing audio file: {audio_filename} ({len(audio_data)} bytes)")
        
        headers = {
            "xi-api-key": self.api_key
        }
        
        # Prepare multipart form data
        files = {
            "file": (audio_filename, audio_data, self._get_content_type(audio_filename))
        }
        
        data = {
            "model_id": model_id
        }
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for long audio
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data=data
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract the text from the response
                transcribed_text = result.get("text", "")
                
                logging.info(f"Transcription complete: '{transcribed_text[:50]}...'")
                
                return {
                    "text": transcribed_text,
                    "language_code": result.get("language_code"),
                    "language_probability": result.get("language_probability"),
                    "words": result.get("words", []),  # Word-level timestamps
                    "full_response": result  # Include full response for advanced use cases
                }
                
        except httpx.HTTPStatusError as e:
            error_detail = "Unknown error"
            try:
                error_detail = e.response.json().get("detail", {}).get("message", str(e))
            except:
                error_detail = str(e)
            
            logging.error(f"STT API error: {error_detail}")
            raise ValueError(f"STT transcription failed: {error_detail}")
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            raise
    
    def _get_content_type(self, filename: str) -> str:
        """
        Get content type based on file extension.
        
        Args:
            filename: Audio file name
        
        Returns:
            Content type string
        """
        extension = filename.lower().split('.')[-1]
        
        content_types = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg",
            "opus": "audio/opus",
            "flac": "audio/flac",
            "webm": "audio/webm",
            "aac": "audio/aac",
            "aiff": "audio/aiff",
            "mp4": "video/mp4",
            "avi": "video/x-msvideo",
            "mkv": "video/x-matroska",
            "mov": "video/quicktime",
            "wmv": "video/x-ms-wmv",
            "flv": "video/x-flv",
            "3gp": "video/3gpp"
        }
        
        return content_types.get(extension, "audio/mpeg")  # Default to MP3
    
    async def transcribe_file(self, file_path: str, model_id: str = "scribe_v1") -> dict:
        """
        Transcribe audio file from disk.
        
        Args:
            file_path: Path to audio file
            model_id: Model ID to use (default: "scribe_v1")
        
        Returns:
            Transcription result dict
        """
        with open(file_path, "rb") as f:
            audio_data = f.read()
        
        filename = os.path.basename(file_path)
        return await self.transcribe_audio(audio_data, filename, model_id)

