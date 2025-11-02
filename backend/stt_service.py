"""
STT Service - Speech-to-Text using OpenAI Whisper API
"""
import os
import logging
from typing import Optional
from openai import OpenAI


class STTService:
    """Speech-to-Text service using OpenAI Whisper API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-transcribe"):
        """
        Initialize STT service.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: "gpt-4o-transcribe")
                   Options: "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe", "gpt-4o-transcribe-diarize"
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = os.getenv("OPENROUTER_BASE_URL")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        
        logging.info(f"STT Service initialized with model: {self.model}")
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_filename: str = "audio.mp3",
        model: Optional[str] = None,
        response_format: str = "json",
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> dict:
        """
        Transcribe audio to text using OpenAI Whisper API.
        
        Args:
            audio_data: Audio file data as bytes
            audio_filename: Name of the audio file (used to determine format)
            model: Model to use (optional, uses instance default if not provided)
            response_format: Response format - "json", "text", "srt", "verbose_json", or "vtt"
            prompt: Optional prompt to improve accuracy
            language: Optional language code (ISO 639-1)
            temperature: Optional temperature (0.0 to 1.0)
        
        Returns:
            Transcription result dict with:
            - text: Full transcribed text
            - language: Detected language (if verbose_json format)
            - segments: List of segments with timestamps (if verbose_json format)
            - words: List of word-level timestamps (if verbose_json with timestamp_granularities)
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")
        
        logging.info(f"Transcribing audio file: {audio_filename} ({len(audio_data)} bytes)")
        
        # Use provided model or instance default
        model_to_use = model or self.model
        
        try:
            # Create a temporary file-like object from bytes
            import io
            
            audio_file = io.BytesIO(audio_data)
            audio_file.name = audio_filename  # Set filename for OpenAI API
            
            # Prepare transcription parameters
            transcription_params = {
                "file": audio_file,
                "model": model_to_use,
                "response_format": response_format
            }
            
            if prompt:
                transcription_params["prompt"] = prompt
            if language:
                transcription_params["language"] = language
            if temperature is not None:
                transcription_params["temperature"] = temperature
            
            # Call OpenAI API
            # Note: OpenAI SDK is synchronous, so we run it in a thread pool
            import asyncio
            
            def _transcribe_sync():
                """Synchronous transcription call."""
                return self.client.audio.transcriptions.create(**transcription_params)
            
            result = await asyncio.to_thread(_transcribe_sync)
            
            # Extract the text from the response
            if isinstance(result, str):
                # Response format is "text"
                transcribed_text = result
                return {
                    "text": transcribed_text
                }
            else:
                # Response format is JSON or verbose_json
                transcribed_text = result.text if hasattr(result, 'text') else str(result)
                
                result_dict = {
                    "text": transcribed_text
                }
                
                # Add additional fields if available
                if hasattr(result, 'language'):
                    result_dict["language"] = result.language
                if hasattr(result, 'segments'):
                    result_dict["segments"] = result.segments
                if hasattr(result, 'words'):
                    result_dict["words"] = result.words
                if hasattr(result, 'duration'):
                    result_dict["duration"] = result.duration
                
                logging.info(f"Transcription complete: '{transcribed_text[:50]}...'")
                
                return result_dict
                
        except Exception as e:
            logging.error(f"Error transcribing audio: {e}")
            raise ValueError(f"STT transcription failed: {str(e)}")
    
    async def transcribe_file(
        self,
        file_path: str,
        model: Optional[str] = None,
        response_format: str = "json",
        prompt: Optional[str] = None,
        language: Optional[str] = None
    ) -> dict:
        """
        Transcribe audio file from disk.
        
        Args:
            file_path: Path to audio file
            model: Model to use (optional, uses instance default if not provided)
            response_format: Response format - "json", "text", "srt", "verbose_json", or "vtt"
            prompt: Optional prompt to improve accuracy
            language: Optional language code (ISO 639-1)
        
        Returns:
            Transcription result dict
        """
        with open(file_path, "rb") as f:
            audio_data = f.read()
        
        filename = os.path.basename(file_path)
        return await self.transcribe_audio(
            audio_data,
            filename,
            model=model,
            response_format=response_format,
            prompt=prompt,
            language=language
        )

