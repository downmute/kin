"""
FastAPI Backend for Kin - Empathetic Voice Agent
Simplified pipeline: Speech-to-Text -> LLM -> TTS
No LiveKit dependencies.
"""
import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

from tts_service import TTSService
from llm_service import LLMService
from stt_service import STTService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Kin Voice Agent API",
    description="Empathetic voice agent for the elderly",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
try:
    tts_service = TTSService()
    logger.info("✓ TTS Service initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize TTS Service: {e}")
    tts_service = None

try:
    llm_service = LLMService()
    logger.info("✓ LLM Service initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize LLM Service: {e}")
    llm_service = None

try:
    stt_service = STTService()
    logger.info("✓ STT Service initialized")
except Exception as e:
    logger.error(f"✗ Failed to initialize STT Service: {e}")
    stt_service = None


# Request/Response models
class ChatRequest(BaseModel):
    text: str
    persona: Optional[str] = None
    backstory: Optional[str] = None
    max_tokens: Optional[int] = 150


class TTSRequest(BaseModel):
    text: str


class ChatResponse(BaseModel):
    message: str
    text_response: str
    audio_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    features: dict


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Kin Voice Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "kin-voice-agent",
        "features": {
            "stt": stt_service is not None,
            "tts": tts_service is not None,
            "llm": llm_service is not None
        }
    }


@app.post("/api/chat", tags=["Chat"])
async def chat(
    audio_file: UploadFile = File(...),
    persona: Optional[str] = Form(None),
    backstory: Optional[str] = Form(None),
    max_tokens: Optional[int] = Form(150)
):
    """
    Chat endpoint: Get LLM response and return audio.
    
    This endpoint:
    1. Converts user audio to text using STT
    2. Sends transcribed text to LLM
    3. Gets LLM response
    4. Converts response to speech using TTS
    5. Returns audio file
    
    Expected multipart/form-data:
    - audio_file: Audio file (MP3, WAV, M4A, etc.)
    - persona: Optional persona/character description
    - backstory: Optional backstory
    - max_tokens: Optional max tokens (default: 150)
    
    Returns:
        Audio file (MP3 format) with X-Transcription and X-Text-Response headers
    """
    if not stt_service:
        raise HTTPException(status_code=500, detail="STT service not configured")
    if not llm_service:
        raise HTTPException(status_code=500, detail="LLM service not configured")
    if not tts_service:
        raise HTTPException(status_code=500, detail="TTS service not configured")
    
    try:
        # Read audio file
        audio_data = await audio_file.read()
        audio_filename = audio_file.filename or "audio.mp3"
        
        logger.info(f"Processing chat request with audio file: {audio_filename} ({len(audio_data)} bytes)")
        
        # Step 1: Convert audio to text using STT
        logger.info("Converting audio to text...")
        transcription_result = await stt_service.transcribe_audio(audio_data, audio_filename)
        transcribed_text = transcription_result["text"]
        
        if not transcribed_text or not transcribed_text.strip():
            raise ValueError("No speech detected in audio file")
        
        logger.info(f"Transcription: '{transcribed_text[:50]}...'")
        
        # Step 2: Get LLM response
        logger.info("Getting LLM response...")
        text_response = await llm_service.get_full_response(
            text=transcribed_text,
            persona=persona,
            backstory=backstory,
            max_tokens=max_tokens or 150
        )
        
        logger.info(f"LLM response received: '{text_response[:50]}...'")
        
        # Step 3: Generate TTS audio (now async with WebSocket)
        logger.info("Generating TTS audio...")
        audio_data = await tts_service.generate_audio(text_response)
        
        # Return audio as streaming response
        return StreamingResponse(
            iter([audio_data]),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=response.mp3",
                "X-Transcription": transcribed_text,
                "X-Text-Response": text_response,
                "X-Language-Code": transcription_result.get("language_code", "unknown")
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/chat/text", tags=["Chat"])
async def chat_text_only(request: ChatRequest):
    """
    Chat endpoint that accepts text input and returns only text (no TTS).
    Useful for testing or when audio is not needed.
    This is a backward compatibility endpoint.
    """
    if not llm_service:
        raise HTTPException(status_code=500, detail="LLM service not configured")
    
    try:
        text_response = await llm_service.get_full_response(
            text=request.text,
            persona=request.persona,
            backstory=request.backstory,
            max_tokens=request.max_tokens or 150
        )
        
        return JSONResponse(content={
            "message": "Chat response received",
            "text_response": text_response
        })
        
    except Exception as e:
        logger.error(f"Error in chat/text endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stt", tags=["STT"])
async def speech_to_text(audio_file: UploadFile = File(...)):
    """
    Direct STT endpoint: Convert audio to text.
    
    Expected multipart/form-data:
    - audio_file: Audio file (MP3, WAV, M4A, etc.)
    
    Returns:
        JSON with transcription result
    """
    if not stt_service:
        raise HTTPException(status_code=500, detail="STT service not configured")
    
    try:
        audio_data = await audio_file.read()
        audio_filename = audio_file.filename or "audio.mp3"
        
        transcription_result = await stt_service.transcribe_audio(audio_data, audio_filename)
        
        return JSONResponse(content={
            "message": "Transcription complete",
            "text": transcription_result["text"],
            "language_code": transcription_result.get("language_code"),
            "language_probability": transcription_result.get("language_probability"),
            "words": transcription_result.get("words", [])
        })
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in STT endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts", tags=["TTS"])
async def text_to_speech(request: TTSRequest):
    """
    Direct TTS endpoint: Convert text to speech.
    
    Expected JSON body:
    {
        "text": "Hello, this is a test."
    }
    
    Returns:
        MP3 audio file
    """
    if not tts_service:
        raise HTTPException(status_code=500, detail="TTS service not configured")
    
    try:
        audio_data = await tts_service.generate_audio(request.text)
        
        return StreamingResponse(
            iter([audio_data]),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=tts.mp3"
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in TTS endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint: Stream LLM response chunks.
    Returns Server-Sent Events (SSE) with text chunks.
    """
    if not llm_service:
        raise HTTPException(status_code=500, detail="LLM service not configured")
    
    async def generate_stream():
        """Generate SSE stream from LLM response."""
        try:
            async for chunk in llm_service.stream_response(
                text=request.text,
                persona=request.persona,
                backstory=request.backstory,
                max_tokens=request.max_tokens or 150
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in stream: {e}", exc_info=True)
            yield f"event: error\ndata: {str(e)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.post("/api/tts/stream", tags=["TTS"])
async def text_to_speech_stream(request: TTSRequest):
    """
    Streaming TTS endpoint: Stream audio chunks as they're generated.
    Returns audio chunks in real-time for better latency.
    
    Expected JSON body:
    {
        "text": "Hello, this is a test."
    }
    
    Returns:
        Streaming audio chunks (MP3 format)
    """
    if not tts_service:
        raise HTTPException(status_code=500, detail="TTS service not configured")
    
    async def generate_audio_stream():
        """Generate audio stream chunks."""
        try:
            async for audio_chunk in tts_service.generate_audio_stream(request.text):
                yield audio_chunk
        except Exception as e:
            logger.error(f"Error in TTS stream: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return StreamingResponse(
        generate_audio_stream(),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "attachment; filename=tts_stream.mp3",
            "X-Accel-Buffering": "no"  # Disable buffering for streaming
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info("=" * 60)
    logger.info("Starting Kin Voice Agent API")
    logger.info("=" * 60)
    logger.info(f"✓ Server: http://{host}:{port}")
    logger.info(f"✓ STT Service: {'Enabled' if stt_service else 'Disabled'}")
    logger.info(f"✓ TTS Service: {'Enabled' if tts_service else 'Disabled'}")
    logger.info(f"✓ LLM Service: {'Enabled' if llm_service else 'Disabled'}")
    logger.info("=" * 60)
    
    uvicorn.run(app, host=host, port=port)

