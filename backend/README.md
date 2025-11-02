# Kin Backend

Simplified FastAPI backend for the Kin voice agent. No LiveKit dependencies.

## Structure

```
backend/
├── main.py           # FastAPI application
├── tts_service.py    # Text-to-Speech service (ElevenLabs)
├── llm_service.py    # LLM service (OpenAI)
└── requirements.txt  # Python dependencies
```

## Services

### TTS Service (`tts_service.py`)
- Handles text-to-speech conversion using ElevenLabs WebSocket API
- Uses multi-context WebSocket for real-time streaming
- Cleans text (removes emojis, markdown, special characters)
- Returns MP3 audio data (streaming or complete)
- Supports configurable voice settings (stability, similarity, style, speaker boost)

### LLM Service (`llm_service.py`)
- Handles LLM interactions via OpenAI API
- Supports streaming and non-streaming responses
- Configurable persona and backstory

### Main Application (`main.py`)
- FastAPI application with REST endpoints
- Integrates TTS and LLM services
- Provides chat, streaming, and TTS endpoints

## Quick Start

1. Create and activate virtual environment:
   ```bash
   # Create venv (if not already created)
   python3 -m venv venv
   
   # Activate venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ELEVENLABS_VOICE_ID=your_voice_id
   ELEVENLABS_MODEL=eleven_turbo_v2_5
   OPENAI_API_KEY=your_openai_api_key
   PORT=8000
   HOST=0.0.0.0
   ```

4. Run the server:
   ```bash
   python main.py
   ```

5. Visit `http://localhost:8000/docs` for API documentation

## API Endpoints

### Chat Endpoints

- `POST /api/chat` - Accept audio file, transcribe with STT, get LLM response, return as audio MP3
- `POST /api/chat/text` - Accept text input, get LLM response as text only (backward compatibility)

### STT Endpoints

- `POST /api/stt` - Direct speech-to-text conversion

### TTS Endpoints

- `POST /api/tts` - Direct text-to-speech conversion (complete audio)
- `POST /api/tts/stream` - Streaming text-to-speech conversion (real-time audio chunks)

### Utility Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check

## Example Usage

### Chat with Audio Response (from audio file)

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -F "audio_file=@your_audio.mp3" \
  -F "persona=You are a caring assistant for elderly people." \
  -F "max_tokens=150" \
  --output response.mp3
```

### Chat Text Only

```bash
curl -X POST "http://localhost:8000/api/chat/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?"
  }'
```

### Direct STT (Speech-to-Text)

```bash
curl -X POST "http://localhost:8000/api/stt" \
  -F "audio_file=@your_audio.mp3" \
  | jq '.text'
```

### Direct TTS

```bash
curl -X POST "http://localhost:8000/api/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test."
  }' \
  --output tts.mp3
```

