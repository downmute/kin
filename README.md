Kin is an empathetic voice agent designed for the elderly. It assists caretakers with daily tasks, such as scheduling appointments, reminding users of medication, and providing assistance in emergencies.

## Architecture

Kin uses a simplified pipeline: **Speech-to-Text → LLM Generation → Text-to-Speech**

The backend is built with FastAPI and provides:
- Speech-to-Text via ElevenLabs
- LLM integration via OpenAI
- Text-to-Speech via ElevenLabs WebSocket API (real-time streaming)
- RESTful API endpoints for chat, STT, and TTS

The frontend is a simple web app that:
- Records audio from the microphone
- Sends audio to the backend
- Displays transcription and response
- Plays audio response

## Backend Setup

### Prerequisites

- Python 3.8+
- ElevenLabs API key
- OpenAI API key

### Installation

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your API keys:
   ```
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   ELEVENLABS_VOICE_ID=your_voice_id
   ELEVENLABS_MODEL=eleven_turbo_v2_5
   OPENAI_API_KEY=your_openai_api_key
   PORT=8000
   HOST=0.0.0.0
   ```

5. Run the server:
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### API Endpoints

- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `POST /api/chat` - Chat endpoint (accepts audio file, returns audio MP3)
- `POST /api/chat/text` - Chat endpoint (text input, text output - backward compatibility)
- `POST /api/stt` - Direct speech-to-text conversion
- `POST /api/tts` - Direct text-to-speech conversion (complete audio)
- `POST /api/tts/stream` - Streaming text-to-speech conversion (real-time audio chunks)

API documentation available at `/docs` when the server is running.

## Frontend Setup

1. Open `frontend/index.html` in a web browser, or serve it using a local web server:

   ```bash
   cd frontend
   python -m http.server 3000
   ```

2. Open `http://localhost:3000` in your browser

3. Grant microphone permissions when prompted

4. Click the record button to start/stop recording

See `frontend/README.md` for more details.
