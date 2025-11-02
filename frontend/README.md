# Kin Frontend

Simple web frontend for the Kin voice agent. Records audio from your microphone and sends it to the backend for processing.

## Features

- One-click recording
- Visual recording indicator
- Displays transcription and response
- Audio playback of response

## Setup

1. Make sure the backend is running on `http://localhost:8000`

2. Open `index.html` in a web browser, or serve it using a local web server:

   ```bash
   # Using Python
   cd frontend
   python -m http.server 3000
   
   # Or using Node.js http-server
   npx http-server -p 3000
   ```

3. Open `http://localhost:3000` in your browser

4. Grant microphone permissions when prompted

5. Click the record button to start/stop recording

## Usage

1. **Click to Record** - Starts recording your speech
2. While recording, the button will pulse and show "Recording..."
3. **Click to Stop** - Stops recording and sends audio to backend
4. Wait for processing (transcription → LLM → TTS)
5. View transcription and response text
6. Audio response will auto-play (if allowed by browser)

## Configuration

Edit `app.js` to change the API endpoint:

```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## Browser Compatibility

- Chrome/Edge: Full support
- Firefox: Full support
- Safari: May require additional configuration for WebM support

## Troubleshooting

- **No microphone access**: Check browser permissions
- **Recording not working**: Ensure you're using HTTPS or localhost
- **Backend errors**: Check that the backend is running and accessible

