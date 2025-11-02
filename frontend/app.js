// Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_CHAT_ENDPOINT = `${API_BASE_URL}/api/chat`;

// DOM elements
const recordBtn = document.getElementById('recordBtn');
const statusDiv = document.getElementById('status');
const recordingIndicator = document.getElementById('recordingIndicator');
const transcriptionDiv = document.getElementById('transcription');
const transcriptionText = document.getElementById('transcriptionText');
const responseDiv = document.getElementById('response');
const responseText = document.getElementById('responseText');
const audioPlayer = document.getElementById('audioPlayer');
const errorDiv = document.getElementById('error');

// State
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// Initialize
async function init() {
    try {
        // Check if browser supports MediaRecorder
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('Your browser does not support audio recording. Please use a modern browser like Chrome, Firefox, or Edge.');
            recordBtn.disabled = true;
            return;
        }

        recordBtn.addEventListener('click', handleRecordClick);
    } catch (error) {
        console.error('Initialization error:', error);
        showError('Failed to initialize recording. Please check your browser permissions.');
    }
}

// Handle record button click
async function handleRecordClick() {
    if (!isRecording) {
        await startRecording();
    } else {
        await stopRecording();
    }
}

// Start recording
async function startRecording() {
    try {
        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Create MediaRecorder
        mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'audio/webm' // WebM is widely supported
        });
        
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = async () => {
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
            
            // Process recorded audio
            await processRecording();
        };
        
        // Start recording
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        recordBtn.classList.add('recording');
        recordBtn.querySelector('.button-text').textContent = 'Click to Stop';
        recordingIndicator.classList.remove('hidden');
        hideError();
        hideStatus();
        hideTranscription();
        hideResponse();
        
    } catch (error) {
        console.error('Error starting recording:', error);
        showError('Failed to start recording. Please check your microphone permissions.');
        isRecording = false;
    }
}

// Stop recording
async function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Update UI
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('.button-text').textContent = 'Click to Record';
        recordingIndicator.classList.add('hidden');
    }
}

// Process recorded audio
async function processRecording() {
    try {
        showStatus('Processing audio...', 'processing');
        recordBtn.disabled = true;
        
        // Convert audio chunks to blob
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        
        // Create FormData for multipart upload
        const formData = new FormData();
        formData.append('audio_file', audioBlob, 'recording.webm');
        
        // Optional: Add persona and backstory
        // formData.append('persona', 'You are a caring assistant for elderly people.');
        // formData.append('max_tokens', '150');
        
        // Send to backend
        const response = await fetch(API_CHAT_ENDPOINT, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }
        
        // Get transcription and response from headers
        const transcription = response.headers.get('X-Transcription') || 'No transcription available';
        const textResponse = response.headers.get('X-Text-Response') || 'No response available';
        
        // Get audio blob
        const audioBlobResponse = await response.blob();
        
        // Create audio URL and play
        const audioUrl = URL.createObjectURL(audioBlobResponse);
        audioPlayer.src = audioUrl;
        
        // Update UI
        transcriptionText.textContent = transcription;
        responseText.textContent = textResponse;
        
        showTranscription();
        showResponse();
        showStatus('Response received!', 'success');
        
        // Auto-play audio
        audioPlayer.play().catch(err => {
            console.warn('Auto-play prevented:', err);
            showStatus('Response received! Click play to listen.', 'success');
        });
        
    } catch (error) {
        console.error('Error processing recording:', error);
        showError(`Error: ${error.message}`);
        showStatus('Failed to process recording.', 'error');
    } finally {
        recordBtn.disabled = false;
    }
}

// UI Helper functions
function showStatus(message, type = '') {
    statusDiv.textContent = message;
    statusDiv.className = `status ${type}`;
    statusDiv.classList.remove('hidden');
}

function hideStatus() {
    statusDiv.classList.add('hidden');
}

function showTranscription() {
    transcriptionDiv.classList.remove('hidden');
}

function hideTranscription() {
    transcriptionDiv.classList.add('hidden');
}

function showResponse() {
    responseDiv.classList.remove('hidden');
}

function hideResponse() {
    responseDiv.classList.add('hidden');
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    errorDiv.classList.add('hidden');
}

// Initialize on page load
init();

