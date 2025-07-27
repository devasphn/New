import torch
import asyncio
import json
import logging
import numpy as np
import warnings
import collections
import time
import librosa
import webrtcvad
import os
import sys
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from datetime import datetime

from aiohttp import web, WSMsgType

from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
import torch.hub

# --- Runpod Environment Detection ---
RUNPOD_POD_ID = os.environ.get('RUNPOD_POD_ID')
if not RUNPOD_POD_ID:
    import socket
    hostname = socket.gethostname()
    if len(hostname) > 8 and hostname != 'localhost':
        RUNPOD_POD_ID = hostname
    else:
        RUNPOD_POD_ID = 'local'

RUNPOD_PUBLIC_IP = os.environ.get('RUNPOD_PUBLIC_IP', '0.0.0.0')
RUNPOD_TCP_PORT_7860 = os.environ.get('RUNPOD_TCP_PORT_7860', '7860')

print(f"🚀 RUNPOD CONTINUOUS VOICE ASSISTANT")
print(f"📍 Pod ID: {RUNPOD_POD_ID}")
print(f"🌐 Public IP: {RUNPOD_PUBLIC_IP}")
print(f"🔌 TCP Port: {RUNPOD_TCP_PORT_7860}")

# --- Enhanced Logging Setup ---
def setup_runpod_logging():
    os.makedirs('/tmp/logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'/tmp/logs/continuous_voice_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    audio_logger = logging.getLogger('audio')
    model_logger = logging.getLogger('models')
    
    for logger_name in ['urllib3', 'requests']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger.info("🔧 Continuous voice assistant logging initialized")
    return logger, audio_logger, model_logger

logger, audio_logger, model_logger = setup_runpod_logging()

# --- Enhanced Setup ---
try:
    import uvloop
    uvloop.install()
    logger.info("🚀 Using uvloop for optimized event loop")
except ImportError:
    logger.warning("⚠️ uvloop not found, using default event loop")

warnings.filterwarnings("ignore")

# --- Global Variables ---
uv_pipe, tts_model = None, None
executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="audio_worker")
active_connections = set()

# --- Continuous Conversation HTML Client ---
def get_continuous_html_client():
    """Generate HTML client for continuous conversation"""
    
    if RUNPOD_POD_ID != 'local':
        public_url = f"https://{RUNPOD_POD_ID}-7860.proxy.runpod.net"
        ws_url = f"wss://{RUNPOD_POD_ID}-7860.proxy.runpod.net/ws"
    else:
        public_url = f"http://localhost:{RUNPOD_TCP_PORT_7860}"
        ws_url = f"ws://localhost:{RUNPOD_TCP_PORT_7860}/ws"
    
    logger.info(f"🌐 Client will connect to: {ws_url}")
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>🚀 Continuous Voice Assistant - Runpod</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff; display: flex; align-items: center; justify-content: center; min-height: 100vh;
        }}
        .container {{ 
            background: rgba(255,255,255,0.1); 
            backdrop-filter: blur(10px);
            padding: 40px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            text-align: center; max-width: 900px; width: 100%; border: 1px solid rgba(255,255,255,0.2);
        }}
        h1 {{ margin-bottom: 30px; font-weight: 300; font-size: 2.5em; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }}
        .runpod-info {{
            background: rgba(0,255,0,0.1); padding: 15px; border-radius: 10px; margin: 20px 0;
            border: 1px solid rgba(0,255,0,0.3);
        }}
        .controls {{ margin: 30px 0; }}
        button {{ 
            background: linear-gradient(45deg, #00c851, #007e33);
            color: white; border: none; padding: 18px 36px; font-size: 18px; font-weight: 600;
            border-radius: 50px; cursor: pointer; margin: 10px; transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2); text-transform: uppercase; letter-spacing: 1px;
        }}
        button:hover {{ transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }}
        button:disabled {{ 
            background: linear-gradient(45deg, #6c757d, #495057); cursor: not-allowed; 
            transform: none; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        .stop-btn {{ background: linear-gradient(45deg, #dc3545, #c82333); }}
        .active {{ background: linear-gradient(45deg, #ff6b35, #f7931e); animation: pulse 1s infinite; }}
        
        .status {{ 
            margin: 25px 0; padding: 20px; border-radius: 15px; font-weight: 600; font-size: 1.1em;
            transition: all 0.5s ease;
        }}
        .status.connected {{ background: linear-gradient(45deg, #28a745, #20c997); }}
        .status.disconnected {{ background: linear-gradient(45deg, #dc3545, #fd7e14); }}
        .status.listening {{ background: linear-gradient(45deg, #17a2b8, #20c997); animation: pulse 2s infinite; }}
        .status.processing {{ background: linear-gradient(45deg, #ffc107, #fd7e14); }}
        .status.speaking {{ background: linear-gradient(45deg, #007bff, #6610f2); }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        .conversation {{ 
            margin-top: 20px; padding: 20px; background: rgba(0,0,0,0.3); 
            border-radius: 15px; text-align: left; max-height: 350px; overflow-y: auto;
        }}
        .message {{ margin: 15px 0; padding: 15px; border-radius: 10px; }}
        .user-msg {{ background: rgba(0, 123, 255, 0.3); margin-left: 20px; }}
        .ai-msg {{ background: rgba(40, 167, 69, 0.3); margin-right: 20px; }}
        
        .metrics {{ 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 15px; margin: 20px 0;
        }}
        .metric {{ 
            padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px; text-align: center;
        }}
        .metric-value {{ font-size: 1.8em; font-weight: bold; color: #00ff88; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.8; margin-top: 5px; }}
        
        .debug {{ 
            margin-top: 15px; padding: 15px; background: rgba(0,0,0,0.2); 
            border-radius: 10px; font-family: 'Courier New', monospace; font-size: 11px;
            max-height: 150px; overflow-y: auto; text-align: left;
        }}
        
        .volume-meter {{
            width: 100%; height: 10px; background: rgba(0,0,0,0.3); border-radius: 5px; margin: 10px 0;
            overflow: hidden;
        }}
        .volume-bar {{
            height: 100%; background: linear-gradient(90deg, #00ff00, #ffff00, #ff0000);
            width: 0%; transition: width 0.1s ease;
        }}
        
        .conversation-hint {{
            background: rgba(255,255,0,0.1); padding: 10px; border-radius: 8px; margin: 10px 0;
            border: 1px solid rgba(255,255,0,0.3); font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Continuous Voice AI - Runpod</h1>
        
        <div class="runpod-info">
            <strong>🏃 Runpod Continuous Mode</strong><br>
            Pod ID: {RUNPOD_POD_ID}<br>
            WebSocket: {ws_url}<br>
            <small>✅ 2-Way Conversation Flow</small>
        </div>
        
        <div class="conversation-hint">
            💡 <strong>How it works:</strong> Start conversation → Speak naturally → AI responds automatically → Continue talking → End when done
        </div>
        
        <div class="controls">
            <button id="startBtn" onclick="startConversation()">🎙️ Start Conversation</button>
            <button id="stopBtn" onclick="stopConversation()" class="stop-btn" disabled>⏹️ End Conversation</button>
        </div>
        
        <div id="status" class="status disconnected">🔌 Disconnected</div>
        
        <div class="volume-meter">
            <div id="volumeBar" class="volume-bar"></div>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div id="latencyValue" class="metric-value">0ms</div>
                <div class="metric-label">Response Time</div>
            </div>
            <div class="metric">
                <div id="connectionValue" class="metric-value">Offline</div>
                <div class="metric-label">Connection</div>
            </div>
            <div class="metric">
                <div id="audioQuality" class="metric-value">-</div>
                <div class="metric-label">Audio Quality</div>
            </div>
        </div>
        
        <div id="conversation" class="conversation"></div>
        <div id="debug" class="debug">Continuous Voice Assistant ready...</div>
        
        <audio id="responseAudio" autoplay style="display: none;"></audio>
    </div>

    <script>
        let ws, mediaRecorder, audioContext, analyser, microphone, stream;
        let isConversationActive = false;
        let isRecording = false;
        let isProcessing = false;
        let isAISpeaking = false;
        let startTime;
        let audioChunks = [];
        let silenceTimer = null;
        let vadThreshold = 25; // Volume threshold for voice activity detection
        let silenceTimeout = 2500; // 2.5 seconds of silence before auto-stop
        let recordingChunkSize = 1000; // 1 second chunks
        
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const debugDiv = document.getElementById('debug');
        const conversationDiv = document.getElementById('conversation');
        const latencyValue = document.getElementById('latencyValue');
        const connectionValue = document.getElementById('connectionValue');
        const audioQuality = document.getElementById('audioQuality');
        const volumeBar = document.getElementById('volumeBar');
        const responseAudio = document.getElementById('responseAudio');

        function log(message) {{
            console.log(message);
            const timestamp = new Date().toLocaleTimeString();
            debugDiv.innerHTML += `${{timestamp}}: ${{message}}<br>`;
            debugDiv.scrollTop = debugDiv.scrollHeight;
            
            if (debugDiv.children.length > 50) {{
                debugDiv.innerHTML = debugDiv.innerHTML.split('<br>').slice(-40).join('<br>');
            }}
        }}

        function addMessage(text, isUser = false) {{
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{isUser ? 'user-msg' : 'ai-msg'}}`;
            messageDiv.innerHTML = `<strong>${{isUser ? '👤 You' : '🤖 AI'}}:</strong> ${{text}}`;
            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
            
            if (conversationDiv.children.length > 20) {{
                conversationDiv.removeChild(conversationDiv.firstChild);
            }}
        }}

        function updateStatus(message, className) {{
            statusDiv.textContent = message;
            statusDiv.className = `status ${{className}}`;
            log(`Status: ${{message}}`);
        }}

        function updateMetrics(latency, connection, quality) {{
            if (latency !== undefined) latencyValue.textContent = `${{latency}}ms`;
            if (connection !== undefined) connectionValue.textContent = connection;
            if (quality !== undefined) audioQuality.textContent = quality;
        }}

        function updateVolumeBar(volume) {{
            const percentage = Math.min(volume * 100, 100);
            volumeBar.style.width = `${{percentage}}%`;
        }}

        async function initializeWebSocket() {{
            try {{
                const wsUrl = '{ws_url}';
                log(`🌐 Connecting to WebSocket: ${{wsUrl}}`);
                
                ws = new WebSocket(wsUrl);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {{
                    log('✅ WebSocket connected');
                    updateStatus('🔌 Connected', 'connected');
                    updateMetrics(undefined, 'Connected', 'Ready');
                    startBtn.disabled = false;
                }};

                ws.onmessage = async (event) => {{
                    try {{
                        if (typeof event.data === 'string') {{
                            const data = JSON.parse(event.data);
                            handleTextMessage(data);
                        }} else {{
                            // Binary audio data
                            await handleAudioResponse(event.data);
                        }}
                    }} catch (err) {{
                        log(`❌ Message processing error: ${{err.message}}`);
                    }}
                }};

                ws.onclose = (event) => {{
                    log(`🔌 WebSocket closed: ${{event.code}}`);
                    updateStatus('🔌 Disconnected', 'disconnected');
                    updateMetrics(undefined, 'Disconnected', '-');
                    startBtn.disabled = true;
                    stopBtn.disabled = true;
                    isConversationActive = false;
                }};

                ws.onerror = (error) => {{
                    log(`❌ WebSocket error`);
                    updateStatus('❌ Connection Error', 'disconnected');
                }};

            }} catch (err) {{
                log(`❌ WebSocket initialization error: ${{err.message}}`);
            }}
        }}

        function handleTextMessage(data) {{
            switch(data.type) {{
                case 'transcription':
                    addMessage(data.text, true);
                    log(`📝 Transcription: ${{data.text}}`);
                    break;
                case 'response':
                    addMessage(data.text, false);
                    log(`🤖 AI Response: ${{data.text}}`);
                    break;
                case 'processing_start':
                    isProcessing = true;
                    updateStatus('🧠 Processing...', 'processing');
                    break;
                case 'audio_ready':
                    isAISpeaking = true;
                    updateStatus('🔊 AI Speaking...', 'speaking');
                    break;
                case 'audio_ended':
                    isAISpeaking = false;
                    isProcessing = false;
                    if (isConversationActive) {{
                        // Resume listening after AI finishes speaking
                        setTimeout(() => {{
                            if (isConversationActive && !isProcessing) {{
                                startListening();
                            }}
                        }}, 500);
                    }}
                    break;
                case 'error':
                    log(`❌ Server error: ${{data.message}}`);
                    isProcessing = false;
                    isAISpeaking = false;
                    if (isConversationActive) {{
                        startListening();
                    }}
                    break;
            }}
        }}

        async function handleAudioResponse(audioData) {{
            try {{
                isAISpeaking = true;
                const audioBlob = new Blob([audioData], {{ type: 'audio/wav' }});
                const audioUrl = URL.createObjectURL(audioBlob);
                
                responseAudio.src = audioUrl;
                
                responseAudio.onended = () => {{
                    URL.revokeObjectURL(audioUrl);
                    isAISpeaking = false;
                    isProcessing = false;
                    
                    if (startTime) {{
                        const totalLatency = Date.now() - startTime;
                        updateMetrics(totalLatency, 'Connected', 'Excellent');
                        log(`⚡ Total latency: ${{totalLatency}}ms`);
                    }}
                    
                    // Send audio ended signal to server
                    if (ws && ws.readyState === WebSocket.OPEN) {{
                        ws.send(JSON.stringify({{ type: 'audio_ended' }}));
                    }}
                    
                    // Resume listening if conversation is still active
                    if (isConversationActive) {{
                        setTimeout(() => {{
                            if (isConversationActive && !isProcessing && !isAISpeaking) {{
                                startListening();
                            }}
                        }}, 500);
                    }}
                }};
                
                await responseAudio.play();
                
            }} catch (err) {{
                log(`❌ Audio playback error: ${{err.message}}`);
                isAISpeaking = false;
                isProcessing = false;
                if (isConversationActive) {{
                    startListening();
                }}
            }}
        }}

        async function startConversation() {{
            try {{
                log('🎙️ Starting continuous conversation...');
                isConversationActive = true;
                
                startBtn.disabled = true;
                stopBtn.disabled = false;
                startBtn.classList.add('active');
                
                updateStatus('🎤 Initializing microphone...', 'connecting');
                
                // Get microphone access
                stream = await navigator.mediaDevices.getUserMedia({{
                    audio: {{
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: 16000
                    }}
                }});

                // Setup audio analysis
                audioContext = new AudioContext({{ sampleRate: 16000 }});
                analyser = audioContext.createAnalyser();
                microphone = audioContext.createMediaStreamSource(stream);
                microphone.connect(analyser);
                
                analyser.fftSize = 256;
                
                log('✅ Continuous conversation started');
                updateMetrics(undefined, 'Connected', 'Excellent');
                
                // Start listening
                startListening();
                
            }} catch (err) {{
                log(`❌ Conversation start error: ${{err.message}}`);
                updateStatus('❌ Microphone Error', 'disconnected');
                stopConversation();
            }}
        }}

        async function startListening() {{
            if (!isConversationActive || isRecording || isProcessing || isAISpeaking) {{
                return;
            }}
            
            try {{
                log('👂 Starting to listen...');
                updateStatus('👂 Listening for speech...', 'listening');
                
                // Setup MediaRecorder for this listening session
                mediaRecorder = new MediaRecorder(stream, {{
                    mimeType: 'audio/webm;codecs=opus'
                }});
                
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {{
                    if (event.data.size > 0) {{
                        audioChunks.push(event.data);
                    }}
                }};
                
                mediaRecorder.onstop = async () => {{
                    if (!isConversationActive) return;
                    
                    log('🎤 Processing recorded audio...');
                    
                    const audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    
                    if (ws && ws.readyState === WebSocket.OPEN && arrayBuffer.byteLength > 1000) {{
                        startTime = Date.now();
                        isProcessing = true;
                        updateStatus('📤 Sending audio...', 'processing');
                        ws.send(arrayBuffer);
                    }} else if (isConversationActive) {{
                        // No significant audio, resume listening
                        setTimeout(() => startListening(), 500);
                    }}
                }};
                
                // Start recording
                mediaRecorder.start();
                isRecording = true;
                
                // Monitor volume and detect speech
                monitorSpeech();
                
            }} catch (err) {{
                log(`❌ Listening error: ${{err.message}}`);
                if (isConversationActive) {{
                    setTimeout(() => startListening(), 1000);
                }}
            }}
        }}

        function monitorSpeech() {{
            if (!isRecording || !isConversationActive) return;
            
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            function checkVolume() {{
                if (!isRecording || !isConversationActive) return;
                
                analyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((a, b) => a + b) / bufferLength;
                const volume = average / 255 * 100;
                
                updateVolumeBar(volume / 100);
                
                // Voice activity detection
                if (volume > vadThreshold) {{
                    // Speech detected, clear silence timer
                    if (silenceTimer) {{
                        clearTimeout(silenceTimer);
                        silenceTimer = null;
                    }}
                }} else {{
                    // Silence detected, start timer if not already started
                    if (!silenceTimer) {{
                        silenceTimer = setTimeout(() => {{
                            if (isRecording && isConversationActive) {{
                                log('🔇 Silence detected, stopping recording');
                                stopCurrentRecording();
                            }}
                        }}, silenceTimeout);
                    }}
                }}
                
                requestAnimationFrame(checkVolume);
            }}
            
            checkVolume();
        }}

        function stopCurrentRecording() {{
            if (mediaRecorder && isRecording) {{
                mediaRecorder.stop();
                isRecording = false;
                
                if (silenceTimer) {{
                    clearTimeout(silenceTimer);
                    silenceTimer = null;
                }}
                
                updateVolumeBar(0);
            }}
        }}

        function stopConversation() {{
            log('🛑 Stopping conversation...');
            isConversationActive = false;
            isRecording = false;
            isProcessing = false;
            isAISpeaking = false;
            
            // Stop current recording
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {{
                mediaRecorder.stop();
            }}
            
            // Clear timers
            if (silenceTimer) {{
                clearTimeout(silenceTimer);
                silenceTimer = null;
            }}
            
            // Stop media stream
            if (stream) {{
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }}
            
            // Close audio context
            if (audioContext) {{
                audioContext.close();
                audioContext = null;
            }}
            
            // Stop audio playback
            if (responseAudio) {{
                responseAudio.pause();
                responseAudio.src = '';
            }}
            
            // Reset UI
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startBtn.classList.remove('active');
            
            updateStatus('🔌 Connected', 'connected');
            updateVolumeBar(0);
            
            log('✅ Conversation ended');
        }}

        // Initialize on page load
        window.addEventListener('load', () => {{
            log('🚀 Continuous Voice Assistant initialized');
            initializeWebSocket();
        }});

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {{
            stopConversation();
            if (ws) {{
                ws.close();
            }}
        }});
    </script>
</body>
</html>
"""#
 --- Enhanced VAD System ---
class ImprovedVAD:
    def __init__(self):
        self.webrtc_vad = webrtcvad.Vad(2)
        self.silero_model = None
        self._load_silero()
        
    def _load_silero(self):
        try:
            audio_logger.info("🎤 Loading Silero VAD...")
            self.silero_model, utils = torch.hub.load(
                'snakers4/silero-vad', 
                'silero_vad', 
                force_reload=False,
                verbose=False
            )
            self.get_speech_timestamps = utils[0]
            audio_logger.info("✅ Silero VAD loaded")
        except Exception as e:
            audio_logger.error(f"❌ Silero VAD loading failed: {e}")
            self.silero_model = None
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        if len(audio) == 0:
            return False
            
        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy < 0.001:
            return False
            
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
        webrtc_detected = self._webrtc_vad_check(audio)
        silero_detected = True
        if self.silero_model is not None:
            silero_detected = self._silero_vad_check(audio)
        
        result = webrtc_detected and silero_detected
        audio_logger.debug(f"VAD: energy={rms_energy:.6f}, result={result}")
        return result
    
    def _webrtc_vad_check(self, audio: np.ndarray) -> bool:
        try:
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            frame_length = 320
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_int16) - frame_length + 1, frame_length):
                frame = audio_int16[i:i + frame_length]
                if len(frame) == frame_length:
                    try:
                        if self.webrtc_vad.is_speech(frame.tobytes(), 16000):
                            speech_frames += 1
                    except:
                        pass
                    total_frames += 1
            
            if total_frames > 0:
                speech_ratio = speech_frames / total_frames
                return speech_ratio > 0.25  # Lower threshold for continuous mode
                
            return False
        except Exception as e:
            audio_logger.debug(f"WebRTC VAD error: {e}")
            return False
    
    def _silero_vad_check(self, audio: np.ndarray) -> bool:
        try:
            audio_tensor = torch.from_numpy(audio)
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.silero_model,
                sampling_rate=16000,
                min_speech_duration_ms=150,  # Shorter for continuous mode
                threshold=0.25  # Lower threshold
            )
            return len(speech_timestamps) > 0
        except Exception as e:
            audio_logger.debug(f"Silero VAD error: {e}")
            return True

# --- Continuous Audio Processing Pipeline ---
class ContinuousAudioProcessor:
    def __init__(self):
        self.vad = ImprovedVAD()
        
    async def process_audio_data(self, audio_data: bytes) -> Tuple[str, np.ndarray]:
        """Process audio data and return transcription and TTS audio"""
        try:
            # Convert WebM/Opus to numpy array
            audio_array = await self._convert_audio_to_numpy(audio_data)
            
            if audio_array is None or len(audio_array) == 0:
                return "", np.array([])
            
            # Check for speech with more lenient threshold
            if not self.vad.detect_speech(audio_array):
                audio_logger.info("⚠️ No speech detected in audio")
                return "", np.array([])
            
            audio_logger.info(f"🎯 Processing {len(audio_array)/16000:.2f}s of audio")
            
            # Run STT
            transcription = await self._run_stt(audio_array)
            if not transcription:
                return "", np.array([])
            
            # Generate AI response (simple echo for now, can be enhanced)
            ai_response = await self._generate_ai_response(transcription)
            
            # Run TTS
            tts_audio = await self._run_tts(ai_response)
            
            return transcription, tts_audio
            
        except Exception as e:
            audio_logger.error(f"❌ Audio processing error: {e}")
            return "", np.array([])
    
    async def _convert_audio_to_numpy(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Convert WebM/Opus audio to numpy array"""
        try:
            import tempfile
            import subprocess
            
            # Save audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input_path = temp_input.name
            
            # Convert to WAV using ffmpeg
            temp_output_path = temp_input_path.replace('.webm', '.wav')
            
            cmd = [
                'ffmpeg', '-i', temp_input_path,
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # Mono
                '-f', 'wav',
                '-y',            # Overwrite output
                temp_output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                audio_logger.error(f"FFmpeg error: {result.stderr}")
                return None
            
            # Load converted audio
            audio_array, sr = librosa.load(temp_output_path, sr=16000, mono=True)
            
            # Cleanup
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
            
            return audio_array.astype(np.float32)
            
        except Exception as e:
            audio_logger.error(f"❌ Audio conversion error: {e}")
            return None
    
    async def _run_stt(self, audio_array: np.ndarray) -> str:
        """Run speech-to-text"""
        try:
            loop = asyncio.get_running_loop()
            
            def _inference():
                with torch.inference_mode():
                    result = uv_pipe({
                        'audio': audio_array,
                        'turns': [],
                        'sampling_rate': 16000
                    }, max_new_tokens=50, do_sample=False, temperature=0.1)
                    
                    text = ""
                    if isinstance(result, list) and len(result) > 0:
                        item = result[0]
                        if isinstance(item, dict) and 'generated_text' in item:
                            text = item['generated_text']
                        elif isinstance(item, str):
                            text = item
                    elif isinstance(result, str):
                        text = result
                    
                    return text.strip() if text else ""
            
            start_time = time.time()
            text = await loop.run_in_executor(executor, _inference)
            stt_time = time.time() - start_time
            
            model_logger.info(f"🧠 STT completed: {stt_time*1000:.0f}ms, text: '{text}'")
            return text
            
        except Exception as e:
            model_logger.error(f"❌ STT error: {e}")
            return ""
    
    async def _generate_ai_response(self, user_text: str) -> str:
        """Generate AI response (can be enhanced with actual AI model)"""
        try:
            # Simple response generation - can be replaced with actual AI model
            responses = [
                f"I heard you say: {user_text}",
                f"That's interesting! You mentioned: {user_text}",
                f"Thanks for sharing: {user_text}",
                f"I understand you said: {user_text}",
                f"Got it! You told me: {user_text}"
            ]
            
            import random
            response = random.choice(responses)
            
            model_logger.info(f"🤖 AI Response: '{response}'")
            return response
            
        except Exception as e:
            model_logger.error(f"❌ AI response generation error: {e}")
            return f"I heard: {user_text}"
    
    async def _run_tts(self, text: str) -> np.ndarray:
        """Run text-to-speech"""
        try:
            if not text.strip():
                return np.array([])
            
            loop = asyncio.get_running_loop()
            
            def _tts_inference():
                with torch.inference_mode():
                    wav = tts_model.generate(text)
                    
                    if hasattr(wav, 'cpu'):
                        wav = wav.cpu().numpy()
                    elif torch.is_tensor(wav):
                        wav = wav.numpy()
                    
                    wav = wav.flatten().astype(np.float32)
                    
                    # Normalize
                    if np.max(np.abs(wav)) > 0:
                        wav = wav / max(np.max(np.abs(wav)), 0.1) * 0.7
                    
                    return wav
            
            start_time = time.time()
            audio_output = await loop.run_in_executor(executor, _tts_inference)
            tts_time = time.time() - start_time
            
            model_logger.info(f"🔊 TTS completed: {tts_time*1000:.0f}ms, {len(audio_output)/24000:.2f}s audio")
            return audio_output
            
        except Exception as e:
            model_logger.error(f"❌ TTS error: {e}")
            return np.array([])

# --- Model Initialization ---
def initialize_models() -> bool:
    global uv_pipe, tts_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_logger.info(f"🚀 Initializing models on device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        model_logger.info(f"🎮 GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB")
    
    try:
        # Load Ultravox
        model_logger.info("📥 Loading Ultravox model...")
        start_time = time.time()
        
        uv_pipe = pipeline(
            model="fixie-ai/ultravox-v0_4",
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            return_tensors="pt"
        )
        
        load_time = time.time() - start_time
        model_logger.info(f"✅ Ultravox loaded in {load_time:.1f}s")
        
        # Load TTS
        model_logger.info("📥 Loading ChatterboxTTS model...")
        start_time = time.time()
        
        tts_model = ChatterboxTTS.from_pretrained(device=device)
        
        load_time = time.time() - start_time
        model_logger.info(f"✅ ChatterboxTTS loaded in {load_time:.1f}s")
        
        # Warmup
        model_logger.info("🔥 Warming up models...")
        dummy = np.random.randn(8000).astype(np.float32) * 0.01
        
        with torch.inference_mode():
            try:
                start_time = time.time()
                uv_pipe({'audio': dummy, 'turns': [], 'sampling_rate': 16000}, max_new_tokens=5)
                warmup_time = time.time() - start_time
                model_logger.info(f"✅ Ultravox warmed up in {warmup_time*1000:.0f}ms")
            except Exception as e:
                model_logger.warning(f"⚠️ Ultravox warmup issue: {e}")
            
            try:
                start_time = time.time()
                tts_model.generate("Test")
                warmup_time = time.time() - start_time
                model_logger.info(f"✅ TTS warmed up in {warmup_time*1000:.0f}ms")
            except Exception as e:
                model_logger.warning(f"⚠️ TTS warmup issue: {e}")
        
        model_logger.info("🎉 All models ready!")
        return True
        
    except Exception as e:
        model_logger.error(f"❌ Model initialization failed: {e}", exc_info=True)
        return False

# --- Continuous WebSocket Handler ---
async def websocket_handler(request):
    """WebSocket handler for continuous conversation"""
    ws = web.WebSocketResponse(heartbeat=30, timeout=300)  # Longer timeout for conversations
    await ws.prepare(request)
    
    client_ip = request.remote
    logger.info(f"🌐 Continuous WebSocket connection from {client_ip}")
    active_connections.add(ws)
    
    processor = ContinuousAudioProcessor()
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                # Audio data received
                audio_data = msg.data
                logger.info(f"📥 Received {len(audio_data)} bytes of audio from {client_ip}")
                
                try:
                    # Send processing start notification
                    await ws.send_json({'type': 'processing_start'})
                    
                    # Process audio
                    transcription, tts_audio = await processor.process_audio_data(audio_data)
                    
                    if transcription:
                        # Send transcription
                        await ws.send_json({
                            'type': 'transcription',
                            'text': transcription
                        })
                        
                        # Send AI response text
                        await ws.send_json({
                            'type': 'response', 
                            'text': transcription  # For now, echo back
                        })
                        
                        if len(tts_audio) > 0:
                            # Convert TTS audio to WAV format
                            wav_data = await convert_to_wav(tts_audio)
                            
                            # Send audio ready notification
                            await ws.send_json({'type': 'audio_ready'})
                            
                            # Send audio data
                            await ws.send_bytes(wav_data)
                            
                            logger.info(f"✅ Sent {len(wav_data)} bytes of TTS audio to {client_ip}")
                        else:
                            await ws.send_json({
                                'type': 'error',
                                'message': 'No audio generated'
                            })
                    else:
                        await ws.send_json({
                            'type': 'error',
                            'message': 'No speech detected or transcription failed'
                        })
                        
                except Exception as e:
                    logger.error(f"❌ Audio processing error for {client_ip}: {e}")
                    await ws.send_json({
                        'type': 'error',
                        'message': f'Processing error: {str(e)}'
                    })
                    
            elif msg.type == WSMsgType.TEXT:
                # Text message handling
                try:
                    data = json.loads(msg.data)
                    if data.get('type') == 'audio_ended':
                        # Client finished playing audio
                        await ws.send_json({'type': 'audio_ended'})
                        logger.debug(f"📢 Audio playback ended for {client_ip}")
                    else:
                        logger.info(f"📝 Text message from {client_ip}: {data}")
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON from {client_ip}")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"❌ WebSocket error from {client_ip}: {ws.exception()}")
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket handler error for {client_ip}: {e}")
    finally:
        logger.info(f"🔚 Continuous WebSocket connection closed for {client_ip}")
        active_connections.discard(ws)
    
    return ws

async def convert_to_wav(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert numpy audio array to WAV bytes"""
    try:
        import io
        import wave
        
        # Convert to int16
        audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_data = wav_buffer.getvalue()
        wav_buffer.close()
        
        return wav_data
        
    except Exception as e:
        logger.error(f"❌ WAV conversion error: {e}")
        return b''

# --- HTTP Handlers ---
async def index_handler(request):
    html_content = get_continuous_html_client()
    return web.Response(
        text=html_content,
        content_type='text/html',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )

async def health_handler(request):
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(0),
            "memory_cached": torch.cuda.memory_reserved(0)
        }
    else:
        gpu_info = {"available": False}
    
    return web.json_response({
        "status": "healthy",
        "mode": "continuous_conversation",
        "runpod": {
            "pod_id": RUNPOD_POD_ID,
            "public_ip": RUNPOD_PUBLIC_IP,
            "tcp_port": RUNPOD_TCP_PORT_7860
        },
        "models": {
            "ultravox": uv_pipe is not None,
            "tts": tts_model is not None
        },
        "connections": len(active_connections),
        "gpu": gpu_info,
        "timestamp": datetime.now().isoformat()
    })

# --- Application ---
async def on_shutdown(app):
    logger.info("🛑 Shutting down continuous voice assistant...")
    
    # Close active connections
    for ws in list(active_connections):
        if not ws.closed:
            await ws.close()
    active_connections.clear()
    
    executor.shutdown(wait=True)
    logger.info("✅ Shutdown complete")

async def main():
    if not initialize_models():
        logger.error("❌ Model initialization failed")
        return
    
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/health', health_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    port = int(RUNPOD_TCP_PORT_7860)
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    if RUNPOD_POD_ID != 'local':
        public_url = f"https://{RUNPOD_POD_ID}-7860.proxy.runpod.net"
    else:
        public_url = f"http://0.0.0.0:{port}"
    
    print("\n" + "="*80)
    print("🚀 CONTINUOUS VOICE ASSISTANT - RUNPOD OPTIMIZED")
    print("="*80)
    print(f"🏃 Runpod Pod ID: {RUNPOD_POD_ID}")
    print(f"📡 Public URL: {public_url}")
    print(f"🔗 Health Check: {public_url}/health")
    print(f"🎯 Mode: Continuous 2-Way Conversation")
    print(f"🧠 GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU Only'}")
    print(f"🔊 TTS: ✅ ChatterboxTTS")
    print(f"🎤 STT: ✅ Ultravox")
    print(f"📝 Logging: ✅ Enhanced")
    print(f"🌐 WebSocket: ✅ Continuous Audio Streaming")
    print(f"💬 Conversation: ✅ Auto-Resume After AI Response")
    print("="*80)
    print("🛑 Press Ctrl+C to stop")
    print("="*80 + "\n")
    
    logger.info(f"🎉 Continuous voice assistant started")
    logger.info(f"📡 Accessible at: {public_url}")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Continuous voice assistant stopped")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}", exc_info=True)
        sys.exit(1)
