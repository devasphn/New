import torch
import asyncio
import json
import logging
import numpy as np
import warnings
import time
import librosa
import webrtcvad
import os
import sys
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from datetime import datetime
import tempfile
import subprocess
import io
import wave

from aiohttp import web, WSMsgType
from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torchaudio
import torchaudio.functional as F

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

print(f"🚀 VOXTRAL + SIMPLE TTS WEBSOCKET ASSISTANT")
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
            logging.FileHandler(f'/tmp/logs/voxtral_simple_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    audio_logger = logging.getLogger('audio')
    model_logger = logging.getLogger('models')
    
    for logger_name in ['urllib3', 'requests', 'transformers']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger.info("🔧 Voxtral + Simple TTS WebSocket assistant logging initialized")
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
voxtral_processor, voxtral_model = None, None
executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="audio_worker")
active_connections = set()

# --- Enhanced VAD System ---
class ImprovedVAD:
    def __init__(self):
        self.webrtc_vad = webrtcvad.Vad(2)
        
    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        if len(audio) == 0:
            return False
            
        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy < 0.001:
            return False
            
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
        return self._webrtc_vad_check(audio)
    
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
                return speech_ratio > 0.3
                
            return False
        except Exception as e:
            audio_logger.debug(f"WebRTC VAD error: {e}")
            return False

# --- Simple TTS Implementation ---
class SimpleTTS:
    """Simple TTS implementation as placeholder"""
    
    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate
        
    def synthesize(self, text: str) -> np.ndarray:
        """Generate simple sine wave based TTS"""
        if not text.strip():
            return np.array([])
        
        words = text.split()
        duration = len(words) * 0.4 + 0.5  # 0.4s per word + 0.5s buffer
        
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Create pleasant multi-tone audio
        freq_base = 220  # A3 note
        audio = 0.3 * np.sin(2 * np.pi * freq_base * t)  # Fundamental
        audio += 0.15 * np.sin(2 * np.pi * freq_base * 1.5 * t)  # Perfect fifth
        audio += 0.1 * np.sin(2 * np.pi * freq_base * 2 * t)   # Octave
        
        # Add envelope
        envelope = np.exp(-t * 0.8)
        audio *= envelope
        
        # Add slight frequency modulation
        modulation = 1 + 0.05 * np.sin(2 * np.pi * 3 * t)
        audio *= modulation
        
        return audio.astype(np.float32)

# --- Audio Processing Pipeline ---
class VoxtralProcessor:
    def __init__(self):
        self.vad = ImprovedVAD()
        self.tts = SimpleTTS()
        
    async def process_audio_data(self, audio_data: bytes) -> Tuple[str, np.ndarray]:
        """Process audio data and return transcription and TTS audio"""
        try:
            # Convert WebM/Opus to numpy array
            audio_array = await self._convert_audio_to_numpy(audio_data)
            
            if audio_array is None or len(audio_array) == 0:
                return "", np.array([])
            
            # Check for speech
            if not self.vad.detect_speech(audio_array):
                audio_logger.info("⚠️ No speech detected in audio")
                return "", np.array([])
            
            audio_logger.info(f"🎯 Processing {len(audio_array)/16000:.2f}s of audio")
            
            # Run Voxtral ASR+LLM
            transcription = await self._run_voxtral(audio_array)
            if not transcription:
                return "", np.array([])
            
            # Run TTS
            tts_audio = await self._run_tts(transcription)
            
            return transcription, tts_audio
            
        except Exception as e:
            audio_logger.error(f"❌ Audio processing error: {e}")
            return "", np.array([])
    
    async def _convert_audio_to_numpy(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Convert WebM/Opus audio to numpy array"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input_path = temp_input.name
            
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
            
            audio_array, sr = librosa.load(temp_output_path, sr=16000, mono=True)
            
            # Cleanup
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
            
            return audio_array.astype(np.float32)
            
        except Exception as e:
            audio_logger.error(f"❌ Audio conversion error: {e}")
            return None
    
    async def _run_voxtral(self, audio_array: np.ndarray) -> str:
        """Run Voxtral ASR+LLM with correct conversation format"""
        try:
            loop = asyncio.get_running_loop()
            
            def _inference():
                with torch.inference_mode():
                    # Correct conversation format for Voxtral
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "audio": audio_array,
                                    "sample_rate": 16000
                                }
                            ]
                        }
                    ]
                    
                    # Process with Voxtral using correct method
                    inputs = voxtral_processor.apply_chat_template(conversation)
                    inputs = inputs.to(voxtral_model.device, dtype=torch.bfloat16)
                    
                    outputs = voxtral_model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.2,
                        do_sample=True,
                        pad_token_id=voxtral_processor.tokenizer.eos_token_id
                    )
                    
                    # Decode response
                    decoded_outputs = voxtral_processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    return decoded_outputs[0].strip() if decoded_outputs else ""
            
            start_time = time.time()
            text = await loop.run_in_executor(executor, _inference)
            stt_time = time.time() - start_time
            
            model_logger.info(f"🧠 Voxtral completed: {stt_time*1000:.0f}ms, text: '{text}'")
            return text
            
        except Exception as e:
            model_logger.error(f"❌ Voxtral error: {e}")
            return ""
    
    async def _run_tts(self, text: str) -> np.ndarray:
        """Run TTS"""
        try:
            if not text.strip():
                return np.array([])
            
            loop = asyncio.get_running_loop()
            
            def _tts_inference():
                return self.tts.synthesize(text)
            
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
    global voxtral_processor, voxtral_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_logger.info(f"🚀 Initializing models on device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        model_logger.info(f"🎮 GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB")
    
    try:
        # Load Voxtral Model with correct imports
        model_logger.info("📥 Loading Voxtral model...")
        start_time = time.time()
        
        repo_id = "mistralai/Voxtral-Mini-3B-2507"
        
        # Load processor and model correctly
        voxtral_processor = AutoProcessor.from_pretrained(repo_id)
        voxtral_model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        load_time = time.time() - start_time
        model_logger.info(f"✅ Voxtral loaded in {load_time:.1f}s")
        
        # Warmup
        model_logger.info("🔥 Warming up Voxtral...")
        try:
            dummy_audio = np.random.randn(8000).astype(np.float32) * 0.01
            
            conversation = [{
                "role": "user",
                "content": [{"type": "audio", "audio": dummy_audio, "sample_rate": 16000}]
            }]
            
            inputs = voxtral_processor.apply_chat_template(conversation)
            inputs = inputs.to(voxtral_model.device, dtype=torch.bfloat16)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = voxtral_model.generate(**inputs, max_new_tokens=5)
            warmup_time = time.time() - start_time
            model_logger.info(f"✅ Voxtral warmed up in {warmup_time*1000:.0f}ms")
            
        except Exception as e:
            model_logger.warning(f"⚠️ Voxtral warmup issue: {e}")
        
        model_logger.info("🎉 All models ready!")
        return True
        
    except Exception as e:
        model_logger.error(f"❌ Model initialization failed: {e}", exc_info=True)
        return False

# --- WebSocket Handler ---
async def websocket_handler(request):
    """WebSocket handler for audio streaming"""
    ws = web.WebSocketResponse(heartbeat=30, timeout=120)
    await ws.prepare(request)
    
    client_ip = request.remote
    logger.info(f"🌐 WebSocket connection from {client_ip}")
    active_connections.add(ws)
    
    processor = VoxtralProcessor()
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                audio_data = msg.data
                logger.info(f"📥 Received {len(audio_data)} bytes of audio from {client_ip}")
                
                try:
                    await ws.send_json({'type': 'processing_start'})
                    
                    transcription, tts_audio = await processor.process_audio_data(audio_data)
                    
                    if transcription:
                        await ws.send_json({
                            'type': 'transcription',
                            'text': transcription
                        })
                        
                        await ws.send_json({
                            'type': 'response', 
                            'text': transcription
                        })
                        
                        if len(tts_audio) > 0:
                            wav_data = await convert_to_wav(tts_audio, sample_rate=24000)
                            await ws.send_json({'type': 'audio_ready'})
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
                try:
                    data = json.loads(msg.data)
                    logger.info(f"📝 Text message from {client_ip}: {data}")
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON from {client_ip}")
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"❌ WebSocket error from {client_ip}: {ws.exception()}")
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket handler error for {client_ip}: {e}")
    finally:
        logger.info(f"🔚 WebSocket connection closed for {client_ip}")
        active_connections.discard(ws)
    
    return ws

async def convert_to_wav(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert numpy audio array to WAV bytes"""
    try:
        audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
        
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

# --- HTML Client ---
def get_websocket_html_client():
    """Generate HTML client for WebSocket audio streaming"""
    
    if RUNPOD_POD_ID != 'local':
        public_url = f"https://{RUNPOD_POD_ID}-7860.proxy.runpod.net"
        ws_url = f"wss://{RUNPOD_POD_ID}-7860.proxy.runpod.net/ws"
    else:
        public_url = f"http://localhost:{RUNPOD_TCP_PORT_7860}"
        ws_url = f"ws://localhost:{RUNPOD_TCP_PORT_7860}/ws"
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <title>🚀 Voxtral WebSocket Assistant</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff; display: flex; align-items: center; justify-content: center; min-height: 100vh;
        }}
        .container {{ 
            background: rgba(255,255,255,0.1); backdrop-filter: blur(10px);
            padding: 40px; border-radius: 20px; box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            text-align: center; max-width: 900px; width: 100%;
        }}
        h1 {{ margin-bottom: 30px; font-weight: 300; font-size: 2.5em; }}
        .model-info {{
            background: rgba(0,255,0,0.1); padding: 15px; border-radius: 10px; margin: 20px 0;
        }}
        button {{ 
            background: linear-gradient(45deg, #00c851, #007e33);
            color: white; border: none; padding: 18px 36px; font-size: 18px;
            border-radius: 50px; cursor: pointer; margin: 10px;
        }}
        .stop-btn {{ background: linear-gradient(45deg, #dc3545, #c82333); }}
        .status {{ margin: 25px 0; padding: 20px; border-radius: 15px; font-weight: 600; }}
        .connected {{ background: linear-gradient(45deg, #28a745, #20c997); }}
        .disconnected {{ background: linear-gradient(45deg, #dc3545, #fd7e14); }}
        .recording {{ background: linear-gradient(45deg, #ff6b35, #f7931e); }}
        .processing {{ background: linear-gradient(45deg, #ffc107, #fd7e14); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Voxtral Assistant</h1>
        
        <div class="model-info">
            <strong>🤖 Model Info</strong><br>
            ASR+LLM: Voxtral-Mini-3B-2507<br>
            TTS: Simple TTS<br>
            Status: Ready
        </div>
        
        <div>
            <button id="startBtn" onclick="startRecording()">🎙️ Start Recording</button>
            <button id="stopBtn" onclick="stopRecording()" class="stop-btn" disabled>⏹️ Stop Recording</button>
        </div>
        
        <div id="status" class="status disconnected">🔌 Disconnected</div>
        <div id="conversation" style="margin-top: 20px; text-align: left; max-height: 300px; overflow-y: auto;"></div>
        <audio id="responseAudio" controls style="width: 100%; margin: 10px 0; display: none;"></audio>
    </div>

    <script>
        let ws, mediaRecorder, stream, isRecording = false;
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const conversationDiv = document.getElementById('conversation');
        const responseAudio = document.getElementById('responseAudio');

        function updateStatus(message, className) {{
            statusDiv.textContent = message;
            statusDiv.className = `status ${{className}}`;
        }}

        function addMessage(text, isUser = false) {{
            const messageDiv = document.createElement('div');
            messageDiv.style.cssText = `margin: 10px 0; padding: 10px; border-radius: 8px; 
                background: ${{isUser ? 'rgba(0,123,255,0.3)' : 'rgba(40,167,69,0.3)'}};`;
            messageDiv.innerHTML = `<strong>${{isUser ? '👤 You' : '🤖 AI'}}:</strong> ${{text}}`;
            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }}

        async function initializeWebSocket() {{
            ws = new WebSocket('{ws_url}');
            ws.binaryType = 'arraybuffer';

            ws.onopen = () => {{
                updateStatus('🔌 Connected', 'connected');
                startBtn.disabled = false;
            }};

            ws.onmessage = async (event) => {{
                if (typeof event.data === 'string') {{
                    const data = JSON.parse(event.data);
                    switch(data.type) {{
                        case 'transcription':
                            addMessage(data.text, true);
                            break;
                        case 'response':
                            addMessage(data.text, false);
                            break;
                        case 'processing_start':
                            updateStatus('🧠 Processing...', 'processing');
                            break;
                        case 'audio_ready':
                            updateStatus('🔊 Playing Response...', 'processing');
                            break;
                        case 'error':
                            updateStatus('❌ Error', 'disconnected');
                            break;
                    }}
                }} else {{
                    const audioBlob = new Blob([event.data], {{ type: 'audio/wav' }});
                    const audioUrl = URL.createObjectURL(audioBlob);
                    responseAudio.src = audioUrl;
                    responseAudio.style.display = 'block';
                    responseAudio.onended = () => {{
                        updateStatus('🎙️ Ready', 'connected');
                        responseAudio.style.display = 'none';
                    }};
                    await responseAudio.play();
                }}
            }};

            ws.onclose = () => {{
                updateStatus('🔌 Disconnected', 'disconnected');
                startBtn.disabled = true;
            }};
        }}

        async function startRecording() {{
            stream = await navigator.mediaDevices.getUserMedia({{
                audio: {{ echoCancellation: true, noiseSuppression: true }}
            }});

            mediaRecorder = new MediaRecorder(stream, {{
                mimeType: 'audio/webm;codecs=opus'
            }});
            
            let audioChunks = [];
            mediaRecorder.ondataavailable = (event) => {{
                if (event.data.size > 0) audioChunks.push(event.data);
            }};
            
            mediaRecorder.onstop = async () => {{
                const audioBlob = new Blob(audioChunks, {{ type: 'audio/webm' }});
                const arrayBuffer = await audioBlob.arrayBuffer();
                if (ws && ws.readyState === WebSocket.OPEN) {{
                    ws.send(arrayBuffer);
                }}
                stream.getTracks().forEach(track => track.stop());
            }};
            
            mediaRecorder.start();
            isRecording = true;
            updateStatus('🎙️ Recording...', 'recording');
            startBtn.disabled = true;
            stopBtn.disabled = false;
        }}

        function stopRecording() {{
            if (mediaRecorder && isRecording) {{
                mediaRecorder.stop();
                isRecording = false;
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }}
        }}

        window.addEventListener('load', initializeWebSocket);
    </script>
</body>
</html>"""

# --- HTTP Handlers ---
async def index_handler(request):
    html_content = get_websocket_html_client()
    return web.Response(text=html_content, content_type='text/html')

async def health_handler(request):
    return web.json_response({
        "status": "healthy",
        "mode": "voxtral_websocket",
        "models": {"voxtral": voxtral_model is not None},
        "connections": len(active_connections),
        "timestamp": datetime.now().isoformat()
    })

# --- Application ---
async def on_shutdown(app):
    logger.info("🛑 Shutting down...")
    for ws in list(active_connections):
        if not ws.closed:
            await ws.close()
    active_connections.clear()
    executor.shutdown(wait=True)

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
    
    print(f"\n{'='*80}")
    print("🚀 VOXTRAL WEBSOCKET ASSISTANT READY")
    print(f"{'='*80}")
    print(f"📡 Public URL: {public_url}")
    print(f"🧠 Model: Voxtral-Mini-3B-2507")
    print(f"🎤 TTS: Simple TTS")
    print(f"🌐 WebSocket: Ready")
    print(f"{'='*80}")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Assistant stopped")
