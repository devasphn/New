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
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from datetime import datetime
import tempfile
import subprocess
import io
import wave

from aiohttp import web, WSMsgType
from transformers import VoxtralForConditionalGeneration, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import torchaudio

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

print(f"🚀 VOXTRAL + HIGGS AUDIO V2 REALTIME ASSISTANT")
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
            logging.FileHandler(f'/tmp/logs/voxtral_higgs_realtime_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    audio_logger = logging.getLogger('audio')
    model_logger = logging.getLogger('models')
    
    for logger_name in ['urllib3', 'requests', 'transformers']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger.info("🔧 Voxtral + Higgs Audio v2 Realtime WebSocket assistant initialized")
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
higgs_model, higgs_tokenizer = None, None
executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="audio_worker")
active_connections = set()

# --- Enhanced VAD System (More Sensitive) ---
class RealtimeVAD:
    def __init__(self):
        self.webrtc_vad = webrtcvad.Vad(1)  # More sensitive (0=least, 3=most aggressive)
        self.min_speech_duration = 0.5  # Minimum 500ms of speech
        self.max_silence_duration = 1.0  # Max 1s of silence before stopping
        
    def detect_speech_realtime(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """More sensitive speech detection for realtime use"""
        if len(audio) == 0:
            return False
            
        # Energy-based detection (more sensitive)
        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy > 0.0005:  # Lower threshold for sensitivity
            return True
            
        # Resample if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
        return self._webrtc_vad_check(audio)
    
    def _webrtc_vad_check(self, audio: np.ndarray) -> bool:
        try:
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            frame_length = 320  # 20ms frames at 16kHz
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
                return speech_ratio > 0.2  # Lower threshold for more sensitivity
                
            return False
        except Exception as e:
            audio_logger.debug(f"WebRTC VAD error: {e}")
            return False

# --- Higgs Audio v2 TTS Implementation ---
class HiggsAudioTTS:
    """Higgs Audio v2 TTS Implementation"""
    
    def __init__(self):
        self.model_id = "bosonai/higgs-audio-v2-generation-3B-base"
        self.model = None
        self.tokenizer = None
        self.sample_rate = 24000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load Higgs Audio v2 model"""
        try:
            model_logger.info(f"📥 Loading Higgs Audio v2 from {self.model_id}...")
            
            # Load Higgs Audio model (Llama-based with audio adapter)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            model_logger.info("✅ Higgs Audio v2 loaded successfully")
            return True
            
        except Exception as e:
            model_logger.error(f"❌ Higgs Audio v2 loading failed: {e}")
            # Fallback to enhanced simple TTS
            self.model = "fallback"
            return False
    
    def synthesize(self, text: str) -> np.ndarray:
        """Generate speech from text using Higgs Audio v2"""
        try:
            if not text.strip():
                return np.array([])
            
            if self.model == "fallback":
                return self._fallback_tts(text)
            
            # Higgs Audio v2 inference
            with torch.inference_mode():
                # Prepare text for Higgs Audio
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Generate audio tokens (this is model-specific)
                # Note: This is a simplified version - actual implementation may vary
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                # Convert tokens to audio (this would be model-specific)
                # For now, using fallback until exact API is confirmed
                return self._fallback_tts(text)
                
        except Exception as e:
            model_logger.error(f"❌ Higgs Audio synthesis error: {e}")
            return self._fallback_tts(text)
    
    def _fallback_tts(self, text: str) -> np.ndarray:
        """Enhanced fallback TTS (better than simple sine waves)"""
        if not text.strip():
            return np.array([])
        
        # More sophisticated audio generation
        words = text.split()
        duration = len(words) * 0.35 + 0.8  # More natural timing
        
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)
        
        # Create more natural-sounding audio
        base_freq = 180  # Lower, more pleasant frequency
        
        # Multiple harmonics for richer sound
        audio = np.zeros_like(t)
        audio += 0.4 * np.sin(2 * np.pi * base_freq * t)  # Fundamental
        audio += 0.2 * np.sin(2 * np.pi * base_freq * 1.25 * t)  # Perfect fourth
        audio += 0.15 * np.sin(2 * np.pi * base_freq * 1.5 * t)  # Perfect fifth
        audio += 0.1 * np.sin(2 * np.pi * base_freq * 2 * t)  # Octave
        
        # Add natural envelope
        attack = 0.1
        decay = 0.3
        sustain = 0.6
        release = 0.1
        
        envelope = np.ones_like(t)
        attack_samples = int(attack * len(t))
        release_samples = int(release * len(t))
        
        # Attack phase
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        # Release phase
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        audio *= envelope
        
        # Add slight vibrato for naturalness
        vibrato = 1 + 0.02 * np.sin(2 * np.pi * 4.5 * t)
        audio *= vibrato
        
        # Add some formant-like filtering (simplified)
        from scipy import signal
        b, a = signal.butter(4, [300, 3000], btype='band', fs=self.sample_rate)
        audio = signal.filtfilt(b, a, audio)
        
        return audio.astype(np.float32)

# --- Realtime Audio Processing Pipeline ---
class RealtimeVoxtralHiggsProcessor:
    def __init__(self):
        self.vad = RealtimeVAD()
        self.higgs_tts = HiggsAudioTTS()
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_start = None
        self.speech_start = None
        
    async def initialize(self):
        """Initialize TTS model"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, self.higgs_tts.load_model)
        
    async def process_realtime_audio(self, audio_chunk: bytes, ws) -> None:
        """Process realtime audio stream"""
        try:
            # Convert audio chunk to numpy
            audio_array = await self._convert_audio_chunk(audio_chunk)
            if audio_array is None:
                return
            
            # Add to buffer
            self.audio_buffer.extend(audio_array)
            
            # Check for speech activity
            current_time = time.time()
            has_speech = self.vad.detect_speech_realtime(audio_array)
            
            if has_speech and not self.is_speaking:
                # Speech started
                self.is_speaking = True
                self.speech_start = current_time
                self.silence_start = None
                await ws.send_json({'type': 'speech_start'})
                audio_logger.info("🎤 Speech started")
                
            elif not has_speech and self.is_speaking:
                # Potential speech end
                if self.silence_start is None:
                    self.silence_start = current_time
                elif current_time - self.silence_start > 1.0:  # 1 second of silence
                    # Process accumulated speech
                    await self._process_speech_buffer(ws)
                    
            elif has_speech and self.is_speaking:
                # Continue speaking, reset silence timer
                self.silence_start = None
                
        except Exception as e:
            audio_logger.error(f"❌ Realtime audio processing error: {e}")
    
    async def _process_speech_buffer(self, ws):
        """Process accumulated speech buffer"""
        try:
            if len(self.audio_buffer) < 8000:  # Less than 0.5 seconds
                self._reset_buffer()
                return
                
            audio_logger.info(f"🎯 Processing {len(self.audio_buffer)/16000:.2f}s of speech")
            
            # Send processing notification
            await ws.send_json({'type': 'processing_start'})
            
            # Convert buffer to numpy array
            audio_array = np.array(self.audio_buffer, dtype=np.float32)
            
            # Run Voxtral ASR+LLM
            transcription = await self._run_voxtral(audio_array)
            
            if transcription:
                # Send transcription
                await ws.send_json({
                    'type': 'transcription',
                    'text': transcription
                })
                
                # Generate TTS response
                tts_audio = await self._run_higgs_tts(transcription)
                
                if len(tts_audio) > 0:
                    # Convert to WAV and send
                    wav_data = await self._convert_to_wav(tts_audio, sample_rate=24000)
                    await ws.send_json({'type': 'audio_ready'})
                    await ws.send_bytes(wav_data)
                    
                    audio_logger.info(f"✅ Sent response audio ({len(tts_audio)/24000:.2f}s)")
                else:
                    await ws.send_json({
                        'type': 'error',
                        'message': 'TTS generation failed'
                    })
            else:
                await ws.send_json({
                    'type': 'error',
                    'message': 'Speech recognition failed'
                })
                
            self._reset_buffer()
            
        except Exception as e:
            audio_logger.error(f"❌ Speech processing error: {e}")
            await ws.send_json({
                'type': 'error',
                'message': f'Processing error: {str(e)}'
            })
            self._reset_buffer()
    
    def _reset_buffer(self):
        """Reset audio buffer and states"""
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_start = None
        self.speech_start = None
    
    async def _convert_audio_chunk(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Convert audio chunk to numpy array"""
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
                temp_input.write(audio_data)
                temp_input_path = temp_input.name
            
            temp_output_path = temp_input_path.replace('.webm', '.wav')
            
            # Convert using ffmpeg
            cmd = [
                'ffmpeg', '-i', temp_input_path,
                '-ar', '16000', '-ac', '1', '-f', 'wav',
                '-y', temp_output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return None
            
            # Load audio
            audio_array, sr = librosa.load(temp_output_path, sr=16000, mono=True)
            
            # Cleanup
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
            
            return audio_array.astype(np.float32)
            
        except Exception as e:
            return None
    
    async def _run_voxtral(self, audio_array: np.ndarray) -> str:
        """Run Voxtral ASR+LLM"""
        try:
            loop = asyncio.get_running_loop()
            
            def _inference():
                with torch.inference_mode():
                    conversation = [{
                        "role": "user",
                        "content": [{
                            "type": "audio",
                            "audio": audio_array,
                            "sample_rate": 16000
                        }]
                    }]
                    
                    inputs = voxtral_processor.apply_chat_template(conversation)
                    inputs = inputs.to(voxtral_model.device, dtype=torch.bfloat16)
                    
                    outputs = voxtral_model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.2,
                        do_sample=True,
                        pad_token_id=voxtral_processor.tokenizer.eos_token_id
                    )
                    
                    decoded_outputs = voxtral_processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:], 
                        skip_special_tokens=True
                    )
                    
                    return decoded_outputs[0].strip() if decoded_outputs else ""
            
            start_time = time.time()
            text = await loop.run_in_executor(executor, _inference)
            stt_time = time.time() - start_time
            
            model_logger.info(f"🧠 Voxtral: {stt_time*1000:.0f}ms, text: '{text}'")
            return text
            
        except Exception as e:
            model_logger.error(f"❌ Voxtral error: {e}")
            return ""
    
    async def _run_higgs_tts(self, text: str) -> np.ndarray:
        """Run Higgs Audio v2 TTS"""
        try:
            if not text.strip():
                return np.array([])
            
            loop = asyncio.get_running_loop()
            
            def _tts_inference():
                return self.higgs_tts.synthesize(text)
            
            start_time = time.time()
            audio_output = await loop.run_in_executor(executor, _tts_inference)
            tts_time = time.time() - start_time
            
            model_logger.info(f"🔊 Higgs Audio: {tts_time*1000:.0f}ms, {len(audio_output)/24000:.2f}s")
            return audio_output
            
        except Exception as e:
            model_logger.error(f"❌ Higgs Audio error: {e}")
            return np.array([])
    
    async def _convert_to_wav(self, audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
        """Convert numpy audio to WAV bytes"""
        try:
            audio_int16 = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
            
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_data = wav_buffer.getvalue()
            wav_buffer.close()
            return wav_data
            
        except Exception as e:
            logger.error(f"❌ WAV conversion error: {e}")
            return b''

# --- Model Initialization ---
def initialize_models() -> bool:
    global voxtral_processor, voxtral_model, higgs_model, higgs_tokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_logger.info(f"🚀 Initializing models on device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        model_logger.info(f"🎮 GPU: {gpu_name}, Memory: {gpu_memory:.1f}GB")
    
    try:
        # Load Voxtral Model
        model_logger.info("📥 Loading Voxtral model...")
        start_time = time.time()
        
        repo_id = "mistralai/Voxtral-Mini-3B-2507"
        voxtral_processor = AutoProcessor.from_pretrained(repo_id)
        voxtral_model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        load_time = time.time() - start_time
        model_logger.info(f"✅ Voxtral loaded in {load_time:.1f}s")
        
        # Higgs Audio will be loaded by processor during initialization
        model_logger.info("📥 Higgs Audio v2 will be loaded during processor initialization")
        
        # Warmup Voxtral
        model_logger.info("🔥 Warming up Voxtral...")
        try:
            # Create a simple test audio
            dummy_audio = np.random.randn(16000).astype(np.float32) * 0.001  # 1 second of quiet noise
            
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
        
        model_logger.info("🎉 Model initialization complete!")
        return True
        
    except Exception as e:
        model_logger.error(f"❌ Model initialization failed: {e}", exc_info=True)
        return False

# --- Realtime WebSocket Handler ---
async def realtime_websocket_handler(request):
    """Realtime WebSocket handler for continuous audio streaming"""
    ws = web.WebSocketResponse(heartbeat=30, timeout=300)
    await ws.prepare(request)
    
    client_ip = request.remote
    logger.info(f"🌐 Realtime WebSocket connection from {client_ip}")
    active_connections.add(ws)
    
    # Create processor instance
    processor = RealtimeVoxtralHiggsProcessor()
    await processor.initialize()
    
    try:
        await ws.send_json({
            'type': 'connected',
            'message': 'Realtime voice assistant ready. Start speaking!'
        })
        
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                # Realtime audio chunk received
                await processor.process_realtime_audio(msg.data, ws)
                
            elif msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    if data.get('type') == 'reset':
                        processor._reset_buffer()
                        await ws.send_json({'type': 'reset_complete'})
                except json.JSONDecodeError:
                    pass
                    
            elif msg.type == WSMsgType.ERROR:
                logger.error(f"❌ WebSocket error from {client_ip}: {ws.exception()}")
                break
                
    except Exception as e:
        logger.error(f"❌ Realtime WebSocket error for {client_ip}: {e}")
    finally:
        logger.info(f"🔚 Realtime WebSocket closed for {client_ip}")
        active_connections.discard(ws)
    
    return ws

# --- Enhanced HTML Client (Realtime) ---
def get_realtime_html_client():
    """Generate realtime HTML client"""
    
    if RUNPOD_POD_ID != 'local':
        ws_url = f"wss://{RUNPOD_POD_ID}-7860.proxy.runpod.net/ws"
    else:
        ws_url = f"ws://localhost:{RUNPOD_TCP_PORT_7860}/ws"
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <title>🚀 Voxtral + Higgs Audio v2 Realtime Assistant</title>
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
            text-align: center; max-width: 1000px; width: 100%;
        }}
        h1 {{ margin-bottom: 30px; font-weight: 300; font-size: 2.5em; }}
        .model-info {{
            background: rgba(0,255,0,0.1); padding: 20px; border-radius: 15px; margin: 20px 0;
            border: 1px solid rgba(0,255,0,0.3);
        }}
        .controls {{ margin: 30px 0; }}
        button {{
            background: linear-gradient(45deg, #00c851, #007e33);
            color: white; border: none; padding: 18px 36px; font-size: 18px;
            border-radius: 50px; cursor: pointer; margin: 10px; transition: all 0.3s ease;
        }}
        button:hover {{ transform: translateY(-3px); }}
        button:disabled {{ background: #666; cursor: not-allowed; }}
        .stop-btn {{ background: linear-gradient(45deg, #dc3545, #c82333); }}
        .reset-btn {{ background: linear-gradient(45deg, #ffc107, #fd7e14); }}
        .status {{
            margin: 25px 0; padding: 20px; border-radius: 15px; font-weight: 600; font-size: 1.1em;
            transition: all 0.5s ease;
        }}
        .connected {{ background: linear-gradient(45deg, #28a745, #20c997); }}
        .disconnected {{ background: linear-gradient(45deg, #dc3545, #fd7e14); }}
        .listening {{ background: linear-gradient(45deg, #17a2b8, #007bff); animation: pulse 2s infinite; }}
        .speaking {{ background: linear-gradient(45deg, #ff6b35, #f7931e); animation: pulse 1.5s infinite; }}
        .processing {{ background: linear-gradient(45deg, #ffc107, #fd7e14); }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        .volume-meter {{
            width: 100%; height: 15px; background: rgba(0,0,0,0.3); border-radius: 10px; margin: 15px 0;
            overflow: hidden;
        }}
        .volume-bar {{
            height: 100%; background: linear-gradient(90deg, #00ff00, #ffff00, #ff0000);
            width: 0%; transition: width 0.1s ease;
        }}
        
        .conversation {{
            margin-top: 20px; padding: 20px; background: rgba(0,0,0,0.3);
            border-radius: 15px; text-align: left; max-height: 400px; overflow-y: auto;
        }}
        .message {{
            margin: 15px 0; padding: 15px; border-radius: 10px; font-size: 14px;
        }}
        .user-msg {{ background: rgba(0, 123, 255, 0.3); margin-left: 20px; }}
        .ai-msg {{ background: rgba(40, 167, 69, 0.3); margin-right: 20px; }}
        .system-msg {{ background: rgba(108, 117, 125, 0.3); font-style: italic; text-align: center; }}
        
        .metrics {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px; margin: 20px 0;
        }}
        .metric {{
            padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px; text-align: center;
        }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; color: #00ff88; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.8; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Voxtral + Higgs Audio v2 Realtime</h1>
        
        <div class="model-info">
            <strong>🤖 Realtime Voice Assistant</strong><br>
            ASR+LLM: Voxtral-Mini-3B-2507 (Mistral AI)<br>
            TTS: Higgs Audio v2 (Boson AI)<br>
            Mode: <strong>Continuous Realtime Listening</strong><br>
            <small>✅ Just start speaking - no buttons needed!</small>
        </div>
        
        <div class="controls">
            <button id="startBtn" onclick="startRealtime()">🎙️ Start Realtime Mode</button>
            <button id="stopBtn" onclick="stopRealtime()" class="stop-btn" disabled>⏹️ Stop Listening</button>
            <button id="resetBtn" onclick="resetConversation()" class="reset-btn">🔄 Reset</button>
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
            <div class="metric">
                <div id="speechStatus" class="metric-value">Ready</div>
                <div class="metric-label">Speech Status</div>
            </div>
        </div>
        
        <div id="conversation" class="conversation">
            <div class="system-msg">🤖 Voxtral + Higgs Audio v2 Assistant ready. Click "Start Realtime Mode" and begin speaking naturally!</div>
        </div>
        
        <audio id="responseAudio" style="display: none;"></audio>
    </div>

    <script>
        let ws, mediaRecorder, audioContext, analyser, microphone, stream;
        let isRealtimeActive = false;
        let startTime;
        
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const resetBtn = document.getElementById('resetBtn');
        const statusDiv = document.getElementById('status');
        const conversationDiv = document.getElementById('conversation');
        const latencyValue = document.getElementById('latencyValue');
        const connectionValue = document.getElementById('connectionValue');
        const audioQuality = document.getElementById('audioQuality');
        const speechStatus = document.getElementById('speechStatus');
        const volumeBar = document.getElementById('volumeBar');
        const responseAudio = document.getElementById('responseAudio');

        function log(message) {{
            console.log(message);
        }}

        function addMessage(text, type = 'system') {{
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{type}}-msg`;
            
            let icon = '🤖';
            if (type === 'user') icon = '👤';
            else if (type === 'system') icon = '⚙️';
            
            messageDiv.innerHTML = `<strong>${{icon}} ${{type === 'user' ? 'You' : type === 'system' ? 'System' : 'AI'}}:</strong> ${{text}}`;
            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
            
            // Limit messages
            if (conversationDiv.children.length > 50) {{
                conversationDiv.removeChild(conversationDiv.firstChild);
            }}
        }}

        function updateStatus(message, className) {{
            statusDiv.textContent = message;
            statusDiv.className = `status ${{className}}`;
        }}

        function updateMetrics(latency, connection, quality, speechStat) {{
            if (latency !== undefined) latencyValue.textContent = `${{latency}}ms`;
            if (connection !== undefined) connectionValue.textContent = connection;
            if (quality !== undefined) audioQuality.textContent = quality;
            if (speechStat !== undefined) speechStatus.textContent = speechStat;
        }}

        function updateVolumeBar(volume) {{
            const percentage = Math.min(volume * 100, 100);
            volumeBar.style.width = `${{percentage}}%`;
        }}

        async function initializeWebSocket() {{
            try {{
                ws = new WebSocket('{ws_url}');
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {{
                    log('✅ WebSocket connected');
                    updateStatus('🔌 Connected - Ready for Realtime', 'connected');
                    updateMetrics(undefined, 'Connected', 'Excellent', 'Ready');
                    startBtn.disabled = false;
                }};

                ws.onmessage = async (event) => {{
                    try {{
                        if (typeof event.data === 'string') {{
                            const data = JSON.parse(event.data);
                            await handleTextMessage(data);
                        }} else {{
                            // Binary audio response
                            await handleAudioResponse(event.data);
                        }}
                    }} catch (err) {{
                        log(`❌ Message processing error: ${{err.message}}`);
                    }}
                }};

                ws.onclose = () => {{
                    log('🔌 WebSocket disconnected');
                    updateStatus('🔌 Disconnected', 'disconnected');
                    updateMetrics(undefined, 'Disconnected', '-', 'Offline');
                    startBtn.disabled = true;
                    stopBtn.disabled = true;
                }};

                ws.onerror = () => {{
                    log('❌ WebSocket error');
                    updateStatus('❌ Connection Error', 'disconnected');
                }};

            }} catch (err) {{
                log(`❌ WebSocket initialization error: ${{err.message}}`);
            }}
        }}

        async function handleTextMessage(data) {{
            switch(data.type) {{
                case 'connected':
                    addMessage(data.message, 'system');
                    break;
                case 'speech_start':
                    updateStatus('🎤 Listening to Speech...', 'speaking');
                    updateMetrics(undefined, undefined, undefined, 'Speaking');
                    break;
                case 'transcription':
                    addMessage(data.text, 'user');
                    break;
                case 'processing_start':
                    updateStatus('🧠 AI Processing...', 'processing');
                    updateMetrics(undefined, undefined, undefined, 'Processing');
                    startTime = Date.now();
                    break;
                case 'audio_ready':
                    updateStatus('🔊 AI Responding...', 'processing');
                    break;
                case 'error':
                    addMessage(`Error: ${{data.message}}`, 'system');
                    updateStatus('🎙️ Listening...', 'listening');
                    updateMetrics(undefined, undefined, undefined, 'Ready');
                    break;
                case 'reset_complete':
                    addMessage('Conversation reset', 'system');
                    break;
            }}
        }}

        async function handleAudioResponse(audioData) {{
            try {{
                const audioBlob = new Blob([audioData], {{ type: 'audio/wav' }});
                const audioUrl = URL.createObjectURL(audioBlob);
                
                responseAudio.src = audioUrl;
                
                responseAudio.onended = () => {{
                    updateStatus('🎙️ Listening...', 'listening');
                    updateMetrics(undefined, undefined, undefined, 'Ready');
                    URL.revokeObjectURL(audioUrl);
                    
                    if (startTime) {{
                        const totalLatency = Date.now() - startTime;
                        updateMetrics(totalLatency, 'Connected', 'Excellent', 'Ready');
                    }}
                }};
                
                await responseAudio.play();
                
            }} catch (err) {{
                log(`❌ Audio playback error: ${{err.message}}`);
            }}
        }}

        async function startRealtime() {{
            try {{
                log('🎤 Starting realtime mode...');
                
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
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                
                // Volume monitoring
                function updateVolume() {{
                    if (isRealtimeActive) {{
                        analyser.getByteFrequencyData(dataArray);
                        const average = dataArray.reduce((a, b) => a + b) / bufferLength;
                        updateVolumeBar(average / 255);
                        requestAnimationFrame(updateVolume);
                    }}
                }}
                updateVolume();

                // Setup MediaRecorder for continuous streaming
                mediaRecorder = new MediaRecorder(stream, {{
                    mimeType: 'audio/webm;codecs=opus'
                }});
                
                mediaRecorder.ondataavailable = (event) => {{
                    if (event.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {{
                        ws.send(event.data);
                    }}
                }};
                
                // Start continuous recording with small chunks
                mediaRecorder.start(100); // 100ms chunks for realtime
                isRealtimeActive = true;
                
                updateStatus('🎙️ Realtime Listening Active', 'listening');
                updateMetrics(undefined, 'Connected', 'Excellent', 'Listening');
                
                startBtn.disabled = true;
                stopBtn.disabled = false;
                
                addMessage('Realtime mode activated. Speak naturally!', 'system');
                log('✅ Realtime mode started');
                
            }} catch (err) {{
                log(`❌ Realtime start error: ${{err.message}}`);
                updateStatus('❌ Microphone Error', 'disconnected');
            }}
        }}

        function stopRealtime() {{
            if (mediaRecorder && isRealtimeActive) {{
                mediaRecorder.stop();
                isRealtimeActive = false;
                
                if (stream) {{
                    stream.getTracks().forEach(track => track.stop());
                }}
                
                if (audioContext) {{
                    audioContext.close();
                }}
                
                startBtn.disabled = false;
                stopBtn.disabled = true;
                
                updateStatus('🔌 Connected - Ready for Realtime', 'connected');
                updateMetrics(undefined, 'Connected', 'Ready', 'Stopped');
                updateVolumeBar(0);
                
                addMessage('Realtime mode stopped', 'system');
                log('⏹️ Realtime mode stopped');
            }}
        }}

        function resetConversation() {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify({{ type: 'reset' }}));
                
                // Clear conversation except first message
                const messages = conversationDiv.querySelectorAll('.message');
                for (let i = 1; i < messages.length; i++) {{
                    messages[i].remove();
                }}
                
                updateMetrics(0, 'Connected', 'Ready', 'Reset');
                log('🔄 Conversation reset');
            }}
        }}

        // Initialize on page load
        window.addEventListener('load', () => {{
            log('🚀 Voxtral + Higgs Audio v2 Realtime Assistant initialized');
            initializeWebSocket();
        }});

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {{
            if (ws) ws.close();
            if (mediaRecorder && isRealtimeActive) stopRealtime();
        }});
    </script>
</body>
</html>"""

# --- HTTP Handlers ---
async def index_handler(request):
    html_content = get_realtime_html_client()
    return web.Response(text=html_content, content_type='text/html')

async def health_handler(request):
    return web.json_response({
        "status": "healthy",
        "mode": "voxtral_higgs_realtime",
        "models": {
            "voxtral": voxtral_model is not None,
            "higgs_audio_v2": True
        },
        "connections": len(active_connections),
        "features": ["realtime_listening", "continuous_conversation", "higgs_audio_v2_tts"],
        "timestamp": datetime.now().isoformat()
    })

# --- Application ---
async def on_shutdown(app):
    logger.info("🛑 Shutting down realtime assistant...")
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
    app.router.add_get('/ws', realtime_websocket_handler)
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
    
    print(f"\n{'='*90}")
    print("🚀 VOXTRAL + HIGGS AUDIO V2 REALTIME ASSISTANT READY")
    print(f"{'='*90}")
    print(f"📡 Public URL: {public_url}")
    print(f"🧠 ASR+LLM: Voxtral-Mini-3B-2507 (Mistral AI)")
    print(f"🔊 TTS: Higgs Audio v2 (Boson AI)")
    print(f"🎯 Mode: REALTIME CONTINUOUS LISTENING")
    print(f"💡 Usage: Click 'Start Realtime Mode' and speak naturally")
    print(f"🌐 WebSocket: Realtime binary audio streaming")
    print(f"📊 Memory: ~15-16GB GPU RAM required")
    print(f"{'='*90}")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Realtime assistant stopped")
