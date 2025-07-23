import torch
import asyncio
import json
import logging
import numpy as np
import fractions
import warnings
import collections
import time
import librosa
import webrtcvad
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple
from datetime import datetime

from aiohttp import web, WSMsgType
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate, RTCConfiguration, RTCIceServer, mediastreams
import av

from transformers import pipeline
from chatterbox.tts import ChatterboxTTS
import torch.hub

# --- Runpod Environment Detection ---
RUNPOD_POD_ID = os.environ.get('RUNPOD_POD_ID', 'local')
RUNPOD_PUBLIC_IP = os.environ.get('RUNPOD_PUBLIC_IP', '0.0.0.0')
RUNPOD_TCP_PORT_7860 = os.environ.get('RUNPOD_TCP_PORT_7860', '7860')

print(f"🚀 RUNPOD ENVIRONMENT DETECTED")
print(f"📍 Pod ID: {RUNPOD_POD_ID}")
print(f"🌐 Public IP: {RUNPOD_PUBLIC_IP}")
print(f"🔌 TCP Port: {RUNPOD_TCP_PORT_7860}")

# --- Enhanced Logging Setup for Runpod ---
def setup_runpod_logging():
    """Setup comprehensive logging for Runpod environment"""
    
    # Create logs directory
    os.makedirs('/tmp/logs', exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'/tmp/logs/ultraandchat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    # Create specific loggers
    logger = logging.getLogger(__name__)
    webrtc_logger = logging.getLogger('webrtc')
    audio_logger = logging.getLogger('audio')
    model_logger = logging.getLogger('models')
    
    # Set levels for noisy libraries
    for logger_name in ['aioice', 'aiortc', 'av.audio.resampler', 'urllib3', 'requests']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger.info("🔧 Runpod logging initialized")
    return logger, webrtc_logger, audio_logger, model_logger

logger, webrtc_logger, audio_logger, model_logger = setup_runpod_logging()

# --- Enhanced Setup for Runpod ---
try:
    import uvloop
    uvloop.install()
    logger.info("🚀 Using uvloop for optimized event loop")
except ImportError:
    logger.warning("⚠️ uvloop not found, using default event loop")

warnings.filterwarnings("ignore")
uv_pipe, tts_model = None, None
executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="audio_worker")
pcs = set()

def get_runpod_ice_servers():
    """Get ICE servers optimized for Runpod environment"""
    ice_servers = [
        # Google STUN servers (TCP fallback)
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
        RTCIceServer(urls="stun:stun1.l.google.com:19302"),
        RTCIceServer(urls="stun:stun2.l.google.com:19302"),
        RTCIceServer(urls="stun:stun3.l.google.com:19302"),
        RTCIceServer(urls="stun:stun4.l.google.com:19302"),
        
        # Cloudflare STUN
        RTCIceServer(urls="stun:stun.cloudflare.com:3478"),
        
        # Additional STUN servers for better connectivity
        RTCIceServer(urls="stun:stun.stunprotocol.org:3478"),
        RTCIceServer(urls="stun:stun.voiparound.com"),
        RTCIceServer(urls="stun:stun.voipbuster.com"),
    ]
    
    webrtc_logger.info(f"🌐 Configured {len(ice_servers)} ICE servers for Runpod")
    return ice_servers

def get_runpod_rtc_config():
    """Get RTCConfiguration optimized for Runpod"""
    config = RTCConfiguration(
        iceServers=get_runpod_ice_servers(),
        iceCandidatePoolSize=10,
        bundlePolicy='max-bundle',
        rtcpMuxPolicy='require',
        iceTransportPolicy='all'  
    )
    webrtc_logger.info("⚙️ RTCConfiguration created for Runpod")
    return config

# --- Enhanced HTML Client with Runpod Optimizations ---
def get_runpod_html_client():
    """Generate HTML client optimized for Runpod environment"""
    
    # Determine the public URL
    if RUNPOD_POD_ID != 'local':
        public_url = f"https://{RUNPOD_TCP_PORT_7860}-{RUNPOD_POD_ID}.proxy.runpod.net"
        ws_url = f"wss://{RUNPOD_TCP_PORT_7860}-{RUNPOD_POD_ID}.proxy.runpod.net/ws"
    else:
        public_url = f"http://localhost:{RUNPOD_TCP_PORT_7860}"
        ws_url = f"ws://localhost:{RUNPOD_TCP_PORT_7860}/ws"
    
    logger.info(f"🌐 Client will connect to: {ws_url}")
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>🚀 UltraFast Voice Assistant - Runpod</title>
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
            -webkit-backdrop-filter: blur(10px);
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
        .stop-btn:hover {{ background: linear-gradient(45deg, #c82333, #a71e2a); }}
        
        .status {{ 
            margin: 25px 0; padding: 20px; border-radius: 15px; font-weight: 600; font-size: 1.1em;
            transition: all 0.5s ease;
        }}
        .status.connected {{ 
            background: linear-gradient(45deg, #28a745, #20c997); 
            box-shadow: 0 0 20px rgba(40, 167, 69, 0.4);
        }}
        .status.disconnected {{ 
            background: linear-gradient(45deg, #dc3545, #fd7e14);
            box-shadow: 0 0 20px rgba(220, 53, 69, 0.4);
        }}
        .status.connecting {{ 
            background: linear-gradient(45deg, #ffc107, #fd7e14);
            animation: pulse 2s infinite;
        }}
        .status.speaking {{ 
            background: linear-gradient(45deg, #007bff, #6610f2);
            animation: speaking 1.5s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        
        @keyframes speaking {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.03); }}
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
        
        .runpod-logs {{
            margin-top: 20px; padding: 15px; background: rgba(255,165,0,0.1);
            border-radius: 10px; border: 1px solid rgba(255,165,0,0.3);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 UltraFast Voice AI - Runpod</h1>
        
        <div class="runpod-info">
            <strong>🏃 Runpod Environment</strong><br>
            Pod ID: {RUNPOD_POD_ID}<br>
            WebSocket: {ws_url}
        </div>
        
        <div class="controls">
            <button id="startBtn" onclick="start()">🎙️ Start Talking</button>
            <button id="stopBtn" onclick="stop()" class="stop-btn" disabled>⏹️ Stop</button>
        </div>
        <div id="status" class="status disconnected">🔌 Disconnected</div>
        
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
                <div id="qualityValue" class="metric-value">-</div>
                <div class="metric-label">Audio Quality</div>
            </div>
        </div>
        
        <div id="conversation" class="conversation"></div>
        
        <div class="runpod-logs">
            <strong>📊 Runpod Connection Logs</strong>
            <div id="runpodDebug" class="debug">Initializing Runpod connection...</div>
        </div>
        
        <div id="debug" class="debug">System ready. Click Start to begin...</div>
        
        <audio id="remoteAudio" autoplay playsinline controls style="width: 100%; margin: 10px 0;"></audio>
    </div>

    <script>
        let pc, ws, localStream, startTime;
        const remoteAudio = document.getElementById('remoteAudio');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const debugDiv = document.getElementById('debug');
        const runpodDebugDiv = document.getElementById('runpodDebug');
        const conversationDiv = document.getElementById('conversation');
        const latencyValue = document.getElementById('latencyValue');
        const connectionValue = document.getElementById('connectionValue');
        const qualityValue = document.getElementById('qualityValue');

        function log(message) {{
            console.log(message);
            const timestamp = new Date().toLocaleTimeString();
            debugDiv.innerHTML += `${{timestamp}}: ${{message}}<br>`;
            debugDiv.scrollTop = debugDiv.scrollHeight;
            
            // Keep debug log manageable
            if (debugDiv.children.length > 50) {{
                debugDiv.innerHTML = debugDiv.innerHTML.split('<br>').slice(-40).join('<br>');
            }}
        }}
        
        function logRunpod(message) {{
            console.log(`[RUNPOD] ${{message}}`);
            const timestamp = new Date().toLocaleTimeString();
            runpodDebugDiv.innerHTML += `${{timestamp}}: ${{message}}<br>`;
            runpodDebugDiv.scrollTop = runpodDebugDiv.scrollHeight;
            
            // Keep runpod log manageable
            if (runpodDebugDiv.children.length > 30) {{
                runpodDebugDiv.innerHTML = runpodDebugDiv.innerHTML.split('<br>').slice(-25).join('<br>');
            }}
        }}

        function addMessage(text, isUser = false) {{
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{isUser ? 'user-msg' : 'ai-msg'}}`;
            messageDiv.innerHTML = `<strong>${{isUser ? '👤 You' : '🤖 AI'}}:</strong> ${{text}}`;
            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
            
            // Keep conversation manageable
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
            if (quality !== undefined) qualityValue.textContent = quality;
        }}

        async function start() {{
            startBtn.disabled = true;
            updateStatus('🔄 Initializing Runpod connection...', 'connecting');
            debugDiv.innerHTML = '';
            conversationDiv.innerHTML = '';
            
            logRunpod('Starting connection to Runpod environment');
            logRunpod('Pod ID: {RUNPOD_POD_ID}');
            logRunpod('Target WebSocket: {ws_url}');
            
            try {{
                log('🎤 Requesting microphone access...');
                
                const constraints = {{
                    audio: {{
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                        sampleRate: {{ ideal: 48000 }},
                        channelCount: 1
                    }}
                }};

                localStream = await navigator.mediaDevices.getUserMedia(constraints);
                log('✅ Microphone access granted');
                logRunpod('Microphone initialized successfully');
                
                const audioTrack = localStream.getAudioTracks()[0];
                const settings = audioTrack.getSettings();
                log(`Audio: ${{settings.sampleRate}}Hz, ${{settings.channelCount}}ch`);
                updateMetrics(undefined, 'Initializing', 'High');

                // Enhanced peer connection configuration for Runpod
                pc = new RTCPeerConnection({{
                    iceServers: [
                        {{ urls: 'stun:stun.l.google.com:19302' }},
                        {{ urls: 'stun:stun1.l.google.com:19302' }},
                        {{ urls: 'stun:stun2.l.google.com:19302' }},
                        {{ urls: 'stun:stun.cloudflare.com:3478' }},
                        {{ urls: 'stun:stun.stunprotocol.org:3478' }}
                    ],
                    iceCandidatePoolSize: 10,
                    bundlePolicy: 'max-bundle',
                    rtcpMuxPolicy: 'require',
                    iceTransportPolicy: 'all'
                }});

                logRunpod('RTCPeerConnection created with Runpod-optimized config');
                log('🔗 RTCPeerConnection created');

                // Add tracks
                localStream.getTracks().forEach(track => {{
                    log(`📤 Adding ${{track.kind}} track`);
                    pc.addTrack(track, localStream);
                }});

                // Enhanced remote audio handling
                pc.ontrack = event => {{
                    log(`🎵 Remote track received: ${{event.track.kind}}`);
                    logRunpod(`Remote ${{event.track.kind}} track established`);
                    if (event.streams && event.streams[0]) {{
                        remoteAudio.srcObject = event.streams[0];
                        
                        remoteAudio.oncanplay = () => {{
                            log('🔊 Audio ready to play');
                            logRunpod('Audio stream ready for playback');
                            remoteAudio.play().then(() => {{
                                log('✅ Audio playback started');
                                logRunpod('Audio playback initiated successfully');
                            }}).catch(err => {{
                                log(`❌ Autoplay failed: ${{err.message}}`);
                                logRunpod(`Autoplay issue: ${{err.message}}`);
                                remoteAudio.controls = true;
                            }});
                        }};

                        remoteAudio.onplaying = () => {{
                            log('🎶 Audio is playing');
                            if (startTime) {{
                                const latency = Date.now() - startTime;
                                updateMetrics(latency, 'Connected', 'Excellent');
                                log(`⚡ Total latency: ${{latency}}ms`);
                                logRunpod(`End-to-end latency: ${{latency}}ms`);
                            }}
                            updateStatus('🤖 AI Speaking...', 'speaking');
                        }};
                        
                        remoteAudio.onended = () => {{
                            log('🔇 Audio playback ended');
                            if (pc && pc.connectionState === 'connected') {{
                                updateStatus('🎙️ Listening...', 'connected');
                            }}
                        }};
                        
                        remoteAudio.onerror = (err) => {{
                            log(`❌ Audio error: ${{err.target.error?.message || 'unknown'}}`);
                            logRunpod(`Audio error: ${{err.target.error?.message || 'unknown'}}`);
                        }};
                    }}
                }};

                // ICE candidate handling with Runpod logging
                pc.onicecandidate = event => {{
                    if (event.candidate) {{
                        log(`📤 Sending ICE candidate: ${{event.candidate.type}}`);
                        logRunpod(`ICE candidate: ${{event.candidate.type}} - ${{event.candidate.protocol}}`);
                        if (ws && ws.readyState === WebSocket.OPEN) {{
                            try {{
                                ws.send(JSON.stringify({{
                                    type: 'ice-candidate',
                                    candidate: event.candidate.toJSON()
                                }}));
                            }} catch (err) {{
                                log(`❌ Failed to send ICE candidate: ${{err.message}}`);
                                logRunpod(`ICE candidate send failed: ${{err.message}}`);
                            }}
                        }}
                    }} else {{
                        log('✅ ICE gathering complete');
                        logRunpod('ICE gathering completed successfully');
                    }}
                }};

                // Connection state monitoring with Runpod specifics
                pc.onconnectionstatechange = () => {{
                    const state = pc.connectionState;
                    log(`🔗 Connection state: ${{state}}`);
                    logRunpod(`WebRTC connection state: ${{state}}`);
                    
                    if (state === 'connected') {{
                        updateStatus('🎙️ Listening...', 'connected');
                        updateMetrics(undefined, 'Connected', 'Excellent');
                        stopBtn.disabled = false;
                        logRunpod('WebRTC connection established successfully');
                    }} else if (state === 'connecting') {{
                        updateStatus('🤝 Connecting...', 'connecting');
                        updateMetrics(undefined, 'Connecting', 'Good');
                        logRunpod('WebRTC connection in progress');
                    }} else if (['failed', 'closed', 'disconnected'].includes(state)) {{
                        log(`❌ Connection ${{state}}`);
                        logRunpod(`WebRTC connection ${{state}} - may be due to Runpod network restrictions`);
                        updateMetrics(undefined, 'Disconnected', 'Poor');
                        stop();
                    }}
                }};

                pc.oniceconnectionstatechange = () => {{
                    log(`🧊 ICE connection state: ${{pc.iceConnectionState}}`);
                    logRunpod(`ICE state: ${{pc.iceConnectionState}}`);
                }};

                pc.onicegatheringstatechange = () => {{
                    log(`❄️ ICE gathering state: ${{pc.iceGatheringState}}`);
                    logRunpod(`ICE gathering: ${{pc.iceGatheringState}}`);
                }};

                // WebSocket connection with Runpod URL
                const wsUrl = '{ws_url}';
                log(`🌐 Connecting to: ${{wsUrl}}`);
                logRunpod(`Establishing WebSocket connection to: ${{wsUrl}}`);
                
                ws = new WebSocket(wsUrl);
                ws.binaryType = 'arraybuffer';

                ws.onopen = async () => {{
                    log('✅ WebSocket connected');
                    logRunpod('WebSocket connection established to Runpod');
                    try {{
                        updateStatus('📋 Creating offer...', 'connecting');
                        const offer = await pc.createOffer({{
                            offerToReceiveAudio: true,
                            offerToReceiveVideo: false
                        }});
                        
                        await pc.setLocalDescription(offer);
                        log('📤 Sending offer');
                        logRunpod('WebRTC offer created and sent');
                        
                        ws.send(JSON.stringify(offer));
                        
                    }} catch (err) {{
                        log(`❌ Offer creation failed: ${{err.message}}`);
                        logRunpod(`Offer creation failed: ${{err.message}}`);
                        throw err;
                    }}
                }};

                ws.onmessage = async event => {{
                    try {{
                        const data = JSON.parse(event.data);
                        log(`📥 Received: ${{data.type}}`);
                        
                        if (data.type === 'answer') {{
                            await pc.setRemoteDescription(new RTCSessionDescription(data));
                            log('✅ Remote description set');
                            logRunpod('WebRTC answer processed successfully');
                        }} else if (data.type === 'speech_start') {{
                            startTime = Date.now();
                            updateStatus('🧠 Processing...', 'connecting');
                            logRunpod('Speech processing started');
                        }} else if (data.type === 'user_speech') {{
                            addMessage(data.text, true);
                            logRunpod(`User speech: ${{data.text}}`);
                        }} else if (data.type === 'ai_response') {{
                            addMessage(data.text, false);
                            logRunpod(`AI response: ${{data.text}}`);
                        }}
                    }} catch (err) {{
                        log(`❌ Message processing error: ${{err.message}}`);
                        logRunpod(`Message processing error: ${{err.message}}`);
                    }}
                }};

                ws.onclose = (event) => {{
                    log(`🔌 WebSocket closed: ${{event.code}} - ${{event.reason || 'No reason'}}`);
                    logRunpod(`WebSocket closed: ${{event.code}} - ${{event.reason || 'Connection terminated'}}`);
                    updateMetrics(undefined, 'Disconnected', 'Poor');
                    if (pc && !['closed', 'failed'].includes(pc.connectionState)) {{
                        setTimeout(() => stop(), 100);
                    }}
                }};

                ws.onerror = (error) => {{
                    log(`❌ WebSocket error`);
                    logRunpod('WebSocket error - check Runpod network connectivity');
                    updateMetrics(undefined, 'Error', 'Poor');
                }};

            }} catch (err) {{
                log(`❌ Initialization error: ${{err.message}}`);
                logRunpod(`Initialization failed: ${{err.message}}`);
                console.error('Full error:', err);
                updateStatus(`❌ Error: ${{err.message}}`, 'disconnected');
                updateMetrics(undefined, 'Error', 'Poor');
                stop();
            }}
        }}

        function stop() {{
            log('🛑 Stopping connection...');
            logRunpod('Terminating Runpod connection');
            
            // Clean up WebSocket
            if (ws) {{
                ws.onclose = ws.onerror = ws.onmessage = null;
                if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {{
                    ws.close(1000, 'User initiated close');
                }}
                ws = null;
            }}
            
            // Clean up peer connection
            if (pc) {{
                pc.onconnectionstatechange = null;
                pc.onicecandidate = null;
                pc.ontrack = null;
                pc.oniceconnectionstatechange = null;
                pc.onicegatheringstatechange = null;
                
                if (pc.connectionState !== 'closed') {{
                    pc.close();
                }}
                pc = null;
            }}
            
            // Clean up media
            if (localStream) {{
                localStream.getTracks().forEach(track => {{
                    log(`⏹️ Stopping ${{track.kind}} track`);
                    track.stop();
                }});
                localStream = null;
            }}
            
            if (remoteAudio.srcObject) {{
                remoteAudio.srcObject = null;
                remoteAudio.controls = false;
            }}
            
            updateStatus('🔌 Disconnected', 'disconnected');
            updateMetrics(0, 'Disconnected', '-');
            startBtn.disabled = false;
            stopBtn.disabled = true;
            startTime = null;
            
            log('✅ Cleanup completed');
            logRunpod('Runpod connection cleanup completed');
        }}

        // Prevent accidental page unload
        window.addEventListener('beforeunload', (e) => {{
            if (pc && pc.connectionState === 'connected') {{
                stop();
                e.preventDefault();
                e.returnValue = '';
            }}
        }});
        
        log('🚀 Interface ready. Click Start to begin...');
        logRunpod('Runpod client interface initialized');
    </script>
</body>
</html>
"""
def parse_ice_candidate_string(candidate_str: str) -> dict:
    """Parse ICE candidate string with better error handling and Runpod logging"""
    try:
        # Remove "candidate:" prefix if present
        if candidate_str.startswith("candidate:"):
            candidate_str = candidate_str[10:]
        
        parts = candidate_str.strip().split()
        if len(parts) < 8:
            webrtc_logger.warning(f"Invalid ICE candidate format: {candidate_str}")
            return {}
        
        # Parse basic fields
        result = {
            'foundation': parts[0],
            'component': int(parts[1]),
            'protocol': parts[2].lower(),
            'priority': int(parts[3]),
            'ip': parts[4],
            'port': int(parts[5]),
            'type': parts[7].lower()
        }
        
        # Parse optional fields
        i = 8
        while i < len(parts) - 1:
            key = parts[i].lower()
            if key == "raddr":
                result['relatedAddress'] = parts[i + 1]
                i += 2
            elif key == "rport":
                result['relatedPort'] = int(parts[i + 1])
                i += 2
            elif key == "tcptype":
                result['tcpType'] = parts[i + 1]
                i += 2
            else:
                i += 1
        
        webrtc_logger.debug(f"Parsed ICE candidate: {result['type']} {result['protocol']} {result['ip']}:{result['port']}")
        return result
        
    except (ValueError, IndexError) as e:
        webrtc_logger.error(f"ICE candidate parsing error: {e}")
        return {}

# --- Enhanced VAD System with Runpod Optimizations ---
class ImprovedVAD:
    def __init__(self):
        self.webrtc_vad = webrtcvad.Vad(2)  # Moderate aggressiveness
        self.silero_model = None
        self._load_silero()
        
    def _load_silero(self):
        try:
            audio_logger.info("🎤 Loading Silero VAD for Runpod...")
            self.silero_model, utils = torch.hub.load(
                'snakers4/silero-vad', 
                'silero_vad', 
                force_reload=False,
                verbose=False
            )
            self.get_speech_timestamps = utils[0]
            audio_logger.info("✅ Silero VAD loaded successfully on Runpod")
        except Exception as e:
            audio_logger.error(f"❌ Silero VAD loading failed on Runpod: {e}")
            self.silero_model = None
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """Detect speech using multiple VAD methods with Runpod optimizations"""
        if len(audio) == 0:
            return False
            
        # Quick energy check
        rms_energy = np.sqrt(np.mean(audio ** 2))
        if rms_energy < 0.001:
            return False
            
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
        # WebRTC VAD
        webrtc_detected = self._webrtc_vad_check(audio)
        
        # Silero VAD (if available)
        silero_detected = True
        if self.silero_model is not None:
            silero_detected = self._silero_vad_check(audio)
        
        # Combined decision - both must agree for reliability
        result = webrtc_detected and silero_detected
        
        audio_logger.debug(f"VAD: energy={rms_energy:.6f}, webrtc={webrtc_detected}, silero={silero_detected}, final={result}")
        return result
    
    def _webrtc_vad_check(self, audio: np.ndarray) -> bool:
        """WebRTC VAD check"""
        try:
            # Convert to int16
            audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            
            # Process in 20ms chunks
            frame_length = 320  # 20ms at 16kHz
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
                return speech_ratio > 0.4  # 40% threshold
                
            return False
            
        except Exception as e:
            audio_logger.debug(f"WebRTC VAD error: {e}")
            return False
    
    def _silero_vad_check(self, audio: np.ndarray) -> bool:
        """Silero VAD check"""
        try:
            audio_tensor = torch.from_numpy(audio)
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.silero_model,
                sampling_rate=16000,
                min_speech_duration_ms=250,
                threshold=0.3
            )
            return len(speech_timestamps) > 0
            
        except Exception as e:
            audio_logger.debug(f"Silero VAD error: {e}")
            return True  # Default to true if Silero fails

# --- Enhanced Audio Buffer with Runpod Optimizations ---
class EnhancedAudioBuffer:
    def __init__(self):
        self.sample_rate = 16000
        self.max_duration = 4.0
        self.max_samples = int(self.max_duration * self.sample_rate)
        self.buffer = collections.deque(maxlen=self.max_samples)
        self.min_speech_duration = 1.0
        self.min_samples = int(self.min_speech_duration * self.sample_rate)
        self.last_process_time = 0
        self.cooldown_period = 1.0
        
        audio_logger.info(f"🔊 Audio buffer initialized: {self.max_duration}s max, {self.min_speech_duration}s min")
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data with preprocessing"""
        # Ensure float32
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Flatten and preprocess
        audio_data = audio_data.flatten()
        
        # Remove DC bias
        audio_data = audio_data - np.mean(audio_data)
        
        # Gentle normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.1:  # Only normalize if loud enough
            audio_data = audio_data * (0.9 / max_val)
        
        self.buffer.extend(audio_data)
    
    def should_process(self, vad: ImprovedVAD) -> Tuple[bool, Optional[np.ndarray]]:
        """Check if audio should be processed"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_process_time < self.cooldown_period:
            return False, None
            
        # Length check
        if len(self.buffer) < self.min_samples:
            return False, None
            
        # Get audio array
        audio_array = np.array(list(self.buffer), dtype=np.float32)
        
        # Energy check
        if np.max(np.abs(audio_array)) < 0.02:
            return False, None
            
        # VAD check
        if vad.detect_speech(audio_array, self.sample_rate):
            self.last_process_time = current_time
            audio_logger.info(f"🎯 Speech detected: {len(audio_array)/16000:.2f}s, energy: {np.max(np.abs(audio_array)):.4f}")
            return True, audio_array
            
        return False, None
    
    def reset(self):
        """Clear the buffer"""
        self.buffer.clear()
        audio_logger.debug("🔄 Audio buffer reset")

# --- Robust Audio Track with Runpod Optimizations ---
class RobustAudioTrack(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self._audio_queue = asyncio.Queue(maxsize=30)
        self._current_audio = None
        self._position = 0
        self._timestamp = 0
        self._sample_rate = 48000
        self._frame_size = 960  # 20ms at 48kHz
        self._silence_frame = np.zeros(self._frame_size, dtype=np.int16)
        
        audio_logger.info(f"🎵 Audio track initialized: {self._sample_rate}Hz, {self._frame_size} samples/frame")
        
    async def recv(self):
        """Generate audio frame"""
        # Get new audio chunk if current is finished
        if self._current_audio is None or self._position >= len(self._current_audio):
            try:
                self._current_audio = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.02
                )
                self._position = 0
            except asyncio.TimeoutError:
                self._current_audio = None
        
        # Prepare frame
        frame_data = self._silence_frame.copy()
        
        # Fill with audio data if available
        if self._current_audio is not None:
            remaining = len(self._current_audio) - self._position
            copy_size = min(self._frame_size, remaining)
            if copy_size > 0:
                frame_data[:copy_size] = self._current_audio[self._position:self._position + copy_size]
                self._position += copy_size
        
        # Create AV frame
        audio_frame = av.AudioFrame.from_ndarray(
            np.array([frame_data]), format="s16", layout="mono"
        )
        audio_frame.pts = self._timestamp
        audio_frame.sample_rate = self._sample_rate
        self._timestamp += self._frame_size
        
        return audio_frame
    
    async def add_audio(self, audio_data: np.ndarray):
        """Add audio to playback queue"""
        if len(audio_data) > 0:
            # Convert to int16
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            
            try:
                # Add to queue without blocking too long
                await asyncio.wait_for(self._audio_queue.put(audio_int16), timeout=0.2)
                audio_logger.debug(f"🔊 Added {len(audio_int16)} audio samples to queue")
            except asyncio.TimeoutError:
                audio_logger.warning("⚠️ Audio queue full, dropping old audio")
                # Clear some old audio
                try:
                    while not self._audio_queue.empty():
                        self._audio_queue.get_nowait()
                        if self._audio_queue.qsize() < 10:
                            break
                except asyncio.QueueEmpty:
                    pass
                # Try adding again
                try:
                    self._audio_queue.put_nowait(audio_int16)
                except asyncio.QueueFull:
                    pass
class RobustAudioProcessor:
    def __init__(self, output_track, executor):
        self.input_track = None
        self.output_track = output_track
        self.buffer = EnhancedAudioBuffer()
        self.vad = ImprovedVAD()
        self.executor = executor
        self.task = None
        self.is_processing = False
        self.ws = None
        self._stop_event = asyncio.Event()
        
        audio_logger.info("🎛️ Audio processor initialized for Runpod")
        
    def set_websocket(self, ws):
        self.ws = ws
        audio_logger.info("🌐 WebSocket connection set for audio processor")
        
    def add_track(self, track):
        self.input_track = track
        audio_logger.info(f"✅ Audio track added: {track.kind}")
        
    async def start(self):
        if not self.task:
            audio_logger.info("🎵 Starting audio processor")
            self.task = asyncio.create_task(self._audio_loop())
            
    async def stop(self):
        if self.task:
            audio_logger.info("🛑 Stopping audio processor")
            self._stop_event.set()
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None
    
    async def _audio_loop(self):
        """Main audio processing loop with enhanced Runpod logging"""
        frame_count = 0
        last_log_time = time.time()
        
        try:
            while not self._stop_event.is_set():
                # Skip processing while AI is speaking
                if self.is_processing:
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    frame = await asyncio.wait_for(self.input_track.recv(), timeout=0.1)
                    frame_count += 1
                    
                    # Log progress every 5 seconds
                    current_time = time.time()
                    if current_time - last_log_time > 5.0:
                        audio_logger.info(f"📊 Processed {frame_count} audio frames in last 5s")
                        last_log_time = current_time
                        frame_count = 0
                    
                except asyncio.TimeoutError:
                    continue
                except mediastreams.MediaStreamError:
                    audio_logger.info("🔚 Audio stream ended")
                    break
                except Exception as e:
                    audio_logger.error(f"❌ Frame receive error: {e}")
                    break
                
                try:
                    # Process audio frame
                    audio_data = frame.to_ndarray().flatten()
                    
                    # Convert to float
                    if audio_data.dtype == np.int16:
                        audio_float = audio_data.astype(np.float32) / 32768.0
                    else:
                        audio_float = audio_data.astype(np.float32)
                    
                    # Resample if needed
                    if frame.sample_rate != 16000:
                        audio_float = librosa.resample(
                            audio_float,
                            orig_sr=frame.sample_rate,
                            target_sr=16000
                        )
                    
                    # Add to buffer
                    self.buffer.add_audio(audio_float)
                    
                    # Check for speech
                    should_process, audio_array = self.buffer.should_process(self.vad)
                    if should_process and audio_array is not None:
                        audio_logger.info(f"🎯 Speech detected: {len(audio_array)/16000:.2f}s")
                        self.buffer.reset()
                        
                        # Process asynchronously
                        asyncio.create_task(self._process_speech(audio_array))
                        
                except Exception as e:
                    audio_logger.error(f"❌ Audio processing error: {e}")
                    continue
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            audio_logger.error(f"❌ Audio loop error: {e}", exc_info=True)
        finally:
            audio_logger.info("🔚 Audio processor stopped")
    
    def _run_inference(self, audio_array: np.ndarray) -> str:
        """Run speech inference with Runpod logging"""
        try:
            start_time = time.time()
            
            with torch.inference_mode():
                result = uv_pipe({
                    'audio': audio_array,
                    'turns': [],
                    'sampling_rate': 16000
                }, max_new_tokens=50, do_sample=False, temperature=0.1)
                
                # Extract text
                text = ""
                if isinstance(result, list) and len(result) > 0:
                    item = result[0]
                    if isinstance(item, dict) and 'generated_text' in item:
                        text = item['generated_text']
                    elif isinstance(item, str):
                        text = item
                elif isinstance(result, str):
                    text = result
                
                inference_time = time.time() - start_time
                model_logger.info(f"🧠 Inference completed: {inference_time*1000:.0f}ms, text: '{text.strip()}'")
                
                return text.strip() if text else ""
                
        except Exception as e:
            model_logger.error(f"❌ Inference error: {e}")
            return ""
    
    def _run_tts(self, text: str) -> np.ndarray:
        """Generate TTS audio with Runpod logging"""
        try:
            if not text.strip():
                return np.array([], dtype=np.float32)
            
            start_time = time.time()
            
            with torch.inference_mode():
                wav = tts_model.generate(text)
                
                # Convert to numpy
                if hasattr(wav, 'cpu'):
                    wav = wav.cpu().numpy()
                elif torch.is_tensor(wav):
                    wav = wav.numpy()
                
                wav = wav.flatten().astype(np.float32)
                
                # Resample to 48kHz
                wav_48k = librosa.resample(wav, orig_sr=24000, target_sr=48000)
                
                # Normalize volume
                if np.max(np.abs(wav_48k)) > 0:
                    wav_48k = wav_48k / max(np.max(np.abs(wav_48k)), 0.1) * 0.7
                
                tts_time = time.time() - start_time
                model_logger.info(f"🔊 TTS completed: {tts_time*1000:.0f}ms, {len(wav_48k)/48000:.2f}s audio")
                
                return wav_48k
                
        except Exception as e:
            model_logger.error(f"❌ TTS error: {e}")
            return np.array([], dtype=np.float32)
    
    async def _process_speech(self, audio_array: np.ndarray):
        """Process detected speech with comprehensive Runpod logging"""
        if self.is_processing:
            audio_logger.warning("⚠️ Already processing speech, skipping")
            return
            
        start_time = time.time()
        self.is_processing = True
        
        try:
            audio_logger.info(f"🎤 Processing speech: {len(audio_array)/16000:.2f}s audio")
            
            # Signal processing start
            if self.ws and not self.ws.closed:
                try:
                    await self.ws.send_json({'type': 'speech_start'})
                    audio_logger.debug("📤 Sent speech_start signal")
                except Exception as e:
                    audio_logger.warning(f"Failed to send speech_start: {e}")
            
            # Run inference
            loop = asyncio.get_running_loop()
            user_text = await loop.run_in_executor(
                self.executor, self._run_inference, audio_array
            )
            
            if not user_text:
                audio_logger.warning("⚠️ No text generated from speech")
                return
                
            stt_time = time.time() - start_time
            logger.info(f"💬 User: '{user_text}' (STT: {stt_time*1000:.0f}ms)")
            
            # Send to client
            if self.ws and not self.ws.closed:
                try:
                    await self.ws.send_json({'type': 'user_speech', 'text': user_text})
                    audio_logger.debug("📤 Sent user_speech to client")
                except Exception as e:
                    audio_logger.warning(f"Failed to send user_speech: {e}")
            
            # Generate TTS
            tts_start = time.time()
            audio_output = await loop.run_in_executor(
                self.executor, self._run_tts, user_text
            )
            
            if audio_output.size > 0:
                tts_time = time.time() - tts_start
                total_time = time.time() - start_time
                
                logger.info(f"⚡ STT: {stt_time*1000:.0f}ms, TTS: {tts_time*1000:.0f}ms, Total: {total_time*1000:.0f}ms")
                
                # Send AI response
                if self.ws and not self.ws.closed:
                    try:
                        await self.ws.send_json({'type': 'ai_response', 'text': user_text})
                        audio_logger.debug("📤 Sent ai_response to client")
                    except Exception as e:
                        audio_logger.warning(f"Failed to send ai_response: {e}")
                
                # Queue audio
                await self.output_track.add_audio(audio_output)
                
                # Wait for playback
                duration = len(audio_output) / 48000
                audio_logger.info(f"🔊 Playing {duration:.2f}s audio")
                await asyncio.sleep(duration + 0.5)
            else:
                audio_logger.warning("⚠️ No audio generated from TTS")
            
        except Exception as e:
            audio_logger.error(f"❌ Speech processing error: {e}", exc_info=True)
        finally:
            self.is_processing = False
            processing_time = time.time() - start_time
            audio_logger.info(f"✅ Speech processing completed: {processing_time*1000:.0f}ms total")

# --- Model Initialization with Runpod Optimizations ---
def initialize_models() -> bool:
    """Initialize models with better error handling and Runpod logging"""
    global uv_pipe, tts_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_logger.info(f"🚀 Initializing models on Runpod device: {device}")
    
    # Log GPU info if available
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
        
        # Optimize if possible
        if hasattr(torch, 'compile') and hasattr(uv_pipe, 'model'):
            try:
                model_logger.info("🔧 Applying torch.compile optimization...")
                uv_pipe.model = torch.compile(uv_pipe.model, mode="reduce-overhead")
                model_logger.info("✅ torch.compile applied successfully")
            except Exception as e:
                model_logger.warning(f"⚠️ torch.compile failed: {e}")
        
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
            
        model_logger.info("🎉 All models ready for Runpod deployment!")
        return True
        
    except Exception as e:
        model_logger.error(f"❌ Model initialization failed on Runpod: {e}", exc_info=True)
        return False# --- 
WebSocket Handler with Runpod Optimizations ---
async def websocket_handler(request):
    """Enhanced WebSocket handler optimized for Runpod"""
    ws = web.WebSocketResponse(
        heartbeat=30, 
        timeout=120,
        max_msg_size=16*1024*1024  # 16MB max message size
    )
    await ws.prepare(request)
    
    client_ip = request.remote
    webrtc_logger.info(f"🌐 WebSocket connection established from {client_ip}")
    
    # Create peer connection with Runpod-optimized config
    config = get_runpod_rtc_config()
    pc = RTCPeerConnection(config)
    pcs.add(pc)
    processor = None
    
    @pc.on("track")
    def on_track(track):
        nonlocal processor
        webrtc_logger.info(f"🎧 Received track: {track.kind} from {client_ip}")
        
        if track.kind == "audio":
            response_track = RobustAudioTrack()
            pc.addTrack(response_track)
            webrtc_logger.info("🔊 Added response audio track")
            
            processor = RobustAudioProcessor(response_track, executor)
            processor.set_websocket(ws)
            processor.add_track(track)
            
            asyncio.create_task(processor.start())
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        webrtc_logger.info(f"🔗 Connection state changed to: {state} for {client_ip}")
        
        if state in ["failed", "closed", "disconnected"]:
            webrtc_logger.info(f"🧹 Cleaning up connection for {client_ip}")
            if processor:
                await processor.stop()
            if pc in pcs:
                pcs.remove(pc)
    
    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        webrtc_logger.debug(f"❄️ ICE gathering state: {pc.iceGatheringState} for {client_ip}")
    
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        webrtc_logger.debug(f"🧊 ICE connection state: {pc.iceConnectionState} for {client_ip}")
    
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "unknown")
                    
                    if msg_type == "offer":
                        webrtc_logger.info(f"📥 Processing WebRTC offer from {client_ip}")
                        
                        await pc.setRemoteDescription(
                            RTCSessionDescription(sdp=data["sdp"], type=data["type"])
                        )
                        
                        answer = await pc.createAnswer()
                        await pc.setLocalDescription(answer)
                        
                        await ws.send_json({
                            "type": "answer",
                            "sdp": pc.localDescription.sdp
                        })
                        
                        webrtc_logger.info(f"📤 WebRTC answer sent to {client_ip}")
                        
                    elif msg_type == "ice-candidate":
                        candidate_data = data.get("candidate", {})
                        if candidate_data:
                            candidate_str = candidate_data.get("candidate", "")
                            parsed = parse_ice_candidate_string(candidate_str)
                            
                            if parsed:
                                try:
                                    candidate = RTCIceCandidate(
                                        component=parsed["component"],
                                        foundation=parsed["foundation"],
                                        ip=parsed["ip"],
                                        port=parsed["port"],
                                        priority=parsed["priority"],
                                        protocol=parsed["protocol"],
                                        type=parsed["type"],
                                        sdpMid=candidate_data.get("sdpMid"),
                                        sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                                        relatedAddress=parsed.get("relatedAddress"),
                                        relatedPort=parsed.get("relatedPort"),
                                        tcpType=parsed.get("tcpType")
                                    )
                                    await pc.addIceCandidate(candidate)
                                    webrtc_logger.debug(f"✅ Added ICE candidate: {parsed['type']} {parsed['protocol']} from {client_ip}")
                                    
                                except Exception as e:
                                    webrtc_logger.error(f"❌ ICE candidate error for {client_ip}: {e}")
                            else:
                                webrtc_logger.warning(f"⚠️ Invalid ICE candidate format from {client_ip}")
                                
                except json.JSONDecodeError as e:
                    webrtc_logger.error(f"❌ JSON decode error from {client_ip}: {e}")
                except Exception as e:
                    webrtc_logger.error(f"❌ Message processing error from {client_ip}: {e}")
                    
            elif msg.type == WSMsgType.ERROR:
                webrtc_logger.error(f"❌ WebSocket error from {client_ip}: {ws.exception()}")
                break
                
    except ConnectionResetError:
        webrtc_logger.info(f"🔌 Client {client_ip} disconnected")
    except Exception as e:
        webrtc_logger.error(f"❌ WebSocket error from {client_ip}: {e}")
    finally:
        webrtc_logger.info(f"🔚 WebSocket closing for {client_ip}")
        
        if processor:
            await processor.stop()
        if pc in pcs:
            pcs.remove(pc)
        if pc.connectionState != "closed":
            await pc.close()
    
    return ws

# --- HTTP Handlers with Runpod Optimizations ---
async def index_handler(request):
    """Serve the main HTML page with Runpod optimizations"""
    html_content = get_runpod_html_client()
    
    return web.Response(
        text=html_content,
        content_type='text/html',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Runpod-Pod-Id': RUNPOD_POD_ID
        }
    )

async def health_handler(request):
    """Health check endpoint with Runpod-specific information"""
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
    
    health_data = {
        "status": "healthy",
        "runpod": {
            "pod_id": RUNPOD_POD_ID,
            "public_ip": RUNPOD_PUBLIC_IP,
            "tcp_port": RUNPOD_TCP_PORT_7860
        },
        "models": {
            "ultravox": uv_pipe is not None,
            "tts": tts_model is not None
        },
        "connections": len(pcs),
        "gpu": gpu_info,
        "timestamp": datetime.now().isoformat()
    }
    
    return web.json_response(health_data)

async def logs_handler(request):
    """Endpoint to retrieve recent logs"""
    try:
        log_files = []
        log_dir = '/tmp/logs'
        
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                if filename.startswith('ultraandchat_') and filename.endswith('.log'):
                    filepath = os.path.join(log_dir, filename)
                    log_files.append({
                        'filename': filename,
                        'size': os.path.getsize(filepath),
                        'modified': os.path.getmtime(filepath)
                    })
        
        # Sort by modification time, newest first
        log_files.sort(key=lambda x: x['modified'], reverse=True)
        
        return web.json_response({
            "status": "success",
            "log_files": log_files,
            "log_directory": log_dir
        })
        
    except Exception as e:
        return web.json_response({
            "status": "error",
            "message": str(e)
        }, status=500)

# --- Application with Runpod Optimizations ---
async def on_shutdown(app):
    logger.info("🛑 Shutting down Runpod application...")
    
    # Close connections
    tasks = [pc.close() for pc in list(pcs)]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    pcs.clear()
    
    # Shutdown executor
    executor.shutdown(wait=True)
    logger.info("✅ Runpod shutdown complete")

async def main():
    """Main function optimized for Runpod deployment"""
    
    # Initialize models
    if not initialize_models():
        logger.error("❌ Model initialization failed - cannot start server")
        return
    
    # Create app
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/health', health_handler)
    app.router.add_get('/logs', logs_handler)
    app.router.add_get('/ws', websocket_handler)
    app.on_shutdown.append(on_shutdown)
    
    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Use the Runpod port
    port = int(RUNPOD_TCP_PORT_7860)
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    # Determine public URL
    if RUNPOD_POD_ID != 'local':
        public_url = f"https://{RUNPOD_TCP_PORT_7860}-{RUNPOD_POD_ID}.proxy.runpod.net"
    else:
        public_url = f"http://0.0.0.0:{port}"
    
    print("\n" + "="*80)
    print("🚀 ULTRA-FAST SPEECH-TO-SPEECH SERVER - RUNPOD OPTIMIZED")
    print("="*80)
    print(f"🏃 Runpod Pod ID: {RUNPOD_POD_ID}")
    print(f"📡 Public URL: {public_url}")
    print(f"🔗 Health Check: {public_url}/health")
    print(f"📊 Logs: {public_url}/logs")
    print(f"🎯 Target Latency: <500ms")
    print(f"🧠 GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU Only'}")
    print(f"🔊 Enhanced TTS Playback: ✅")
    print(f"🎤 Advanced VAD: ✅")
    print(f"📝 Logging: ✅ Enhanced for Runpod")
    print(f"🌐 WebRTC: ✅ Optimized for Runpod network")
    print("="*80)
    print("🛑 Press Ctrl+C to stop")
    print("="*80 + "\n")
    
    # Log startup completion
    logger.info(f"🎉 Server started successfully on Runpod")
    logger.info(f"📡 Accessible at: {public_url}")
    
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Runpod server stopped")
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}", exc_info=True)
        sys.exit(1)
