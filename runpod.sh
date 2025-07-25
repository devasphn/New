#!/bin/bash

# Runpod Deployment Script for Ultra-Fast Voice Assistant
echo "🚀 Starting Runpod deployment for Ultra-Fast Voice Assistant"

# Set environment variables for Runpod
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export RUNPOD_TCP_PORT_7860=7860

# Create logs directory
mkdir -p /tmp/logs

# Update system and install dependencies (matching commands.txt)
echo "📦 Installing system dependencies..."
apt-get update && apt-get install -y \
    libsox-dev \
    libsndfile1-dev \
    portaudio19-dev \
    ffmpeg \
    python3-dev \
    build-essential \
    git \
    wget \
    curl \
    nodejs \
    npm

# Install PM2 globally if not already installed
echo "📦 Installing PM2 process manager..."
if ! command -v pm2 &> /dev/null; then
    npm install -g pm2
    echo "✅ PM2 installed successfully"
else
    echo "✅ PM2 already installed"
fi

# Create and activate Python virtual environment
echo "🐍 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip in virtual environment
echo "🔧 Upgrading pip in virtual environment..."
pip install --upgrade pip

# Install PyTorch with CUDA support (matching commands.txt)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python dependencies
echo "🐍 Installing other Python dependencies..."
pip install transformers librosa chatterbox-tts gradio numpy scipy huggingface-hub peft accelerate
pip install torch-audiomentations silero-vad aiortc websockets aiohttp aiofiles soundfile webrtcvad pyaudio uvloop av

# Pre-download models to avoid cold start delays
echo "📥 Pre-downloading models..."
venv/bin/python -c "
import torch
import os
import sys

try:
    print('🔥 Checking CUDA availability...')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    print('📥 Downloading Ultravox model...')
    from transformers import pipeline
    # Use the correct task and parameters for Ultravox
    uv_pipe = pipeline(
        model='fixie-ai/ultravox-v0_4',
        trust_remote_code=True,
        device_map='auto',
        torch_dtype=torch.float16,
        return_tensors='pt'
    )
    print('✅ Ultravox downloaded')

    print('📥 Downloading ChatterboxTTS model...')
    from chatterbox.tts import ChatterboxTTS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts_model = ChatterboxTTS.from_pretrained(device=device)
    print('✅ ChatterboxTTS downloaded')

    print('📥 Downloading Silero VAD model...')
    torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    print('✅ Silero VAD downloaded')

    print('🎉 All models pre-downloaded successfully!')
    
except Exception as e:
    print(f'❌ Model download error: {e}')
    print('⚠️ Continuing anyway - models will be loaded at runtime...')
"

# Set proper permissions
chmod +x ultraandchat_runpod.py
chmod +x ultraandchat_runpod_websocket.py
chmod +x ultraandchat_runpod_continuous.py

echo "✅ Runpod deployment setup complete"

# Function to display deployment options
show_deployment_options() {
    echo ""
    echo "="*60
    echo "🎯 VOICE ASSISTANT DEPLOYMENT OPTIONS"
    echo "="*60
    echo "1️⃣  WebRTC Version (ultraandchat_runpod.py)"
    echo "    • Ultra-low latency (<500ms)"
    echo "    • May have issues with Runpod UDP restrictions"
    echo "    • Best for: Local development, unrestricted networks"
    echo ""
    echo "2️⃣  WebSocket Version (ultraandchat_runpod_websocket.py) [OLD_BUT_WORKING]"
    echo "    • UDP-free audio streaming"
    echo "    • Reliable on Runpod (TCP-only)"
    echo "    • Moderate latency (~1-2s)"
    echo "    • Best for: Runpod deployment, restricted networks"
    echo ""
    echo "3️⃣  Continuous Conversation (ultraandchat_runpod_continuous.py) [RECOMMENDED NEW]"
    echo "    • Natural 2-way conversation flow"
    echo "    • Auto-resume listening after AI response"
    echo "    • UDP-free with enhanced user experience"
    echo "    • Best for: Interactive voice assistants"
    echo ""
    echo "4️⃣  Deploy All (for testing/comparison)"
    echo "    • WebRTC on port 7860"
    echo "    • WebSocket on port 7861"
    echo "    • Continuous on port 7862"
    echo ""
    echo "="*60
}

# Function to start WebRTC version
start_webrtc() {
    echo "🚀 Starting WebRTC Voice Assistant with PM2..."
    
    # Stop existing process if running
    pm2 delete ultraandchat 2>/dev/null || true
    
    # Start with PM2 using virtual environment
    pm2 start venv/bin/python --name "ultraandchat" -- ultraandchat_runpod.py
    
    echo "✅ WebRTC Voice Assistant started with PM2"
    echo "📡 Access at: https://$RUNPOD_POD_ID-7860.proxy.runpod.net"
}

# Function to start WebSocket version
start_websocket() {
    echo "🚀 Starting WebSocket Voice Assistant with PM2..."
    
    # Stop existing process if running
    pm2 delete ultraandchat-ws 2>/dev/null || true
    
    # Start with PM2 using virtual environment
    pm2 start venv/bin/python --name "ultraandchat-ws" -- ultraandchat_runpod_websocket.py
    
    echo "✅ WebSocket Voice Assistant started with PM2"
    echo "📡 Access at: https://$RUNPOD_POD_ID-7860.proxy.runpod.net"
}

# Function to start Continuous Conversation version
start_continuous() {
    echo "🚀 Starting Continuous Conversation Voice Assistant with PM2..."
    
    # Stop existing process if running
    pm2 delete ultraandchat-continuous 2>/dev/null || true
    
    # Start with PM2 using virtual environment
    pm2 start venv/bin/python --name "ultraandchat-continuous" -- ultraandchat_runpod_continuous.py
    
    echo "✅ Continuous Conversation Voice Assistant started with PM2"
    echo "📡 Access at: https://$RUNPOD_POD_ID-7860.proxy.runpod.net"
}

# Function to start all versions
start_all() {
    echo "🚀 Starting All Voice Assistants with PM2..."
    
    # Stop existing processes if running
    pm2 delete ultraandchat 2>/dev/null || true
    pm2 delete ultraandchat-ws 2>/dev/null || true
    pm2 delete ultraandchat-continuous 2>/dev/null || true
    
    # Start WebRTC version on port 7860
    pm2 start venv/bin/python --name "ultraandchat" -- ultraandchat_runpod.py
    
    # Start WebSocket version on port 7861
    export RUNPOD_TCP_PORT_7860=7861
    pm2 start venv/bin/python --name "ultraandchat-ws" -- ultraandchat_runpod_websocket.py
    
    # Start Continuous version on port 7862
    export RUNPOD_TCP_PORT_7860=7862
    pm2 start venv/bin/python --name "ultraandchat-continuous" -- ultraandchat_runpod_continuous.py
    
    # Reset environment variable
    export RUNPOD_TCP_PORT_7860=7860
    
    echo "✅ All Voice Assistants started with PM2"
    echo "📡 WebRTC Version: https://$RUNPOD_POD_ID-7860.proxy.runpod.net"
    echo "📡 WebSocket Version: https://$RUNPOD_POD_ID-7861.proxy.runpod.net"
    echo "📡 Continuous Version: https://$RUNPOD_POD_ID-7862.proxy.runpod.net"
}

# Check if deployment option is provided as argument
if [ "$1" = "webrtc" ] || [ "$1" = "1" ]; then
    start_webrtc
elif [ "$1" = "websocket" ] || [ "$1" = "2" ]; then
    start_websocket
elif [ "$1" = "continuous" ] || [ "$1" = "3" ]; then
    start_continuous
elif [ "$1" = "all" ] || [ "$1" = "4" ]; then
    start_all
else
    # Interactive mode
    show_deployment_options
    
    echo "🤔 Which version would you like to deploy?"
    echo "Enter your choice (1, 2, 3, or 4): "
    read -r choice
    
    case $choice in
        1|webrtc)
            start_webrtc
            ;;
        2|websocket)
            start_websocket
            ;;
        3|continuous)
            start_continuous
            ;;
        4|all)
            start_all
            ;;
        *)
            echo "❌ Invalid choice. Defaulting to Continuous version (recommended for best experience)..."
            start_continuous
            ;;
    esac
fi

# Show PM2 status
echo ""
echo "📊 PM2 Process Status:"
pm2 list

echo ""
echo "🔧 Useful PM2 Commands:"
echo "  pm2 list                           # Show all processes"
echo "  pm2 logs                           # Show all logs"
echo "  pm2 logs ultraandchat              # Show WebRTC logs"
echo "  pm2 logs ultraandchat-ws           # Show WebSocket logs"
echo "  pm2 logs ultraandchat-continuous   # Show Continuous logs"
echo "  pm2 restart ultraandchat           # Restart WebRTC version"
echo "  pm2 restart ultraandchat-ws        # Restart WebSocket version"
echo "  pm2 restart ultraandchat-continuous # Restart Continuous version"
echo "  pm2 stop all                       # Stop all processes"
echo "  pm2 delete all                     # Delete all processes"

echo ""
echo "🎉 Deployment complete! Your voice assistant is running with PM2."
