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

# Install Python dependencies from requirements file
echo "🐍 Installing Python dependencies from requirements_runpod.txt..."
pip install -r requirements.txt

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
chmod +x websockets.py

echo "✅ Runpod deployment setup complete"

# Function to start WebSocket Voice Assistant
start_websocket() {
    echo "🚀 Starting WebSocket Voice Assistant with PM2..."

    # Stop existing process if running
    pm2 delete ultraandchat-websocket 2>/dev/null || true

    # Start with PM2 using virtual environment
    pm2 start venv/bin/python --name "ultraandchat-websocket" -- ultraandchat_runpod_websocket.py

    echo "✅ WebSocket Voice Assistant started with PM2"
    echo "📡 Access at: https://$RUNPOD_POD_ID-7860.proxy.runpod.net"
}

# Start the WebSocket voice assistant
start_websocket

# Show PM2 status
echo ""
echo "📊 PM2 Process Status:"
pm2 list

echo ""
echo "🔧 Useful PM2 Commands:"
echo "  pm2 list                           # Show all processes"
echo "  pm2 logs                           # Show all logs"
echo "  pm2 logs ultraandchat-websocket    # Show WebSocket logs"
echo "  pm2 restart ultraandchat-websocket # Restart WebSocket version"
echo "  pm2 stop ultraandchat-websocket    # Stop WebSocket process"
echo "  pm2 delete ultraandchat-websocket  # Delete WebSocket process"

echo ""
echo "🎉 WebSocket Voice Assistant deployment complete! Your voice assistant is running with PM2."
