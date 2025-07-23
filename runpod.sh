#!/bin/bash

# Runpod Deployment Script for Ultra-Fast Voice Assistant
echo "🚀 Starting Runpod deployment for Ultra-Fast Voice Assistant"

# Set environment variables for Runpod
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p /tmp/logs

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    portaudio19-dev \
    python3-dev \
    build-essential

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Pre-download models to avoid cold start delays
echo "📥 Pre-downloading models..."
python3 -c "
import torch
from transformers import pipeline
from chatterbox.tts import ChatterboxTTS

print('Downloading Ultravox model...')
pipeline('automatic-speech-recognition', model='fixie-ai/ultravox-v0_4', trust_remote_code=True)

print('Downloading ChatterboxTTS model...')
ChatterboxTTS.from_pretrained()

print('Downloading Silero VAD model...')
torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)

print('✅ All models pre-downloaded')
"

# Set proper permissions
chmod +x main.py

echo "✅ Runpod deployment setup complete"
echo "🎯 Run: python3 main.py"
