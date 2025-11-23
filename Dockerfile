# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio and Whisper
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    alsa-utils \
    pulseaudio \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY live_transcribe_gui.py .
COPY live_transcribe.py .

# Create directory for transcriptions
RUN mkdir -p /app/transcriptions

# Expose Gradio port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "live_transcribe_gui.py"]

