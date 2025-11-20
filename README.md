# 🎙️ Whisper Live Transcription

Real-time speech transcription using OpenAI's Whisper model with a modern web interface.

## Features

- 🎤 Real-time audio transcription from your microphone
- 🖥️ Clean, modern web interface
- 📝 Full transcription history with timestamps
- 💾 Save transcriptions to text files
- 🚀 Uses Apple Silicon (MPS) acceleration when available
- ⚡ Low-latency processing with overlapping chunks

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For Apple Silicon users:** Make sure you have PyTorch with MPS support:
   ```bash
   pip install --upgrade torch torchvision torchaudio
   ```

## Usage

### Web Interface (Recommended)

Run the GUI version with a modern web interface:

```bash
python live_transcribe_gui.py
```

Then open your browser to `http://127.0.0.1:7860`

**Features:**
- Start/Stop recording with a button
- See live transcription as you speak
- View complete transcription history with timestamps
- Save transcriptions to file
- Download transcriptions directly from the browser

### Command Line Interface

Run the terminal version for a minimal interface:

```bash
python live_transcribe.py
```

Press `Ctrl+C` to stop.

## How It Works

1. **Audio Capture**: Records audio from your microphone in real-time
2. **Buffering**: Maintains a rolling buffer of recent audio
3. **Chunking**: Processes audio in 2-second chunks with 0.5s overlap for accuracy
4. **Transcription**: Uses Whisper model to transcribe each chunk
5. **Display**: Shows results in real-time

## Configuration

You can adjust these parameters in the script:

- `CHUNK_DURATION`: Length of each audio chunk (default: 2.0 seconds)
- `OVERLAP`: Overlap between chunks for better accuracy (default: 0.5 seconds)
- `BUFFER_DURATION`: How much audio history to keep (default: 5.0 seconds)
- Model size: Change `"base"` to `"tiny"`, `"small"`, `"medium"`, or `"large"` for different accuracy/speed tradeoffs

## Requirements

- Python 3.8+
- Microphone access
- ~1GB RAM for the base model
- Apple Silicon recommended for best performance (falls back to CPU)

## Troubleshooting

**No audio input:**
- Check microphone permissions in System Preferences
- Verify your microphone is working with other apps

**Slow transcription:**
- Try using a smaller model (`tiny` or `small`)
- Reduce `CHUNK_DURATION` for faster (but potentially less accurate) results

**Port already in use (GUI):**
- Change the `server_port` parameter in `live_transcribe_gui.py`

## License

MIT License - Feel free to use and modify as needed.
