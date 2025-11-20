# 🎙️ Whisper Live Transcription

Real-time speech transcription using OpenAI's Whisper model with a modern web interface.

**Built with:** OpenAI Whisper • PyTorch • Gradio • NumPy • SoundDevice

## Features

- 🎤 Real-time audio transcription from your microphone
- 🖥️ Clean, modern web interface
- 📝 Live preview of speech as you talk
- 💾 Accurate final transcription when you stop recording
- 🚀 Uses Apple Silicon (MPS) acceleration when available
- ⚡ Low-latency live display with complete audio transcription on stop

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

Press `Ctrl+C` in the terminal to stop the server cleanly.

**Features:**
- Start/Stop recording with a button
- See live transcription preview as you speak
- Get accurate complete transcription when you stop recording
- Save transcriptions to file
- Download transcriptions directly from the browser
- Clean transcriptions without hallucinations or repeated phrases

### Command Line Interface

Run the terminal version for a minimal interface:

```bash
python live_transcribe.py
```

Press `Ctrl+C` to stop.

## How It Works

The application uses a two-stage transcription approach for optimal accuracy:

### Live Preview (While Recording)
1. **Audio Capture**: Records audio from your microphone in real-time
2. **Buffering**: Maintains a rolling buffer of recent audio (5 seconds)
3. **Chunking**: Processes audio in 2-second chunks with 0.5s overlap
4. **Live Display**: Shows real-time transcription preview as you speak

### Final Transcription (On Stop)
1. **Complete Audio Buffer**: All recorded audio is stored throughout the session
2. **Single-Pass Transcription**: When you stop, the entire recording is transcribed at once
3. **Accurate Results**: Full context produces clean transcription without hallucinations or repeated phrases

This approach gives you the best of both worlds: immediate feedback while recording and highly accurate final results.

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

**Final transcription takes time:**
- The final transcription processes the entire recording for accuracy
- Longer recordings will take a few seconds to process
- This is normal and ensures you get clean, accurate results

**Slow transcription:**
- Try using a smaller model (`tiny` or `small`)
- Reduce `CHUNK_DURATION` for faster (but potentially less accurate) live preview

**Port already in use (GUI):**
- Change the `server_port` parameter in `live_transcribe_gui.py`

**Program won't stop with Ctrl+C:**
- The cleanup should now work properly
- If it still hangs, use `Ctrl+Z` then `kill %1` as a last resort

## License

MIT License - Feel free to use and modify as needed.
