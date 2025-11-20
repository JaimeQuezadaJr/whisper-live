import whisper
import sounddevice as sd
import numpy as np
import torch
import queue
import threading
import tempfile
import wave
import time
import os

# -------------------------------
# SETUP
# -------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Using device: {device}")

print("🔊 Loading Whisper model (base)...")
model = whisper.load_model("base", device=device)

SAMPLE_RATE = 16000
CHUNK_DURATION = 2.0  # seconds per chunk
OVERLAP = 0.5  # overlap between chunks (in seconds)
BUFFER_DURATION = 5.0  # how much past audio to keep

audio_queue = queue.Queue()
running = True
last_text = ""


# -------------------------------
# AUDIO CAPTURE
# -------------------------------
def record_audio():
    """Continuously records from mic and places audio chunks in a queue."""

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_queue.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=callback, dtype="float32"
    ):
        print("\n🎙️ Listening... Press Ctrl+C to stop.\n")
        while running:
            sd.sleep(100)


# -------------------------------
# AUDIO PROCESSING
# -------------------------------
def process_audio():
    global last_text
    buffer = np.zeros(int(BUFFER_DURATION * SAMPLE_RATE), dtype=np.float32)
    step_size = int(CHUNK_DURATION * SAMPLE_RATE)
    overlap_size = int(OVERLAP * SAMPLE_RATE)

    last_transcribed = 0

    while running:
        try:
            new_data = audio_queue.get(timeout=1)
            new_len = len(new_data)

            buffer = np.roll(buffer, -new_len)
            buffer[-new_len:] = new_data.flatten()

            last_transcribed += new_len
            if last_transcribed >= step_size - overlap_size:
                last_transcribed = 0
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    filename = tmp.name
                    with wave.open(filename, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes((buffer * 32767).astype(np.int16).tobytes())

                result = model.transcribe(filename, fp16=False, language="en")
                text = result["text"].strip()

                if text != last_text:
                    os.system("clear" if os.name == "posix" else "cls")
                    print("🎧 Live Transcription (Whisper Medium):\n")
                    print(text)
                    last_text = text

                os.remove(filename)

        except queue.Empty:
            continue


# -------------------------------
# MAIN
# -------------------------------
record_thread = threading.Thread(target=record_audio)
process_thread = threading.Thread(target=process_audio)

try:
    record_thread.start()
    process_thread.start()
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    running = False
    print("\n🛑 Stopping live transcription...")
    record_thread.join()
    process_thread.join()
