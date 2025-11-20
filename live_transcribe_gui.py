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
import gradio as gr
from datetime import datetime
import signal
import sys

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
running = False
full_transcription = []
current_text = ""
record_thread = None
process_thread = None
complete_audio_buffer = []  # Store all recorded audio for final transcription


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
        print("\n🎙️ Listening...")
        while running:
            sd.sleep(100)


# -------------------------------
# AUDIO PROCESSING
# -------------------------------
def process_audio():
    global current_text, complete_audio_buffer
    buffer = np.zeros(int(BUFFER_DURATION * SAMPLE_RATE), dtype=np.float32)
    step_size = int(CHUNK_DURATION * SAMPLE_RATE)
    overlap_size = int(OVERLAP * SAMPLE_RATE)

    last_transcribed = 0

    while running:
        try:
            new_data = audio_queue.get(timeout=1)
            new_len = len(new_data)

            # Store all audio for final transcription
            complete_audio_buffer.append(new_data.copy())

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

                # Transcribe ONLY for live display (not saved to full transcription)
                result = model.transcribe(
                    filename,
                    fp16=False,
                    language="en",
                    condition_on_previous_text=False,  # Prevent hallucinations
                    temperature=0.0,  # More deterministic output
                    no_speech_threshold=0.6,  # Higher threshold to filter out silence
                )
                text = result["text"].strip()

                # Update live display only
                if text and len(text) > 3:
                    current_text = text

                os.remove(filename)

        except queue.Empty:
            continue


# -------------------------------
# GRADIO INTERFACE
# -------------------------------
def start_recording():
    """Start the recording and transcription process."""
    global running, record_thread, process_thread, full_transcription, current_text, complete_audio_buffer

    if running:
        return (
            "⚠️ Already recording!",
            "",
            "⏸️ Stop Recording",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    # Reset state
    full_transcription = []
    current_text = ""
    complete_audio_buffer = []
    running = True

    # Clear the queue
    while not audio_queue.empty():
        audio_queue.get()

    # Start threads
    record_thread = threading.Thread(target=record_audio, daemon=True)
    process_thread = threading.Thread(target=process_audio, daemon=True)

    record_thread.start()
    process_thread.start()

    return (
        "🎙️ Recording... Speak into your microphone",
        "",
        "⏸️ Stop Recording",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def stop_recording():
    """Stop the recording and transcription process."""
    global running, record_thread, process_thread, complete_audio_buffer

    if not running:
        return (
            "⚠️ Not currently recording!",
            "",
            "🎙️ Start Recording",
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    running = False

    # Give threads time to see the flag
    time.sleep(0.3)

    # Wait for threads to finish
    if record_thread and record_thread.is_alive():
        record_thread.join(timeout=2)
    if process_thread and process_thread.is_alive():
        process_thread.join(timeout=2)

    # Now transcribe the COMPLETE audio buffer for accurate final transcription
    if complete_audio_buffer:
        print("🔄 Generating final accurate transcription...")

        # Concatenate all audio chunks
        full_audio = np.concatenate(complete_audio_buffer)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            filename = tmp.name
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(
                    (full_audio.flatten() * 32767).astype(np.int16).tobytes()
                )

        # Transcribe the entire recording at once
        result = model.transcribe(
            filename,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
        )

        final_text = result["text"].strip()
        os.remove(filename)

        if not final_text:
            final_text = "No speech detected in recording."
    else:
        final_text = "No audio recorded."

    return (
        "✅ Recording stopped. Final transcription ready!",
        final_text,
        "🎙️ Start Recording",
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def clear_transcription():
    """Clear the current transcription."""
    global full_transcription, current_text, running, complete_audio_buffer

    if running:
        return "⚠️ Stop recording before clearing!", gr.update(), gr.update()

    full_transcription = []
    current_text = ""
    complete_audio_buffer = []

    return "🗑️ Transcription cleared. Ready to start new recording.", "", ""


def update_live_text():
    """Generator function to update live transcription."""
    while True:
        if running:
            yield current_text
        else:
            yield current_text
        time.sleep(0.5)


def save_transcription(text):
    """Save transcription to a file."""
    if not text or text == "No transcription recorded.":
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcription_{timestamp}.txt"
    filepath = os.path.join(os.getcwd(), filename)

    with open(filepath, "w") as f:
        f.write(text)

    return filepath


# -------------------------------
# BUILD INTERFACE
# -------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Live Transcription") as demo:
    gr.Markdown(
        """
        # 🎙️ Live Transcription with Whisper
        Start recording to see real-time transcription. Stop to get the full text and save it.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="Status", value="Ready to start", interactive=False, lines=1
            )

        with gr.Column(scale=1):
            toggle_btn = gr.Button("🎙️ Start Recording", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔴 Live Transcription")
            live_text = gr.Textbox(
                label="Current Speech",
                placeholder="Start recording to see live transcription...",
                lines=6,
                max_lines=10,
                interactive=False,
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Full Transcription")
            full_text = gr.Textbox(
                label="Complete Recording",
                placeholder="Stop recording to see full transcription...",
                lines=15,
                max_lines=25,
                interactive=False,
            )

    with gr.Row():
        save_btn = gr.Button("💾 Save Transcription", variant="secondary", size="lg")
        clear_btn = gr.Button("🗑️ Clear Transcription", variant="stop", size="lg")

    with gr.Row():
        download_file = gr.File(label="Download", interactive=False)

    # Event handlers
    is_recording = gr.State(False)

    def toggle_recording(is_rec):
        if not is_rec:
            result = start_recording()
            return result + (True,)
        else:
            result = stop_recording()
            return result + (False,)

    toggle_btn.click(
        fn=toggle_recording,
        inputs=[is_recording],
        outputs=[status_box, full_text, toggle_btn, save_btn, clear_btn, is_recording],
    )

    clear_btn.click(fn=clear_transcription, outputs=[status_box, full_text, live_text])

    # Update live transcription every 0.5 seconds using timer
    timer = gr.Timer(0.5)
    timer.tick(fn=lambda: current_text, outputs=live_text)

    save_btn.click(fn=save_transcription, inputs=[full_text], outputs=[download_file])

    # Stop recording when demo closes
    demo.load(lambda: None)
    demo.unload(lambda: stop_recording() if running else None)


# -------------------------------
# CLEANUP & SIGNAL HANDLING
# -------------------------------
def cleanup():
    """Clean up resources and stop threads."""
    global running, record_thread, process_thread, complete_audio_buffer
    print("\n🛑 Shutting down...")
    running = False

    # Give threads a moment to see the flag
    time.sleep(0.5)

    # Wait for threads to finish
    if record_thread and record_thread.is_alive():
        print("⏳ Waiting for record thread...")
        record_thread.join(timeout=3)
    if process_thread and process_thread.is_alive():
        print("⏳ Waiting for process thread...")
        process_thread.join(timeout=3)

    # Clear audio buffer
    complete_audio_buffer = []

    # Close Gradio demo
    try:
        if "demo" in globals():
            print("🔒 Closing Gradio server...")
            demo.close()
    except Exception as e:
        print(f"⚠️ Error closing Gradio: {e}")

    print("✅ Cleanup complete")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n⚠️ Interrupt received...")
    cleanup()
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# -------------------------------
# LAUNCH
# -------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 Starting Live Transcription Interface...")
    print("=" * 50 + "\n")
    print("💡 Press Ctrl+C in terminal to stop the server")
    print("=" * 50 + "\n")

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=False,
            prevent_thread_lock=False,
        )
    except KeyboardInterrupt:
        print("\n⚠️ KeyboardInterrupt detected")
        cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        cleanup()
    finally:
        cleanup()
