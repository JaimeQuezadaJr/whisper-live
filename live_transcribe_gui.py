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
BUFFER_DURATION = 5.0  # how much past audio to keep for preview

audio_queue = queue.Queue()
running = False
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
            print(f"⚠️ Audio status: {status}")
        try:
            if running:
                audio_queue.put(indata.copy())
        except Exception as e:
            print(f"❌ Error in audio callback: {e}")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, callback=callback, dtype="float32"
        ):
            print("\n🎙️ Listening...")
            while running:
                sd.sleep(100)
    except Exception as e:
        print(f"❌ Error in record_audio: {e}")
    finally:
        print("🔄 Recording thread stopped")


# -------------------------------
# AUDIO PROCESSING
# -------------------------------
def process_audio():
    """Process audio chunks for live preview and store for final transcription."""
    global current_text, complete_audio_buffer
    buffer = np.zeros(int(BUFFER_DURATION * SAMPLE_RATE), dtype=np.float32)
    step_size = int(CHUNK_DURATION * SAMPLE_RATE)
    overlap_size = int(OVERLAP * SAMPLE_RATE)

    last_transcribed = 0
    chunks_processed = 0

    print("🔄 Audio processing thread started")

    while running:
        try:
            new_data = audio_queue.get(timeout=1)

            if new_data is None or len(new_data) == 0:
                continue

            new_len = len(new_data)
            chunks_processed += 1

            # Store all audio for final transcription when stopped
            complete_audio_buffer.append(new_data.copy())

            # Update rolling buffer for live preview
            buffer = np.roll(buffer, -new_len)
            buffer[-new_len:] = new_data.flatten()

            last_transcribed += new_len

            # Live preview transcription every ~2 seconds
            if last_transcribed >= step_size - overlap_size:
                last_transcribed = 0
                filename = None
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp:
                        filename = tmp.name
                        with wave.open(filename, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes((buffer * 32767).astype(np.int16).tobytes())

                    # Quick transcription for live preview only
                    result = model.transcribe(
                        filename,
                        fp16=False,
                        language="en",
                        condition_on_previous_text=False,
                        temperature=0.0,
                        no_speech_threshold=0.6,
                    )
                    text = result["text"].strip()

                    # Update live preview
                    if text and len(text) > 3:
                        current_text = text

                    if chunks_processed % 50 == 0:
                        print(
                            f"📊 Processed {chunks_processed} chunks, buffer: {len(complete_audio_buffer)}"
                        )

                except Exception as e:
                    print(f"⚠️ Error in preview transcription: {e}")
                finally:
                    if filename and os.path.exists(filename):
                        try:
                            os.remove(filename)
                        except:
                            pass

        except queue.Empty:
            # Normal during silence - just continue
            continue
        except Exception as e:
            print(f"❌ Critical error in process_audio: {e}")
            import traceback

            traceback.print_exc()
            # Don't exit - keep running
            continue

    print(f"🔄 Audio processing thread stopped (processed {chunks_processed} chunks)")


# -------------------------------
# GRADIO INTERFACE
# -------------------------------
def start_recording():
    """Start the recording and transcription process."""
    global running, record_thread, process_thread, current_text, complete_audio_buffer

    if running:
        print("⚠️ Already recording!")
        return (
            "⚠️ Already recording!",
            "",
            "⏸️ Stop Recording",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    print("🎬 Starting new recording session...")

    # Reset state
    current_text = ""
    complete_audio_buffer = []

    # Clear the queue
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except:
            break

    running = True

    # Start threads
    record_thread = threading.Thread(target=record_audio, daemon=True)
    process_thread = threading.Thread(target=process_audio, daemon=True)

    record_thread.start()
    process_thread.start()

    print("✅ Recording threads started")

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

    print("⏸️ Stopping recording...")
    running = False

    # Give threads time to finish
    time.sleep(0.5)

    # Wait for threads to finish
    if record_thread and record_thread.is_alive():
        print("⏳ Waiting for record thread...")
        record_thread.join(timeout=3)
    if process_thread and process_thread.is_alive():
        print("⏳ Waiting for process thread...")
        process_thread.join(timeout=3)

    # Now transcribe the complete recording for final accurate transcription
    final_text = ""
    if complete_audio_buffer and len(complete_audio_buffer) > 0:
        try:
            print(
                f"🔄 Generating final transcription from {len(complete_audio_buffer)} chunks..."
            )

            # Concatenate all audio chunks
            full_audio = np.concatenate(complete_audio_buffer)

            if len(full_audio) > SAMPLE_RATE * 0.3:  # At least 0.3 seconds
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

                # Transcribe entire recording with full context
                result = model.transcribe(
                    filename,
                    fp16=False,
                    language="en",
                    condition_on_previous_text=False,
                    temperature=0.0,
                    no_speech_threshold=0.6,
                )

                final_text = result["text"].strip()
                print(f"✅ Final transcription: {len(final_text)} characters")

                try:
                    os.remove(filename)
                except:
                    pass

                if not final_text:
                    final_text = "No speech detected in recording."
            else:
                final_text = "Recording too short."
        except Exception as e:
            print(f"⚠️ Error generating final transcription: {e}")
            import traceback

            traceback.print_exc()
            final_text = "Error processing recording."
    else:
        final_text = "No audio recorded."

    print("✅ Recording stopped")

    return (
        "✅ Recording stopped. Final transcription ready!",
        final_text,
        "🎙️ Start Recording",
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def clear_transcription():
    """Clear the current transcription."""
    global current_text, running, complete_audio_buffer

    if running:
        return "⚠️ Stop recording before clearing!", gr.update(), gr.update()

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
        Start recording to see real-time transcription preview. Stop recording to get your complete, accurate transcription.
        **Live Preview** shows what you're saying now • **Full Transcription** appears when you stop
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
            gr.Markdown("### 🔴 Live Preview")
            live_text = gr.Textbox(
                label="Current Speech",
                placeholder="Start recording to see live transcription preview...",
                lines=6,
                max_lines=10,
                interactive=False,
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Full Transcription")
            full_text = gr.Textbox(
                label="Complete Recording",
                placeholder="Stop recording to see complete transcription...",
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

    # Update live preview every 0.5 seconds
    live_timer = gr.Timer(0.5)
    live_timer.tick(fn=lambda: current_text, outputs=live_text)

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

    # Clear the queue to help threads exit
    try:
        while not audio_queue.empty():
            audio_queue.get_nowait()
    except:
        pass

    # Wait for threads to finish
    if record_thread and record_thread.is_alive():
        print("⏳ Waiting for record thread...")
        record_thread.join(timeout=3)
        if record_thread.is_alive():
            print("⚠️ Record thread still alive, forcing exit...")

    if process_thread and process_thread.is_alive():
        print("⏳ Waiting for process thread...")
        process_thread.join(timeout=3)
        if process_thread.is_alive():
            print("⚠️ Process thread still alive, forcing exit...")

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
    print("\n⚠️ Interrupt received (Ctrl+C)...")
    cleanup()
    print("👋 Exiting...")
    os._exit(0)  # Force exit if cleanup doesn't work


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
    except Exception as e:
        print(f"\n❌ Error during launch: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cleanup()
        print("👋 Goodbye!")
        os._exit(0)
