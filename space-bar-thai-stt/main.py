#!/usr/bin/env python3
"""
Space Bar Thai STT Demo

Hold down the space bar to record audio, which is then:
1. Transcribed to Thai text using OpenAI Whisper
2. Translated to English using GPT-4o-mini
3. Spoken aloud using Cartesia TTS

This demo does not use pipecat - it directly uses OpenAI and Cartesia APIs.
"""

import io
import os
import sys
import wave
import struct
import threading
import tempfile
from pathlib import Path

import pyaudio

# Hard-coded audio device indices (run select_audio_device.py to find correct values)
INPUT_DEVICE_INDEX = 1
OUTPUT_DEVICE_INDEX = 2
import requests
import evdev
from evdev import ecodes
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv(override=True)

# Configure logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# Audio recording settings
SAMPLE_RATE = 16000  # Whisper works well with 16kHz
CHANNELS = 1
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

# OpenAI settings
WHISPER_MODEL = "whisper-1"
TRANSLATION_MODEL = "gpt-4o-mini"

# Cartesia TTS settings
CARTESIA_VOICE_ID = "5ee9feff-1265-424a-9d7f-8e4d431a12c7"
CARTESIA_MODEL_ID = "sonic-2"
CARTESIA_OUTPUT_SAMPLE_RATE = 48000


class SpaceBarRecorder:
    """Records audio while space bar is held, then processes with OpenAI and Cartesia."""

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cartesia_api_key = os.getenv("CARTESIA_API_KEY")
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.recording_lock = threading.Lock()

        # Use hard-coded audio device indices
        self.input_device = INPUT_DEVICE_INDEX
        self.output_device = OUTPUT_DEVICE_INDEX

        logger.info(f"Using input device: {self.input_device}")
        logger.info(f"Using output device: {self.output_device}")

    def start_recording(self):
        """Start recording audio."""
        with self.recording_lock:
            if self.is_recording:
                return

            self.is_recording = True
            self.frames = []

            try:
                self.stream = self.audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    input_device_index=self.input_device,
                    frames_per_buffer=CHUNK_SIZE,
                )
                logger.info("Recording started... (release space bar to stop)")
            except Exception as e:
                logger.error(f"Failed to start recording: {e}")
                self.is_recording = False
                return

        # Record in a separate thread
        self.record_thread = threading.Thread(target=self._record_audio)
        self.record_thread.start()

    def _record_audio(self):
        """Record audio frames while recording flag is set."""
        while self.is_recording:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                logger.error(f"Error reading audio: {e}")
                break

    def stop_recording(self):
        """Stop recording and process the audio."""
        with self.recording_lock:
            if not self.is_recording:
                return

            self.is_recording = False

        # Wait for recording thread to finish
        if hasattr(self, 'record_thread'):
            self.record_thread.join()

        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        if not self.frames:
            logger.warning("No audio recorded")
            return

        logger.info(f"Recording stopped. Captured {len(self.frames)} chunks.")

        # Process the recorded audio
        self._process_audio()

    def _process_audio(self):
        """Process recorded audio: STT -> Translate -> TTS."""
        # Save audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            with wave.open(tmp_file, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(self.frames))

        try:
            # Step 1: Transcribe Thai audio using Whisper
            logger.info("Transcribing Thai audio...")
            thai_text = self._transcribe_thai(tmp_path)

            if not thai_text:
                logger.warning("No speech detected in recording")
                return

            logger.info(f"Thai transcription: {thai_text}")
            print(f"\n{'='*50}")
            print(f"Thai: {thai_text}")

            # Step 2: Translate to English
            logger.info("Translating to English...")
            english_text = self._translate_to_english(thai_text)
            logger.info(f"English translation: {english_text}")
            print(f"English: {english_text}")
            print(f"{'='*50}\n")

            # Step 3: Speak the English translation
            logger.info("Generating speech...")
            self._speak_text(english_text)

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    def _transcribe_thai(self, audio_path: str) -> str:
        """Transcribe audio to Thai text using Whisper."""
        with open(audio_path, "rb") as audio_file:
            transcript = self.openai_client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                language="th",  # Thai language code
            )
        return transcript.text.strip()

    def _translate_to_english(self, thai_text: str) -> str:
        """Translate Thai text to English using GPT-4o-mini."""
        response = self.openai_client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Thai to English translator. Translate the given Thai text to natural English. Only output the translation, nothing else."
                },
                {
                    "role": "user",
                    "content": thai_text
                }
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()

    def _speak_text(self, text: str):
        """Convert text to speech and play it using Cartesia TTS."""
        response = requests.post(
            "https://api.cartesia.ai/tts/bytes",
            headers={
                "X-API-Key": self.cartesia_api_key,
                "Cartesia-Version": "2024-06-10",
                "Content-Type": "application/json",
            },
            json={
                "model_id": CARTESIA_MODEL_ID,
                "transcript": text,
                "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": CARTESIA_OUTPUT_SAMPLE_RATE,
                },
            },
        )

        if response.status_code != 200:
            logger.error(f"Cartesia TTS error: {response.status_code} - {response.text}")
            return

        # Get raw PCM audio data
        audio_data = response.content

        # Play the audio
        self._play_pcm_audio(audio_data)

    def _play_pcm_audio(self, audio_data: bytes):
        """Play raw PCM audio data through the speaker."""
        output_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=CARTESIA_OUTPUT_SAMPLE_RATE,
            output=True,
            output_device_index=self.output_device,
        )

        # Play in chunks
        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            output_stream.write(audio_data[i:i + chunk_size])

        output_stream.stop_stream()
        output_stream.close()

        logger.info("Playback complete")

    def cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.close()
        self.audio.terminate()


def find_keyboard_device():
    """Find the keyboard input device."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    
    for device in devices:
        capabilities = device.capabilities()
        # Check if device has EV_KEY capability and has keyboard keys
        if ecodes.EV_KEY in capabilities:
            keys = capabilities[ecodes.EV_KEY]
            # Check for typical keyboard keys (space, escape, letters)
            if ecodes.KEY_SPACE in keys and ecodes.KEY_ESC in keys:
                logger.info(f"Found keyboard: {device.name} ({device.path})")
                return device
    
    # If no keyboard found, list available devices for debugging
    logger.error("No keyboard device found. Available devices:")
    for device in devices:
        logger.error(f"  {device.path}: {device.name}")
    return None


def main():
    """Main entry point."""
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        logger.error("Please set it in your .env file or environment")
        sys.exit(1)

    if not os.getenv("CARTESIA_API_KEY"):
        logger.error("CARTESIA_API_KEY environment variable is not set")
        logger.error("Please set it in your .env file or environment")
        sys.exit(1)

    # Find keyboard device
    keyboard_device = find_keyboard_device()
    if not keyboard_device:
        logger.error("Could not find keyboard. Make sure you're running with proper permissions.")
        logger.error("Try: sudo python main.py")
        logger.error("Or add your user to the 'input' group: sudo usermod -aG input $USER")
        sys.exit(1)

    logger.info("Space Bar Thai STT Demo")
    logger.info("=" * 40)
    logger.info("Hold SPACE to record Thai speech")
    logger.info("Release to transcribe, translate, and speak")
    logger.info("Press ESC to exit")
    logger.info("=" * 40)

    recorder = SpaceBarRecorder()
    space_pressed = False

    try:
        for event in keyboard_device.read_loop():
            if event.type == ecodes.EV_KEY:
                key_event = evdev.categorize(event)
                
                # Space bar pressed
                if key_event.scancode == ecodes.KEY_SPACE:
                    if key_event.keystate == evdev.KeyEvent.key_down and not space_pressed:
                        space_pressed = True
                        recorder.start_recording()
                    elif key_event.keystate == evdev.KeyEvent.key_up and space_pressed:
                        space_pressed = False
                        recorder.stop_recording()
                
                # ESC pressed - exit
                elif key_event.scancode == ecodes.KEY_ESC and key_event.keystate == evdev.KeyEvent.key_down:
                    logger.info("Exiting...")
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted...")
    finally:
        recorder.cleanup()
        keyboard_device.close()
    
    logger.info("Goodbye!")


if __name__ == "__main__":
    main()
