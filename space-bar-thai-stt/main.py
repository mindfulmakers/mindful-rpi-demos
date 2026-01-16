#!/usr/bin/env python3
"""
Space Bar Thai STT Demo

Hold down the space bar to record audio, which is then:
1. Transcribed to Thai text using OpenAI Whisper
2. Translated to English using GPT-4o-mini
3. Spoken aloud using OpenAI TTS

This demo does not use pipecat - it directly uses OpenAI APIs.
"""

import io
import os
import sys
import wave
import threading
import tempfile
from pathlib import Path

import pyaudio
from pynput import keyboard
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
TTS_MODEL = "tts-1"
TTS_VOICE = "alloy"


class SpaceBarRecorder:
    """Records audio while space bar is held, then processes with OpenAI."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.recording_lock = threading.Lock()

        # Select audio devices
        self.input_device, self.output_device = self._select_audio_devices()

    def _select_audio_devices(self) -> tuple[int | None, int | None]:
        """List and select audio devices."""
        logger.info("Available audio devices:")

        input_device = None
        output_device = None

        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            name = device_info["name"]
            max_input = device_info["maxInputChannels"]
            max_output = device_info["maxOutputChannels"]

            device_type = []
            if max_input > 0:
                device_type.append("input")
            if max_output > 0:
                device_type.append("output")

            logger.info(f"  [{i}] {name} ({', '.join(device_type)})")

            # Auto-select first available input/output devices
            if input_device is None and max_input > 0:
                input_device = i
            if output_device is None and max_output > 0:
                output_device = i

        logger.info(f"Selected input device: {input_device}")
        logger.info(f"Selected output device: {output_device}")

        return input_device, output_device

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
            transcript = self.client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                language="th",  # Thai language code
            )
        return transcript.text.strip()

    def _translate_to_english(self, thai_text: str) -> str:
        """Translate Thai text to English using GPT-4o-mini."""
        response = self.client.chat.completions.create(
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
        """Convert text to speech and play it using OpenAI TTS."""
        response = self.client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="wav",
        )

        # Get audio data
        audio_data = response.content

        # Play the audio
        self._play_audio(audio_data)

    def _play_audio(self, audio_data: bytes):
        """Play WAV audio data through the speaker."""
        # Read WAV data
        wav_io = io.BytesIO(audio_data)
        with wave.open(wav_io, 'rb') as wf:
            output_stream = self.audio.open(
                format=self.audio.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                output_device_index=self.output_device,
            )

            # Read and play chunks
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            while data:
                output_stream.write(data)
                data = wf.readframes(chunk_size)

            output_stream.stop_stream()
            output_stream.close()

        logger.info("Playback complete")

    def cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.close()
        self.audio.terminate()


def main():
    """Main entry point."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        logger.error("Please set it in your .env file or environment")
        sys.exit(1)

    logger.info("Space Bar Thai STT Demo")
    logger.info("=" * 40)
    logger.info("Hold SPACE to record Thai speech")
    logger.info("Release to transcribe, translate, and speak")
    logger.info("Press ESC to exit")
    logger.info("=" * 40)

    recorder = SpaceBarRecorder()
    space_pressed = False

    def on_press(key):
        nonlocal space_pressed
        if key == keyboard.Key.space and not space_pressed:
            space_pressed = True
            recorder.start_recording()
        elif key == keyboard.Key.esc:
            logger.info("Exiting...")
            return False  # Stop listener

    def on_release(key):
        nonlocal space_pressed
        if key == keyboard.Key.space and space_pressed:
            space_pressed = False
            recorder.stop_recording()

    # Start keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    recorder.cleanup()
    logger.info("Goodbye!")


if __name__ == "__main__":
    main()
