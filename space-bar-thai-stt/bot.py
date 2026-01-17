#!/usr/bin/env python3
"""
Space Bar Thai STT Demo - Pipecat Version

Hold down the space bar to record Thai audio, which is then:
1. Transcribed to Thai text using Deepgram Nova 2
2. Translated to English using OpenAI GPT-4o-mini
3. Spoken aloud using Cartesia TTS

This version uses Pipecat for the audio pipeline.
"""

import asyncio
import os
import sys
import threading

from deepgram import LiveOptions
from dotenv import load_dotenv
from keyboard_handler import (
    IS_MACOS,
    find_keyboard_device,
)
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Hard-coded audio device indices (run select_audio_device.py to find correct values)
INPUT_DEVICE_INDEX = 0
OUTPUT_DEVICE_INDEX = 1

OUTPUT_SAMPLE_RATE = 48000

# Shared state for push-to-talk
spacebar_pressed = False
spacebar_lock = threading.Lock()
pipeline_task: PipelineTask | None = None


def set_spacebar_state(pressed: bool):
    """Thread-safe setter for spacebar state."""
    global spacebar_pressed
    with spacebar_lock:
        spacebar_pressed = pressed
    if pressed:
        logger.info("Recording... (release space bar to stop)")
    else:
        logger.info("Recording stopped.")


def get_spacebar_state() -> bool:
    """Thread-safe getter for spacebar state."""
    with spacebar_lock:
        return spacebar_pressed


class PushToTalkGate(FrameProcessor):
    """Gate that only passes audio frames when spacebar is pressed."""

    def __init__(self):
        super().__init__()
        self._logged_blocking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Block audio-related frames when spacebar is not pressed
        if isinstance(
            frame, (InputAudioRawFrame, UserStartedSpeakingFrame, InterruptionFrame)
        ):
            if get_spacebar_state():
                self._logged_blocking = False
                await self.push_frame(frame, direction)
            else:
                # Log once when we start blocking
                if not self._logged_blocking and isinstance(frame, InputAudioRawFrame):
                    logger.debug(
                        "PushToTalkGate: Blocking audio (spacebar not pressed)"
                    )
                    self._logged_blocking = True
                # Drop the frame (don't push it)
            return

        # Let all other frames through
        await self.push_frame(frame, direction)


def run_keyboard_listener_macos():
    """Run macOS keyboard listener in a separate thread."""
    from pynput import keyboard

    space_held = False

    def on_press(key):
        nonlocal space_held
        if key == keyboard.Key.space and not space_held:
            space_held = True
            set_spacebar_state(True)
        elif key == keyboard.Key.esc:
            logger.info("ESC pressed, exiting...")
            if pipeline_task:
                asyncio.run_coroutine_threadsafe(
                    pipeline_task.cancel(), asyncio.get_event_loop()
                )
            return False

    def on_release(key):
        nonlocal space_held
        if key == keyboard.Key.space and space_held:
            space_held = False
            set_spacebar_state(False)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def run_keyboard_listener_linux(keyboard_device):
    """Run Linux keyboard listener in a separate thread."""
    import evdev
    from evdev import ecodes

    space_held = False

    try:
        for event in keyboard_device.read_loop():
            if event.type == ecodes.EV_KEY:
                key_event = evdev.categorize(event)

                if key_event.scancode == ecodes.KEY_SPACE:
                    if key_event.keystate == evdev.KeyEvent.key_down and not space_held:
                        space_held = True
                        set_spacebar_state(True)
                    elif key_event.keystate == evdev.KeyEvent.key_up and space_held:
                        space_held = False
                        set_spacebar_state(False)

                elif (
                    key_event.scancode == ecodes.KEY_ESC
                    and key_event.keystate == evdev.KeyEvent.key_down
                ):
                    logger.info("ESC pressed, exiting...")
                    if pipeline_task:
                        asyncio.run_coroutine_threadsafe(
                            pipeline_task.cancel(), asyncio.get_event_loop()
                        )
                    break
    except Exception as e:
        logger.error(f"Keyboard listener error: {e}")
    finally:
        keyboard_device.close()


async def main(input_device: int, output_device: int, keyboard_device=None):
    global pipeline_task

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=input_device,
            output_device_index=output_device,
        )
    )

    logger.info(f"Input device: {input_device}")
    logger.info(f"Output device: {output_device}")

    # Deepgram STT with Nova 2 and Thai language
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        live_options=LiveOptions(
            model="nova-2",
            language="th",
            encoding="linear16",
            sample_rate=16000,
            interim_results=True,
            smart_format=True,
            punctuate=True,
        ),
    )

    # OpenAI LLM for translation
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    # Cartesia TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="5ee9feff-1265-424a-9d7f-8e4d431a12c7",
        sample_rate=OUTPUT_SAMPLE_RATE,
    )

    # System prompt for translation
    messages = [
        {
            "role": "system",
            "content": "Translate anything you hear directly into English and say nothing else. Do not add explanations, commentary, or any other text - only output the English translation of what was said.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # Push-to-talk gate
    ptt_gate = PushToTalkGate()

    pipeline = Pipeline(
        [
            transport.input(),
            ptt_gate,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    pipeline_task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        ),
    )

    # Start keyboard listener in a separate thread
    if IS_MACOS:
        keyboard_thread = threading.Thread(
            target=run_keyboard_listener_macos, daemon=True
        )
    else:
        keyboard_thread = threading.Thread(
            target=run_keyboard_listener_linux, args=(keyboard_device,), daemon=True
        )
    keyboard_thread.start()

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Audio transport connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Audio transport disconnected")
        await pipeline_task.cancel()

    runner = PipelineRunner()

    logger.info("=" * 50)
    logger.info("Space Bar Thai STT Demo (Pipecat)")
    logger.info("=" * 50)
    logger.info("Hold SPACE to record Thai speech")
    logger.info("Release to transcribe, translate, and speak")
    logger.info("Press ESC to exit")
    logger.info("=" * 50)

    await runner.run(pipeline_task)


if __name__ == "__main__":
    # Check for API keys
    if not os.getenv("DEEPGRAM_API_KEY"):
        logger.error("DEEPGRAM_API_KEY environment variable is not set")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    if not os.getenv("CARTESIA_API_KEY"):
        logger.error("CARTESIA_API_KEY environment variable is not set")
        sys.exit(1)

    # Find keyboard device (Linux only)
    keyboard_device = None
    if not IS_MACOS:
        keyboard_device = find_keyboard_device()
        if not keyboard_device:
            logger.error("Could not find keyboard device.")
            logger.error("Try: sudo python bot.py")
            logger.error(
                "Or add your user to the 'input' group: sudo usermod -aG input $USER"
            )
            sys.exit(1)

    asyncio.run(main(INPUT_DEVICE_INDEX, OUTPUT_DEVICE_INDEX, keyboard_device))
