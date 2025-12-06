#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import sys
from typing import Any, Tuple

from dotenv import load_dotenv
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from select_audio_device import AudioDevice, run_device_selector

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

import os

from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")

logger.info("✅ Silero VAD model loaded")

from pipecat.frames.frames import LLMRunFrame

logger.info("Loading pipeline components...")
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService

OUTPUT_SAMPLE_RATE = 48000


async def main(input_device: int, output_device: int):
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=input_device,
            output_device_index=output_device,
        )
    )

    async def check_user_schedule_for_today() -> list[dict[str, Any]]:
        """Check user schedule for today. Returns a JSON object with the user's schedule for today."""
        return [
            {"type": "text", "text": "User has a meeting with Sally at 11:00 AM today."},
            {
                "type": "text",
                "text": "User has a doctor's appointment with Dr. Johnson at 2:00 PM today.",
            },
        ]

    tools = [check_user_schedule_for_today]

    logger.info(f"Starting bot")
    logger.info(f"Input device: {input_device}")
    logger.info(f"Output device: {output_device}")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="5ee9feff-1265-424a-9d7f-8e4d431a12c7",
        sample_rate=OUTPUT_SAMPLE_RATE,
    )

    llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), tools=tools)

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. Respond naturally and keep your answers conversational.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner()

    await asyncio.gather(runner.run(task))


if __name__ == "__main__":
    # Comment out the lines below to save time debugging once you know the indexes of the input and output devices.
    input_output_devices: Tuple[AudioDevice, AudioDevice, int] = asyncio.run(
        run_device_selector()  # runs the textual app that allows to select input device
    )
    input_index = input_output_devices[0].index
    output_index = input_output_devices[1].index

    # Uncomment below once you know the indexes of the input and output devices.
    # input_index = 1
    # output_index = 2

    asyncio.run(main(input_index, output_index))

