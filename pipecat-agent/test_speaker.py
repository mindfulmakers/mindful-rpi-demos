#!/usr/bin/env python3
#
# PyAudio test script to play an audio file
#

import asyncio
import audioop
import os
import sys
import wave
from typing import Tuple

import pyaudio
from select_audio_device import AudioDevice, run_device_selector


def resample_audio(
    data: bytes, sample_width: int, channels: int, in_rate: int, out_rate: int
) -> bytes:
    """
    Resample audio data from one sample rate to another.

    Args:
        data: Raw audio data
        sample_width: Sample width in bytes
        channels: Number of channels
        in_rate: Input sample rate
        out_rate: Output sample rate

    Returns:
        Resampled audio data
    """
    if in_rate == out_rate:
        return data

    # Use audioop.ratecv for resampling
    converted, _ = audioop.ratecv(data, sample_width, channels, in_rate, out_rate, None)
    return converted


def play_audio_file(file_path: str, output_device: AudioDevice, chunk_size: int = 1024):
    """
    Play an audio file using PyAudio.

    Args:
        file_path: Path to the audio file (WAV format)
        output_device: The output AudioDevice to use
        chunk_size: Size of audio chunks to read (default: 1024)
    """

    print(f"Output device: {output_device}")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return False

    try:
        # Open the audio file
        wf = wave.open(file_path, "rb")

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        # Get audio file properties
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        file_sample_rate = wf.getframerate()
        frames = wf.getnframes()

        # Get device's supported sample rate
        device_sample_rate = int(output_device.default_sample_rate)

        print(f"Playing: {file_path}")
        print(f"File sample rate: {file_sample_rate} Hz")
        print(f"Device sample rate: {device_sample_rate} Hz")
        print(f"Channels: {channels}")
        print(f"Sample width: {sample_width} bytes")
        print(f"Duration: {frames / file_sample_rate:.2f} seconds")
        print(f"Output device: {output_device.name} (Index: {output_device.index})")

        needs_resample = file_sample_rate != device_sample_rate
        if needs_resample:
            print(f"Resampling from {file_sample_rate} Hz to {device_sample_rate} Hz")
        print()

        # Open audio stream with device's supported sample rate
        stream = p.open(
            format=p.get_format_from_width(sample_width),
            channels=channels,
            rate=device_sample_rate,
            output=True,
            output_device_index=output_device.index,
        )

        # Read and play audio data
        data = wf.readframes(chunk_size)
        while data:
            if needs_resample:
                data = resample_audio(
                    data, sample_width, channels, file_sample_rate, device_sample_rate
                )
            stream.write(data)
            data = wf.readframes(chunk_size)

        # Clean up
        stream.stop_stream()
        stream.close()
        wf.close()
        p.terminate()

        print("Playback complete!")
        return True

    except wave.Error as e:
        print(f"Error: Not a valid WAV file. {e}")
        return False
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False


async def main(file_path: str):
    """Main async function that runs the device selector and plays the audio file."""
    # Run the device selector (same as bot.py)
    res: Tuple[AudioDevice, AudioDevice, int] = await run_device_selector()

    input_device = res[0]
    output_device = res[1]
    host_api = res[2]

    print(f"Selected input device: {input_device.name} (Index: {input_device.index})")
    print(f"Selected output device: {output_device.name} (Index: {output_device.index})")
    print(f"Host API: {host_api}")
    print()

    # Play the audio file using the selected output device
    play_audio_file(file_path, output_device)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        asyncio.run(main(file_path))
    else:
        print("Usage: python test.py <audio_file.wav>")
        print()
        print("Example: python test.py test_audio.wav")
        print()
        print("Note: You will be prompted to select input and output audio devices.")
        sys.exit(1)
