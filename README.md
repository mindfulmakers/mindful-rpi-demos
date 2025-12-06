# Mindful Raspberry Pi Demos

There are two demos in this repo:
- `simple-keyboard-journal`: Designed to play sounds and take input from a USB keyboard
- `pipecat-agent`: Uses Pipecat to run an agent that you can interact with through voice (given you have a microphone and speaker connected via the USB ports)

## Setup (might be already done)
- Optional: SSH into your Raspberry Pi - if not, everything will still run on a Mac or Linux machine.
- `cd` into the root of this repo
- Run `python3 -m venv venv` to create a virtual environment.
- Reload the Cursor window Shift - Command - P -> Reload Window
- Cursor terminal will pick up on the virtual environment and activate it when you create new terminal tabs.
- With the `venv` activated, run `pip install -r requirements.txt`.

## Running `simple-keyboard-journal`
- Run `python3 simple-keyboard-journal/main.py` or use the launch config.  Make sure you have a keyboard connected to the Raspberry Pi.

## Running `pipecat-agent`
- Run `python3 pipecat-agent/bot.py` or use the launch config in Cursor
- Always choose ALSA as the output system.
- Choose the USB output device (you may see multiple options)

## Troubleshooting
- You may have to figure out what the sample rate is for your output speaker (or input microphone)
- For the output speaker, test_speaker.py will print the correct output sample rate for your speaker.  Then modify bot.py to use the correct sample rate.
