# Mindful Raspberry Pi Demos

There are three demos in this repo:
- `simple-keyboard-journal`: Designed to play sounds and take input from a USB keyboard
- `pipecat-agent`: Uses Pipecat to run an agent that you can interact with through voice (given you have a microphone and speaker connected via the USB ports)
- `space-bar-thai-stt`: Hold space bar to record Thai speech, which is transcribed, translated to English, and spoken aloud (uses OpenAI APIs directly, no Pipecat)

## Setup 1: Connect to a Raspberry Pi via SSH
- I recommend following the steps below in Setup 2 on your local computer first - it will all run on a Mac or Linux and you can test it on a Raspberry Pi later.
- **Prerequisites**
  - Only works if:
    - The Raspberry Pi is already connected to the internet via a monitor
    - Your computer is connected to a phone hotspot

- **Download the SSH key**
  - Download `rpi_hackathon_key` shared on WhatsApp
  - Ensure it is located in your `Downloads` folder

- **Move the SSH key and set permissions**
  - Open Terminal and run:
    ```bash
    mv ~/Downloads/rpi_hackathon_key ~/.ssh/
    chmod 600 ~/.ssh/rpi_hackathon_key
    ```

- **Install VS Code Remote SSH**
  - Open Cursor (or VS Code)
  - Go to **Extensions** (left sidebar)
  - Install **“Remote – SSH”** by Microsoft

- **Connect the Raspberry Pi**
  - Plug in the Raspberry Pi given to you
  - Open the Command Palette:
    - `Shift + Cmd + P`
    - Select **“Remote-SSH: Open SSH Configuration File…”**
  - Paste the following configuration (based on your working SSH command):
    ```ssh
    Host mindfulmakers-rpi
        HostName ssh.mindfulmakers.xyz
        User mindfulmakers
        Port <insert port here>
        IdentityFile ~/.ssh/rpi_hackathon_key
    ```
  - Replace `<insert port here>` with the larger number written on top of the Raspberry Pi

- **Connect via Remote SSH**
  - Open the Command Palette again (`Shift + Cmd + P`)
  - Select **“Remote-SSH: Connect to Host…”**
  - Choose: `mindfulmakers-rpi`

- **Open the project folder**
  - On the left, press **Open Folder**
  - Select **Desktop** → **mindful-rpi-demos**
  - Click **OK**

- **Set up environment variables**
  - Copy `.env.example` to a new file called `.env`
  - Fill in your API keys

## Setup 2: Once you have shell access
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

## Running `space-bar-thai-stt`
- Ensure `OPENAI_API_KEY` is set in your `.env` file
- Run `python3 space-bar-thai-stt/main.py` or use the launch config
- Hold the **space bar** to record Thai speech
- Release to transcribe, translate to English, and hear the spoken translation
- Press **ESC** to exit

## Troubleshooting
- You may have to figure out what the sample rate is for your output speaker (or input microphone)
- For the output speaker, test_speaker.py will print the correct output sample rate for your speaker.  Then modify bot.py to use the correct sample rate.
