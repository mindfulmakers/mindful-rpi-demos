import fileinput
import os
import signal
from datetime import datetime
from pathlib import Path

from metric import Metric
from mpyg321.mpyg321 import MPyg123Player

ENTRY_EXPIRES_AFTER_SECONDS = 60 * 60  # one hour
NOTES = ["c", "e", "g", "a", "c2"]
SIGKILL_MIN_SECONDS = 1
SIGKILL_MAX_SECONDS = SIGKILL_MIN_SECONDS + 2
METRICS = [Metric("happiness", "Hap 5"), Metric("ph", "P H 5")]

player = MPyg123Player()
entry_time = None
entry_name = None
note_index = 0
journal_updated = datetime.utcnow()
quit_window_start = None
blanks = 0
metric_index = None

project_path = Path(__file__).parent
sounds_path = project_path / "sounds"


def attempt_quit(signum, frame):
    global quit_window_start
    if quit_window_start is None:
        quit_window_start = datetime.utcnow()
    age = (datetime.utcnow() - quit_window_start).total_seconds()
    print(age)

    if age > SIGKILL_MAX_SECONDS:
        quit_window_start = None
    elif age > SIGKILL_MIN_SECONDS:
        exit(0)


def time_to_entry_name(dt):
    return "entry_" + dt.strftime("%Y_%m_%dT%H_%M_%S%z")


def new_entry():
    global entry_time, entry_name
    entry_time = datetime.utcnow()
    entry_name = time_to_entry_name(entry_time)


def play_sound():
    global note_index
    note = NOTES[note_index]
    note_index = (note_index + 1) % len(NOTES)
    player.play_song((sounds_path / f"xylophone-{note}.mp3").as_posix())


def play_error_sound():
    player.play_song("sounds/boink.mp3")


def print_current_metric():
    metric = METRICS[metric_index]
    print(metric.name)


signal.signal(signal.SIGINT, attempt_quit)

new_entry()

while True:
    for line in fileinput.input():
        journal_updated = datetime.utcnow()
        if (journal_updated - entry_time).total_seconds() > ENTRY_EXPIRES_AFTER_SECONDS:
            new_entry()
            metric_index = None

        if line == "\n":
            blanks += 1
            if note_index == len(NOTES) - 1 and blanks > 2:
                print("ACTIVATING METRICS")
                metric_index = 0
                print_current_metric()
                blanks = 0
                continue
        else:
            blanks = 0

        if metric_index is None:
            output = line
            play_sound()
        else:
            output = ""
            try:
                metric = METRICS[metric_index]
                value = metric.typ(line)
                output = f"METRIC {metric.name}: {value}\n"
                play_sound()

                metric_index += 1
                if metric_index >= len(METRICS):
                    print("DEACTIVATING METRICS")
                    metric_index = None
                else:
                    print_current_metric()
            except ValueError:
                if line == "?\n":
                    os.system(f'espeak -ven-us "{metric.name_to_read}" 2>/dev/null')
                else:
                    play_error_sound()

        with open(project_path / "entries" / (entry_name + ".txt"), "a") as entry:
            entry.write(output)
