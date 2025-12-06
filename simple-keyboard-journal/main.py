import fileinput
from datetime import datetime
from pathlib import Path

from mpyg321.mpyg321 import MPyg123Player

ENTRY_EXPIRES_AFTER_SECONDS = 60 * 60  # one hour
NOTES = ["c", "e", "g", "a", "c2"]

player = MPyg123Player()
entry_time = None
entry_name = None
note_index = 0
journal_updated = datetime.utcnow()

project_path = Path(__file__).parent
sounds_path = project_path / "sounds"


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


new_entry()

while True:
    for line in fileinput.input():
        journal_updated = datetime.utcnow()
        if (journal_updated - entry_time).total_seconds() > ENTRY_EXPIRES_AFTER_SECONDS:
            new_entry()

        output = line
        play_sound()

        with open(project_path / "entries" / (entry_name + ".txt"), "a") as entry:
            entry.write(output)
