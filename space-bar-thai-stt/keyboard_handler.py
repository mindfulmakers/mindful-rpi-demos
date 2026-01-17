#!/usr/bin/env python3
"""
Keyboard handling for Space Bar Thai STT Demo.

Uses pynput on macOS and evdev on Linux.
"""

import platform

from loguru import logger

IS_MACOS = platform.system() == "Darwin"

if not IS_MACOS:
    import evdev
    from evdev import ecodes


def find_keyboard_device():
    """Find the keyboard input device (Linux only)."""
    if IS_MACOS:
        return None

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


def run_macos_main(recorder):
    """Main loop for macOS using pynput."""
    from pynput import keyboard

    space_pressed = False
    any_key_received = False

    def on_press(key):
        nonlocal space_pressed, any_key_received
        any_key_received = True
        logger.debug(f"Key pressed: {key}")
        if key == keyboard.Key.space and not space_pressed:
            space_pressed = True
            logger.info("Space pressed - starting recording")
            recorder.start_recording()
        elif key == keyboard.Key.esc:
            logger.info("ESC pressed - exiting...")
            return False  # Stop listener

    def on_release(key):
        nonlocal space_pressed
        logger.debug(f"Key released: {key}")
        if key == keyboard.Key.space and space_pressed:
            space_pressed = False
            logger.info("Space released - stopping recording")
            recorder.stop_recording()

    logger.info("Starting keyboard listener (pynput)...")
    logger.info(
        "If you don't see key events, check Accessibility permissions in System Settings"
    )

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def run_linux_main(recorder, keyboard_device):
    """Main loop for Linux using evdev."""
    space_pressed = False

    try:
        for event in keyboard_device.read_loop():
            if event.type == ecodes.EV_KEY:
                key_event = evdev.categorize(event)

                # Space bar pressed
                if key_event.scancode == ecodes.KEY_SPACE:
                    if (
                        key_event.keystate == evdev.KeyEvent.key_down
                        and not space_pressed
                    ):
                        space_pressed = True
                        recorder.start_recording()
                    elif key_event.keystate == evdev.KeyEvent.key_up and space_pressed:
                        space_pressed = False
                        recorder.stop_recording()

                # ESC pressed - exit
                elif (
                    key_event.scancode == ecodes.KEY_ESC
                    and key_event.keystate == evdev.KeyEvent.key_down
                ):
                    logger.info("Exiting...")
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted...")
    finally:
        keyboard_device.close()
