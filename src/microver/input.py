import time

import pygame
from pygame.joystick import Joystick, JoystickType
from serial import Serial, SerialException


def connect_serial(port: str, baud: int) -> Serial:
    ser: Serial | None = None
    while ser is None:
        try:
            ser = Serial(port, baud, timeout=None)
        except SerialException:
            print(f"{port} not available. Retrying in 5 seconds...")
            time.sleep(5)
    print(f"Serial port {port} connected successfully.\n")
    return ser


def send_command(ser: Serial, direction: str, speed: int):
    command = f"{direction},{speed}"
    ser.write((command + "\n").encode())
    print(f"Send {command}")


def close_serial(ser: Serial | None) -> None:
    if not ser:
        return
    try:
        if ser.is_open:
            ser.close()
            print("Serial closed. Program exited successfully.")
    except Exception as e:
        print(f"Error closing serial port: {e}")


def connect_joystick() -> JoystickType | None:
    if pygame.joystick.get_count() > 0:
        joystick = Joystick(0)
        print("Joystick connected successfully.")
        return joystick
    return None
