import pygame
from pygame.joystick import JoystickType
from serial import Serial

from microver import input
from microver.camera import CameraRelay

PORT = "/dev/ttyACM0"
BAUD = 115200


def handle_input(joystick: JoystickType, ser: Serial):
    x_axis = joystick.get_axis(0)
    y_axis = joystick.get_axis(1)

    left_speed = -y_axis + x_axis
    right_speed = -y_axis - x_axis

    left_speed = max(-1, min(1, left_speed))
    right_speed = max(-1, min(1, right_speed))

    left_cmd = round(left_speed * 255)
    right_cmd = round(right_speed * 255)

    input.send_command(ser, "L", left_cmd)
    input.send_command(ser, "R", right_cmd)


def main():
    pygame.init()
    joystick = input.connect_joystick()
    ser: Serial | None = None
    camera_relay = CameraRelay()

    if not joystick:
        camera_relay.serve()
        return

    try:
        ser = input.connect_serial(PORT, BAUD)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    handle_input(joystick, ser)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Exiting program.")
    finally:
        input.close_serial(ser)
        pygame.quit()
