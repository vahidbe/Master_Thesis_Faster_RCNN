# SPDX-FileCopyrightText: 2021 ladyada for Adafruit Industries
# SPDX-License-Identifier: MIT
 
"""Simple test for using adafruit_motorkit with a stepper motor"""

def close_trap(kit):
    for i in range(50):
        kit.stepper1.onestep(direction=stepper.FORWARD)
        time.sleep(0.01)

def open_trap(kit):
    for i in range(50):
        kit.stepper1.onestep(direction=stepper.BACKWARD)
        time.sleep(0.01)


def trap_insect(duration, kit):
    close_trap(kit)
    time.sleep(duration)
    open_trap(kit)

if __name__ == '__main__':
    import time
    import board
    from adafruit_motorkit import MotorKit
    from adafruit_motor import stepper

    kit = MotorKit(i2c=board.I2C())
    trap_insect(2, kit)
