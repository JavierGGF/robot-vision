"""
Connect to the UFACTORY simulator and read its position.

Author: Javier G. Fontanet
"""

from xarm.wrapper import XArmAPI


def main():
    arm = None

    try:
        arm = XArmAPI("127.0.0.1", is_radian=False)

        print(f"Connected: {arm.connected}")

        if not arm.connected:
            print("Could not connect to the simulator.")
            return

        code, position = arm.get_position()

        if code != 0:
            print(f"Could not read position. Error code: {code}")
            return

        print("Position [X, Y, Z, roll, pitch, yaw]:")
        print(position)
        print("Units: millimeters and degrees.")

    finally:
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main()