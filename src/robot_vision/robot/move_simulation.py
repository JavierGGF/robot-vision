"""
Move the local UFACTORY simulator to a test target.

Author: Javier G. Fontanet
Simulation only. Reference placement is assumed, not physically calibrated.
"""

import time

from xarm.wrapper import XArmAPI

from coordinates import load_config, reference_to_robot


def check(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation} failed. Code: {code}")


def main():
    config = load_config()

    if (
        config["mode"] != "simulation"
        or config["controller_ip"] != "127.0.0.1"
    ):
        raise ValueError("This script requires the local simulator.")

    # Temporary vision result, entered manually for this test.
    x, y = reference_to_robot(54.88, 72.82, config)
    z = config["approach_z_mm"]
    roll, pitch, yaw = config["tool_orientation_deg"]
    target = [x, y, z, roll, pitch, yaw]

    print("SIMULATION ONLY")
    print(f"Target: {target}")

    arm = None

    try:
        arm = XArmAPI(config["controller_ip"], is_radian=False)

        if not arm.connected:
            raise RuntimeError("Could not connect to the simulator.")

        code, _ = arm.get_inverse_kinematics(
            target,
            input_is_radian=False,
            return_is_radian=False
        )
        check(code, "Inverse kinematics")

        check(arm.motion_enable(enable=True), "Motion enable")
        check(arm.set_mode(0), "Position mode")
        check(arm.set_state(state=0), "Ready state")
        time.sleep(1)

        print("Moving simulated arm...")

        code = arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=config["speed_mm_s"],
            mvacc=100,
            is_radian=False,
            wait=True,
            timeout=30
        )
        check(code, "Movement")

        code, position = arm.get_position()
        check(code, "Position reading")
        print(f"Reported final position: {position}")

    finally:
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main()