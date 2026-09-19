"""
Check inverse kinematics for a simulated target without moving.

Author: Javier G. Fontanet
"""

from xarm.wrapper import XArmAPI

from coordinates import load_config, reference_to_robot


def main():
    config = load_config()

    if (
        config["mode"] != "simulation"
        or config["controller_ip"] != "127.0.0.1"
    ):
        raise ValueError("This test requires the local simulator.")

    x, y = reference_to_robot(54.88, 72.82, config)

    target = [
        x,
        y,
        config["approach_z_mm"],
        *config["tool_orientation_deg"]
    ]

    print("Target [X, Y, Z, roll, pitch, yaw]:")
    print(target)

    arm = None

    try:
        arm = XArmAPI(config["controller_ip"], is_radian=False)

        if not arm.connected:
            print("Could not connect to the simulator.")
            return

        code, angles = arm.get_inverse_kinematics(
            target,
            input_is_radian=False,
            return_is_radian=False
        )

        print(f"Response code: {code}")

        if code == 0:
            print("Inverse kinematics solution found.")
            print("Joint angles in degrees:")
            print(angles)
        else:
            print("Could not obtain an inverse kinematics solution.")

        print("No movement command sent.")

    finally:
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main()