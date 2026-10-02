"""
Lite6 robot controller.

The robot IP and operating mode are loaded from the
project configuration.

Author: Javier G. Fontanet
"""

import json
import math
from pathlib import Path

from xarm.wrapper import XArmAPI
from xarm.core.config.x_config import XCONF


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.json"


class Lite6Controller:

    def __init__(self):

        self.arm = None
        self.config = self._load_config()

        self.mode = self.config["mode"]
        self.robot_ip = self.config["controller_ip"]

    def _load_config(self):

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Robot configuration not found: {CONFIG_PATH}"
            )

        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    def connect(self):

        print(f"Mode: {self.mode}")
        print(f"Connecting to Lite6 at {self.robot_ip}...")

        self.arm = XArmAPI(
            self.robot_ip,
            is_radian=False
        )

        if not self.arm.connected:
            raise RuntimeError(
                f"Could not connect to Lite6 at {self.robot_ip}"
            )

        print("Robot connected successfully.")

    def disconnect(self):

        if self.arm is not None:
            self.arm.disconnect()
            self.arm = None

        print("Robot disconnected.")

    def _validate_joint_limits(self, joints):

        limits = XCONF.Robot.JOINT_LIMITS[
            self.arm._arm.axis
        ][
            self.arm._arm.device_type
        ]

        for index, (angle, limit) in enumerate(
            zip(joints, limits)
        ):

            lower, upper = limit

            if not lower <= math.radians(angle) <= upper:
                raise ValueError(
                    f"Joint J{index + 1} is outside "
                    f"the allowed range: "
                    f"{math.degrees(lower):.2f} to "
                    f"{math.degrees(upper):.2f} degrees. "
                    f"Received: {angle:.2f} degrees."
                )

        print("Joint-limit validation: OK.")

    def calculate_ik(self, target):

        if self.arm is None:
            raise RuntimeError("Robot is not connected.")

        if len(target) != 6:
            raise ValueError(
                "Target must contain X, Y, Z, roll, pitch, and yaw."
            )

        if not all(math.isfinite(value) for value in target):
            raise ValueError("Target contains an invalid value.")

        code, angles = self.arm.get_inverse_kinematics(
            target,
            input_is_radian=False,
            return_is_radian=False
        )

        if code != 0:
            raise RuntimeError(
                f"Inverse kinematics failed with code {code}."
            )

        joints = angles[:6]

        print("Target:")
        print(target)

        print("IK solution [J1, J2, J3, J4, J5, J6]:")
        print(joints)

        self._validate_joint_limits(joints)

        return joints

    def move_to_target(
        self,
        target,
        speed=20,
        mvacc=200,
        wait=True
    ):

        joints = self.calculate_ik(target)

        if self.mode == "simulation":
            self.arm._arm._check_joint_limit = False

        result = self.arm.set_servo_angle(
            angle=joints,
            speed=speed,
            mvacc=mvacc,
            is_radian=False,
            wait=wait
        )

        if result != 0:
            raise RuntimeError(
                f"Robot movement failed with code {result}."
            )

        print("Robot movement completed.")

        return joints


if __name__ == "__main__":

    controller = Lite6Controller()

    try:

        controller.connect()

        test_target = [
            250.0,
            0.0,
            150.0,
            180.0,
            0.0,
            0.0
        ]

        controller.move_to_target(test_target)

    finally:

        controller.disconnect()