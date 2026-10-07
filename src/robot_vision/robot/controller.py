"""
Robot connection and motion control for the vision project.

The controller provides:
- Connection using the existing station configuration.
- Cartesian pose reading.
- Cartesian linear motion without joint-motion fallback.
- Lite6 gripper opening and closing.
- The existing inverse-kinematics and joint-motion interface.

Motion commands never clear errors, enable motors, or retry automatically.
Executing this file directly only reads the robot pose.

Author: Javier G. Fontanet
"""

import json
import math
from pathlib import Path

from xarm.wrapper import XArmAPI
from xarm.core.config.x_config import XCONF


# Keep the existing configuration path used by the project.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.json"


class Lite6Controller:
    """Manage the robot connection and explicit robot commands."""

    def __init__(self):
        # Creating the controller does not connect or move the robot.
        self.arm = None
        self.config = self._load_config()

        self.mode = self.config["mode"]
        self.robot_ip = self.config["controller_ip"]

    def _load_config(self):
        """Read the existing station configuration without modifying it."""

        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Robot configuration not found: {CONFIG_PATH}"
            )

        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _require_connection(self):
        """Reject commands when the robot is not connected."""

        if self.arm is None or not self.arm.connected:
            raise RuntimeError("Robot is not connected.")

    @staticmethod
    def _check_code(code, action):
        """Stop the calling sequence when an SDK command reports an error."""

        if code != 0:
            raise RuntimeError(
                f"{action} failed with code {code}."
            )

    @staticmethod
    def _validate_pose(target):
        """Validate a six-value pose in millimeters and degrees."""

        if len(target) != 6:
            raise ValueError(
                "Target must contain X, Y, Z, roll, pitch, and yaw."
            )

        pose = [float(value) for value in target]

        if not all(math.isfinite(value) for value in pose):
            raise ValueError("Target contains an invalid value.")

        return pose

    @staticmethod
    def _validate_positive(value, name):
        """Validate a finite positive speed or acceleration."""

        number = float(value)

        if not math.isfinite(number) or number <= 0:
            raise ValueError(
                f"{name} must be a finite positive number."
            )

        return number

    def connect(self):
        """Connect without enabling motion or changing the robot state."""

        if self.arm is not None and self.arm.connected:
            print("Robot is already connected.")
            return

        print(f"Mode: {self.mode}")
        print(f"Connecting to Lite6 at {self.robot_ip}...")

        self.arm = XArmAPI(
            self.robot_ip,
            is_radian=False,
        )

        if not self.arm.connected:
            raise RuntimeError(
                f"Could not connect to Lite6 at {self.robot_ip}"
            )

        print("Robot connected successfully.")

    def disconnect(self):
        """Disconnect without moving the robot or operating the gripper."""

        if self.arm is not None:
            try:
                self.arm.disconnect()
            finally:
                self.arm = None

        print("Robot disconnected.")

    def get_pose(self):
        """Return the measured X,Y,Z,roll,pitch,yaw in millimeters/degrees."""

        self._require_connection()

        code, pose = self.arm.get_position(is_radian=False)
        self._check_code(code, "Position reading")

        return self._validate_pose(pose)

    def move_linear(
        self,
        target,
        speed=10.0,
        mvacc=20.0,
        wait=True,
    ):
        """
        Execute Cartesian linear motion.

        Positive and negative Z refer to the robot base coordinate system.
        Orientation values are in degrees.

        motion_type=0 requires linear planning.
        No fallback to joint motion is permitted.
        """

        self._require_connection()

        pose = self._validate_pose(target)
        speed = self._validate_positive(speed, "Speed")
        mvacc = self._validate_positive(mvacc, "Acceleration")

        print(
            "Linear target:",
            [round(value, 3) for value in pose],
        )
        print(f"Speed: {speed:.1f} mm/s")

        # Use the same linear command exercised in the pick demonstrations.
        # A negative radius selects a line rather than blended arc motion.
        code = self.arm.set_position(
            x=pose[0],
            y=pose[1],
            z=pose[2],
            roll=pose[3],
            pitch=pose[4],
            yaw=pose[5],
            radius=-1.0,
            motion_type=0,
            relative=False,
            is_radian=False,
            speed=speed,
            mvacc=mvacc,
            wait=wait,
        )

        # Do not retry on timeout: the physical movement may have occurred.
        self._check_code(code, "Linear movement")

        return code

    def open_gripper(self):
        """
        Command the Lite6 gripper to open.

        This sends the command immediately.
        The calling cycle must allow time for the fingers to move.
        A successful return does not independently verify finger position.
        """

        self._require_connection()

        print("Opening gripper.")
        code = self.arm.open_lite6_gripper(sync=False)
        self._check_code(code, "Gripper opening")

        return code

    def close_gripper(self):
        """
        Command the Lite6 gripper to close.

        The calling cycle must allow time for the fingers to settle.
        A successful return does not verify that a part is retained.
        """

        self._require_connection()

        print("Closing gripper.")
        code = self.arm.close_lite6_gripper(sync=False)
        self._check_code(code, "Gripper closing")

        return code

    def _validate_joint_limits(self, joints):
        """Preserve the existing joint-limit check for joint-motion commands."""

        self._require_connection()

        limits = XCONF.Robot.JOINT_LIMITS[
            self.arm._arm.axis
        ][
            self.arm._arm.device_type
        ]

        for index, (angle, limit) in enumerate(zip(joints, limits)):
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
        """Preserve the existing inverse-kinematics interface."""

        self._require_connection()
        pose = self._validate_pose(target)

        code, angles = self.arm.get_inverse_kinematics(
            pose,
            input_is_radian=False,
            return_is_radian=False,
        )
        self._check_code(code, "Inverse kinematics")

        joints = angles[:6]

        print("Target:")
        print(pose)
        print("IK solution [J1, J2, J3, J4, J5, J6]:")
        print(joints)

        self._validate_joint_limits(joints)

        return joints

    def move_to_target(
        self,
        target,
        speed=20,
        mvacc=200,
        wait=True,
    ):
        """
        Preserve the existing joint-motion interface.

        This method uses inverse kinematics followed by set_servo_angle().
        It does not guarantee a straight Cartesian path.

        The pick cycle must use move_linear() instead.
        """

        joints = self.calculate_ik(target)

        # Retain the existing simulator-specific behavior.
        if self.mode == "simulation":
            self.arm._arm._check_joint_limit = False

        code = self.arm.set_servo_angle(
            angle=joints,
            speed=speed,
            mvacc=mvacc,
            is_radian=False,
            wait=wait,
        )
        self._check_code(code, "Joint movement")

        print("Robot movement completed.")

        return joints


def main():
    """Read the robot pose when this module is executed directly."""

    controller = Lite6Controller()

    try:
        controller.connect()

        print("Position [X, Y, Z, roll, pitch, yaw]:")
        print(controller.get_pose())
        print("Units: millimeters and degrees.")
        print("No movement or gripper command was issued.")

    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()