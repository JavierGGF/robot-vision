"""
Robot connection and guarded movement control.

Every movement is checked by the controller before execution.
Complete linear routes can also be checked without moving the robot.

The controller does not clear errors, enable motors, disable collision
protection, or fall back from linear to joint motion.
Checks use the robot's configured geometry, not external obstacles.
"""

import json
import math
from pathlib import Path

from xarm.wrapper import XArmAPI


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.json"


class Lite6Controller:
    """Connect, inspect, and issue explicitly checked robot commands."""

    def __init__(self):
        self.arm = None
        self.config = json.loads(
            CONFIG_PATH.read_text(encoding="utf-8")
        )
        self.mode = self.config["mode"]
        self.robot_ip = self.config["controller_ip"]
        self.last_path_check = None

    def _require_connection(self):
        if self.arm is None or not self.arm.connected:
            raise RuntimeError("Robot is not connected.")

    @staticmethod
    def _check_code(code, action):
        if code != 0:
            raise RuntimeError(f"{action} failed with code {code}.")

    @staticmethod
    def _vector(values, length, name):
        if len(values) != length:
            raise ValueError(f"{name} requires {length} values.")

        if any(isinstance(value, bool) for value in values):
            raise ValueError(f"{name} requires numeric values.")

        result = [float(value) for value in values]

        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} contains invalid values.")

        return result

    @staticmethod
    def _validate_pose(target):
        return Lite6Controller._vector(target, 6, "Pose")

    @staticmethod
    def _validate_positive(value, name):
        if isinstance(value, bool):
            raise ValueError(f"{name} requires a number.")

        value = float(value)

        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite.")

        return value

    def connect(self):
        """Connect without enabling motion or resetting the robot."""

        if self.arm is not None and self.arm.connected:
            return

        print(f"Mode: {self.mode}")
        print(f"Connecting to Lite6 at {self.robot_ip}...")

        self.arm = XArmAPI(self.robot_ip, is_radian=False)
        self._require_connection()

        print("Robot connected successfully.")

    def disconnect(self):
        """Disconnect without requesting movement or release."""

        if self.arm is not None:
            try:
                self.arm.disconnect()
            finally:
                self.arm = None

        print("Robot disconnected.")

    def get_pose(self):
        """Read the actual Cartesian pose, including when stopped."""

        self._require_connection()
        code, pose = self.arm.get_position(is_radian=False)
        self._check_code(code, "Position reading")

        return self._validate_pose(pose)

    def get_joints(self):
        """Read six real joints; ignore the unused seventh SDK entry."""

        self._require_connection()

        if self.arm.axis != 6:
            raise RuntimeError(
                "This controller requires a six-axis robot."
            )

        code, joints = self.arm.get_servo_angle(is_radian=False)
        self._check_code(code, "Joint reading")

        return self._vector(joints[:6], 6, "Joint angles")

    def _require_motion_ready(self):
        """Reject commands in an error, stopped, paused, or moving state."""

        self._require_connection()

        code, diagnostics = self.arm.get_err_warn_code()
        self._check_code(code, "Diagnostic reading")

        if diagnostics[0] != 0:
            raise RuntimeError(
                f"Controller error is active: {diagnostics[0]}."
            )

        code, state = self.arm.get_state()
        self._check_code(code, "State reading")

        if state not in (0, 2):
            raise RuntimeError(
                f"Robot is not stationary and ready: state={state}."
            )

        # Reject firmware that cannot perform non-moving path checks.
        version = tuple(self.arm.version_number)

        if version < (1, 11, 100):
            raise RuntimeError(
                "Firmware does not support non-moving path checks."
            )

        # Never operate with self-collision protection disabled.
        params = self.arm.self_collision_params

        if not params or params[0] != 1:
            raise RuntimeError(
                "Self-collision protection is not confirmed enabled."
            )

    def _validate_joint_limits(self, joints):
        """Ask the actual controller to validate joint-angle limits."""

        joints = self._vector(joints, 6, "Joint angles")

        code, outside = self.arm.is_joint_limit(
            joints,
            is_radian=False,
        )
        self._check_code(code, "Joint-limit checking")

        if outside is not False and outside != 0:
            raise RuntimeError(
                f"Joint limits rejected or unknown: {outside}."
            )

        return joints

    def calculate_ik(self, target):
        """Check endpoint IK; this alone does not validate a trajectory."""

        self._require_connection()
        pose = self._validate_pose(target)

        code, outside = self.arm.is_tcp_limit(
            pose,
            is_radian=False,
        )
        self._check_code(code, "Cartesian-limit checking")

        if outside is not False and outside != 0:
            raise RuntimeError(
                f"Cartesian limits rejected or unknown: {outside}."
            )

        # Firmware 2.7.1 does not support the newer ref_angles argument.
        code, angles = self.arm.get_inverse_kinematics(
            pose,
            input_is_radian=False,
            return_is_radian=False,
        )
        self._check_code(code, "Inverse kinematics")

        joints = self._validate_joint_limits(angles[:6])

        print(
            "Endpoint IK joints:",
            [round(value, 3) for value in joints],
        )

        return joints

    def _send_linear(self, pose, speed, mvacc, check_type, wait):
        """Use explicit per-command check mode, not global SDK mode."""

        return self.arm.set_position(
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
            only_check_type=check_type,
            wait=wait,
        )

    def check_linear_route(self, segments):
        """
        Check an ordered route from the actual start without movement.

        Each segment contains pose, speed, mvacc, and an optional label.
        Later planning checks continue from the virtual previous endpoint.
        """

        self._require_motion_ready()

        if not segments:
            raise ValueError("A route requires at least one segment.")

        # Validate every argument before making planning queries.
        prepared = []

        for index, item in enumerate(segments):
            prepared.append({
                "label": str(
                    item.get("label", f"segment_{index + 1}")
                ),
                "pose": self._validate_pose(item["pose"]),
                "speed": self._validate_positive(
                    item["speed"],
                    "Speed",
                ),
                "mvacc": self._validate_positive(
                    item["mvacc"],
                    "Acceleration",
                ),
            })

        report = {
            "status": "checking",
            "start_pose_mm_deg": self.get_pose(),
            "start_joints_deg": self.get_joints(),
            "segments": [],
            "external_obstacles_checked": False,
        }
        self.last_path_check = report

        try:
            for index, item in enumerate(prepared):
                self._require_motion_ready()

                entry = dict(item)
                report["segments"].append(entry)

                entry["endpoint_ik_joints_deg"] = self.calculate_ik(
                    item["pose"]
                )

                # Type 1 starts from the actual robot configuration.
                # Type 3 continues from the previous virtual endpoint.
                # Type 2 finishes the chain and restores virtual state.
                check_type = 1 if index == 0 else (
                    2 if index == len(prepared) - 1 else 3
                )

                code = self._send_linear(
                    item["pose"],
                    item["speed"],
                    item["mvacc"],
                    check_type=check_type,
                    wait=False,
                )
                result = self.arm.only_check_result

                entry["return_code"] = code
                entry["planner_result"] = result

                if code != 0 or result != 0:
                    raise RuntimeError(
                        f"Path rejected at '{item['label']}': "
                        f"code={code}, planner_result={result}."
                    )

                print(f"PATH CHECK OK: {item['label']}")

            report["status"] = "passed"
            return report

        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error)
            raise

    def move_linear(self, target, speed=10.0, mvacc=20.0, wait=True):
        """Check the actual-to-target path before executing it."""

        if wait is not True:
            raise ValueError("Guarded movements require wait=True.")

        pose = self._validate_pose(target)
        speed = self._validate_positive(speed, "Speed")
        mvacc = self._validate_positive(mvacc, "Acceleration")

        # Every cycle using this method receives a pre-movement check.
        self.check_linear_route([{
            "label": "linear_movement",
            "pose": pose,
            "speed": speed,
            "mvacc": mvacc,
        }])

        self._require_motion_ready()

        print("Linear target:", [round(value, 3) for value in pose])
        print(f"Speed: {speed:.1f} mm/s")

        # Type 0 executes only after the non-moving check passed.
        code = self._send_linear(
            pose,
            speed,
            mvacc,
            check_type=0,
            wait=True,
        )
        self._check_code(code, "Linear movement")

        return code

    def open_gripper(self):
        """Open only when no controller stop or error is active."""

        self._require_motion_ready()
        print("Opening gripper.")

        code = self.arm.open_lite6_gripper(sync=False)
        self._check_code(code, "Gripper opening")

        return code

    def close_gripper(self):
        """Close without claiming that a part is securely retained."""

        self._require_motion_ready()
        print("Closing gripper.")

        code = self.arm.close_lite6_gripper(sync=False)
        self._check_code(code, "Gripper closing")

        return code

    def move_to_target(self, target, speed=20, mvacc=200, wait=True):
        """Preserve explicit joint motion with a non-moving path check."""

        if wait is not True:
            raise ValueError("Guarded movements require wait=True.")

        self._require_motion_ready()

        speed = self._validate_positive(speed, "Joint speed")
        mvacc = self._validate_positive(mvacc, "Joint acceleration")
        joints = self.calculate_ik(target)

        # Check the joint trajectory without executing it.
        code = self.arm.set_servo_angle(
            angle=joints,
            speed=speed,
            mvacc=mvacc,
            is_radian=False,
            wait=False,
            only_check_type=1,
        )
        result = self.arm.only_check_result

        if code != 0 or result != 0:
            raise RuntimeError(
                f"Joint path rejected: "
                f"code={code}, planner_result={result}."
            )

        self._require_motion_ready()

        code = self.arm.set_servo_angle(
            angle=joints,
            speed=speed,
            mvacc=mvacc,
            is_radian=False,
            wait=True,
            only_check_type=0,
        )
        self._check_code(code, "Joint movement")

        return joints


def main():
    """Print diagnostics without issuing motion or gripper commands."""

    controller = Lite6Controller()

    try:
        controller.connect()

        print("POSE:", controller.get_pose())
        print("JOINTS:", controller.get_joints())
        print("STATE:", controller.arm.get_state())
        print("ERROR / WARNING:", controller.arm.get_err_warn_code())
        print("SELF-COLLISION:", controller.arm.self_collision_params)
        print("No motion or gripper command was issued.")

    finally:
        controller.disconnect()


if __name__ == "__main__":
    main()