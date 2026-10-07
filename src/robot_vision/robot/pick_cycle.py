"""
Reusable pick cycle for Robot 1.

Receive a target already expressed in robot X,Y coordinates.
Load the tested motion parameters from the Robot 1 profile.

Sequence:
1. Raise from the measured starting pose.
2. Restore the taught tool orientation at height.
3. Open the gripper.
4. Approach the target.
5. Descend vertically.
6. Close the gripper.
7. Lift by the configured distance.

All movements use the controller's Cartesian linear interface.
No joint-motion fallback, automatic retry, or automatic release is used.

Executing this module directly only displays the profile.
"""

import copy
import json
import math
import time
from datetime import datetime
from pathlib import Path


# Resolve paths relative to the project instead of the current terminal folder.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "config" / "robot1_pick.json"

# Check the measured position before closing and after lifting.
# These checks verify robot position, not whether the part is retained.
POSITION_TOLERANCE_MM = 1.0
ORIENTATION_TOLERANCE_DEG = 2.0


class PickCycleError(RuntimeError):
    """Report a stopped cycle together with its stage and recorded parameters."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def _number(value, name):
    """Convert a setting to a finite number."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")

    return number


def _positive(value, name):
    """Validate positive speeds, accelerations, and distances."""

    number = _number(value, name)

    if number <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return number


def _nonnegative(value, name):
    """Validate a delay that may be zero."""

    number = _number(value, name)

    if number < 0:
        raise ValueError(f"{name} must not be negative.")

    return number


def _validated_profile(profile):
    """Validate all required motion settings before issuing any commands."""

    # Work with a copy so validation does not modify the caller's profile.
    validated = copy.deepcopy(profile)

    if validated.get("profile_version") != 1:
        raise ValueError("Unsupported pick profile version.")

    motion = validated.get("motion")

    if not isinstance(motion, dict):
        raise ValueError("Pick profile must contain a motion section.")

    orientation = motion.get("tool_orientation_deg")

    if not isinstance(orientation, (list, tuple)) or len(orientation) != 3:
        raise ValueError(
            "Tool orientation must contain roll, pitch, and yaw."
        )

    motion["tool_orientation_deg"] = [
        _number(value, "Tool orientation")
        for value in orientation
    ]

    # Heights are absolute robot-base Z coordinates in millimeters.
    for name in ("pick_z_mm", "approach_z_mm"):
        motion[name] = _number(motion[name], name)

    if motion["approach_z_mm"] <= motion["pick_z_mm"]:
        raise ValueError(
            "Approach height must be above the grasp height."
        )

    # Retain the profile settings exercised in the demonstration.
    for name in (
        "lift_mm",
        "travel_speed_mm_s",
        "orientation_speed_deg_s",
        "descent_speed_mm_s",
        "lift_speed_mm_s",
        "travel_acceleration",
        "descent_acceleration",
    ):
        motion[name] = _positive(motion[name], name)

    for name in ("open_delay_s", "close_delay_s"):
        motion[name] = _nonnegative(motion[name], name)

    return validated


def load_pick_profile(path=None):
    """Read the Robot 1 profile without connecting or modifying any files."""

    profile_path = (
        DEFAULT_PROFILE_PATH if path is None else Path(path)
    )

    if not profile_path.is_absolute():
        profile_path = PROJECT_ROOT / profile_path

    if not profile_path.is_file():
        raise FileNotFoundError(
            f"Pick profile not found: {profile_path}"
        )

    with profile_path.open("r", encoding="utf-8") as file:
        profile = json.load(file)

    return _validated_profile(profile)


def _validated_target(target_xy_mm):
    """Validate a target already converted to robot-base coordinates."""

    if len(target_xy_mm) != 2:
        raise ValueError("Target must contain robot X and Y.")

    return [
        _number(target_xy_mm[0], "Target X"),
        _number(target_xy_mm[1], "Target Y"),
    ]


def _check_measured_pose(measured, expected):
    """Verify position and orientation before continuing the sequence."""

    # Compare Cartesian position in millimeters.
    position_error = math.sqrt(
        sum(
            (float(measured[index]) - float(expected[index])) ** 2
            for index in range(3)
        )
    )

    if position_error > POSITION_TOLERANCE_MM:
        raise RuntimeError(
            f"Measured position differs from target by "
            f"{position_error:.3f} mm."
        )

    # Compare angles using the shortest difference across the +/-180 boundary.
    orientation_errors = [
        abs(
            (
                float(measured[index])
                - float(expected[index])
                + 180.0
            ) % 360.0 - 180.0
        )
        for index in range(3, 6)
    ]

    if max(orientation_errors) > ORIENTATION_TOLERANCE_DEG:
        raise RuntimeError(
            "Measured orientation differs from the requested orientation."
        )


def run_pick_cycle(controller, target_xy_mm, profile=None):
    """
    Execute one pick cycle using an already connected controller.

    Parameters:
        controller:
            Connected Lite6Controller.
        target_xy_mm:
            Robot-base [X, Y] calculated by the vision layer.
        profile:
            Loaded pick profile, or None to load Robot 1's default profile.

    Returns:
        A serializable result describing the executed sequence.

    Raises:
        PickCycleError if the sequence stops during a robot operation.

    Connection lifetime and file logging belong to the calling program.
    Completing the cycle does not verify that the part remains held.
    """

    # Validate the destination and all motion settings before moving.
    target_xy = _validated_target(target_xy_mm)

    settings = (
        load_pick_profile()
        if profile is None
        else _validated_profile(profile)
    )
    motion = settings["motion"]

    orientation = motion["tool_orientation_deg"]
    pick_z = motion["pick_z_mm"]

    result = {
        "started_at": datetime.now().astimezone().isoformat(),
        "robot_name": settings.get("robot_name", "Robot 1"),
        "part_name": settings.get("part_name", "Part 1"),
        "target_xy_mm": target_xy,
        "motion": copy.deepcopy(motion),
        "status": "started",
        "stage": "read_start_pose",
        "completed_steps": [],
        "part_retention_verified": False,
    }

    try:
        # The caller connects explicitly; this function never reconnects.
        start_pose = controller.get_pose()
        result["start_pose_mm_deg"] = start_pose

        # Do not lower a robot that already starts above the approach height.
        approach_z = max(
            motion["approach_z_mm"],
            float(start_pose[2]),
        )
        result["effective_approach_z_mm"] = approach_z

        # Raise at the starting X,Y before changing orientation.
        result["stage"] = "raise"
        controller.move_linear(
            [
                start_pose[0],
                start_pose[1],
                approach_z,
                *start_pose[3:6],
            ],
            speed=motion["travel_speed_mm_s"],
            mvacc=motion["travel_acceleration"],
            wait=True,
        )
        result["completed_steps"].append("raise")

        # Restore the taught orientation while elevated.
        result["stage"] = "orient"
        controller.move_linear(
            [
                start_pose[0],
                start_pose[1],
                approach_z,
                *orientation,
            ],
            speed=motion["orientation_speed_deg_s"],
            mvacc=motion["travel_acceleration"],
            wait=True,
        )
        result["completed_steps"].append("orient")

        # Always open before descending, regardless of the prior pin state.
        result["stage"] = "open_gripper"
        controller.open_gripper()
        time.sleep(motion["open_delay_s"])
        result["completed_steps"].append("open_gripper")

        # Approach the supplied target at height.
        result["stage"] = "approach"
        controller.move_linear(
            [*target_xy, approach_z, *orientation],
            speed=motion["travel_speed_mm_s"],
            mvacc=motion["travel_acceleration"],
            wait=True,
        )
        result["completed_steps"].append("approach")

        # Descend vertically with unchanged X,Y and orientation.
        result["stage"] = "descend"
        grasp_target = [*target_xy, pick_z, *orientation]

        controller.move_linear(
            grasp_target,
            speed=motion["descent_speed_mm_s"],
            mvacc=motion["descent_acceleration"],
            wait=True,
        )
        result["completed_steps"].append("descend")

        # Confirm the measured pose before issuing the close command.
        result["stage"] = "verify_grasp_pose"
        grasp_pose = controller.get_pose()
        result["grasp_pose_mm_deg"] = grasp_pose

        _check_measured_pose(grasp_pose, grasp_target)
        result["completed_steps"].append("verify_grasp_pose")

        # Close and allow the fingers to settle.
        result["stage"] = "close_gripper"
        controller.close_gripper()
        time.sleep(motion["close_delay_s"])
        result["completed_steps"].append("close_gripper")

        # Lift relative to the actual measured grasp position.
        result["stage"] = "lift"
        lift_target = [
            grasp_pose[0],
            grasp_pose[1],
            grasp_pose[2] + motion["lift_mm"],
            *grasp_pose[3:6],
        ]

        controller.move_linear(
            lift_target,
            speed=motion["lift_speed_mm_s"],
            mvacc=motion["descent_acceleration"],
            wait=True,
        )
        result["completed_steps"].append("lift")

        # Verify the final robot pose without claiming successful retention.
        result["stage"] = "verify_lift_pose"
        final_pose = controller.get_pose()
        result["final_pose_mm_deg"] = final_pose

        _check_measured_pose(final_pose, lift_target)
        result["completed_steps"].append("verify_lift_pose")

        result["stage"] = "completed"
        result["status"] = "cycle_completed_grasp_not_verified"
        result["finished_at"] = datetime.now().astimezone().isoformat()

        print("Pick cycle completed.")
        print("Gripper remains closed.")
        print("Part retention requires visual verification.")

        return result

    except Exception as error:
        # Never retry a motion or release a possibly suspended part.
        result["status"] = "failed"
        result["error"] = str(error)
        result["finished_at"] = datetime.now().astimezone().isoformat()

        message = (
            f"Pick cycle stopped during '{result['stage']}': {error}"
        )
        raise PickCycleError(message, result) from error


def main():
    """Display the saved profile without connecting or executing a cycle."""

    profile = load_pick_profile()

    print()
    print("ROBOT 1 PICK PROFILE")
    print(json.dumps(profile, indent=2))
    print()
    print("No robot connection, movement, or gripper command was issued.")


if __name__ == "__main__":
    main()