"""
Reusable Robot 1 picking and placement.

Build one explicit plan for both route checking and execution.
Keep picking and placement descents vertical.
Use the configured intermediate turn above Zone 2.

All movements remain Cartesian linear movements.
Do not retry failed movements or fall back to joint motion.

Controller checks do not model external obstacles or verify that
the gripper retains the part.
"""

import copy
import json
import math
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "config" / "robot1_pick.json"
DEFAULT_TRANSFER_PATH = PROJECT_ROOT / "config" / "robot1_transfer.json"

POSITION_TOLERANCE_MM = 1.0
ORIENTATION_TOLERANCE_DEG = 2.0


class PickCycleError(RuntimeError):
    """Preserve the failed stage and completed operations."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def _number(value, name):
    """Accept finite numbers and reject Boolean values."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    return value


def _positive(value, name):
    """Validate a strictly positive setting."""

    value = _number(value, name)

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


def _nonnegative(value, name):
    """Validate delays, including zero."""

    value = _number(value, name)

    if value < 0:
        raise ValueError(f"{name} must not be negative.")

    return value


def _vector(value, length, name):
    """Validate a pose, orientation, or coordinate vector."""

    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} requires {length} values.")

    return [_number(item, name) for item in value]


def _wrap_angle(angle):
    """Represent an angle in the interval [-180, 180)."""

    return (angle + 180.0) % 360.0 - 180.0


def _read_profile(path):
    """Read configuration without changing the original file."""

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return json.loads(path.read_text(encoding="utf-8"))


def _validated_profile(profile):
    """Validate pick settings before planning or motion."""

    settings = copy.deepcopy(profile)

    if settings.get("profile_version") != 1:
        raise ValueError("Unsupported pick profile version.")

    motion = settings.get("motion")

    if not isinstance(motion, dict):
        raise ValueError("Pick profile requires a motion section.")

    motion["tool_orientation_deg"] = _vector(
        motion["tool_orientation_deg"], 3, "Tool orientation"
    )

    for name in ("pick_z_mm", "approach_z_mm"):
        motion[name] = _number(motion[name], name)

    if motion["approach_z_mm"] <= motion["pick_z_mm"]:
        raise ValueError("Approach height must exceed pick height.")

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

    return settings


def _validated_transfer(profile):
    """Validate the selected transfer strategy and taught poses."""

    settings = copy.deepcopy(profile)

    if settings.get("profile_version") != 1:
        raise ValueError("Unsupported transfer profile version.")

    if settings.get("transfer_strategy") != "after_travel_longer_turn":
        raise ValueError(
            "Transfer strategy must be after_travel_longer_turn."
        )

    for name in ("placement_pose_mm_deg", "retreat_pose_mm_deg"):
        settings[name] = _vector(settings[name], 6, name)

    motion = settings.get("motion")

    if not isinstance(motion, dict):
        raise ValueError("Transfer profile requires a motion section.")

    motion["transfer_z_mm"] = _number(
        motion["transfer_z_mm"], "Transfer height"
    )

    for name in (
        "travel_speed_mm_s",
        "orientation_speed",
        "placement_speed_mm_s",
        "clearance_speed_mm_s",
        "travel_acceleration",
        "placement_acceleration",
    ):
        motion[name] = _positive(motion[name], name)

    motion["release_delay_s"] = _nonnegative(
        motion["release_delay_s"], "Release delay"
    )

    for name in ("placement_pose_mm_deg", "retreat_pose_mm_deg"):
        if motion["transfer_z_mm"] <= settings[name][2]:
            raise ValueError(f"Transfer height must exceed {name}.")

    return settings


def load_pick_profile(path=None):
    """Load pick settings without connecting to the robot."""

    return _validated_profile(
        _read_profile(DEFAULT_PROFILE_PATH if path is None else path)
    )


def load_transfer_profile(path=None):
    """Load transfer settings without connecting to the robot."""

    return _validated_transfer(
        _read_profile(DEFAULT_TRANSFER_PATH if path is None else path)
    )


def _check_measured_pose(measured, expected):
    """Require the commanded pose before continuing the sequence."""

    measured = _vector(list(measured), 6, "Measured pose")
    expected = _vector(list(expected), 6, "Expected pose")

    position_error = math.sqrt(
        sum((measured[i] - expected[i]) ** 2 for i in range(3))
    )

    if position_error > POSITION_TOLERANCE_MM:
        raise RuntimeError(
            f"Position error is {position_error:.3f} mm."
        )

    angle_errors = [
        abs(_wrap_angle(measured[i] - expected[i]))
        for i in range(3, 6)
    ]

    if max(angle_errors) > ORIENTATION_TOLERANCE_DEG:
        raise RuntimeError(
            "Measured tool orientation differs from target."
        )


def build_cycle_plan(
    start_pose,
    target_xy_mm,
    profile,
    transfer_profile=None,
    above_only=False,
):
    """
    Build the exact ordered operations without issuing commands.

    An elevated preview contains movement operations only.
    The full cycle includes explicit gripper operations.
    """

    settings = _validated_profile(profile)
    transfer = (
        None if transfer_profile is None
        else _validated_transfer(transfer_profile)
    )

    start = _vector(list(start_pose), 6, "Starting pose")
    target = _vector(list(target_xy_mm), 2, "Target X,Y")
    motion = settings["motion"]
    orientation = motion["tool_orientation_deg"]

    # Use the same elevated plane throughout the transfer plan.
    height = max(start[2], motion["approach_z_mm"])

    if transfer is not None:
        height = max(
            height,
            transfer["motion"]["transfer_z_mm"],
            motion["pick_z_mm"] + motion["lift_mm"],
        )

    plan = []

    def move(label, pose, speed, acceleration):
        """Append one explicit Cartesian movement."""

        plan.append({
            "operation": "move",
            "label": label,
            "pose": list(pose),
            "speed": speed,
            "mvacc": acceleration,
        })

    def gripper(label, opening, delay):
        """Append a gripper operation only for the physical cycle."""

        if not above_only:
            plan.append({
                "operation": "gripper",
                "label": label,
                "opening": opening,
                "delay_s": delay,
            })

    # Raise vertically before approaching the detected target.
    move(
        "raise",
        [start[0], start[1], height, *start[3:6]],
        motion["travel_speed_mm_s"],
        motion["travel_acceleration"],
    )

    gripper("open_before_pick", True, motion["open_delay_s"])

    move(
        "above_zone1_and_orient",
        [*target, height, *orientation],
        motion["travel_speed_mm_s"],
        motion["travel_acceleration"],
    )

    if not above_only:
        # Keep X,Y and orientation fixed throughout the descent.
        move(
            "descend_to_pick",
            [*target, motion["pick_z_mm"], *orientation],
            motion["descent_speed_mm_s"],
            motion["descent_acceleration"],
        )

        gripper("close_gripper", False, motion["close_delay_s"])

        # Use the planned height so checking and execution agree.
        move(
            "lift_after_pick",
            [
                *target,
                motion["pick_z_mm"] + motion["lift_mm"],
                *orientation,
            ],
            motion["lift_speed_mm_s"],
            motion["descent_acceleration"],
        )

    if transfer is None:
        return plan

    placement = transfer["placement_pose_mm_deg"]
    retreat = transfer["retreat_pose_mm_deg"]
    travel = transfer["motion"]

    if not above_only:
        move(
            "raise_for_transfer",
            [*target, height, *orientation],
            travel["clearance_speed_mm_s"],
            travel["placement_acceleration"],
        )

    # Travel while preserving the grasp orientation.
    move(
        "travel_to_zone2_with_pick_orientation",
        [placement[0], placement[1], height, *orientation],
        travel["travel_speed_mm_s"],
        travel["travel_acceleration"],
    )

    # Split the longer yaw turn using the previously tested construction.
    # Every new target still requires a fresh controller route check.
    shortest_turn = _wrap_angle(placement[5] - orientation[2])
    longer_turn = (
        shortest_turn - 360.0
        if shortest_turn >= 0.0
        else shortest_turn + 360.0
    )
    midpoint_yaw = _wrap_angle(
        orientation[2] + longer_turn / 2.0
    )

    move(
        "intermediate_orientation",
        [
            placement[0],
            placement[1],
            height,
            placement[3],
            placement[4],
            midpoint_yaw,
        ],
        travel["orientation_speed"],
        travel["travel_acceleration"],
    )

    move(
        "set_zone2_orientation",
        [placement[0], placement[1], height, *placement[3:6]],
        travel["orientation_speed"],
        travel["travel_acceleration"],
    )

    if not above_only:
        # Place only after reaching the complete elevated orientation.
        move(
            "descend_to_place",
            placement,
            travel["placement_speed_mm_s"],
            travel["placement_acceleration"],
        )

        gripper("release_in_zone2", True, travel["release_delay_s"])

        move(
            "clear_zone2",
            [placement[0], placement[1], height, *placement[3:6]],
            travel["clearance_speed_mm_s"],
            travel["placement_acceleration"],
        )

    move(
        "approach_retreat_and_orient",
        [retreat[0], retreat[1], height, *retreat[3:6]],
        travel["travel_speed_mm_s"],
        travel["travel_acceleration"],
    )

    if not above_only:
        move(
            "finish_retreat",
            retreat,
            travel["clearance_speed_mm_s"],
            travel["placement_acceleration"],
        )

    return plan


def movement_segments(plan):
    """Extract the ordered movements for controller route checking."""

    return [
        {
            "label": step["label"],
            "pose": list(step["pose"]),
            "speed": step["speed"],
            "mvacc": step["mvacc"],
        }
        for step in plan
        if step["operation"] == "move"
    ]


def _run_cycle(controller, target_xy_mm, settings, transfer=None):
    """Check the complete plan before any movement or gripper action."""

    result = {
        "started_at": datetime.now().astimezone().isoformat(),
        "robot_name": settings.get("robot_name", "Robot 1"),
        "part_name": settings.get("part_name", "Part 1"),
        "target_xy_mm": list(target_xy_mm),
        "motion": copy.deepcopy(settings["motion"]),
        "transfer": copy.deepcopy(transfer),
        "stage": "read_start_pose",
        "status": "started",
        "completed_steps": [],
        "measured_poses": {},
        "part_retention_verified": False,
        "placement_verified": False,
    }

    try:
        start = list(controller.get_pose())
        result["start_pose_mm_deg"] = start

        plan = build_cycle_plan(
            start, target_xy_mm, settings, transfer
        )
        result["plan"] = copy.deepcopy(plan)

        # A rejected later segment prevents even the initial gripper action.
        result["stage"] = "check_complete_route"
        result["path_check"] = controller.check_linear_route(
            movement_segments(plan)
        )

        # Reject a changed starting pose before executing the saved plan.
        result["stage"] = "verify_start_pose"
        _check_measured_pose(controller.get_pose(), start)

        for step in plan:
            label = step["label"]
            result["stage"] = label
            print(label)

            if step["operation"] == "move":
                # The controller also checks each segment immediately
                # before sending its actual movement command.
                controller.move_linear(
                    step["pose"],
                    speed=step["speed"],
                    mvacc=step["mvacc"],
                    wait=True,
                )

                measured = list(controller.get_pose())
                result["measured_poses"][label] = measured
                _check_measured_pose(measured, step["pose"])
                result["final_pose_mm_deg"] = measured

                if label == "descend_to_pick":
                    result["grasp_pose_mm_deg"] = measured

            elif step["operation"] == "gripper":
                if step["opening"]:
                    controller.open_gripper()
                else:
                    controller.close_gripper()

                time.sleep(step["delay_s"])

            else:
                raise RuntimeError("Unknown plan operation.")

            result["completed_steps"].append(label)

        result["stage"] = "completed"
        result["status"] = (
            "cycle_completed_grasp_not_verified"
            if transfer is None
            else "pick_place_completed_not_verified"
        )
        result["finished_at"] = datetime.now().astimezone().isoformat()

        print("Cycle completed.")
        print("Part retention and placement require visual verification.")
        return result

    except Exception as error:
        # Preserve diagnostics without retrying or opening the gripper.
        result["status"] = "failed"
        result["error"] = str(error)
        result["finished_at"] = datetime.now().astimezone().isoformat()

        if result["stage"] == "check_complete_route":
            result["path_check"] = copy.deepcopy(
                getattr(controller, "last_path_check", None)
            )

        raise PickCycleError(
            f"Cycle stopped during '{result['stage']}': {error}",
            result,
        ) from error


def run_pick_cycle(controller, target_xy_mm, profile=None):
    """Preserve the existing pick-only interface."""

    settings = (
        load_pick_profile()
        if profile is None
        else _validated_profile(profile)
    )

    return _run_cycle(controller, target_xy_mm, settings)


def run_pick_and_place_cycle(
    controller,
    target_xy_mm,
    profile=None,
    transfer_profile=None,
):
    """Pick, transfer through the intermediate turn, place, and retreat."""

    settings = (
        load_pick_profile()
        if profile is None
        else _validated_profile(profile)
    )
    transfer = (
        load_transfer_profile()
        if transfer_profile is None
        else _validated_transfer(transfer_profile)
    )

    return _run_cycle(controller, target_xy_mm, settings, transfer)


def main():
    """Print configuration only; do not connect to the robot."""

    print("ROBOT 1 PICK PROFILE")
    print(json.dumps(load_pick_profile(), indent=2))

    if DEFAULT_TRANSFER_PATH.is_file():
        print("\nROBOT 1 TRANSFER PROFILE")
        print(json.dumps(load_transfer_profile(), indent=2))

    print("\nNo robot connection or movement was issued.")


if __name__ == "__main__":
    main()