"""
Check alternative elevated routes to Zone 2 without robot movement.

The routes are planning proposals, not executable approved trajectories.
This program never operates the gripper or executes movement.

The retreat route is deliberately excluded until the Zone 2 transfer
has a valid planned trajectory.
"""

import argparse
import copy
import json
from datetime import datetime

from robot_vision.robot.controller import Lite6Controller
from robot_vision.robot.pick_cycle import load_pick_profile
from robot_vision.robot.test_pick_motion import (
    PROJECT_ROOT,
    project_path,
    read_json,
)


def wrap(angle):
    """Express an angle within [-180, 180)."""

    return (float(angle) + 180.0) % 360.0 - 180.0


def segment(label, pose, speed, acceleration):
    """Create one non-moving planning segment."""

    return {
        "label": label,
        "pose": list(pose),
        "speed": speed,
        "mvacc": acceleration,
    }


def make_candidates(saved_route, start, profile):
    """Separate travel from rotation and examine both rotation directions."""

    if len(saved_route) != 4:
        raise ValueError("Expected the saved four-segment elevated route.")

    # Reuse the exact detected target and taught Zone 2 destination.
    common = copy.deepcopy(saved_route[:2])
    destination = copy.deepcopy(saved_route[2])
    source = common[-1]["pose"]

    # Rebuild the initial raise from the actual current robot pose.
    height = max(
        float(start[2]),
        float(common[0]["pose"][2]),
        float(destination["pose"][2]),
    )
    common[0]["pose"] = [
        start[0], start[1], height, *start[3:6]
    ]
    common[1]["pose"][2] = height
    destination["pose"][2] = height

    source = common[-1]["pose"]
    desired = destination["pose"]

    speed = profile["motion"]["orientation_speed_deg_s"]
    acceleration = profile["motion"]["travel_acceleration"]

    # A midpoint divides the longer rotation into two shorter commands.
    shortest = wrap(desired[5] - source[5])
    longer = shortest - 360.0 if shortest >= 0 else shortest + 360.0
    midpoint_yaw = wrap(source[5] + longer / 2.0)

    candidates = []

    for rotate_before_travel in (False, True):
        for use_midpoint in (False, True):
            route = copy.deepcopy(common)

            if rotate_before_travel:
                rotation_xy = source[:2]
                location = "before_travel"
            else:
                # Travel while retaining the pick orientation.
                held_orientation = [
                    desired[0], desired[1], height, *source[3:6]
                ]
                route.append(segment(
                    "travel_to_zone2_with_pick_orientation",
                    held_orientation,
                    destination["speed"],
                    destination["mvacc"],
                ))
                rotation_xy = desired[:2]
                location = "after_travel"

            if use_midpoint:
                route.append(segment(
                    "intermediate_orientation",
                    [
                        *rotation_xy,
                        height,
                        desired[3],
                        desired[4],
                        midpoint_yaw,
                    ],
                    speed,
                    acceleration,
                ))

            route.append(segment(
                "set_zone2_orientation",
                [*rotation_xy, height, *desired[3:6]],
                speed,
                acceleration,
            ))

            if rotate_before_travel:
                route.append(destination)

            turn = "longer_turn" if use_midpoint else "direct_turn"
            candidates.append({
                "name": f"{location}_{turn}",
                "route": route,
            })

    return candidates


def main():
    """Ask the controller to check proposals, without executing them."""

    parser = argparse.ArgumentParser(
        description="Check alternative Zone 2 routes without motion."
    )
    parser.add_argument(
        "--trial",
        default=(
            "data/demo_trials/"
            "20261007_163201_066364/results.json"
        ),
        help="Saved elevated-route trial used as the planning reference.",
    )
    args = parser.parse_args()

    trial_path = project_path(args.trial)
    saved = read_json(trial_path)
    profile = load_pick_profile()

    if not saved.get("above_only"):
        raise ValueError("Use an elevated-route trial.")

    controller = Lite6Controller()
    report = {
        "started_at": datetime.now().astimezone().isoformat(),
        "source_trial": str(trial_path),
        "mode": "non_moving_route_checks",
        "retreat_checked": False,
        "external_obstacles_checked": False,
        "candidates": [],
    }

    try:
        controller.connect()
        start = controller.get_pose()
        report["actual_start_pose_mm_deg"] = start
        report["actual_start_joints_deg"] = controller.get_joints()

        candidates = make_candidates(
            saved["planned_route"],
            start,
            profile,
        )

        print("PLANNING ONLY: no movement or gripper commands.")

        for candidate in candidates:
            print(f"\nCHECKING: {candidate['name']}")
            entry = copy.deepcopy(candidate)
            report["candidates"].append(entry)

            try:
                entry["path_check"] = controller.check_linear_route(
                    candidate["route"]
                )
                entry["status"] = "passed"
                print("RESULT: PASSED")

            except Exception as error:
                entry["status"] = "rejected"
                entry["error"] = str(error)
                entry["path_check"] = copy.deepcopy(
                    controller.last_path_check
                )
                print("RESULT: REJECTED")
                print(error)

                # Do not reset a real controller error to continue testing.
                code, diagnostics = controller.arm.get_err_warn_code()
                state_code, state = controller.arm.get_state()

                if (
                    code != 0
                    or state_code != 0
                    or diagnostics[0] != 0
                    or state not in (0, 2)
                ):
                    print("Checks stopped: controller is not ready.")
                    report["stopped_early"] = True
                    break

        passed = [
            entry["name"]
            for entry in report["candidates"]
            if entry["status"] == "passed"
        ]

        print("\nPASSED PROPOSALS:", passed)
        print("No proposal was executed.")
        print("The retreat and complete pick cycle still need checking.")

    finally:
        try:
            controller.disconnect()
        finally:
            output_dir = PROJECT_ROOT / "data" / "route_checks"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output = output_dir / f"zone2_options_{timestamp}.json"
            output.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )

            print("Planning report saved:", output)


if __name__ == "__main__":
    main()