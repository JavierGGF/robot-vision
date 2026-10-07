"""
Check complete pick, placement, and retreat trajectories without motion.

Reuse the transfer alternatives that passed the previous planning check.
Insert the configured pick and placement heights and the taught retreat.

No movement or gripper command is executed.
The saved part position is used only for planning, not for execution.
"""

import argparse
import copy
import json
from datetime import datetime

from robot_vision.robot.controller import Lite6Controller
from robot_vision.robot.pick_cycle import (
    load_pick_profile,
    load_transfer_profile,
)
from robot_vision.robot.test_pick_motion import (
    PROJECT_ROOT,
    project_path,
    read_json,
)


def movement(label, pose, speed, acceleration):
    """Describe a linear segment without executing it."""

    return {
        "label": label,
        "pose": list(pose),
        "speed": speed,
        "mvacc": acceleration,
    }


def complete_route(elevated_route, start, pick_profile, transfer_profile):
    """Extend a passed elevated route with the vertical operations."""

    pick = pick_profile["motion"]
    travel = transfer_profile["motion"]
    placement = transfer_profile["placement_pose_mm_deg"]
    retreat = transfer_profile["retreat_pose_mm_deg"]

    route = copy.deepcopy(elevated_route)

    if route[1]["label"] != "above_zone1_and_orient":
        raise ValueError("Unexpected saved route structure.")

    # Rebuild the planning start from the actual robot configuration.
    height = max(
        start[2],
        pick["approach_z_mm"],
        travel["transfer_z_mm"],
        pick["pick_z_mm"] + pick["lift_mm"],
    )

    for item in route:
        item["pose"][2] = height

    route[0]["pose"] = [
        start[0], start[1], height, *start[3:6]
    ]

    pick_above = route[1]["pose"]
    pick_xy = pick_above[:2]
    pick_orientation = pick_above[3:6]

    # These are planning segments only. The gripper is not operated.
    pick_segments = [
        movement(
            "descend_to_pick",
            [*pick_xy, pick["pick_z_mm"], *pick_orientation],
            pick["descent_speed_mm_s"],
            pick["descent_acceleration"],
        ),
        movement(
            "lift_after_pick",
            [
                *pick_xy,
                pick["pick_z_mm"] + pick["lift_mm"],
                *pick_orientation,
            ],
            pick["lift_speed_mm_s"],
            pick["descent_acceleration"],
        ),
        movement(
            "raise_for_transfer",
            [*pick_xy, height, *pick_orientation],
            travel["clearance_speed_mm_s"],
            travel["placement_acceleration"],
        ),
    ]

    # Preserve the passed intermediate-orientation transfer sequence.
    route = route[:2] + pick_segments + route[2:]

    route.extend([
        movement(
            "descend_to_place",
            placement,
            travel["placement_speed_mm_s"],
            travel["placement_acceleration"],
        ),
        movement(
            "clear_zone2",
            [
                placement[0],
                placement[1],
                height,
                *placement[3:6],
            ],
            travel["clearance_speed_mm_s"],
            travel["placement_acceleration"],
        ),
        movement(
            "approach_retreat_and_orient",
            [
                retreat[0],
                retreat[1],
                height,
                *retreat[3:6],
            ],
            travel["travel_speed_mm_s"],
            travel["travel_acceleration"],
        ),
        movement(
            "finish_retreat",
            retreat,
            travel["clearance_speed_mm_s"],
            travel["placement_acceleration"],
        ),
    ])

    return route


def main():
    """Check both passed alternatives as complete trajectories."""

    parser = argparse.ArgumentParser(
        description="Check complete Robot 1 routes without movement."
    )
    parser.add_argument(
        "--report",
        default=(
            "data/route_checks/"
            "zone2_options_20261007_163630_841472.json"
        ),
        help="Previous report containing passed transfer proposals.",
    )
    args = parser.parse_args()

    source_path = project_path(args.report)
    source = read_json(source_path)

    candidates = [
        item
        for item in source["candidates"]
        if item.get("status") == "passed"
    ]

    if not candidates:
        raise ValueError("The source report has no passed proposals.")

    pick_profile = load_pick_profile()
    transfer_profile = load_transfer_profile()

    controller = Lite6Controller()
    report = {
        "started_at": datetime.now().astimezone().isoformat(),
        "source_report": str(source_path),
        "mode": "complete_route_check_without_motion",
        "external_obstacles_checked": False,
        "part_retention_verified": False,
        "placement_verified": False,
        "candidates": [],
    }

    try:
        controller.connect()
        start = controller.get_pose()

        report["actual_start_pose_mm_deg"] = start
        report["actual_start_joints_deg"] = controller.get_joints()

        print("COMPLETE ROUTE PLANNING ONLY.")
        print("The saved part position is used for this check.")
        print("No movement or gripper command will be executed.")

        for candidate in candidates:
            print(f"\nCHECKING COMPLETE ROUTE: {candidate['name']}")

            route = complete_route(
                candidate["route"],
                start,
                pick_profile,
                transfer_profile,
            )

            entry = {
                "name": candidate["name"],
                "route": route,
            }
            report["candidates"].append(entry)

            try:
                entry["path_check"] = controller.check_linear_route(
                    route
                )
                entry["status"] = "passed"
                print("COMPLETE ROUTE: PASSED")

            except Exception as error:
                entry["status"] = "rejected"
                entry["error"] = str(error)
                entry["path_check"] = copy.deepcopy(
                    controller.last_path_check
                )

                print("COMPLETE ROUTE: REJECTED")
                print(error)

                # Stop instead of clearing a controller error automatically.
                code, diagnostics = controller.arm.get_err_warn_code()
                state_code, state = controller.arm.get_state()

                if (
                    code != 0
                    or state_code != 0
                    or diagnostics[0] != 0
                    or state not in (0, 2)
                ):
                    report["stopped_early"] = True
                    print("Checks stopped: controller is not ready.")
                    break

        passed = [
            item["name"]
            for item in report["candidates"]
            if item["status"] == "passed"
        ]

        print("\nPASSED COMPLETE ROUTES:", passed)
        print("No robot movement or gripper operation was executed.")

    finally:
        try:
            controller.disconnect()
        finally:
            output_dir = PROJECT_ROOT / "data" / "route_checks"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output = output_dir / f"complete_options_{timestamp}.json"

            output.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )

            print("Planning report saved:", output)


if __name__ == "__main__":
    main()