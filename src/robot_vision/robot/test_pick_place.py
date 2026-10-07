"""
Preview, check, or execute Robot 1 pick and placement.

Use the shared cycle plan for route construction.
Default operation is a vision preview without a robot connection.

--check-route performs controller planning checks without movement.
--execute --above-only follows the elevated route without gripper actions.
--execute performs the complete checked pick and placement cycle.

Controller checks do not model external obstacles or verify part retention.
"""

import argparse
import copy
import json
from datetime import datetime

import cv2

from robot_vision.robot.controller import Lite6Controller
from robot_vision.robot.pick_cycle import (
    PickCycleError,
    _check_measured_pose,
    build_cycle_plan,
    load_pick_profile,
    load_transfer_profile,
    movement_segments,
    run_pick_and_place_cycle,
)
from robot_vision.robot.test_pick_motion import (
    PROJECT_ROOT,
    capture_frame,
    load_sheet_calibration,
    project_path,
    save_images,
    show_preview,
)
from robot_vision.robot.test_rotated_pick import calculate_rotated_target


def build_route(
    start,
    target,
    orientation,
    pick_profile,
    transfer_profile,
    above_only=False,
):
    """Preserve the route-builder interface using the shared plan."""

    runtime_profile = copy.deepcopy(pick_profile)
    runtime_profile["motion"]["tool_orientation_deg"] = list(orientation)

    plan = build_cycle_plan(
        start,
        target,
        runtime_profile,
        transfer_profile,
        above_only=above_only,
    )
    return movement_segments(plan)


def run_elevated_route(controller, route, pick_profile=None):
    """
    Check and execute elevated movements only.

    Keep the optional profile argument for existing callers.
    Do not open or close the gripper.
    """

    result = {
        "status": "started",
        "stage": "read_start_pose",
        "completed_steps": [],
        "measured_poses": {},
        "planned_route": copy.deepcopy(route),
        "part_retention_verified": False,
    }

    try:
        start = list(controller.get_pose())
        result["start_pose_mm_deg"] = start

        # Check every segment before issuing the first movement.
        result["stage"] = "check_complete_route"
        result["path_check"] = controller.check_linear_route(route)

        # Stop if the starting pose changed during planning.
        result["stage"] = "verify_start_pose"
        _check_measured_pose(controller.get_pose(), start)

        for segment in route:
            label = segment["label"]
            result["stage"] = label
            print(f"{label}: {segment['pose']}")

            # The controller also checks each individual movement.
            controller.move_linear(
                segment["pose"],
                speed=segment["speed"],
                mvacc=segment["mvacc"],
                wait=True,
            )

            measured = list(controller.get_pose())
            result["measured_poses"][label] = measured
            _check_measured_pose(measured, segment["pose"])
            result["completed_steps"].append(label)
            result["final_pose_mm_deg"] = measured

        result["status"] = "elevated_route_completed"
        result["stage"] = "completed"

        print("Elevated route completed.")
        print("No pick or placement descent was performed.")
        print("No gripper command was issued.")
        return result

    except Exception as error:
        # Preserve failure evidence without retrying or recovering.
        result["status"] = "failed"
        result["error"] = str(error)

        if result["stage"] == "check_complete_route":
            result["path_check"] = copy.deepcopy(
                getattr(controller, "last_path_check", None)
            )

        raise PickCycleError(
            f"Elevated route stopped during '{result['stage']}': {error}",
            result,
        ) from error


def save_record(trial_dir, record):
    """Store the trial without modifying calibration files."""

    (trial_dir / "results.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    """Capture the part, then preview, check, or execute."""

    parser = argparse.ArgumentParser(
        description="Checked Robot 1 pick and Zone 2 placement."
    )

    # Planning and execution are separate command-line selections.
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument(
        "--check-route",
        action="store_true",
        help="Check the route without movement or gripper commands.",
    )

    parser.add_argument(
        "--above-only",
        action="store_true",
        help="Use elevated movements only, with no gripper commands.",
    )
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--image",
        help="Offline image from the current calibrated camera view.",
    )
    args = parser.parse_args()

    # Saved images cannot be used to command a physical cycle.
    if args.image and (args.execute or args.check_route):
        parser.error("Execution and route checking require live capture.")

    if args.above_only and not (args.execute or args.check_route):
        parser.error("--above-only requires --execute or --check-route.")

    pick_profile = load_pick_profile()
    transfer_profile = load_transfer_profile()

    calibration = load_sheet_calibration(
        "data/calibration/zone1_camera.json"
    )
    reference_calibration = load_sheet_calibration(
        "data/calibration/zone1_reference_camera.json"
    )

    trial_dir = (
        PROJECT_ROOT
        / "data"
        / "demo_trials"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    trial_dir.mkdir(parents=True, exist_ok=False)

    mode = (
        "path_check" if args.check_route
        else "execute" if args.execute
        else "preview"
    )

    controller = None
    record = {
        "started_at": datetime.now().astimezone().isoformat(),
        "mode": mode,
        "above_only": args.above_only,
        "status": "started",
        "pick_profile": pick_profile,
        "transfer_profile": transfer_profile,
        "current_calibration": calibration,
        "reference_calibration": reference_calibration,
    }

    try:
        if args.image:
            image_path = project_path(args.image)
            frame = cv2.imread(str(image_path))

            if frame is None:
                raise FileNotFoundError(f"Could not read {image_path}")

            record["source_image"] = str(image_path)

        else:
            print("Keep the robot outside the camera view.")
            print("Keep the camera, sheet, and part stationary.")
            frame = capture_frame(calibration)
            record["source_image"] = "live_camera"

        # Retain the original capture even if target analysis fails.
        save_images(trial_dir, {"camera.png": frame})

        analysis, images = calculate_rotated_target(
            frame,
            calibration,
            reference_calibration,
            pick_profile,
        )
        record["analysis"] = analysis
        save_images(trial_dir, images)

        target = analysis["target_xy_mm"]
        orientation = analysis["target_orientation_deg"]

        # Apply the detected orientation only to this trial's profile.
        runtime_profile = copy.deepcopy(pick_profile)
        runtime_profile["motion"]["tool_orientation_deg"] = list(
            orientation
        )
        record["runtime_pick_profile"] = runtime_profile

        print("\nROBOT 1 PICK AND PLACE PLAN")
        print("Pick X,Y:", target)
        print("Pick orientation:", orientation)
        print("Pick Z:", runtime_profile["motion"]["pick_z_mm"])
        print("Transfer strategy:", transfer_profile["transfer_strategy"])
        print(
            "Zone 2 placement:",
            transfer_profile["placement_pose_mm_deg"],
        )
        print("Retreat:", transfer_profile["retreat_pose_mm_deg"])

        if not (args.execute or args.check_route):
            # Vision preview never creates a robot connection.
            record["status"] = "preview_completed"
            print("\nPREVIEW ONLY: no robot connection or movement.")

            if not args.no_display:
                show_preview(images, analysis)

            return

        controller = Lite6Controller()
        controller.connect()

        # Construct the route from the actual starting pose.
        start = list(controller.get_pose())
        plan = build_cycle_plan(
            start,
            target,
            runtime_profile,
            transfer_profile,
            above_only=args.above_only,
        )
        route = movement_segments(plan)

        record["start_pose_mm_deg"] = start
        record["plan"] = plan
        record["planned_route"] = route
        record["status"] = "ready_for_route_check"

        # Save the intended operations before checking or executing.
        save_record(trial_dir, record)

        if args.check_route:
            record["path_check"] = controller.check_linear_route(route)
            record["status"] = "path_check_passed"

            print("\nROUTE CHECK PASSED.")
            print("No movement or gripper command was executed.")

        elif args.above_only:
            print("\nEXECUTING CHECKED ELEVATED ROUTE.")
            print("No descent or gripper operation is included.")

            cycle = run_elevated_route(controller, route)
            record["cycle"] = cycle
            record["path_check"] = cycle["path_check"]
            record["status"] = cycle["status"]

        else:
            print("\nCHECKING AND EXECUTING COMPLETE CYCLE.")

            # The shared executor rebuilds from its actual start and
            # checks the complete plan before any physical operation.
            cycle = run_pick_and_place_cycle(
                controller,
                target,
                runtime_profile,
                transfer_profile,
            )

            # Retain the exact plan used by the executor.
            record["cycle"] = cycle
            record["plan"] = cycle["plan"]
            record["planned_route"] = movement_segments(cycle["plan"])
            record["path_check"] = cycle["path_check"]
            record["status"] = cycle["status"]

    except PickCycleError as error:
        record["status"] = "failed"
        record["cycle"] = error.result
        record["error"] = str(error)

        # A failed cycle may still contain its exact execution plan.
        if "plan" in error.result:
            record["plan"] = error.result["plan"]
            record["planned_route"] = movement_segments(
                error.result["plan"]
            )

        if "path_check" in error.result:
            record["path_check"] = error.result["path_check"]

        print(f"Sequence stopped: {error}")
        raise

    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        print(f"Test stopped: {error}")
        raise

    finally:
        # Keep rejected controller reports as well as successful ones.
        last_check = (
            getattr(controller, "last_path_check", None)
            if controller is not None
            else None
        )
        if last_check is not None:
            record["last_controller_path_check"] = copy.deepcopy(
                last_check
            )
            if "path_check" not in record:
                record["path_check"] = copy.deepcopy(last_check)

        try:
            if controller is not None:
                controller.disconnect()
        finally:
            record["finished_at"] = (
                datetime.now().astimezone().isoformat()
            )
            save_record(trial_dir, record)
            print(f"Trial saved: {trial_dir}")


if __name__ == "__main__":
    main()