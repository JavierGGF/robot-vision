"""Extend the currently inspected chain using existing vision and robot controls."""
import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_vision.robot.pick_cycle import build_cycle_plan
from robot_vision.robot.robot1_two_link_cycle import (
    PhysicalCouplingController, confirm, inspect_image, run_block, save_record,
)
from robot_vision.robot.test_pick_motion import (
    PROJECT_ROOT, read_json, capture_frame, save_images,
)
from robot_vision.vision.preview_zone2_horseshoe import analyze
from robot_vision.vision.preview_zone2_targets import (
    displacement_matrix, matching_calibration,
)

PICK = "data/orientation_tests/20261013_141042_884411"
REFERENCE = "data/zone2_chain_tests/20261013_152227_815452"
MARKERS = "data/workspace_calibrations/robot1_zone2_20261013_151241_356980/robot_mapping.json"


def read(path):
    return json.loads(Path(path).read_text())


def compensate(pose, matrix, origin, q0, q1, angle, yaw_change):
    """Transform a taught TCP around the observed opening in the sheet plane."""
    theta = math.radians(angle)
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    target = np.asarray(pose, dtype=float).copy()
    pivot = origin + matrix @ q0
    new_pivot = origin + matrix @ q1
    target[:2] = new_pivot + matrix @ rotation @ np.linalg.solve(matrix, target[:2] - pivot)
    target[5] = (target[5] + yaw_change + 180) % 360 - 180
    if not np.isfinite(target).all() or not 0 < target[2] < 190:
        raise ValueError("Invalid compensated target")
    return target.tolist()


def assembly_plan(start, xy, pick, entry, final):
    if start[2] > 191:
        raise ValueError("Start above the current 190 mm clearance; no movement issued")
    pick = copy.deepcopy(pick)
    pick["motion"]["approach_z_mm"] = 190.0
    pick["motion"]["travel_speed_mm_s"] = 20.0
    plan = build_cycle_plan(start, xy, pick)
    for step in plan:
        if step["operation"] == "move" and step["pose"][2] > 190:
            raise ValueError("Plan exceeds 190 mm clearance")
    orientation = pick["motion"]["tool_orientation_deg"]
    def move(label, pose, speed):
        plan.append({"operation": "move", "label": label, "pose": pose,
                     "speed": speed, "mvacc": 20.0})
    move("raise_for_transfer", [*xy, 190.0, *orientation], 10.0)
    move("travel_to_zone2", [*entry[:2], 190.0, *orientation], 20.0)
    turn = (entry[5] - orientation[2] + 180) % 360 - 180
    midpoint = (orientation[2] + turn / 2 + 180) % 360 - 180
    move("intermediate_orientation", [*entry[:2], 190.0, *entry[3:5], midpoint], 10.0)
    move("set_zone2_orientation", [*entry[:2], 190.0, *entry[3:]], 10.0)
    move("above_entry", [*entry[:2], min(entry[2] + 20, 190.0), *entry[3:]], 10.0)
    move("coupling_entry", entry, 5.0)
    move("couple_next_link", final, 1.0)
    # Mandatory release in the SAME plan before either retreat movement.
    # Existing PhysicalCouplingController sends two immediate open commands.
    plan.append({"operation": "gripper", "label": "release_in_zone2",
                 "opening": True, "delay_s": 1.0})
    move("clear_zone2", [*final[:2], 190.0, *final[3:]], 10.0)
    move("travel_to_retreat", [-120.758293, -155.405518, 190.0, *final[3:]], 20.0)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approach-only", action="store_true",
                        help="Pick and stop 20 mm above entry; no insertion or release")
    args = parser.parse_args()
    if args.execute:
        confirm("Pick and elevated approach only; keep part held." if args.approach_only else "One chain extension. Clear the route; support the chain; camera view unobstructed.", "VERIFIED")
    output = PROJECT_ROOT / "data/chain_extension_trials" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    record = {"status": "started", "blocks": [], "physical_assembly_verified": False,
              "pointer_offset_corrected": False, "external_obstacles_checked": False,
              "mode": "physical" if args.execute else "preview"}
    controller = None
    try:
        cal1 = read_json("data/calibration/zone1_camera.json")
        cal2 = read_json("data/calibration/zone2_camera.json")
        if cal1["camera_index"] != cal2["camera_index"]:
            raise ValueError("This trial requires the same camera for both zones")
        reference = PROJECT_ROOT / REFERENCE
        ref_cal2 = read(reference / "calibration.json")
        if not matching_calibration(cal2, ref_cal2):
            raise ValueError("Zone 2 calibration differs from the taught reference")
        previews = sorted((PROJECT_ROOT / "data/chain_destination_previews").glob("*/results.json"))
        if not previews:
            raise ValueError("Run preview_chain_destination.py and inspect the FREE opening first")
        approved_folder = previews[-1].parent
        approved = read(previews[-1])
        if approved.get("status") != "candidate_not_verified":
            raise ValueError("Latest destination preview was rejected")
        if not matching_calibration(cal2, read(approved_folder / "calibration.json")):
            raise ValueError("Destination preview calibration changed")
        frame = capture_frame(cal1)
        for zone, cal in ((1, cal1), (2, cal2)):
            folder = output / f"zone{zone}"
            folder.mkdir()
            save_images(folder, {"camera.png": frame})
            (folder / "calibration.json").write_text(json.dumps(cal, indent=2) + "\n")
        preview, _, current_geometry = analyze(frame, cal2)
        save_images(output / "zone2", {"opening_preview.png": preview})
        old_geometry = approved["geometry"]
        preview_shift = np.linalg.norm(np.asarray(current_geometry["mouth_center_px"]) - old_geometry["mouth_center_px"]) / 4
        preview_turn = (current_geometry["direction_image_deg"] - old_geometry["direction_image_deg"] + 180) % 360 - 180
        if preview_shift > 3 or abs(preview_turn) > 2:
            raise ValueError("Opening changed since the inspected preview; run its preview again")
        ref_frame = cv2.imread(str(reference / "camera.png"))
        if ref_frame is None:
            raise FileNotFoundError(reference / "camera.png")
        _, _, ref_geometry = analyze(ref_frame, ref_cal2)
        q0 = np.asarray(ref_geometry["mouth_center_px"], dtype=float) / 4
        q1 = np.asarray(current_geometry["mouth_center_px"], dtype=float) / 4
        angle = (current_geometry["direction_image_deg"] - ref_geometry["direction_image_deg"] + 180) % 360 - 180
        shift = float(np.linalg.norm(q1 - q0))
        record.update(reference_geometry=ref_geometry,
                      destination_geometry=current_geometry,
                      rotation_sheet_deg=float(angle),
                      opening_shift_sheet_mm=shift,
                      reference_mouth_sheet_mm=q0.tolist(),
                      current_mouth_sheet_mm=q1.tolist(),
                      inspected_preview=str(approved_folder),
                      inspected_preview_shift_mm=float(preview_shift),
                      inspected_preview_turn_deg=float(preview_turn))
        print(f"REFERENCE DIRECTION: {ref_geometry['direction_image_deg']:.3f} deg")
        print(f"CURRENT DIRECTION: {current_geometry['direction_image_deg']:.3f} deg")
        print(f"ROTATION CHANGE: {angle:.3f} deg | OPENING SHIFT: {shift:.3f} mm")
        print("REFERENCE MOUTH [sheet mm]:", q0.tolist())
        print("CURRENT MOUTH [sheet mm]:", q1.tolist())
        within_trial_range = abs(angle) <= 10 and shift <= 100
        record["within_physical_trial_range"] = bool(within_trial_range)
        save_record(output, record)
        # Preview reports the full transformation; physical limits remain.
        if not within_trial_range:
            print("OUTSIDE PHYSICAL TRIAL RANGE: preview allowed; execution blocked.")
            elevated_range = args.approach_only and abs(angle) <= 20 and shift <= 100
            if args.execute and not elevated_range:
                raise ValueError(f"Destination outside local trial range: rotation={angle:.3f} deg (max 10); shift={shift:.3f} mm (max 100)")
            if args.execute and elevated_range:
                print("ELEVATED APPROACH ONLY: insertion and release disabled.")
        markers = read(PROJECT_ROOT / MARKERS)
        matrix = displacement_matrix(markers, cal2)
        tl = next(item["pose_mm_deg"] for item in markers["markers"] if item["marker"] == "TL")
        origin = np.asarray(tl[:2]) - matrix @ np.asarray(cal2["reference_points_mm"][0])
        def direction(value):
            a = math.radians(value)
            v = matrix @ np.array([math.cos(a), math.sin(a)])
            return math.degrees(math.atan2(v[1], v[0]))
        yaw_change = (direction(current_geometry["direction_image_deg"]) - direction(ref_geometry["direction_image_deg"]) + 180) % 360 - 180
        targets = {label: compensate(read(reference / f"taught_coupling_{label}.json")["pose_mm_deg"],
                                   matrix, origin, q0, q1, angle, yaw_change)
                   for label in ("entry", "final")}
        pick_folder = PROJECT_ROOT / PICK
        pick = read(pick_folder / "pick_profile.json")
        analysis, runtime = inspect_image(frame, cal1, read(pick_folder / "reference_camera.json"),
                                          pick, output / "zone1", "fresh_live_frame")
        record.update(pick_analysis=analysis, destination_geometry=current_geometry,
                      targets=targets, rotation_sheet_deg=angle, yaw_change_deg=yaw_change,
                      inspected_preview=str(approved_folder))
        print("PICK:", analysis["target_xy_mm"], analysis["target_orientation_deg"])
        print("ENTRY:", targets["entry"])
        print("FINAL:", targets["final"])
        if not args.execute:
            record["status"] = "targets_calculated_no_robot_connection"
            return
        controller = PhysicalCouplingController()
        if controller.robot_ip != "192.168.1.185":
            raise ValueError("Unexpected Robot 1 IP")
        controller.connect()
        controller.wait_stationary()
        start = controller.get_pose()
        plan = assembly_plan(start, analysis["target_xy_mm"], runtime, targets["entry"], targets["final"])
        if args.approach_only:
            end = next(i for i, step in enumerate(plan) if step["label"] == "above_entry")
            plan = plan[:end + 1]
            record["approach_only"] = True
        record["plan"] = plan
        save_record(output, record)
        # run_block checks the COMPLETE route before opening or picking.
        run_block(controller, "pick_couple_release_retreat", plan, start, record, output)
        if args.approach_only:
            record["status"] = "elevated_approach_completed_part_held"
            print("STOPPED ABOVE ENTRY. Gripper remains closed; no insertion or release.")
        else:
            controller.stop_gripper_after_completion()
            record["status"] = "motion_completed_physical_fit_unverified"
    except BaseException as error:
        record.update(status="stopped", error=str(error) or "Interrupted")
        if controller is not None and controller.arm is not None:
            controller.arm.set_state(4)
        raise
    finally:
        if controller is not None:
            controller.disconnect()
        save_record(output, record)
        print("SAVED:", output)


if __name__ == "__main__":
    main()
