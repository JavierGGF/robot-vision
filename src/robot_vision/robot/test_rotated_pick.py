"""
Preview or execute a pick using the horseshoe orientation.

The saved grasp point is rotated around the detected part center.
The tool yaw follows the part direction in robot-base coordinates.

Camera rotation is removed through the two sheet calibrations.
The original taught pose, height, speeds, and pick cycle are preserved.

Default operation is preview only.
An elevated approach can be requested before testing the complete pick.
"""

import argparse
import copy
import json
import math
from datetime import datetime

import cv2
import numpy as np

from robot_vision.robot.controller import Lite6Controller
from robot_vision.robot.pick_cycle import (
    PickCycleError,
    load_pick_profile,
    run_pick_cycle,
)
from robot_vision.robot.test_pick_motion import (
    PIXELS_PER_MM,
    PROJECT_ROOT,
    analyze_part,
    capture_frame,
    load_sheet_calibration,
    project_path,
    read_json,
    save_images,
    show_preview,
    transform_points,
)


def wrapped_angle(angle):
    """Express a signed angle within [-180, 180)."""

    return (float(angle) + 180.0) % 360.0 - 180.0


def direction_vector(geometry):
    """Express the horseshoe opening direction in sheet coordinates."""

    angle = math.radians(float(geometry["direction_image_deg"]))
    return np.array([math.cos(angle), math.sin(angle)])


def recover_reference_grasp(saved_record, reference_calibration):
    """Convert the original saved grasp to physical sheet coordinates."""

    # The original grasp was stored in the old calibration's millimeters.
    old_matrix = np.asarray(
        saved_record["planar_calibration"]["pixel_to_mm_matrix"],
        dtype=np.float64,
    )
    old_grasp = np.array(
        [
            saved_record["grasp_x_mm"],
            saved_record["grasp_y_mm"],
            1.0,
        ],
        dtype=np.float64,
    )

    # Recover its original camera pixel before applying the new registration.
    pixel = np.linalg.solve(old_matrix, old_grasp)

    if abs(pixel[2]) < 1e-12:
        raise ValueError("Invalid saved grasp conversion.")

    point = pixel[:2] / pixel[2]

    return transform_points(
        [point],
        reference_calibration,
    )[0]


def sheet_to_robot_matrix(profile, reference_calibration):
    """Reuse the original paired marker measurements."""

    markers = read_json(
        profile["local_reference"]["marker_record"]
    )
    image_points = np.asarray(
        markers["selected_image_points_px"][:3],
        dtype=np.float64,
    )
    robot_points = np.asarray(
        markers["robot_marker_points_xy_mm"],
        dtype=np.float64,
    )

    if image_points.shape != (3, 2) or robot_points.shape != (3, 2):
        raise ValueError("Three paired marker measurements are required.")

    if not np.isfinite(robot_points).all():
        raise ValueError("Invalid robot marker measurements.")

    sheet_points = transform_points(
        image_points,
        reference_calibration,
    )

    sheet_axes = np.column_stack(
        (
            sheet_points[1] - sheet_points[0],
            sheet_points[2] - sheet_points[0],
        )
    )
    robot_axes = np.column_stack(
        (
            robot_points[1] - robot_points[0],
            robot_points[2] - robot_points[0],
        )
    )

    if np.linalg.cond(sheet_axes) > 1000:
        raise ValueError("Invalid sheet marker geometry.")

    matrix = robot_axes @ np.linalg.inv(sheet_axes)

    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1000:
        raise ValueError("Invalid sheet-to-robot transformation.")

    return matrix


def mark_grasp(image, point):
    """Draw the predicted physical grasp point on a rectified image."""

    pixel = tuple(
        np.rint(np.asarray(point) * PIXELS_PER_MM).astype(int)
    )

    cv2.drawMarker(
        image,
        pixel,
        (255, 0, 255),
        cv2.MARKER_TILTED_CROSS,
        26,
        2,
    )
    cv2.putText(
        image,
        "GRASP",
        (pixel[0] + 12, pixel[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
    )


def calculate_rotated_target(
    frame,
    calibration,
    reference_calibration,
    profile,
):
    """Rotate the grasp offset and tool yaw using the horseshoe direction."""

    local = profile["local_reference"]
    reference_dir = project_path(local["capture_directory"])

    if (
        project_path(reference_calibration["reference_capture"]).resolve()
        != reference_dir.resolve()
    ):
        raise ValueError("Registered image does not match the taught grasp.")

    # Both views must describe the same physical sheet coordinates.
    for name in ("reference_width_mm", "reference_height_mm"):
        if not math.isclose(
            float(calibration[name]),
            float(reference_calibration[name]),
        ):
            raise ValueError("Sheet dimensions differ between calibrations.")

    if not np.allclose(
        calibration["reference_points_mm"],
        reference_calibration["reference_points_mm"],
    ):
        raise ValueError("Physical fiducial identities differ.")

    saved_record = read_json(reference_dir / "results.json")

    if saved_record.get("status") != "ok":
        raise ValueError("The original taught reference is invalid.")

    reference_frame = cv2.imread(
        str(project_path(reference_calibration["source_image"]))
    )

    if reference_frame is None:
        raise FileNotFoundError("Could not read the registered reference.")

    ref_detection, ref_center, ref_geometry, ref_images = analyze_part(
        reference_frame,
        reference_calibration,
    )
    detection, center, geometry, images = analyze_part(
        frame,
        calibration,
    )

    reference_grasp = recover_reference_grasp(
        saved_record,
        reference_calibration,
    )
    reference_offset = reference_grasp - ref_center

    # Sheet coordinates follow image-style axes: X right, Y down.
    # Use current minus reference direction for the standard 2-D rotation.
    sheet_angle = wrapped_angle(
        float(geometry["direction_image_deg"])
        - float(ref_geometry["direction_image_deg"])
    )
    theta = math.radians(sheet_angle)

    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ]
    )

    grasp = center + rotation @ reference_offset

    # Keep the predicted grasp within the calibrated fiducial region.
    fiducials = np.asarray(
        calibration["reference_points_mm"],
        dtype=np.float64,
    )

    if (
        np.any(grasp < fiducials.min(axis=0))
        or np.any(grasp > fiducials.max(axis=0))
    ):
        raise ValueError("Predicted grasp is outside the working region.")

    mapping = sheet_to_robot_matrix(
        profile,
        reference_calibration,
    )

    # The taught robot anchor corresponds to the saved reference grasp.
    robot_translation = mapping @ (grasp - reference_grasp)
    translation_distance = float(np.linalg.norm(robot_translation))
    maximum_translation = float(local["max_translation_mm"])

    if not math.isfinite(maximum_translation) or maximum_translation <= 0:
        raise ValueError("Invalid local translation limit.")

    if translation_distance > maximum_translation:
        raise ValueError(
            f"Local translation is {translation_distance:.1f} mm. "
            f"The limit is {maximum_translation:.1f} mm."
        )

    taught_xy = np.asarray(
        local["taught_robot_xy_mm"],
        dtype=np.float64,
    )

    if taught_xy.shape != (2,) or not np.isfinite(taught_xy).all():
        raise ValueError("Invalid taught robot grasp anchor.")

    target = taught_xy + robot_translation

    # Convert both opening directions to robot coordinates.
    # This determines the correct yaw sign without a camera-angle assumption.
    reference_direction = mapping @ direction_vector(ref_geometry)
    current_direction = mapping @ direction_vector(geometry)

    reference_robot_angle = math.degrees(
        math.atan2(reference_direction[1], reference_direction[0])
    )
    current_robot_angle = math.degrees(
        math.atan2(current_direction[1], current_direction[0])
    )
    yaw_change = wrapped_angle(
        current_robot_angle - reference_robot_angle
    )

    taught_orientation = profile["motion"]["tool_orientation_deg"]
    orientation = [
        float(taught_orientation[0]),
        float(taught_orientation[1]),
        wrapped_angle(float(taught_orientation[2]) + yaw_change),
    ]

    if not np.isfinite(target).all() or not np.isfinite(orientation).all():
        raise ValueError("Calculated robot pose is invalid.")

    # Preserve the sign convention used by the existing preview display.
    analysis = {
        "mapping": "rotated_grasp_with_preserved_taught_anchor",
        "detection": detection,
        "reference_detection": ref_detection,
        "current_geometry": geometry,
        "reference_geometry": ref_geometry,
        "current_center_sheet_mm": center.tolist(),
        "reference_center_sheet_mm": ref_center.tolist(),
        "reference_grasp_sheet_mm": reference_grasp.tolist(),
        "reference_grasp_offset_sheet_mm": reference_offset.tolist(),
        "grasp_sheet_mm": grasp.tolist(),
        "rotation_change_deg": wrapped_angle(-sheet_angle),
        "robot_yaw_change_deg": yaw_change,
        "target_xy_mm": target.tolist(),
        "target_orientation_deg": orientation,
        "translation_distance_mm": translation_distance,
        "tool_rotation_enabled": True,
    }

    # Retain the grasp overlays for visual inspection and documentation.
    mark_grasp(images["detection.png"], grasp)
    mark_grasp(ref_images["detection.png"], reference_grasp)

    images["reference_detection.png"] = ref_images["detection.png"]
    images["reference_geometry.png"] = ref_images["current_geometry.png"]

    return analysis, images


def approach_only(controller, target, orientation, profile):
    """Approach at height without descending or closing the gripper."""

    motion = profile["motion"]
    start = controller.get_pose()
    height = max(float(start[2]), motion["approach_z_mm"])

    # Raise before changing the tool orientation.
    controller.move_linear(
        [start[0], start[1], height, *start[3:6]],
        speed=motion["travel_speed_mm_s"],
        mvacc=motion["travel_acceleration"],
        wait=True,
    )

    # Set the calculated orientation while elevated.
    controller.move_linear(
        [start[0], start[1], height, *orientation],
        speed=motion["orientation_speed_deg_s"],
        mvacc=motion["travel_acceleration"],
        wait=True,
    )

    controller.open_gripper()

    # Move over the predicted grasp without descending.
    controller.move_linear(
        [*target, height, *orientation],
        speed=motion["travel_speed_mm_s"],
        mvacc=motion["travel_acceleration"],
        wait=True,
    )

    return {
        "status": "elevated_approach_completed",
        "target_xy_mm": target,
        "orientation_deg": orientation,
        "approach_z_mm": height,
        "measured_pose_mm_deg": controller.get_pose(),
        "part_retention_verified": False,
    }


def main():
    """Preview by default; execute only through an explicit option."""

    parser = argparse.ArgumentParser(
        description="Pick using the geometric horseshoe orientation."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approach-only", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument(
        "--image",
        help="Offline image from the current calibrated camera view.",
    )
    args = parser.parse_args()

    if args.execute and args.image:
        parser.error("--execute requires live capture.")

    if args.approach_only and not args.execute:
        parser.error("--approach-only requires --execute.")

    profile = load_pick_profile()
    calibration = load_sheet_calibration(
        "data/calibration/zone1_camera.json"
    )
    reference_calibration = load_sheet_calibration(
        "data/calibration/zone1_reference_camera.json"
    )

    controller = None
    trial_dir = (
        PROJECT_ROOT
        / "data"
        / "demo_trials"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    trial_dir.mkdir(parents=True, exist_ok=False)

    record = {
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "started",
        "mode": "execute" if args.execute else "preview",
        "approach_only": args.approach_only,
        "profile": profile,
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
            print("Keep the calibrated camera and sheet stationary.")
            print("Keep the part stationary during capture.")
            frame = capture_frame(calibration)
            record["source_image"] = "live_camera"

        save_images(trial_dir, {"camera.png": frame})

        analysis, images = calculate_rotated_target(
            frame,
            calibration,
            reference_calibration,
            profile,
        )
        record["analysis"] = analysis
        save_images(trial_dir, images)

        target = analysis["target_xy_mm"]
        orientation = analysis["target_orientation_deg"]

        print("\nROTATED PICK TARGET")
        print(f"X: {target[0]:.3f} mm")
        print(f"Y: {target[1]:.3f} mm")
        print(f"Pick Z: {profile['motion']['pick_z_mm']:.3f} mm")
        print("Tool roll / pitch / yaw:", orientation)
        print(
            f"Sheet rotation change: "
            f"{analysis['rotation_change_deg']:.2f} deg"
        )
        print(
            f"Robot yaw change: "
            f"{analysis['robot_yaw_change_deg']:.2f} deg"
        )
        print(
            f"Local translation: "
            f"{analysis['translation_distance_mm']:.2f} mm"
        )

        # Save the intended destination before making any robot connection.
        record["status"] = "target_calculated"
        (trial_dir / "results.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

        if not args.execute:
            record["status"] = "preview_completed"
            print("\nPREVIEW ONLY: no robot connection or movement.")

            if not args.no_display:
                show_preview(images, analysis)

        else:
            controller = Lite6Controller()
            controller.connect()

            if args.approach_only:
                record["cycle"] = approach_only(
                    controller,
                    target,
                    orientation,
                    profile,
                )

            else:
                # Supply the calculated orientation through a runtime copy.
                # Do not overwrite the original taught profile.
                runtime_profile = copy.deepcopy(profile)
                runtime_profile["motion"]["tool_orientation_deg"] = (
                    orientation
                )
                record["cycle"] = run_pick_cycle(
                    controller,
                    target,
                    runtime_profile,
                )

            record["status"] = record["cycle"]["status"]

    except PickCycleError as error:
        record["status"] = "failed"
        record["cycle"] = error.result
        record["error"] = str(error)
        print(f"Sequence stopped: {error}")
        raise

    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        print(f"Test stopped: {error}")
        raise

    finally:
        # Never retry movement or automatically release a suspended part.
        try:
            if controller is not None:
                controller.disconnect()

        finally:
            record["finished_at"] = datetime.now().astimezone().isoformat()
            (trial_dir / "results.json").write_text(
                json.dumps(record, indent=2),
                encoding="utf-8",
            )
            print(f"Trial saved: {trial_dir}")


if __name__ == "__main__":
    main()