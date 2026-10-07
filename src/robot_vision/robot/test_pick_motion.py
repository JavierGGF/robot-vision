"""
Preview or execute the reusable Robot 1 pick cycle.

Default behavior:
- Capture the part or load a supplied image.
- Calculate the target using the saved local reference.
- Save images and target information.
- Do not connect to or move the robot.

Explicit execution:
- Use --execute with a live camera capture.
- Connect and call run_pick_cycle().
- Stop on any reported error without retrying movements.

The local mapping remains approximate.
Camera position, paper position, and tool mounting must match the reference.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_vision.robot.controller import Lite6Controller
from robot_vision.robot.pick_cycle import (
    PickCycleError,
    load_pick_profile,
    run_pick_cycle,
)
from robot_vision.vision.detection import (
    load_calibration,
    process_frame,
)
from robot_vision.vision.test_orientation import (
    center_object_mask,
    horseshoe_direction,
    geometry_preview,
)


# Resolve project paths independently of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(value):
    """Resolve a profile path relative to the project when necessary."""

    path = Path(value)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def read_json(path):
    """Read an existing JSON file without changing it."""

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def capture_frame(calibration):
    """Capture at the calibrated resolution after exposure stabilization."""

    camera = cv2.VideoCapture(calibration["camera_index"])

    try:
        if not camera.isOpened():
            raise RuntimeError(
                "Could not open camera. Close other camera programs."
            )

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            calibration["image_width_px"],
        )
        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            calibration["image_height_px"],
        )

        frame = None

        # Allow automatic exposure to settle before retaining the last frame.
        for _ in range(30):
            success, candidate = camera.read()

            if not success or candidate is None:
                raise RuntimeError("Could not capture camera frame.")

            frame = candidate

        return frame

    finally:
        camera.release()


def camera_pixel(detection, calibration):
    """Convert planar detection coordinates back to original camera pixels."""

    matrix = np.asarray(
        calibration["pixel_to_mm_matrix"],
        dtype=np.float64,
    )

    point = np.array([
        detection["reference_x_mm"],
        detection["reference_y_mm"],
        1.0,
    ], dtype=np.float64)

    camera_point = np.linalg.solve(matrix, point)

    if abs(camera_point[2]) < 1e-12:
        raise ValueError("Invalid camera-coordinate conversion.")

    pixel = camera_point[:2] / camera_point[2]

    if not np.all(np.isfinite(pixel)):
        raise ValueError("Camera coordinates are invalid.")

    return pixel


def calculate_local_target(frame, calibration, profile):
    """
    Calculate a robot target using the preserved local demonstration method.

    This function does not connect to the robot.
    It deliberately keeps the existing local mapping separate from the cycle.
    """

    local = profile["local_reference"]

    reference_dir = project_path(local["capture_directory"])
    marker_path = project_path(local["marker_record"])

    reference = read_json(reference_dir / "results.json")
    markers = read_json(marker_path)

    if reference.get("status") != "ok":
        raise ValueError("The saved reference detection is not valid.")

    # Verify the calibration values used to interpret the reference image.
    old_calibration = reference["planar_calibration"]

    for name in (
        "image_width_px",
        "image_height_px",
        "reference_width_mm",
        "reference_height_mm",
    ):
        if calibration[name] != old_calibration[name]:
            raise ValueError(
                f"Calibration setting '{name}' changed since teaching."
            )

    old_matrix = np.asarray(
        old_calibration["pixel_to_mm_matrix"],
        dtype=np.float64,
    )
    current_matrix = np.asarray(
        calibration["pixel_to_mm_matrix"],
        dtype=np.float64,
    )

    if not np.allclose(old_matrix, current_matrix):
        raise ValueError(
            "Planar calibration changed. Update the local reference "
            "before using this mapping."
        )

    # Reuse the current detection and geometric orientation functions.
    detection, rectified, mask = process_frame(frame, calibration)

    grasp_config = reference["grasp_calibration"]
    scale = float(grasp_config["pixels_per_mm"])

    if not math.isclose(scale, 4.0):
        raise ValueError(
            "Reference scale does not match the current detector."
        )

    current_mask = center_object_mask(
        mask,
        detection["reference_x_mm"] * scale,
        detection["reference_y_mm"] * scale,
        int(grasp_config["silhouette_size_px"]),
    )

    geometry = horseshoe_direction(current_mask)

    reference_direction = float(
        reference["current_geometry"]["direction_image_deg"]
    )
    current_direction = float(geometry["direction_image_deg"])

    # Express the rotation difference in the range [-180, 180).
    rotation_change = (
        reference_direction - current_direction + 180.0
    ) % 360.0 - 180.0

    maximum_rotation = float(local["max_rotation_change_deg"])

    if not math.isfinite(maximum_rotation) or maximum_rotation <= 0:
        raise ValueError("Invalid rotation limit in the local profile.")

    if abs(rotation_change) > maximum_rotation:
        raise ValueError(
            f"Part rotated {rotation_change:.1f} degrees. "
            f"The local limit is {maximum_rotation:.1f} degrees."
        )

    # Use the three previously selected marker corners.
    image_markers = np.asarray(
        markers["selected_image_points_px"][:3],
        dtype=np.float64,
    )
    robot_markers = np.asarray(
        markers["robot_marker_points_xy_mm"],
        dtype=np.float64,
    )

    if image_markers.shape != (3, 2):
        raise ValueError("Expected three image marker points.")

    if robot_markers.shape != (3, 2):
        raise ValueError("Expected three robot marker points.")

    if (
        not np.all(np.isfinite(image_markers))
        or not np.all(np.isfinite(robot_markers))
    ):
        raise ValueError("Marker measurements contain invalid values.")

    image_axes = np.column_stack((
        image_markers[1] - image_markers[0],
        image_markers[2] - image_markers[0],
    ))
    robot_axes = np.column_stack((
        robot_markers[1] - robot_markers[0],
        robot_markers[2] - robot_markers[0],
    ))

    if np.linalg.cond(image_axes) > 1000:
        raise ValueError("Invalid image marker geometry.")

    reference_pixel = camera_pixel(
        reference["detection"],
        old_calibration,
    )
    current_pixel = camera_pixel(detection, calibration)

    # Keep the detected center inside the marked working region.
    region_uv = np.linalg.solve(
        image_axes,
        current_pixel - image_markers[0],
    )

    if np.any(region_uv < 0.0) or np.any(region_uv > 1.0):
        raise ValueError("Part is outside the marked working region.")

    # Anchor the translation to the accurately taught grasp position.
    translation_uv = np.linalg.solve(
        image_axes,
        current_pixel - reference_pixel,
    )
    robot_translation = robot_axes @ translation_uv

    maximum_translation = float(local["max_translation_mm"])

    if (
        not math.isfinite(maximum_translation)
        or maximum_translation <= 0
    ):
        raise ValueError("Invalid translation limit in the local profile.")

    translation_distance = float(np.linalg.norm(robot_translation))

    if translation_distance > maximum_translation:
        raise ValueError(
            f"Local translation is {translation_distance:.1f} mm. "
            f"The limit is {maximum_translation:.1f} mm."
        )

    taught_xy = np.asarray(
        local["taught_robot_xy_mm"],
        dtype=np.float64,
    )

    if taught_xy.shape != (2,) or not np.all(np.isfinite(taught_xy)):
        raise ValueError("Invalid taught robot X,Y reference.")

    target_xy = taught_xy + robot_translation

    if not np.all(np.isfinite(target_xy)):
        raise ValueError("Calculated robot target is invalid.")

    # Return the target independently of any robot connection.
    analysis = {
        "detection": detection,
        "rotation_change_deg": float(rotation_change),
        "translation_robot_xy_mm": robot_translation.tolist(),
        "translation_distance_mm": translation_distance,
        "target_xy_mm": target_xy.tolist(),
        "current_geometry": geometry,
        "reference_capture": str(reference_dir),
        "mapping": "approximate_local_translation",
    }

    images = {
        "camera.png": frame,
        "detection.png": rectified,
        "segmentation.png": mask,
        "current_mask.png": current_mask,
        "current_geometry.png": geometry_preview(
            current_mask,
            geometry,
        ),
    }

    return analysis, images


def show_preview(images, analysis):
    """Display the captured detection and geometric opening direction."""

    display = images["detection.png"].copy()
    target = analysis["target_xy_mm"]

    cv2.putText(
        display,
        f"Robot target: X={target[0]:.2f}, Y={target[1]:.2f} mm",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        display,
        "PREVIEW ONLY - no robot movement",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    try:
        cv2.imshow("Robot 1 Target Preview", display)
        cv2.imshow(
            "Current Horseshoe",
            images["current_geometry.png"],
        )
        print("Press any key in an image window to close the preview.")
        cv2.waitKey(0)

    finally:
        cv2.destroyAllWindows()


def parse_arguments():
    """Provide preview as the default and require explicit motion selection."""

    parser = argparse.ArgumentParser(
        description="Preview or execute the reusable Robot 1 pick cycle."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Capture live and execute robot motion and gripper commands.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Use a saved camera image for an offline preview.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("config/robot1_pick.json"),
        help="Pick profile path, relative to the project or absolute.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Print and save the preview without opening image windows.",
    )

    arguments = parser.parse_args()

    # Do not execute a pick from an old image whose scene may have changed.
    if arguments.execute and arguments.image is not None:
        parser.error(
            "--execute requires a live capture and cannot use --image."
        )

    return arguments


def main():
    """Calculate a target, then preview it or call the reusable pick cycle."""

    args = parse_arguments()
    profile = load_pick_profile(args.profile)
    calibration = load_calibration()

    controller = None
    trial_dir = None

    record = {
        "started_at": datetime.now().astimezone().isoformat(),
        "mode": "execute" if args.execute else "preview",
        "profile": profile,
        "status": "started",
    }

    try:
        if args.image is not None:
            # Offline preview uses a saved raw camera image.
            image_path = project_path(args.image)
            frame = cv2.imread(str(image_path))

            if frame is None:
                raise FileNotFoundError(
                    f"Could not read image: {image_path}"
                )

            record["source_image"] = str(image_path)

        else:
            # Live capture takes place before connecting to the robot.
            print("Keep the robot outside the camera view.")
            print("Keep the paper and part stationary.")
            frame = capture_frame(calibration)
            record["source_image"] = "live_camera"

        analysis, images = calculate_local_target(
            frame,
            calibration,
            profile,
        )
        record["analysis"] = analysis

        motion = profile["motion"]
        target = analysis["target_xy_mm"]

        print()
        print("ROBOT 1 PICK TARGET")
        print(f"X: {target[0]:.3f} mm")
        print(f"Y: {target[1]:.3f} mm")
        print(f"Pick Z: {motion['pick_z_mm']:.3f} mm")
        print(f"Approach Z: at least {motion['approach_z_mm']:.3f} mm")
        print(f"Lift: {motion['lift_mm']:.1f} mm")
        print(
            "Tool roll / pitch / yaw:",
            motion["tool_orientation_deg"],
        )
        print(
            f"Rotation change: {analysis['rotation_change_deg']:.2f} deg"
        )
        print(
            f"Local translation: "
            f"{analysis['translation_distance_mm']:.2f} mm"
        )

        # Save a separate trial without changing the calibration or reference.
        trial_dir = (
            PROJECT_ROOT
            / "data"
            / "demo_trials"
            / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
        trial_dir.mkdir(parents=True, exist_ok=False)

        for filename, image in images.items():
            if not cv2.imwrite(str(trial_dir / filename), image):
                raise RuntimeError(f"Could not save {filename}")

        # Preserve the calculated target before making any robot connection.
        record["status"] = "target_calculated"
        (trial_dir / "results.json").write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

        if not args.execute:
            # Preview mode never creates or connects a robot controller.
            record["status"] = "preview_completed"
            print()
            print("PREVIEW ONLY: no robot connection or movement.")

            if not args.no_display:
                show_preview(images, analysis)

        else:
            # Use the reusable function instead of copying the cycle here.
            print()
            print("EXECUTING ROBOT MOTION AND GRIPPER COMMANDS.")

            controller = Lite6Controller()
            controller.connect()

            cycle_result = run_pick_cycle(
                controller,
                target,
                profile,
            )

            record["cycle"] = cycle_result
            record["status"] = cycle_result["status"]

    except PickCycleError as error:
        # Retain the cycle stage and completed steps for diagnosis.
        record["status"] = "failed"
        record["cycle"] = error.result
        record["error"] = str(error)
        print(f"Sequence stopped: {error}")
        raise

    except Exception as error:
        # Never retry movements or automatically release a suspended part.
        record["status"] = "failed"
        record["error"] = str(error)
        print(f"Test stopped: {error}")
        raise

    finally:
        # Disconnect before writing the final report.
        try:
            if controller is not None:
                controller.disconnect()

        finally:
            record["finished_at"] = (
                datetime.now().astimezone().isoformat()
            )

            if trial_dir is not None:
                (trial_dir / "results.json").write_text(
                    json.dumps(record, indent=2),
                    encoding="utf-8",
                )
                print(f"Trial saved: {trial_dir}")


if __name__ == "__main__":
    main()