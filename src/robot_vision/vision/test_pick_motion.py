"""
Preview or execute the preserved Robot 1 pick cycle.

Both the saved reference and the current image are interpreted in
printed-sheet coordinates. Camera rotation is therefore separated
from actual part rotation.

Default operation captures and saves a preview without robot motion.
Use --execute explicitly to perform the existing pick cycle.

The robot mapping retains the approximate marker measurements and
the accurately taught grasp anchor from the original demonstration.
Actual tool rotation for substantially rotated parts is not enabled.
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
from robot_vision.vision.detection import process_frame
from robot_vision.vision.test_orientation import (
    center_object_mask,
    geometry_preview,
    horseshoe_direction,
)


# Resolve paths independently of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# The existing detector produces four pixels per physical millimeter.
PIXELS_PER_MM = 4.0

# Use a larger template to avoid clipping the physically rectified part.
TEMPLATE_SIZE_PX = 801


def project_path(value):
    """Resolve an absolute path or a project-relative path."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(value):
    """Read a saved record without changing its contents."""

    path = project_path(value)
    return json.loads(path.read_text(encoding="utf-8"))


def load_sheet_calibration(value):
    """Require the new calibration based on printed fiducial centers."""

    calibration = read_json(value)

    if calibration.get("reference_type") != "printed_fiducial_centers":
        raise ValueError("A fiducial-center calibration is required.")

    if calibration.get("zone") != 1:
        raise ValueError("This program requires the Zone 1 calibration.")

    if calibration.get("point_order") != ["TL", "TR", "BR", "BL"]:
        raise ValueError("Invalid physical fiducial order.")

    matrix = np.asarray(
        calibration["pixel_to_mm_matrix"],
        dtype=np.float64,
    )

    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid sheet transformation.")

    return calibration


def capture_frame(calibration):
    """Capture at the calibrated resolution after exposure settles."""

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

        # Retain the final frame after automatic exposure stabilization.
        for _ in range(30):
            ok, frame = camera.read()

            if not ok or frame is None:
                raise RuntimeError("Could not capture a camera frame.")

        return frame

    finally:
        camera.release()


def transform_points(points, calibration):
    """Convert original camera pixels to printed-sheet millimeters."""

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected an array of image X,Y coordinates.")

    if not np.isfinite(points).all():
        raise ValueError("Image coordinates are invalid.")

    matrix = np.asarray(
        calibration["pixel_to_mm_matrix"],
        dtype=np.float64,
    )

    homogeneous = np.column_stack(
        (points, np.ones(len(points)))
    )
    projected = homogeneous @ matrix.T

    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError("Invalid sheet-coordinate conversion.")

    result = projected[:, :2] / projected[:, 2, None]

    if not np.isfinite(result).all():
        raise ValueError("Sheet coordinates are invalid.")

    return result


def analyze_part(frame, calibration):
    """Detect the part and measure its horseshoe direction on the sheet."""

    detection, rectified, mask = process_frame(frame, calibration)

    center = np.array(
        [
            detection["reference_x_mm"],
            detection["reference_y_mm"],
        ],
        dtype=np.float64,
    )

    # Restrict the detected center to the region between the fiducials.
    fiducials = np.asarray(
        calibration["reference_points_mm"],
        dtype=np.float64,
    )

    if (
        np.any(center < fiducials.min(axis=0))
        or np.any(center > fiducials.max(axis=0))
    ):
        raise ValueError(
            "Detected object center is outside the fiducial working region."
        )

    # Center the exterior silhouette before analyzing its concavity.
    silhouette = center_object_mask(
        mask,
        center[0] * PIXELS_PER_MM,
        center[1] * PIXELS_PER_MM,
        TEMPLATE_SIZE_PX,
    )

    # A clipped silhouette cannot provide a reliable geometric direction.
    if (
        np.any(silhouette[0, :])
        or np.any(silhouette[-1, :])
        or np.any(silhouette[:, 0])
        or np.any(silhouette[:, -1])
    ):
        raise ValueError("The part silhouette exceeds the analysis template.")

    geometry = horseshoe_direction(silhouette)

    images = {
        "detection.png": rectified,
        "segmentation.png": mask,
        "current_mask.png": silhouette,
        "current_geometry.png": geometry_preview(
            silhouette,
            geometry,
        ),
    }

    return detection, center, geometry, images


def calculate_target(frame, calibration, reference_calibration, profile):
    """Calculate translation using a common physical sheet reference."""

    local = profile["local_reference"]
    reference_dir = project_path(local["capture_directory"])

    # Ensure the registered image belongs to the existing taught grasp.
    registered_dir = project_path(
        reference_calibration["reference_capture"]
    )

    if registered_dir.resolve() != reference_dir.resolve():
        raise ValueError(
            "Registered reference does not match the pick profile."
        )

    # Both calibrations must describe the same printed sheet geometry.
    for field in ("reference_width_mm", "reference_height_mm"):
        if not math.isclose(
            float(calibration[field]),
            float(reference_calibration[field]),
        ):
            raise ValueError("Current and reference sheet dimensions differ.")

    if not np.allclose(
        calibration["reference_points_mm"],
        reference_calibration["reference_points_mm"],
    ):
        raise ValueError("Current and reference fiducial identities differ.")

    reference_image = project_path(
        reference_calibration["source_image"]
    )
    reference_frame = cv2.imread(str(reference_image))

    if reference_frame is None:
        raise FileNotFoundError(
            f"Could not read reference image: {reference_image}"
        )

    # Reprocess the original image with its newly registered calibration.
    reference_detection, reference_center, reference_geometry, ref_images = (
        analyze_part(reference_frame, reference_calibration)
    )

    detection, center, geometry, images = analyze_part(
        frame,
        calibration,
    )

    # Compare physical sheet directions instead of camera-image directions.
    rotation_change = (
        float(reference_geometry["direction_image_deg"])
        - float(geometry["direction_image_deg"])
        + 180.0
    ) % 360.0 - 180.0

    maximum_rotation = float(local["max_rotation_change_deg"])

    if not math.isfinite(maximum_rotation) or maximum_rotation <= 0:
        raise ValueError("Invalid local rotation limit.")

    if abs(rotation_change) > maximum_rotation:
        raise ValueError(
            f"Part rotated {rotation_change:.1f} degrees on the sheet. "
            f"The local limit is {maximum_rotation:.1f} degrees. "
            "Actual tool rotation has not been enabled."
        )

    markers = read_json(local["marker_record"])

    # Preserve the original measured robot points and selected marker corners.
    # Convert their old image coordinates to the common sheet coordinate frame.
    image_markers = np.asarray(
        markers["selected_image_points_px"][:3],
        dtype=np.float64,
    )
    robot_markers = np.asarray(
        markers["robot_marker_points_xy_mm"],
        dtype=np.float64,
    )

    if image_markers.shape != (3, 2) or robot_markers.shape != (3, 2):
        raise ValueError("Three paired marker measurements are required.")

    if not np.isfinite(robot_markers).all():
        raise ValueError("Invalid robot marker measurements.")

    sheet_markers = transform_points(
        image_markers,
        reference_calibration,
    )

    sheet_axes = np.column_stack(
        (
            sheet_markers[1] - sheet_markers[0],
            sheet_markers[2] - sheet_markers[0],
        )
    )
    robot_axes = np.column_stack(
        (
            robot_markers[1] - robot_markers[0],
            robot_markers[2] - robot_markers[0],
        )
    )

    if np.linalg.cond(sheet_axes) > 1000:
        raise ValueError("Invalid marker geometry for robot mapping.")

    # Anchor translation to the precisely taught grasp rather than to
    # the absolute origin of the approximate marker measurements.
    sheet_translation = center - reference_center
    robot_translation = robot_axes @ np.linalg.solve(
        sheet_axes,
        sheet_translation,
    )

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
        raise ValueError("Invalid taught grasp anchor.")

    target_xy = taught_xy + robot_translation

    if not np.isfinite(target_xy).all():
        raise ValueError("Calculated robot target is invalid.")

    # Preserve both rectified views for comparison and student documentation.
    images["reference_detection.png"] = ref_images["detection.png"]
    images["reference_geometry.png"] = ref_images["current_geometry.png"]

    analysis = {
        "mapping": "approximate_sheet_translation_with_taught_anchor",
        "detection": detection,
        "reference_detection": reference_detection,
        "current_center_sheet_mm": center.tolist(),
        "reference_center_sheet_mm": reference_center.tolist(),
        "current_geometry": geometry,
        "reference_geometry": reference_geometry,
        "rotation_change_deg": rotation_change,
        "translation_sheet_mm": sheet_translation.tolist(),
        "translation_robot_xy_mm": robot_translation.tolist(),
        "translation_distance_mm": translation_distance,
        "target_xy_mm": target_xy.tolist(),
        "tool_rotation_enabled": False,
    }

    return analysis, images


def save_images(directory, images):
    """Save the captured evidence without overwriting previous trials."""

    for filename, image in images.items():
        if not cv2.imwrite(str(directory / filename), image):
            raise RuntimeError(f"Could not save {filename}")


def show_preview(images, analysis):
    """Display the physical sheet view and the calculated robot target."""

    display = images["detection.png"].copy()
    target = analysis["target_xy_mm"]

    for row, text in enumerate(
        (
            f"Robot X={target[0]:.2f}, Y={target[1]:.2f} mm",
            f"Sheet rotation change: {analysis['rotation_change_deg']:.2f} deg",
            "PREVIEW ONLY - no robot movement",
        )
    ):
        cv2.putText(
            display,
            text,
            (10, 80 + row * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    try:
        cv2.imshow("Robot 1 Target Preview", display)
        cv2.imshow("Current Horseshoe", images["current_geometry.png"])
        cv2.imshow("Reference Horseshoe", images["reference_geometry.png"])
        print("Press any key in an image window to close.")
        cv2.waitKey(0)

    finally:
        cv2.destroyAllWindows()


def parse_arguments():
    """Require an explicit option for physical robot execution."""

    parser = argparse.ArgumentParser(
        description="Preview or execute the preserved Robot 1 pick cycle."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--image",
        type=Path,
        help="Offline image captured with the current camera calibration.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("config/robot1_pick.json"),
    )
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    # An old image must never directly trigger physical motion.
    if args.execute and args.image is not None:
        parser.error("--execute requires live capture; do not use --image.")

    return args


def main():
    """Save a target preview or call the unchanged reusable pick cycle."""

    args = parse_arguments()
    profile = load_pick_profile(args.profile)

    calibration = load_sheet_calibration(
        "data/calibration/zone1_camera.json"
    )
    reference_calibration = load_sheet_calibration(
        "data/calibration/zone1_reference_camera.json"
    )

    controller = None

    # Create the record before analysis so rejected captures are retained.
    trial_dir = (
        PROJECT_ROOT
        / "data"
        / "demo_trials"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    trial_dir.mkdir(parents=True, exist_ok=False)

    record = {
        "started_at": datetime.now().astimezone().isoformat(),
        "mode": "execute" if args.execute else "preview",
        "status": "started",
        "profile": profile,
        "current_calibration": calibration,
        "reference_calibration": reference_calibration,
    }

    try:
        if args.image is not None:
            image_path = project_path(args.image)
            frame = cv2.imread(str(image_path))

            if frame is None:
                raise FileNotFoundError(f"Could not read {image_path}")

            record["source_image"] = str(image_path)

        else:
            print("Keep the robot outside the camera view.")
            print("Keep the calibrated sheet and camera stationary.")
            print("Keep the part stationary during capture.")
            frame = capture_frame(calibration)
            record["source_image"] = "live_camera"

        save_images(trial_dir, {"camera.png": frame})

        analysis, images = calculate_target(
            frame,
            calibration,
            reference_calibration,
            profile,
        )
        record["analysis"] = analysis
        save_images(trial_dir, images)

        target = analysis["target_xy_mm"]
        motion = profile["motion"]

        print("\nROBOT 1 PICK TARGET")
        print(f"X: {target[0]:.3f} mm")
        print(f"Y: {target[1]:.3f} mm")
        print(f"Pick Z: {motion['pick_z_mm']:.3f} mm")
        print(f"Approach Z: at least {motion['approach_z_mm']:.3f} mm")
        print(f"Lift: {motion['lift_mm']:.1f} mm")
        print("Tool roll / pitch / yaw:", motion["tool_orientation_deg"])
        print(
            f"Sheet rotation change: "
            f"{analysis['rotation_change_deg']:.2f} deg"
        )
        print(
            f"Local translation: "
            f"{analysis['translation_distance_mm']:.2f} mm"
        )

        # Record the intended target before any possible robot connection.
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
            print("\nEXECUTING ROBOT MOTION AND GRIPPER COMMANDS.")
            controller = Lite6Controller()
            controller.connect()

            # Preserve the existing linear approach, descent, close, and lift.
            cycle = run_pick_cycle(controller, target, profile)
            record["cycle"] = cycle
            record["status"] = cycle["status"]

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
        # Do not retry motion or automatically release the gripper on errors.
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