"""
Shared camera and preview functions for Robot 1 pick tests.

The rotated-pick program uses these functions to capture an image,
interpret the printed-sheet calibration, and inspect the horseshoe.

Executing this module delegates to the rotated-pick program.
Default operation remains preview only.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from robot_vision.vision.detection import process_frame
from robot_vision.vision.test_orientation import (
    center_object_mask,
    geometry_preview,
    horseshoe_direction,
)


# Resolve paths independently of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Match the scale used by the existing detector.
PIXELS_PER_MM = 4.0

# Provide enough room for the physically rectified part silhouette.
TEMPLATE_SIZE_PX = 801


def project_path(value):
    """Resolve an absolute path or a project-relative path."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(value):
    """Read a JSON record without modifying it."""

    path = project_path(value)
    return json.loads(path.read_text(encoding="utf-8"))


def load_sheet_calibration(value):
    """Require calibration using the printed fiducial centers."""

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
    fiducials = np.asarray(
        calibration["reference_points_mm"],
        dtype=np.float64,
    )

    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid sheet transformation.")

    if fiducials.shape != (4, 2) or not np.isfinite(fiducials).all():
        raise ValueError("Invalid physical fiducial coordinates.")

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

        # Allow automatic exposure to stabilize before retaining a frame.
        for _ in range(30):
            ok, frame = camera.read()

            if not ok or frame is None:
                raise RuntimeError("Could not capture a camera frame.")

        return frame

    finally:
        # Release the camera before any possible robot connection.
        camera.release()


def transform_points(points, calibration):
    """Convert original camera pixels to printed-sheet millimeters."""

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected image X,Y coordinates.")

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
    fiducials = np.asarray(
        calibration["reference_points_mm"],
        dtype=np.float64,
    )

    # Restrict the detected center to the calibrated working region.
    if (
        np.any(center < fiducials.min(axis=0))
        or np.any(center > fiducials.max(axis=0))
    ):
        raise ValueError(
            "Detected object center is outside the fiducial working region."
        )

    silhouette = center_object_mask(
        mask,
        center[0] * PIXELS_PER_MM,
        center[1] * PIXELS_PER_MM,
        TEMPLATE_SIZE_PX,
    )

    # Reject a clipped silhouette before estimating its direction.
    if (
        np.any(silhouette[0, :])
        or np.any(silhouette[-1, :])
        or np.any(silhouette[:, 0])
        or np.any(silhouette[:, -1])
    ):
        raise ValueError("The silhouette exceeds the analysis template.")

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


def save_images(directory, images):
    """Save trial evidence and report any image-writing failure."""

    directory = Path(directory)

    for filename, image in images.items():
        if not cv2.imwrite(str(directory / filename), image):
            raise RuntimeError(f"Could not save {filename}")


def show_preview(images, analysis):
    """Display the grasp overlay, opening direction, and robot target."""

    display = images["detection.png"].copy()
    target = analysis["target_xy_mm"]

    instructions = [
        f"Robot X={target[0]:.2f}, Y={target[1]:.2f} mm",
        f"Sheet rotation: {analysis['rotation_change_deg']:.2f} deg",
    ]

    if "target_orientation_deg" in analysis:
        yaw = analysis["target_orientation_deg"][2]
        instructions.append(f"Tool yaw: {yaw:.2f} deg")

    instructions.append("PREVIEW ONLY - no robot movement")

    for row, text in enumerate(instructions):
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
        cv2.imshow(
            "Current Horseshoe",
            images["current_geometry.png"],
        )

        if "reference_geometry.png" in images:
            cv2.imshow(
                "Reference Horseshoe",
                images["reference_geometry.png"],
            )

        print("Press any key in an image window to close.")
        cv2.waitKey(0)

    finally:
        cv2.destroyAllWindows()


def main():
    """Keep the original command available through the rotated-pick runner."""

    # Import only when executed, after the shared functions are defined.
    from robot_vision.robot.test_rotated_pick import main as rotated_main

    rotated_main()


if __name__ == "__main__":
    main()