"""
Calibrate a workspace using the four printed fiducial centers.

The calibration uses the physical orientation of the printed sheet.
Camera rotation is removed by mapping the image to sheet coordinates.

This program records images and calibration data.
It does not connect to or move a robot.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# Locate the project independently of the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Dimensions of the printed US Letter calibration sheet.
PAPER_WIDTH_MM = 215.9
PAPER_HEIGHT_MM = 279.4

# Identify corners according to the printed sheet, not the screen.
LABELS = ("TL", "TR", "BR", "BL")

# Physical centers of the four 12 mm black fiducials.
FIDUCIAL_CENTERS_MM = np.array(
    [
        [21.0, 21.0],
        [194.9, 21.0],
        [194.9, 258.4],
        [21.0, 258.4],
    ],
    dtype=np.float32,
)

# Leave room above the camera image for instructions.
HEADER_HEIGHT = 100
WINDOW_NAME = "Workspace Calibration"


def make_calibration(frame, points, camera_index, zone):
    """Map the selected image centers to the printed sheet."""

    image_points = np.asarray(points, dtype=np.float32)

    # Reject incomplete, crossed, or collapsed selections.
    if image_points.shape != (4, 2):
        raise ValueError("Select exactly four fiducial centers.")

    contour = image_points.reshape(4, 1, 2)

    if not cv2.isContourConvex(contour):
        raise ValueError("The points must follow TL, TR, BR, BL.")

    if abs(cv2.contourArea(contour)) < 100:
        raise ValueError("The selected region is too small.")

    matrix = cv2.getPerspectiveTransform(
        image_points,
        FIDUCIAL_CENTERS_MM,
    )

    if not np.isfinite(matrix).all():
        raise ValueError("Could not calculate a valid transformation.")

    height, width = frame.shape[:2]

    # Preserve the existing detector's calibration field names.
    return {
        "calibration_version": 2,
        "zone": zone,
        "camera_index": camera_index,
        "image_width_px": width,
        "image_height_px": height,
        "reference_width_mm": PAPER_WIDTH_MM,
        "reference_height_mm": PAPER_HEIGHT_MM,
        "reference_type": "printed_fiducial_centers",
        "point_order": list(LABELS),
        "image_points": image_points.tolist(),
        "reference_points_mm": FIDUCIAL_CENTERS_MM.tolist(),
        "pixel_to_mm_matrix": matrix.tolist(),
    }


def rectified_preview(frame, calibration):
    """Show the sheet upright in its physical coordinate system."""

    # Use a fixed scale for a clear preview of the full sheet.
    scale = 3.0
    matrix = np.asarray(
        calibration["pixel_to_mm_matrix"],
        dtype=np.float64,
    )

    output = cv2.warpPerspective(
        frame,
        np.diag([scale, scale, 1.0]) @ matrix,
        (
            round(PAPER_WIDTH_MM * scale),
            round(PAPER_HEIGHT_MM * scale),
        ),
    )

    # Mark where each fiducial center should appear after rectification.
    for label, point in zip(LABELS, FIDUCIAL_CENTERS_MM):
        center = tuple(np.rint(point * scale).astype(int))
        cv2.drawMarker(
            output,
            center,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            20,
            2,
        )
        cv2.putText(
            output,
            label,
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return output


def main():
    """Capture, select, inspect, and save one workspace calibration."""

    parser = argparse.ArgumentParser(
        description="Calibrate a workspace from four fiducial centers."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--zone", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)

    # Request the resolution used by the current camera setup.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    frozen = None
    points = []
    calibration = None
    preview = None
    display_scale = 1.0
    message = "SPACE: freeze image | ESC: exit"

    def on_mouse(event, x, y, flags, parameter):
        """Convert display clicks back to original camera pixels."""

        nonlocal calibration, preview, message

        if frozen is None:
            return

        # Right-click removes the most recent selected center.
        if event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
            calibration = None
            preview = None
            message = "Selection updated. Continue clicking centers."
            return

        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return

        image_x = x / display_scale
        image_y = (y - HEADER_HEIGHT) / display_scale
        height, width = frozen.shape[:2]

        # Ignore clicks in the instruction area or outside the image.
        if not (0 <= image_x < width and 0 <= image_y < height):
            return

        points.append([image_x, image_y])
        calibration = None
        preview = None

        if len(points) == 4:
            message = "ENTER: inspect calibration | Right-click: undo"
        else:
            message = f"Click {LABELS[len(points)]} fiducial center."

    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {args.camera}. "
                "Close other camera programs or change --camera."
            )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        print("Use the physical orientation of the printed sheet.")
        print("The printed footer belongs at the bottom.")
        print("Click centers in this order: TL, TR, BR, BL.")
        print("SPACE: freeze | ENTER: preview | S: save | R: restart")

        while True:
            # Read live frames until the user freezes one for selection.
            if frozen is None:
                ok, frame = cap.read()

                if not ok:
                    raise RuntimeError("Could not read a camera frame.")
            else:
                frame = frozen

            height, width = frame.shape[:2]

            # Fit the full image on screen without changing its coordinates.
            display_scale = min(1200 / width, 675 / height, 1.0)
            view = cv2.resize(
                frame,
                (
                    round(width * display_scale),
                    round(height * display_scale),
                ),
            )

            # Label every selected center in the displayed image.
            for label, point in zip(LABELS, points):
                center = (
                    round(point[0] * display_scale),
                    round(point[1] * display_scale),
                )
                cv2.drawMarker(
                    view,
                    center,
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )
                cv2.putText(
                    view,
                    label,
                    (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            canvas = cv2.copyMakeBorder(
                view,
                HEADER_HEIGHT,
                0,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(30, 30, 30),
            )

            instructions = (
                f"Zone {args.zone}: physical sheet order TL, TR, BR, BL",
                message,
                "R: live view | Right-click: undo | ESC: exit",
            )

            for row, text in enumerate(instructions):
                cv2.putText(
                    canvas,
                    text,
                    (12, 25 + row * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                break

            # Freeze only on SPACE; the camera remains open until exit.
            if key == ord(" ") and frozen is None:
                frozen = frame.copy()
                points.clear()
                calibration = None
                preview = None
                message = "Click TL fiducial center."

            # Restart selection from the current live camera view.
            elif key == ord("r"):
                frozen = None
                points.clear()
                calibration = None
                preview = None
                message = "SPACE: freeze image | ESC: exit"
                cv2.destroyAllWindows()
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(WINDOW_NAME, on_mouse)

            # Calculate only after all four centers have been selected.
            elif key in (10, 13) and frozen is not None:
                try:
                    calibration = make_calibration(
                        frozen,
                        points,
                        args.camera,
                        args.zone,
                    )
                    preview = rectified_preview(frozen, calibration)
                    cv2.imshow("Rectified Workspace", preview)
                    message = "Check upright sheet and red centers. S: save"
                except ValueError as error:
                    calibration = None
                    preview = None
                    message = str(error)
                    print(error)

            # Save both an archived record and the new zone calibration.
            elif key == ord("s") and calibration is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                archive = (
                    PROJECT_ROOT
                    / "data"
                    / "workspace_calibrations"
                    / f"zone{args.zone}_{timestamp}"
                )
                archive.mkdir(parents=True, exist_ok=False)

                raw_path = archive / "camera.png"
                preview_path = archive / "rectified.png"

                if not cv2.imwrite(str(raw_path), frozen):
                    raise RuntimeError("Could not save the camera image.")

                if not cv2.imwrite(str(preview_path), preview):
                    raise RuntimeError("Could not save the rectified image.")

                calibration["created_at"] = datetime.now().isoformat()
                calibration["source_image"] = str(
                    raw_path.relative_to(PROJECT_ROOT)
                )

                content = json.dumps(calibration, indent=2) + "\n"
                (archive / "calibration.json").write_text(
                    content,
                    encoding="utf-8",
                )

                destination = (
                    PROJECT_ROOT
                    / "data"
                    / "calibration"
                    / f"zone{args.zone}_camera.json"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding="utf-8")

                print("\nCALIBRATION SAVED")
                print(destination)
                print("Images and archived calibration:")
                print(archive)
                print("The robot mapping must be updated before movement.")
                break

    finally:
        # Always release the camera, including after a capture failure.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()