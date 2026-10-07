"""
Register the saved pick reference in printed-sheet coordinates.

The original image and taught robot pose remain unchanged.
The selected fiducial centers provide a common coordinate system
for the old reference and the new camera view.

This program does not connect to the camera or the robot.
"""

import json
from datetime import datetime
from pathlib import Path

import cv2

from robot_vision.vision.calibrate_zone import (
    LABELS,
    make_calibration,
    rectified_preview,
)


# Locate the project independently of the terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Keep image coordinates separate from the display header.
HEADER_HEIGHT = 100
WINDOW_NAME = "Saved Pick Reference"


def project_path(value):
    """Resolve a path stored in the project configuration."""

    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    """Select the four physical fiducial centers in the saved image."""

    # Use the reference already linked to the taught robot grasp.
    profile_path = PROJECT_ROOT / "config" / "robot1_pick.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    reference_dir = project_path(
        profile["local_reference"]["capture_directory"]
    )
    image_path = reference_dir / "camera.png"
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise FileNotFoundError(f"Could not read {image_path}")

    # Read the new calibration only to retain the camera and zone identity.
    current_path = (
        PROJECT_ROOT / "data" / "calibration" / "zone1_camera.json"
    )
    current = json.loads(current_path.read_text(encoding="utf-8"))

    height, width = frame.shape[:2]
    scale = min(1200 / width, 675 / height, 1.0)

    points = []
    calibration = None
    preview = None
    message = "Click physical TL center in the saved image."

    def on_mouse(event, x, y, flags, parameter):
        """Record original image coordinates from display clicks."""

        nonlocal calibration, preview, message

        # Removing a point also invalidates the previous preview.
        if event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
            calibration = None
            preview = None
            message = f"Click physical {LABELS[len(points)]} center."
            return

        if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
            return

        image_x = x / scale
        image_y = (y - HEADER_HEIGHT) / scale

        if not (0 <= image_x < width and 0 <= image_y < height):
            return

        points.append([image_x, image_y])
        calibration = None
        preview = None

        if len(points) == 4:
            message = "ENTER: inspect | Right-click: undo"
        else:
            message = f"Click physical {LABELS[len(points)]} center."

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("This is the saved image, not the live camera.")
    print("Use the same physical sheet identities as the new calibration.")
    print("Click TL, TR, BR, BL centers.")
    print("ENTER: inspect | S: save | Right-click: undo | ESC: exit")

    try:
        while True:
            # Draw labels without modifying the original reference image.
            view = cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
            )

            for label, point in zip(LABELS, points):
                center = (
                    round(point[0] * scale),
                    round(point[1] * scale),
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
                "SAVED IMAGE: identify the same physical sheet corners.",
                message,
                "ENTER: inspect | S: save | Right-click: undo | ESC: exit",
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

            # Reuse the same sheet geometry as the current calibration.
            if key in (10, 13):
                try:
                    calibration = make_calibration(
                        frame,
                        points,
                        current["camera_index"],
                        1,
                    )
                    preview = rectified_preview(frame, calibration)
                    cv2.imshow("Rectified Saved Reference", preview)
                    message = "Check sheet orientation and centers. S: save"

                except ValueError as error:
                    calibration = None
                    preview = None
                    message = str(error)
                    print(error)

            elif key == ord("s") and calibration is not None:
                # Save a separate record; preserve the original reference.
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                archive = (
                    PROJECT_ROOT
                    / "data"
                    / "workspace_calibrations"
                    / f"zone1_reference_{timestamp}"
                )
                archive.mkdir(parents=True, exist_ok=False)

                if not cv2.imwrite(
                    str(archive / "rectified.png"),
                    preview,
                ):
                    raise RuntimeError("Could not save reference preview.")

                calibration["created_at"] = datetime.now().isoformat()
                calibration["source_image"] = str(
                    image_path.relative_to(PROJECT_ROOT)
                )
                calibration["reference_capture"] = str(
                    reference_dir.relative_to(PROJECT_ROOT)
                )
                calibration["purpose"] = "preserved_pick_reference"

                content = json.dumps(calibration, indent=2) + "\n"

                (archive / "calibration.json").write_text(
                    content,
                    encoding="utf-8",
                )

                destination = (
                    PROJECT_ROOT
                    / "data"
                    / "calibration"
                    / "zone1_reference_camera.json"
                )
                destination.write_text(content, encoding="utf-8")

                print("\nSAVED REFERENCE REGISTERED")
                print(destination)
                print("The original taught pose and image are preserved.")
                break

    finally:
        # Close only the preview windows; no hardware was connected.
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()