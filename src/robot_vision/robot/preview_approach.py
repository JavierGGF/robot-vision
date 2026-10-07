"""
Preview an approximate robot approach using three taught markers.

The user selects the marker corners in a saved camera image,
then selects the desired approach point on the part.

A planar affine mapping converts image pixels to robot X,Y.
This preliminary test ignores tool-offset and orientation differences.

The program does not connect to or move the robot.
Existing calibration files are not modified.
"""

import json
from pathlib import Path

import cv2
import numpy as np


# Robot positions read while the same pointer touched each marker.
# The marker order must match the image-click order.
ROBOT_POINTS_XY_MM = np.array([
    [-27.293282, -424.635712],   # Upper-left marker.
    [-38.432098, -182.537369],   # Upper-right marker.
    [137.816635, -418.713806],   # Lower-left marker.
], dtype=np.float64)

# This height is a preview value, not an instruction to move.
APPROACH_Z_MM = 200.0

# Use the first marker's orientation as the provisional tool orientation.
TOOL_ORIENTATION_DEG = [176.814317, 0.011516, -2.116678]

CLICK_LABELS = [
    "1: TOP-LEFT corner of TOP-LEFT marker",
    "2: TOP-LEFT corner of TOP-RIGHT marker",
    "3: TOP-LEFT corner of BOTTOM-LEFT marker",
    "4: Desired approach point on the part",
]


def main():
    """Select image references and print the approximate robot target."""

    project_root = Path(__file__).resolve().parents[3]
    tests_root = project_root / "data" / "orientation_tests"

    # Select the most recent saved camera capture.
    captures = sorted(tests_root.glob("*/camera.png"))

    if not captures:
        raise FileNotFoundError("No saved orientation captures found.")

    image_path = captures[-1]
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError("Could not open the saved capture.")

    print(f"Using capture: {image_path}")
    print("Confirm the paper and part have not moved since this capture.")
    print("Click the four points in order.")
    print("R: reset selections. ENTER: calculate. ESC: cancel.")

    # Display at a convenient size while retaining original pixel coordinates.
    height, width = image.shape[:2]
    scale = min(1200.0 / width, 800.0 / height, 1.0)
    display_width = round(width * scale)
    display_height = round(height * scale)

    preview_base = cv2.resize(
        image, (display_width, display_height)
    )
    selected_points = []
    window = "Approximate Approach Preview"

    def on_click(event, x, y, flags, param):
        """Convert display clicks back to original camera coordinates."""

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(selected_points) >= 4:
            return

        selected_points.append([
            float(x * width / display_width),
            float(y * height / display_height),
        ])
        print(f"Selected point {len(selected_points)}.")

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_click)

    try:
        while True:
            preview = preview_base.copy()

            # Number the selected points so their order can be checked visually.
            for index, point in enumerate(selected_points):
                x = round(point[0] * display_width / width)
                y = round(point[1] * display_height / height)

                cv2.drawMarker(
                    preview,
                    (x, y),
                    (0, 0, 255),
                    cv2.MARKER_CROSS,
                    16,
                    2,
                )
                cv2.putText(
                    preview,
                    str(index + 1),
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            # Show the next required selection.
            instruction = (
                CLICK_LABELS[len(selected_points)]
                if len(selected_points) < 4
                else "ENTER: calculate | R: reset | ESC: cancel"
            )
            cv2.rectangle(
                preview, (0, 0), (display_width, 40), (0, 0, 0), -1
            )
            cv2.putText(
                preview,
                instruction,
                (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
            )

            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                print("Preview cancelled.")
                return

            if key in (ord("r"), ord("R")):
                selected_points.clear()

            if key in (10, 13) and len(selected_points) == 4:
                break

        points = np.asarray(selected_points, dtype=np.float64)
        top_left, top_right, bottom_left, target_pixel = points

        # Express the target as fractions along the two marker directions.
        image_axes = np.column_stack((
            top_right - top_left,
            bottom_left - top_left,
        ))

        if np.linalg.cond(image_axes) > 1000:
            raise ValueError(
                "Marker geometry is invalid. Repeat the selections."
            )

        u, v = np.linalg.solve(
            image_axes, target_pixel - top_left
        )

        # Keep this first preview inside the region bounded by the markers.
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError(
                "Target is outside the marker region. Check click order."
            )

        # Apply the same fractions to the taught robot positions.
        robot_origin = ROBOT_POINTS_XY_MM[0]
        robot_x_direction = ROBOT_POINTS_XY_MM[1] - robot_origin
        robot_y_direction = ROBOT_POINTS_XY_MM[2] - robot_origin

        target_xy = (
            robot_origin
            + u * robot_x_direction
            + v * robot_y_direction
        )

        record = {
            "status": "approximate_preview_only",
            "camera_image": str(image_path),
            "selected_image_points_px": points.tolist(),
            "robot_marker_points_xy_mm": ROBOT_POINTS_XY_MM.tolist(),
            "target_pose_mm_deg": [
                float(target_xy[0]),
                float(target_xy[1]),
                APPROACH_Z_MM,
                *TOOL_ORIENTATION_DEG,
            ],
            "limitations": [
                "Affine mapping approximates the camera perspective.",
                "Tool offset and marker orientation changes are not corrected.",
                "Target has not been checked for reachability or clearance.",
            ],
        }

        # Save a separate preview record, leaving existing calibrations intact.
        output_path = image_path.parent / "approximate_approach_preview.json"
        output_path.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

        print()
        print("APPROXIMATE TARGET — NO MOVEMENT")
        print(f"X: {target_xy[0]:.3f} mm")
        print(f"Y: {target_xy[1]:.3f} mm")
        print(f"Z: {APPROACH_Z_MM:.3f} mm")
        print(f"Roll / Pitch / Yaw: {TOOL_ORIENTATION_DEG}")
        print(f"Preview saved: {output_path}")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()