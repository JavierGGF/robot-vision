"""
Detect a dark object from a live webcam feed and measure its position
on the calibrated reference plane.

This version assumes:
- the part is dark / black
- the background is white or light gray

Author: Javier G. Fontanet
"""

import json
from pathlib import Path

import cv2
import numpy as np


# Pixels darker than this value are considered part of the object.
# This can be adjusted later for each station if needed.
DARK_THRESHOLD = 90


def load_calibration():
    project_root = Path(__file__).resolve().parents[3]

    calibration_path = (
        project_root
        / "data"
        / "calibration"
        / "planar_calibration.json"
    )

    return json.loads(
        calibration_path.read_text(encoding="utf-8")
    )


def process_frame(image, calibration=None):
    """
    Detect the main dark object in an existing camera frame.

    Returns:
        result, rectified, mask
    """

    if calibration is None:
        calibration = load_calibration()

    height, width = image.shape[:2]

    if (
        width != calibration["image_width_px"]
        or height != calibration["image_height_px"]
    ):
        raise ValueError(
            "Camera resolution does not match calibration."
        )

    # ---------------------------------------------------------
    # Rectify calibrated reference area
    # ---------------------------------------------------------

    matrix = np.asarray(
        calibration["pixel_to_mm_matrix"],
        dtype=np.float64
    )

    width_mm = calibration["reference_width_mm"]
    height_mm = calibration["reference_height_mm"]

    pixels_per_mm = 4.0

    scale_matrix = np.diag([
        pixels_per_mm,
        pixels_per_mm,
        1.0
    ])

    rectified = cv2.warpPerspective(
        image,
        scale_matrix @ matrix,
        (
            round(width_mm * pixels_per_mm),
            round(height_mm * pixels_per_mm)
        )
    )

    # ---------------------------------------------------------
    # Convert to grayscale
    # ---------------------------------------------------------

    gray = cv2.cvtColor(
        rectified,
        cv2.COLOR_BGR2GRAY
    )

    # Slight blur to reduce sensor noise
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ---------------------------------------------------------
    # Dark-object segmentation
    #
    # Dark pixels -> white in the mask
    # Gray/white background -> black in the mask
    # ---------------------------------------------------------

    _, mask = cv2.threshold(
        blurred,
        DARK_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    # ---------------------------------------------------------
    # Clean the mask
    # ---------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    # Fill small holes inside the part
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Remove isolated noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # ---------------------------------------------------------
    # Find contours
    # ---------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    margin = round(
        3 * pixels_per_mm
    )

    mask_height, mask_width = mask.shape

    candidates = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)

        area_mm2 = (
            cv2.contourArea(contour)
            / pixels_per_mm**2
        )

        if (
            area_mm2 >= 10
            and x > margin
            and y > margin
            and x + w < mask_width - margin
            and y + h < mask_height - margin
        ):
            candidates.append(contour)

    if not candidates:
        raise ValueError(
            "No suitable object detected."
        )

    # ---------------------------------------------------------
    # Select largest dark object
    # ---------------------------------------------------------

    contour = max(
        candidates,
        key=cv2.contourArea
    )

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        raise ValueError(
            "Could not calculate object center."
        )

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    reference_x_mm = float(
        cx / pixels_per_mm
    )

    reference_y_mm = float(
        cy / pixels_per_mm
    )

    result = {
        "status": "ok",
        "message": "Object detected",
        "reference_x_mm": reference_x_mm,
        "reference_y_mm": reference_y_mm
    }

    # ---------------------------------------------------------
    # Draw detection preview
    # ---------------------------------------------------------

    cv2.drawContours(
        rectified,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    cv2.drawMarker(
        rectified,
        (round(cx), round(cy)),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        25,
        2
    )

    cv2.putText(
        rectified,
        f"X: {reference_x_mm:.1f} mm",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        rectified,
        f"Y: {reference_y_mm:.1f} mm",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    return result, rectified, mask


def detect_piece():
    """
    Capture one webcam frame and detect the object.

    Kept for compatibility with app.py.
    """

    calibration = load_calibration()

    camera_index = calibration["camera_index"]

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        calibration["image_width_px"]
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        calibration["image_height_px"]
    )

    frame = None
    ret = False

    # Give the camera time to stabilize exposure
    for _ in range(30):
        ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        raise RuntimeError(
            "Could not capture image from camera."
        )

    return process_frame(
        frame,
        calibration
    )


def main():
    """
    Run continuous live vision.

    Press Q to quit.
    """

    calibration = load_calibration()

    camera_index = calibration["camera_index"]

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        calibration["image_width_px"]
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        calibration["image_height_px"]
    )

    print("Live vision started.")
    print(f"Dark threshold: {DARK_THRESHOLD}")
    print("Press Q to quit.")

    pixels_per_mm = 4.0

    matrix = np.asarray(
        calibration["pixel_to_mm_matrix"],
        dtype=np.float64
    )

    scale_matrix = np.diag([
        pixels_per_mm,
        pixels_per_mm,
        1.0
    ])

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            try:

                result, rectified, mask = process_frame(
                    frame,
                    calibration
                )

                cv2.imshow(
                    "Live Vision",
                    rectified
                )

                cv2.imshow(
                    "Object Mask",
                    mask
                )

            except ValueError:

                rectified = cv2.warpPerspective(
                    frame,
                    scale_matrix @ matrix,
                    (
                        round(
                            calibration["reference_width_mm"]
                            * pixels_per_mm
                        ),
                        round(
                            calibration["reference_height_mm"]
                            * pixels_per_mm
                        )
                    )
                )

                cv2.putText(
                    rectified,
                    "No object detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "Live Vision",
                    rectified
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()