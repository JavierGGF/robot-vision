"""
Detect an object and measure its center on the reference plane.

Author: Javier G. Fontanet
"""

import json
from pathlib import Path

import cv2
import numpy as np


def detect_piece():
    """Return measurements and preview images without opening windows."""
    project_root = Path(__file__).resolve().parents[3]
    calibration_path = (
        project_root / "data" / "calibration"
        / "planar_calibration.json"
    )

    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )

    # Use the calibrated test image for this first integration.
    image_path = (
        project_root / "data" / "examples"
        / calibration["source_image"]
    )
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    height, width = image.shape[:2]

    if (
        width != calibration["image_width_px"]
        or height != calibration["image_height_px"]
    ):
        raise ValueError("Image dimensions do not match calibration.")

    matrix = np.asarray(
        calibration["pixel_to_mm_matrix"], dtype=np.float64
    )
    width_mm = calibration["reference_width_mm"]
    height_mm = calibration["reference_height_mm"]

    pixels_per_mm = 4.0
    scale_matrix = np.diag([pixels_per_mm, pixels_per_mm, 1.0])

    rectified = cv2.warpPerspective(
        image,
        scale_matrix @ matrix,
        (
            round(width_mm * pixels_per_mm),
            round(height_mm * pixels_per_mm)
        )
    )

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    margin = round(3 * pixels_per_mm)
    mask_height, mask_width = mask.shape
    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_mm2 = cv2.contourArea(contour) / pixels_per_mm**2

        if (
            area_mm2 >= 10
            and x > margin
            and y > margin
            and x + w < mask_width - margin
            and y + h < mask_height - margin
        ):
            candidates.append(contour)

    if not candidates:
        raise ValueError("No suitable object detected.")

    # This version assumes one main object.
    contour = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(contour)

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    result = {
        "status": "ok",
        "message": "Object detected",
        "reference_x_mm": float(cx / pixels_per_mm),
        "reference_y_mm": float(cy / pixels_per_mm)
    }

    cv2.drawContours(rectified, [contour], -1, (0, 255, 0), 2)
    cv2.drawMarker(
        rectified, (round(cx), round(cy)),
        (0, 0, 255), cv2.MARKER_CROSS, 25, 2
    )

    return result, rectified, mask


def main():
    result, rectified, mask = detect_piece()
    print(json.dumps(result, indent=2))

    try:
        cv2.imshow("Rectified reference plane", rectified)
        cv2.imshow("Object mask", mask)
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()