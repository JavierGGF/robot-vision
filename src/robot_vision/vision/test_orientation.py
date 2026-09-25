"""
Test 0-360 degree part orientation using the saved reference silhouette.

The program captures the same part in a new orientation, compares its
silhouette with the saved reference, and predicts the new grasp point.

Author: Javier G. Fontanet
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

from detection import load_calibration, process_frame


def center_object_mask(mask, center_x_px, center_y_px, template_size):
    """Extract the detected object and center it in a square image."""

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No contour found.")

    best_contour = None
    best_distance = None

    for contour in contours:

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        distance = (
            (cx - center_x_px) ** 2
            + (cy - center_y_px) ** 2
        )

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_contour = contour

    if best_contour is None:
        raise RuntimeError("Could not identify the part contour.")

    object_mask = np.zeros_like(mask)

    cv2.drawContours(
        object_mask,
        [best_contour],
        -1,
        255,
        cv2.FILLED
    )

    template_center = template_size // 2

    translation = np.array(
        [
            [1, 0, template_center - center_x_px],
            [0, 1, template_center - center_y_px]
        ],
        dtype=np.float32
    )

    centered = cv2.warpAffine(
        object_mask,
        translation,
        (template_size, template_size),
        flags=cv2.INTER_NEAREST
    )

    return centered


def similarity(mask_a, mask_b):
    """Intersection-over-union similarity."""

    a = mask_a > 0
    b = mask_b > 0

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return intersection / union


def find_rotation(reference_mask, current_mask):
    """Find the 0-360 degree rotation with the best silhouette match."""

    size = reference_mask.shape[0]
    center = (size // 2, size // 2)

    best_angle = 0
    best_score = -1

    # First search in 5 degree steps
    for angle in range(0, 360, 5):

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            reference_mask,
            matrix,
            (size, size),
            flags=cv2.INTER_NEAREST
        )

        score = similarity(
            rotated,
            current_mask
        )

        if score > best_score:
            best_score = score
            best_angle = angle

    # Refine around the best result in 1 degree steps
    refined_angle = best_angle
    refined_score = best_score

    for offset in range(-5, 6):

        angle = (best_angle + offset) % 360

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            reference_mask,
            matrix,
            (size, size),
            flags=cv2.INTER_NEAREST
        )

        score = similarity(
            rotated,
            current_mask
        )

        if score > refined_score:
            refined_score = score
            refined_angle = angle

    return float(refined_angle), float(refined_score)


def main():

    project_root = Path(__file__).resolve().parents[3]

    calibration = load_calibration()

    grasp_path = (
        project_root
        / "data"
        / "calibration"
        / "grasp_calibration.json"
    )

    grasp_config = json.loads(
        grasp_path.read_text(encoding="utf-8")
    )

    reference_path = (
        project_root
        / "data"
        / "calibration"
        / grasp_config["reference_silhouette"]
    )

    reference_mask = cv2.imread(
        str(reference_path),
        cv2.IMREAD_GRAYSCALE
    )

    if reference_mask is None:
        raise FileNotFoundError(
            "Reference silhouette was not found."
        )

    _, reference_mask = cv2.threshold(
        reference_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    template_size = grasp_config["silhouette_size_px"]
    pixels_per_mm = grasp_config["pixels_per_mm"]

    camera_index = calibration["camera_index"]

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        calibration["image_width_px"]
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        calibration["image_height_px"]
    )

    print()
    print("Place the rotated part inside the work area.")
    print("Press SPACE to test orientation.")
    print("Press ESC to cancel.")

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        cv2.imshow(
            "Orientation Test - SPACE: analyze | ESC: cancel",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == 32:

            try:

                result, rectified, mask = process_frame(
                    frame,
                    calibration
                )

                center_x_mm = result["reference_x_mm"]
                center_y_mm = result["reference_y_mm"]

                center_x_px = (
                    center_x_mm * pixels_per_mm
                )

                center_y_px = (
                    center_y_mm * pixels_per_mm
                )

                current_mask = center_object_mask(
                    mask,
                    center_x_px,
                    center_y_px,
                    template_size
                )

                angle_deg, score = find_rotation(
                    reference_mask,
                    current_mask
                )

                # ---------------------------------------------
                # Transform the taught grasp point
                # ---------------------------------------------

                grasp_dx_mm, grasp_dy_mm = (
                    grasp_config["grasp_offset_mm"]
                )

                theta = math.radians(angle_deg)

                # Same rotation convention used by OpenCV
                rotated_dx = (
                    math.cos(theta) * grasp_dx_mm
                    + math.sin(theta) * grasp_dy_mm
                )

                rotated_dy = (
                    -math.sin(theta) * grasp_dx_mm
                    + math.cos(theta) * grasp_dy_mm
                )

                grasp_x_mm = (
                    center_x_mm + rotated_dx
                )

                grasp_y_mm = (
                    center_y_mm + rotated_dy
                )

                grasp_x_px = round(
                    grasp_x_mm * pixels_per_mm
                )

                grasp_y_px = round(
                    grasp_y_mm * pixels_per_mm
                )

                # ---------------------------------------------
                # Display result
                # ---------------------------------------------

                cv2.drawMarker(
                    rectified,
                    (grasp_x_px, grasp_y_px),
                    (255, 0, 255),
                    cv2.MARKER_TILTED_CROSS,
                    30,
                    3
                )

                cv2.putText(
                    rectified,
                    f"Rotation: {angle_deg:.1f} deg",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    rectified,
                    f"Match: {score:.2f}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                print()
                print("==============================")
                print(" ORIENTATION RESULT")
                print("==============================")
                print(f"Rotation: {angle_deg:.1f} deg")
                print(f"Match score: {score:.3f}")
                print()
                print(
                    f"New grasp point: "
                    f"X={grasp_x_mm:.2f} mm, "
                    f"Y={grasp_y_mm:.2f} mm"
                )

                cv2.imshow(
                    "Orientation Result",
                    rectified
                )

                cv2.imshow(
                    "Current Silhouette",
                    current_mask
                )

                cv2.waitKey(0)
                break

            except Exception as error:

                print(f"ERROR: {error}")
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()