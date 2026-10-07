"""
Test part orientation using the horseshoe opening.

The large external concavity defines a directed geometric reference.
The relative rotation is used to rotate the student-taught grasp offset.

IoU and PCA are retained for comparison only.
Each capture is saved for later analysis and student documentation.

This program uses the existing planar and grasp calibrations.
It does not connect to or move the robot.
"""

import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_vision.vision.detection import load_calibration, process_frame


# Initial geometric acceptance limits.
# These values are relative to part size and must be checked with real captures.
MIN_DEPTH_RATIO = 0.10
MAX_SECOND_DEPTH_RATIO = 0.75

# Average points near the bottom instead of relying on one contour pixel.
BOTTOM_DEPTH_FRACTION = 0.90


def center_object_mask(mask, center_x_px, center_y_px, template_size):
    """Extract the detected part and center it without changing its scale."""

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_contour = None
    best_distance = float("inf")

    # Match the contour to the center returned by the existing detector.
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

        if distance < best_distance:
            best_distance = distance
            best_contour = contour

    if best_contour is None:
        raise ValueError("Could not identify the part contour.")

    # Reject captures that would lose geometry in the reference-sized template.
    points = best_contour[:, 0].astype(np.float64)
    shifted = points + np.array([
        template_size // 2 - center_x_px,
        template_size // 2 - center_y_px,
    ])

    if np.any(shifted < 1) or np.any(shifted > template_size - 2):
        raise ValueError("Part does not fit inside the silhouette template.")

    # Filling only the external contour removes internal circular holes.
    # The horseshoe opening remains because it belongs to the outer boundary.
    object_mask = np.zeros_like(mask)
    cv2.drawContours(
        object_mask, [best_contour], -1, 255, cv2.FILLED
    )

    template_center = template_size // 2
    translation = np.array([
        [1, 0, template_center - center_x_px],
        [0, 1, template_center - center_y_px],
    ], dtype=np.float32)

    return cv2.warpAffine(
        object_mask,
        translation,
        (template_size, template_size),
        flags=cv2.INTER_NEAREST,
    )


def similarity(mask_a, mask_b):
    """Calculate intersection-over-union for two binary masks."""

    a = mask_a > 0
    b = mask_b > 0
    union = np.logical_or(a, b).sum()

    if union == 0:
        return 0.0

    return float(np.logical_and(a, b).sum() / union)


def rotate_mask(mask, angle_deg):
    """Rotate a silhouette using the existing OpenCV angle convention."""

    height, width = mask.shape
    matrix = cv2.getRotationMatrix2D(
        (width // 2, height // 2), angle_deg, 1.0
    )

    return cv2.warpAffine(
        mask, matrix, (width, height), flags=cv2.INTER_NEAREST
    )


def find_rotation(reference_mask, current_mask):
    """Find the best IoU rotation; retained as a comparison method."""

    best_angle = 0.0
    best_score = -1.0

    # Search the full circle first.
    for angle in range(0, 360, 5):
        score = similarity(
            rotate_mask(reference_mask, angle), current_mask
        )

        if score > best_score:
            best_angle = float(angle)
            best_score = score

    # Refine around the best coarse result.
    coarse_angle = best_angle

    for offset in range(-5, 6):
        angle = (coarse_angle + offset) % 360.0
        score = similarity(
            rotate_mask(reference_mask, angle), current_mask
        )

        if score > best_score:
            best_angle = angle
            best_score = score

    return float(best_angle), float(best_score)


def pca_orientation(mask):
    """Calculate the principal image axis, with 180-degree ambiguity."""

    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys)).astype(np.float64)

    if len(points) < 2:
        raise ValueError("Not enough pixels for PCA orientation.")

    _, eigenvectors = cv2.PCACompute(
        points, mean=None, maxComponents=2
    )

    vx, vy = eigenvectors[0]

    # In image coordinates, positive y points downward.
    return float(math.degrees(math.atan2(vy, vx)) % 180.0)


def mask_diagnostics(mask):
    """Return measurements useful when comparing saved silhouettes."""

    binary = (mask > 0).astype(np.uint8)
    x, y, width, height = cv2.boundingRect(binary)
    moments = cv2.moments(binary)

    center = None
    if moments["m00"] != 0:
        center = [
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        ]

    return {
        "shape": list(mask.shape),
        "bounding_box_px": [x, y, width, height],
        "white_pixels": int(binary.sum()),
        "centroid_px": center,
    }


def horseshoe_direction(mask):
    """Identify the dominant external concavity and its opening direction."""

    # A dense contour allows averaging several pixels near the concavity floor.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        raise ValueError("No part contour found.")

    contour = max(contours, key=cv2.contourArea)

    if len(contour) < 4:
        raise ValueError("Insufficient contour geometry.")

    # Convexity defects measure recesses between the contour and its hull.
    # Internal holes are excluded by RETR_EXTERNAL.
    hull_indices = cv2.convexHull(contour, returnPoints=False)

    if hull_indices is None or len(hull_indices) < 3:
        raise ValueError("Could not calculate convex hull.")

    defects = cv2.convexityDefects(contour, hull_indices)

    if defects is None:
        raise ValueError("Horseshoe opening not found.")

    candidates = sorted(
        defects.reshape(-1, 4),
        key=lambda item: int(item[3]),
        reverse=True,
    )

    start_i, end_i, far_i, depth_fixed = map(int, candidates[0])

    # OpenCV stores defect depth with eight fractional bits.
    depth_px = depth_fixed / 256.0
    _, _, width, height = cv2.boundingRect(contour)
    diagonal_px = math.hypot(width, height)
    depth_ratio = depth_px / diagonal_px

    if depth_ratio < MIN_DEPTH_RATIO:
        raise ValueError("No sufficiently deep horseshoe opening.")

    second_depth = (
        int(candidates[1][3]) / 256.0
        if len(candidates) > 1 else 0.0
    )

    if second_depth > MAX_SECOND_DEPTH_RATIO * depth_px:
        raise ValueError("Multiple similar concavities: orientation unclear.")

    points = contour[:, 0].astype(np.float64)
    start = points[start_i]
    end = points[end_i]
    mouth = (start + end) / 2.0
    mouth_vector = end - start
    mouth_width = float(np.linalg.norm(mouth_vector))

    if mouth_width < 1.0:
        raise ValueError("Opening mouth is too narrow.")

    # Follow the contour segment that contains the selected defect.
    if start_i <= end_i:
        arc = points[start_i:end_i + 1]
    else:
        arc = np.concatenate((points[start_i:], points[:end_i + 1]))

    # Measure perpendicular distance from each arc point to the mouth line.
    relative = arc - start
    distances = np.abs(
        mouth_vector[0] * relative[:, 1]
        - mouth_vector[1] * relative[:, 0]
    ) / mouth_width

    # Averaging the deepest region reduces sensitivity to a single pixel.
    floor_points = arc[
        distances >= BOTTOM_DEPTH_FRACTION * distances.max()
    ]
    bottom = floor_points.mean(axis=0)
    direction = mouth - bottom

    if np.linalg.norm(direction) < 1.0:
        raise ValueError("Opening direction is undefined.")

    angle_deg = math.degrees(
        math.atan2(direction[1], direction[0])
    ) % 360.0

    # Lists and floats keep the result directly serializable to JSON.
    return {
        "direction_image_deg": float(angle_deg),
        "mouth_start_px": start.tolist(),
        "mouth_end_px": end.tolist(),
        "mouth_center_px": mouth.tolist(),
        "bottom_px": bottom.tolist(),
        "deepest_contour_point_px": points[far_i].tolist(),
        "depth_px": float(depth_px),
        "depth_ratio": float(depth_ratio),
        "second_depth_px": float(second_depth),
        "mouth_width_px": mouth_width,
    }


def geometry_preview(mask, geometry):
    """Draw the opening mouth and its directed geometric reference."""

    preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    def point(key):
        return tuple(np.rint(geometry[key]).astype(int))

    # Yellow joins the two sides of the opening.
    cv2.line(
        preview,
        point("mouth_start_px"),
        point("mouth_end_px"),
        (0, 255, 255),
        2,
    )

    # Red points from the concavity floor toward the opening.
    cv2.arrowedLine(
        preview,
        point("bottom_px"),
        point("mouth_center_px"),
        (0, 0, 255),
        2,
        tipLength=0.25,
    )

    return preview


def calculate_grasp(result, grasp_config, angle_deg):
    """Rotate the taught offset around the existing detected center."""

    dx, dy = grasp_config["grasp_offset_mm"]
    theta = math.radians(angle_deg)

    # Preserve the rotation convention already used by the project.
    rotated_dx = math.cos(theta) * dx + math.sin(theta) * dy
    rotated_dy = -math.sin(theta) * dx + math.cos(theta) * dy

    return (
        float(result["reference_x_mm"] + rotated_dx),
        float(result["reference_y_mm"] + rotated_dy),
    )


def put_label(image, text, y, color=(0, 255, 255)):
    """Draw a readable diagnostic label on the rectified image."""

    cv2.putText(
        image, text, (10, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
    )


def save_capture(project_root, images, record):
    """Save a unique test folder without replacing earlier captures."""

    capture_dir = (
        project_root
        / "data"
        / "orientation_tests"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    capture_dir.mkdir(parents=True, exist_ok=False)

    # PNG preserves the masks without JPEG compression artifacts.
    for filename, image in images.items():
        if not cv2.imwrite(str(capture_dir / filename), image):
            raise RuntimeError(f"Could not save {filename}")

    # Store calibration snapshots alongside the test for reproducibility.
    (capture_dir / "results.json").write_text(
        json.dumps(record, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    return capture_dir


def analyze_capture(
    frame, project_root, calibration, grasp_config, reference_mask
):
    """Analyze one capture, saving both successful and rejected tests."""

    images = {
        "camera.png": frame.copy(),
        "reference_mask.png": reference_mask,
    }
    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "status": "pending",
        "angle_convention": "OpenCV: positive counterclockwise in image",
        "planar_calibration": calibration,
        "grasp_calibration": grasp_config,
        "geometry_limits": {
            "minimum_depth_ratio": MIN_DEPTH_RATIO,
            "maximum_second_depth_ratio": MAX_SECOND_DEPTH_RATIO,
            "bottom_depth_fraction": BOTTOM_DEPTH_FRACTION,
        },
    }

    try:
        # Use the existing segmentation, rectification, and center calculation.
        result, rectified, mask = process_frame(frame, calibration)
        images["rectified_detection.png"] = rectified.copy()
        images["segmentation.png"] = mask
        record["detection"] = result

        pixels_per_mm = float(grasp_config["pixels_per_mm"])
        template_size = int(grasp_config["silhouette_size_px"])

        current_mask = center_object_mask(
            mask,
            result["reference_x_mm"] * pixels_per_mm,
            result["reference_y_mm"] * pixels_per_mm,
            template_size,
        )
        images["current_mask.png"] = current_mask

        # Keep silhouette diagnostics and both previous methods for comparison.
        record["reference_mask_diagnostics"] = mask_diagnostics(
            reference_mask
        )
        record["current_mask_diagnostics"] = mask_diagnostics(
            current_mask
        )

        iou_angle, iou_score = find_rotation(
            reference_mask, current_mask
        )
        reference_pca = pca_orientation(reference_mask)
        current_pca = pca_orientation(current_mask)

        # Retain the original PCA difference for comparison with earlier tests.
        # Also report the equivalent difference using the OpenCV convention.
        pca_original = (current_pca - reference_pca) % 180.0
        pca_opencv = (reference_pca - current_pca) % 180.0

        record.update({
            "iou_rotation_deg": iou_angle,
            "iou_score": iou_score,
            "reference_pca_image_deg": reference_pca,
            "current_pca_image_deg": current_pca,
            "pca_original_rotation_deg": pca_original,
            "pca_opencv_rotation_deg": pca_opencv,
        })

        # Extract the same physical opening from reference and current masks.
        reference_geometry = horseshoe_direction(reference_mask)
        record["reference_geometry"] = reference_geometry
        images["reference_geometry.png"] = geometry_preview(
            reference_mask, reference_geometry
        )

        current_geometry = horseshoe_direction(current_mask)
        record["current_geometry"] = current_geometry
        images["current_geometry.png"] = geometry_preview(
            current_mask, current_geometry
        )

        # Image atan2 increases clockwise because image y points downward.
        # Reverse the difference to obtain the existing OpenCV convention.
        angle_deg = (
            reference_geometry["direction_image_deg"]
            - current_geometry["direction_image_deg"]
        ) % 360.0

        grasp_x_mm, grasp_y_mm = calculate_grasp(
            result, grasp_config, angle_deg
        )

        # IoU at the geometric angle is diagnostic, not an acceptance score.
        geometric_iou = similarity(
            rotate_mask(reference_mask, angle_deg), current_mask
        )

        record.update({
            "status": "ok",
            "geometric_rotation_deg": float(angle_deg),
            "iou_at_geometric_angle": geometric_iou,
            "grasp_x_mm": grasp_x_mm,
            "grasp_y_mm": grasp_y_mm,
        })

        # Draw the predicted grasp using the same calibrated millimeter scale.
        grasp_pixel = (
            round(grasp_x_mm * pixels_per_mm),
            round(grasp_y_mm * pixels_per_mm),
        )
        cv2.drawMarker(
            rectified, grasp_pixel, (255, 0, 255),
            cv2.MARKER_TILTED_CROSS, 30, 3,
        )
        cv2.putText(
            rectified, "GRASP",
            (grasp_pixel[0] + 10, grasp_pixel[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
        )

        put_label(rectified, f"Geometry: {angle_deg:.1f} deg", 75)
        put_label(
            rectified,
            f"IoU: {iou_angle:.1f} deg / score {iou_score:.3f}",
            100,
        )
        put_label(
            rectified,
            f"PCA original: {pca_original:.1f} deg (mod 180)",
            125,
            (255, 255, 0),
        )
        put_label(
            rectified,
            f"IoU at geometry angle: {geometric_iou:.3f}",
            150,
        )
        images["result.png"] = rectified

        print()
        print("ORIENTATION RESULT")
        print(f"Geometry rotation: {angle_deg:.1f} deg")
        print(f"IoU rotation: {iou_angle:.1f} deg")
        print(f"IoU score: {iou_score:.3f}")
        print(f"PCA original rotation: {pca_original:.1f} deg")
        print(f"PCA OpenCV rotation: {pca_opencv:.1f} deg")
        print(
            f"Grasp: X={grasp_x_mm:.2f} mm, "
            f"Y={grasp_y_mm:.2f} mm"
        )

    except (ValueError, RuntimeError, cv2.error) as error:
        # Preserve rejected captures so segmentation and geometry can be reviewed.
        # Do not substitute an IoU angle when the geometric estimate fails.
        record["status"] = "rejected"
        record["error"] = str(error)
        print(f"Analysis rejected: {error}")

    # Save before displaying so closing a window does not lose the test.
    capture_dir = save_capture(project_root, images, record)
    print(f"Test saved: {capture_dir}")

    # Display available results, including partial results from rejected tests.
    for filename, window in (
        ("result.png", "Orientation Result"),
        ("current_mask.png", "Current Silhouette"),
        ("reference_geometry.png", "Reference Horseshoe"),
        ("current_geometry.png", "Current Horseshoe"),
    ):
        if filename in images:
            cv2.imshow(window, images[filename])

    return record


def main():
    """Load existing calibration and run repeated manual webcam tests."""

    project_root = Path(__file__).resolve().parents[3]
    calibration = load_calibration()

    grasp_path = (
        project_root / "data" / "calibration" / "grasp_calibration.json"
    )
    grasp_config = json.loads(grasp_path.read_text(encoding="utf-8"))

    reference_path = (
        project_root
        / "data"
        / "calibration"
        / grasp_config["reference_silhouette"]
    )
    reference_mask = cv2.imread(
        str(reference_path), cv2.IMREAD_GRAYSCALE
    )

    if reference_mask is None:
        raise FileNotFoundError("Reference silhouette was not found.")

    _, reference_mask = cv2.threshold(
        reference_mask, 127, 255, cv2.THRESH_BINARY
    )

    # Confirm template dimensions before opening the camera.
    template_size = int(grasp_config["silhouette_size_px"])

    if reference_mask.shape != (template_size, template_size):
        raise ValueError("Reference mask dimensions do not match calibration.")

    # The existing detector rectifies at 4 pixels per millimeter.
    if not math.isclose(float(grasp_config["pixels_per_mm"]), 4.0):
        raise ValueError("Grasp scale does not match the existing detector.")

    cap = cv2.VideoCapture(calibration["camera_index"])

    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")

        # Request the same resolution used by the existing planar calibration.
        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, calibration["image_width_px"]
        )
        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT, calibration["image_height_px"]
        )

        print()
        print("HORSESHOE ORIENTATION TEST")
        print("Place the part inside the calibrated area.")
        print("SPACE: analyze and save a capture.")
        print("ESC: exit.")
        print("Positive rotation is counterclockwise in the image.")
        print("The red arrow should point toward the horseshoe opening.")

        consecutive_read_failures = 0

        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 60:
                    raise RuntimeError("Could not read camera frames.")
                if cv2.waitKey(10) & 0xFF == 27:
                    break
                continue

            consecutive_read_failures = 0

            cv2.imshow(
                "Orientation Test - SPACE: analyze | ESC: exit", frame
            )
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key == 32:
                analyze_capture(
                    frame,
                    project_root,
                    calibration,
                    grasp_config,
                    reference_mask,
                )

                # Pause for inspection, then allow another physical rotation.
                print("Press any key to resume, or ESC to exit.")
                if cv2.waitKey(0) & 0xFF == 27:
                    break

    finally:
        # Always release the webcam, including after an analysis or save error.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()