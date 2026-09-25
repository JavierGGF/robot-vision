"""
Teach the desired grasp point and save the reference silhouette.

The student clicks once on the location where the gripper
should grasp the part.

The silhouette of the part is automatically saved so that
its orientation can later be recognized from 0 to 360 degrees.

Author: Javier G. Fontanet
"""

import json
from pathlib import Path

import cv2
import numpy as np

from detection import load_calibration, process_frame


def main():

    project_root = Path(__file__).resolve().parents[3]

    calibration = load_calibration()
    camera_index = calibration["camera_index"]

    pixels_per_mm = 4.0

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
    print("======================================")
    print("          GRASP TEACHING")
    print("======================================")
    print()
    print("1. Place the part inside the calibrated area.")
    print("2. Use the same part that the robot will pick.")
    print("3. Press SPACE when the part is clearly visible.")
    print("4. Press ESC to cancel.")
    print()

    result = None
    rectified = None
    mask = None

    # ---------------------------------------------------------
    # Capture reference part
    # ---------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        cv2.imshow(
            "Grasp Teaching - SPACE: capture | ESC: cancel",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            print("Teaching cancelled.")
            return

        if key == 32:

            try:

                result, rectified, mask = process_frame(
                    frame,
                    calibration
                )

                break

            except ValueError:

                print("No valid part detected.")
                print("Make sure the part is clearly visible.")

    cap.release()
    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # Detected center
    # ---------------------------------------------------------

    center_x_mm = result["reference_x_mm"]
    center_y_mm = result["reference_y_mm"]

    center_x_px = center_x_mm * pixels_per_mm
    center_y_px = center_y_mm * pixels_per_mm

    # ---------------------------------------------------------
    # Find the contour corresponding to the detected part
    # ---------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError(
            "Could not extract reference silhouette."
        )

    # Select the contour whose center is closest to
    # the detected object center.
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

        if (
            best_distance is None
            or distance < best_distance
        ):
            best_distance = distance
            best_contour = contour

    if best_contour is None:
        raise RuntimeError(
            "Could not identify reference part contour."
        )

    # ---------------------------------------------------------
    # Create clean object-only mask
    # ---------------------------------------------------------

    object_mask = np.zeros_like(mask)

    cv2.drawContours(
        object_mask,
        [best_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # ---------------------------------------------------------
    # Create centered square silhouette template
    #
    # A square image allows the reference part to be rotated
    # later without clipping its geometry.
    # ---------------------------------------------------------

    x, y, w, h = cv2.boundingRect(best_contour)

    template_size = int(
        np.ceil(
            np.sqrt(w**2 + h**2)
        )
    ) + 30

    # Make template size odd so it has an exact center pixel.
    if template_size % 2 == 0:
        template_size += 1

    template_center = template_size // 2

    translation_matrix = np.array(
        [
            [
                1,
                0,
                template_center - center_x_px
            ],
            [
                0,
                1,
                template_center - center_y_px
            ]
        ],
        dtype=np.float32
    )

    reference_template = cv2.warpAffine(
        object_mask,
        translation_matrix,
        (
            template_size,
            template_size
        ),
        flags=cv2.INTER_NEAREST
    )

    # ---------------------------------------------------------
    # Save reference silhouette
    # ---------------------------------------------------------

    output_dir = (
        project_root
        / "data"
        / "calibration"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    silhouette_path = (
        output_dir
        / "reference_part_mask.png"
    )

    cv2.imwrite(
        str(silhouette_path),
        reference_template
    )

    # ---------------------------------------------------------
    # Ask student for desired grasp point
    # ---------------------------------------------------------

    selected_point = []

    display = rectified.copy()

    window = (
        "Click grasp point - "
        "R: reset | Enter: save | Esc: cancel"
    )

    def redraw():

        nonlocal display

        display = rectified.copy()

        # Detected center
        cv2.drawMarker(
            display,
            (
                round(center_x_px),
                round(center_y_px)
            ),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            22,
            2
        )

        if selected_point:

            gx, gy = selected_point[0]

            # Student-selected grasp point
            cv2.drawMarker(
                display,
                (gx, gy),
                (255, 0, 255),
                cv2.MARKER_TILTED_CROSS,
                25,
                3
            )

            cv2.putText(
                display,
                "GRASP",
                (gx + 10, gy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                2
            )

            cv2.line(
                display,
                (
                    round(center_x_px),
                    round(center_y_px)
                ),
                (gx, gy),
                (255, 0, 255),
                2
            )

    def on_click(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:

            selected_point.clear()
            selected_point.append((x, y))

            redraw()

            print()
            print("Grasp point selected.")
            print("Press ENTER to save.")

    cv2.namedWindow(
        window,
        cv2.WINDOW_AUTOSIZE
    )

    cv2.setMouseCallback(
        window,
        on_click
    )

    redraw()

    print()
    print("The red cross is the detected part center.")
    print()
    print(
        "Click exactly where you want your gripper "
        "to grab the part."
    )
    print()
    print("Purple X = selected grasp point")
    print()
    print("Press ENTER to save.")
    print("Press R to select again.")
    print("Press ESC to cancel.")

    while True:

        cv2.imshow(
            window,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        if key == 27:

            print("Teaching cancelled.")
            break

        if key in (ord("r"), ord("R")):

            selected_point.clear()
            redraw()

            print("Selection reset.")

        if key in (10, 13):

            if not selected_point:

                print(
                    "Select a grasp point first."
                )
                continue

            grasp_x_px, grasp_y_px = (
                selected_point[0]
            )

            grasp_x_mm = (
                grasp_x_px / pixels_per_mm
            )

            grasp_y_mm = (
                grasp_y_px / pixels_per_mm
            )

            grasp_offset_x_mm = (
                grasp_x_mm - center_x_mm
            )

            grasp_offset_y_mm = (
                grasp_y_mm - center_y_mm
            )

            # Preserve PICK_Z if it was already calibrated.
            existing_pick_z = None

            config_path = (
                output_dir
                / "grasp_calibration.json"
            )

            if config_path.exists():

                try:

                    old_config = json.loads(
                        config_path.read_text(
                            encoding="utf-8"
                        )
                    )

                    existing_pick_z = (
                        old_config.get(
                            "pick_z_mm"
                        )
                    )

                except Exception:
                    pass

            grasp_config = {

                "reference_part_center_mm": [
                    center_x_mm,
                    center_y_mm
                ],

                "grasp_point_mm": [
                    grasp_x_mm,
                    grasp_y_mm
                ],

                "grasp_offset_mm": [
                    grasp_offset_x_mm,
                    grasp_offset_y_mm
                ],

                "reference_silhouette":
                    "reference_part_mask.png",

                "silhouette_size_px":
                    template_size,

                "pixels_per_mm":
                    pixels_per_mm,

                "part_height_mm":
                    25.0,

                "pick_z_mm":
                    existing_pick_z,

                "gripper_angle_offset_deg":
                    None
            }

            config_path.write_text(
                json.dumps(
                    grasp_config,
                    indent=2
                ),
                encoding="utf-8"
            )

            print()
            print("======================================")
            print("        GRASP TEACHING SAVED")
            print("======================================")
            print()
            print(
                f"Grasp offset X: "
                f"{grasp_offset_x_mm:.2f} mm"
            )
            print(
                f"Grasp offset Y: "
                f"{grasp_offset_y_mm:.2f} mm"
            )
            print()
            print(
                "Reference silhouette saved automatically."
            )
            print()

            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()