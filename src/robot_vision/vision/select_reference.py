"""
Capture an image from the webcam, select the four corners
of a US Letter reference sheet, and save the planar calibration.

Reference sheet:
US Letter, portrait orientation
215.9 mm x 279.4 mm

Author: Javier G. Fontanet
"""

import json
from pathlib import Path

import cv2

from calibration import (
    create_calibration,
    REFERENCE_WIDTH_MM,
    REFERENCE_HEIGHT_MM
)


def main():

    project_root = Path(__file__).resolve().parents[3]

    # ---------------------------------------------------------
    # 1. Open USB camera
    # ---------------------------------------------------------

    camera_index = 0

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print()
    print("======================================")
    print("       CAMERA CALIBRATION")
    print("======================================")
    print()
    print("Place a US Letter sheet in the vision area.")
    print("Use portrait orientation.")
    print()
    print("Sheet dimensions:")
    print(f"Width  = {REFERENCE_WIDTH_MM} mm")
    print(f"Height = {REFERENCE_HEIGHT_MM} mm")
    print()
    print("Make sure all four corners are visible.")
    print("Press SPACE to capture the calibration image.")
    print("Press ESC to cancel.")
    print()

    image = None

    window_camera = (
        "Camera Calibration - SPACE: capture | ESC: cancel"
    )

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        cv2.imshow(
            window_camera,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key == 32:
            image = frame.copy()
            break

    cap.release()
    cv2.destroyAllWindows()

    if image is None:
        print("Calibration cancelled.")
        return

    # ---------------------------------------------------------
    # 2. Save captured image
    # ---------------------------------------------------------

    examples_dir = (
        project_root
        / "data"
        / "examples"
    )

    examples_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = (
        examples_dir
        / "camera_calibration.jpg"
    )

    cv2.imwrite(
        str(image_path),
        image
    )

    print(
        f"Calibration image saved: {image_path}"
    )

    # ---------------------------------------------------------
    # 3. Prepare preview
    # ---------------------------------------------------------

    height, width = image.shape[:2]

    scale = min(
        1000 / width,
        750 / height,
        1.0
    )

    preview = cv2.resize(
        image,
        (
            round(width * scale),
            round(height * scale)
        )
    )

    preview_height, preview_width = (
        preview.shape[:2]
    )

    points = []

    window = (
        "Select Letter Sheet Corners - "
        "R: reset | Enter: save | Esc: cancel"
    )

    # ---------------------------------------------------------
    # 4. Mouse callback
    # ---------------------------------------------------------

    def on_click(event, x, y, flags, param):

        if (
            event == cv2.EVENT_LBUTTONDOWN
            and len(points) < 4
        ):

            original_x = (
                x * width / preview_width
            )

            original_y = (
                y * height / preview_height
            )

            points.append([
                original_x,
                original_y
            ])

            cv2.circle(
                preview,
                (x, y),
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                preview,
                str(len(points)),
                (x + 10, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    cv2.namedWindow(
        window,
        cv2.WINDOW_AUTOSIZE
    )

    cv2.setMouseCallback(
        window,
        on_click
    )

    print()
    print("Select the four paper corners in this order:")
    print()
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print()
    print("Press ENTER after selecting all four corners.")
    print("Press R if you need to start again.")
    print()

    # ---------------------------------------------------------
    # 5. Select corners and create calibration
    # ---------------------------------------------------------

    try:

        while True:

            cv2.imshow(
                window,
                preview
            )

            key = cv2.waitKey(20) & 0xFF

            if key == 27:

                print(
                    "Calibration cancelled."
                )

                break

            if key in (
                ord("r"),
                ord("R")
            ):

                points.clear()

                preview = cv2.resize(
                    image,
                    (
                        preview_width,
                        preview_height
                    )
                )

                print(
                    "Points reset."
                )

            if key in (10, 13):

                if len(points) != 4:

                    print(
                        "Select all four corners first."
                    )

                    continue

                try:

                    matrix = create_calibration(
                        points,
                        REFERENCE_WIDTH_MM,
                        REFERENCE_HEIGHT_MM
                    )

                except ValueError as error:

                    print(
                        f"{error} "
                        "Press R to try again."
                    )

                    continue

                # ---------------------------------------------
                # 6. Save calibration
                # ---------------------------------------------

                output_dir = (
                    project_root
                    / "data"
                    / "calibration"
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                calibration = {

                    "source_image":
                        image_path.name,

                    "camera_index":
                        camera_index,

                    "image_width_px":
                        width,

                    "image_height_px":
                        height,

                    "reference_width_mm":
                        REFERENCE_WIDTH_MM,

                    "reference_height_mm":
                        REFERENCE_HEIGHT_MM,

                    "image_points":
                        points,

                    "pixel_to_mm_matrix":
                        matrix.tolist()
                }

                output_path = (
                    output_dir
                    / "planar_calibration.json"
                )

                output_path.write_text(
                    json.dumps(
                        calibration,
                        indent=2
                    ),
                    encoding="utf-8"
                )

                print()
                print("======================================")
                print("       CALIBRATION SAVED")
                print("======================================")
                print()
                print(
                    f"Reference width: "
                    f"{REFERENCE_WIDTH_MM} mm"
                )
                print(
                    f"Reference height: "
                    f"{REFERENCE_HEIGHT_MM} mm"
                )
                print()
                print(
                    f"Saved to: {output_path}"
                )

                break

    finally:

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()