"""
Select four reference points and save the planar calibration.

Author: Javier G. Fontanet
"""

import json
from pathlib import Path

import cv2

from calibration import create_calibration


def main():
    project_root = Path(__file__).resolve().parents[3]
    image_path = project_root / "data" / "examples" / "part_02.jpg"

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    # Resize only the preview; calibration uses original image coordinates.
    height, width = image.shape[:2]
    scale = min(1000 / width, 750 / height, 1.0)
    preview = cv2.resize(
        image, (round(width * scale), round(height * scale))
    )
    preview_height, preview_width = preview.shape[:2]

    points = []
    window = "Select corners - R: reset | Enter: save | Esc: cancel"

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([
                x * width / preview_width,
                y * height / preview_height
            ])
            cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                preview, str(len(points)), (x + 8, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_click)

    print("Click: top-left, top-right, bottom-right, bottom-left.")
    print("Reference dimensions: width 100 mm, height 150 mm.")

    try:
        while True:
            cv2.imshow(window, preview)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:
                print("Cancelled.")
                break

            if key in (ord("r"), ord("R")):
                points.clear()
                preview = cv2.resize(
                    image, (preview_width, preview_height)
                )

            if key in (10, 13):
                if len(points) != 4:
                    print("Select all four points first.")
                    continue

                try:
                    matrix = create_calibration(points)
                except ValueError as error:
                    print(f"{error} Press R to try again.")
                    continue

                output = project_root / "data" / "calibration"
                output.mkdir(parents=True, exist_ok=True)

                calibration = {
                    "source_image": image_path.name,
                    "image_width_px": width,
                    "image_height_px": height,
                    "reference_width_mm": 100.0,
                    "reference_height_mm": 150.0,
                    "image_points": points,
                    "pixel_to_mm_matrix": matrix.tolist()
                }

                output_path = output / "planar_calibration.json"
                output_path.write_text(
                    json.dumps(calibration, indent=2),
                    encoding="utf-8"
                )
                print(f"Calibration saved: {output_path}")
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()