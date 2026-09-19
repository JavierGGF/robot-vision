"""
Load and display a test image.

Author: Javier G. Fontanet
"""

from pathlib import Path

import cv2


def main():
    project_root = Path(__file__).resolve().parents[3]
    image_path = project_root / "data" / "examples" / "part_01.jpg"

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    print(f"Image loaded: {width} x {height} pixels")
        # Convert to grayscale and reduce noise.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Separate the dark object from the light background.
    _, mask = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    cv2.namedWindow("Object mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Object mask", mask)
        # Find the outlines of the white regions.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("No object detected.")
    else:
        # For this single-object test, select the largest region.
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)

        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]

            print(f"Object center: X={cx:.1f}, Y={cy:.1f} pixels")

            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            cv2.drawMarker(
                image,
                (round(cx), round(cy)),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                25,
                2
            )
    cv2.namedWindow("Test image", cv2.WINDOW_NORMAL)
    cv2.imshow("Test image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()