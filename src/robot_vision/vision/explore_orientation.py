"""
Interactive orientation activity using a saved part silhouette.

Rotate the silhouette and inspect the horseshoe direction.
Set a practice zero and select a practice grasp point.

This activity does not open a camera, import a robot controller,
or modify calibration and grasp configuration files.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from robot_vision.vision.test_orientation import horseshoe_direction


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMAGE = (
    PROJECT_ROOT
    / "data/demo_trials/20261007_135341_548823/current_mask.png"
)

WINDOW = "Explore Part Orientation - Offline Practice"
SIZE = 440
CENTER = (SIZE // 2, SIZE // 2)
RADIUS = 185
HEADER = 130
FOOTER = 105


def load_silhouette(path):
    """Center a saved silhouette with enough room for a full rotation."""

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise FileNotFoundError(f"Could not read silhouette: {path}")

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("The image contains no part silhouette.")

    # Keep the largest external contour, matching the orientation method.
    contour = max(contours, key=cv2.contourArea)
    isolated = np.zeros_like(mask)
    cv2.drawContours(isolated, [contour], -1, 255, cv2.FILLED)

    x, y, width, height = cv2.boundingRect(contour)
    crop = isolated[y:y + height, x:x + width]

    scale = min(240.0 / width, 240.0 / height)
    crop = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )

    # Padding prevents the silhouette from being clipped when rotated.
    canvas = np.zeros((SIZE, SIZE), dtype=np.uint8)
    height, width = crop.shape
    top = (SIZE - height) // 2
    left = (SIZE - width) // 2
    canvas[top:top + height, left:left + width] = crop
    return canvas


def point_on_circle(angle_deg, radius):
    """Use mathematical angles: positive rotation is counterclockwise."""

    angle = math.radians(angle_deg)
    return (
        round(CENTER[0] + radius * math.cos(angle)),
        round(CENTER[1] - radius * math.sin(angle)),
    )


def put_text(image, text, position, color=(225, 225, 225), scale=0.48):
    """Draw consistent labels on the activity screen."""

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_protractor(image, zero_angle):
    """Draw the overlay only after silhouette analysis."""

    cv2.circle(image, CENTER, RADIUS, (110, 110, 110), 1)

    for angle in range(0, 360, 15):
        absolute = zero_angle + angle
        inner_radius = RADIUS - (12 if angle % 30 == 0 else 6)
        cv2.line(
            image,
            point_on_circle(absolute, inner_radius),
            point_on_circle(absolute, RADIUS),
            (150, 150, 150),
            1,
        )

        if angle % 30 == 0:
            x, y = point_on_circle(absolute, RADIUS + 19)
            label = str(angle)
            width = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )[0][0]
            put_text(image, label, (x - width // 2, y + 4), scale=0.4)

    # The zero line belongs only to this practice activity.
    cv2.line(
        image,
        CENTER,
        point_on_circle(zero_angle, RADIUS),
        (190, 140, 40),
        1,
    )


def main():
    """Run a local image exercise without hardware access."""

    parser = argparse.ArgumentParser(
        description="Explore horseshoe orientation using a saved silhouette."
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
        help="Path to a saved binary part silhouette.",
    )
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    base = load_silhouette(path)
    reference_geometry = horseshoe_direction(base)

    # Convert downward-positive image angles to counterclockwise angles.
    initial_angle = (
        -reference_geometry["direction_image_deg"]
    ) % 360.0
    zero_angle = initial_angle

    state = {
        "matrix": cv2.getRotationMatrix2D(CENTER, 0, 1.0),
        "grasp": None,
    }

    def select_grasp(event, x, y, flags, parameter):
        """Store a clicked practice point in the original image frame."""

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        local_y = y - HEADER
        if not (0 <= x < SIZE and 0 <= local_y < SIZE):
            return

        inverse = cv2.invertAffineTransform(state["matrix"])
        original = inverse @ np.array([x, local_y, 1.0])

        if 0 <= original[0] < SIZE and 0 <= original[1] < SIZE:
            state["grasp"] = original

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Rotate CCW", WINDOW, 0, 359, lambda value: None)
    cv2.setMouseCallback(WINDOW, select_grasp)

    print("OFFLINE PRACTICE - no camera or robot connection.")
    print("Move the rotation slider or press A / D.")
    print("Click to select a practice grasp point.")
    print("Z: set practice zero. R: reset. Q or ESC: exit.")
    print("Practice changes are not saved to robot configuration.")

    try:
        while True:
            rotation = cv2.getTrackbarPos("Rotate CCW", WINDOW)
            matrix = cv2.getRotationMatrix2D(CENTER, rotation, 1.0)
            state["matrix"] = matrix

            # Analyze only the binary image, never the graphical overlay.
            rotated = cv2.warpAffine(
                base,
                matrix,
                (SIZE, SIZE),
                flags=cv2.INTER_NEAREST,
                borderValue=0,
            )

            panel = np.full((SIZE, SIZE, 3), 24, dtype=np.uint8)
            panel[rotated > 0] = (200, 205, 210)

            contours, _ = cv2.findContours(
                rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(panel, contours, -1, (70, 220, 90), 2)

            current_angle = None
            message = "Orientation unavailable"

            try:
                geometry = horseshoe_direction(rotated)
                current_angle = (
                    -geometry["direction_image_deg"]
                ) % 360.0
                relative_angle = (current_angle - zero_angle) % 360.0
                message = f"Detected from practice zero: {relative_angle:6.1f} deg"

                # Draw the identified opening and its direction.
                start = tuple(np.rint(geometry["mouth_start_px"]).astype(int))
                end = tuple(np.rint(geometry["mouth_end_px"]).astype(int))
                bottom = tuple(np.rint(geometry["bottom_px"]).astype(int))
                mouth = tuple(np.rint(geometry["mouth_center_px"]).astype(int))
                cv2.line(panel, start, end, (0, 220, 255), 2)
                cv2.arrowedLine(
                    panel, bottom, mouth, (70, 70, 255), 2, tipLength=0.25
                )

            except ValueError as error:
                message = str(error)

            # A centroid marker makes the detected part center visible.
            moments = cv2.moments(rotated)
            if moments["m00"]:
                centroid = (
                    round(moments["m10"] / moments["m00"]),
                    round(moments["m01"] / moments["m00"]),
                )
                cv2.drawMarker(
                    panel, centroid, (255, 180, 60),
                    cv2.MARKER_CROSS, 14, 1,
                )

            draw_protractor(panel, zero_angle)

            if current_angle is not None:
                cv2.arrowedLine(
                    panel,
                    CENTER,
                    point_on_circle(current_angle, RADIUS - 20),
                    (70, 70, 255),
                    2,
                    tipLength=0.08,
                )

            # Rotate the selected point with the image for practice.
            # This is not a robot target or a metric grasp calibration.
            if state["grasp"] is not None:
                point = matrix @ np.array([*state["grasp"], 1.0])
                point = tuple(np.rint(point).astype(int))
                cv2.drawMarker(
                    panel, point, (230, 70, 230),
                    cv2.MARKER_TILTED_CROSS, 18, 2,
                )

            screen = np.full(
                (HEADER + SIZE + FOOTER, SIZE, 3), 24, dtype=np.uint8
            )
            screen[HEADER:HEADER + SIZE] = panel

            put_text(screen, "EXPLORE PART ORIENTATION", (15, 27), scale=0.65)
            put_text(screen, "OFFLINE PRACTICE - NO ROBOT", (15, 52), (80, 220, 255))
            put_text(screen, message, (15, 79))
            put_text(screen, f"Applied image rotation: {rotation} deg CCW", (15, 103))
            put_text(screen, "Red arrow: horseshoe opening direction", (15, 123), scale=0.43)

            bottom_y = HEADER + SIZE
            put_text(screen, "Slider / A D: rotate    Z: set practice zero", (12, bottom_y + 22))
            put_text(screen, "Click: practice grasp point (magenta)", (12, bottom_y + 44))
            put_text(screen, "R: reset    Q / ESC: exit", (12, bottom_y + 66))
            put_text(screen, "Robot calibration is unchanged.", (12, bottom_y + 90))

            cv2.imshow(WINDOW, screen)
            key = cv2.waitKey(30) & 0xFF

            if key in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord("z") and current_angle is not None:
                zero_angle = current_angle
            elif key == ord("r"):
                zero_angle = initial_angle
                state["grasp"] = None
                cv2.setTrackbarPos("Rotate CCW", WINDOW, 0)
            elif key == ord("a"):
                cv2.setTrackbarPos("Rotate CCW", WINDOW, (rotation + 5) % 360)
            elif key == ord("d"):
                cv2.setTrackbarPos("Rotate CCW", WINDOW, (rotation - 5) % 360)

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()