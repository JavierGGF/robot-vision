"""
Planar calibration using four reference points.

Default reference area:
US Letter paper, portrait orientation
215.9 mm x 279.4 mm

Author: Javier G. Fontanet
"""

import cv2
import numpy as np


REFERENCE_WIDTH_MM = 215.9
REFERENCE_HEIGHT_MM = 279.4


def create_calibration(
    image_points,
    width_mm=REFERENCE_WIDTH_MM,
    height_mm=REFERENCE_HEIGHT_MM
):
    """
    Calculate the transformation from image pixels to millimeters.

    Select the reference points in this order:

    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left

    Default reference:
    US Letter paper in portrait orientation.

    Width  = 215.9 mm
    Height = 279.4 mm
    """

    image_points = np.asarray(
        image_points,
        dtype=np.float32
    )

    if image_points.shape != (4, 2):
        raise ValueError(
            "Exactly four points with X and Y are required."
        )

    if not np.isfinite(image_points).all():
        raise ValueError(
            "Point coordinates must be finite."
        )

    if (
        not np.isfinite(
            [width_mm, height_mm]
        ).all()
        or width_mm <= 0
        or height_mm <= 0
    ):
        raise ValueError(
            "Reference dimensions must be positive and finite."
        )

    contour = image_points.reshape(
        4,
        1,
        2
    )

    if not cv2.isContourConvex(contour):
        raise ValueError(
            "Select four distinct corners "
            "in the specified order."
        )

    reference_points = np.array(
        [
            [0, 0],
            [width_mm, 0],
            [width_mm, height_mm],
            [0, height_mm]
        ],
        dtype=np.float32
    )

    return cv2.getPerspectiveTransform(
        image_points,
        reference_points
    )


def pixel_to_mm(x, y, matrix):
    """
    Convert an image point to coordinates
    on the reference plane.
    """

    point = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    result = cv2.perspectiveTransform(
        point,
        matrix
    )

    return (
        float(result[0, 0, 0]),
        float(result[0, 0, 1])
    )