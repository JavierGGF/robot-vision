"""
Estimate part orientation from the saved reference silhouette
and calculate the taught grasp point.

Author: Javier G. Fontanet
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np


def load_grasp_config():

    project_root = Path(__file__).resolve().parents[3]

    config_path = (
        project_root
        / "data"
        / "calibration"
        / "grasp_calibration.json"
    )

    return json.loads(
        config_path.read_text(encoding="utf-8")
    )


def load_reference_mask(grasp_config):

    project_root = Path(__file__).resolve().parents[3]

    mask_path = (
        project_root
        / "data"
        / "calibration"
        / grasp_config["reference_silhouette"]
    )

    reference_mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if reference_mask is None:
        raise FileNotFoundError(
            "Reference part silhouette was not found."
        )

    _, reference_mask = cv2.threshold(
        reference_mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return reference_mask


def center_object_mask(
    mask,
    center_x_px,
    center_y_px,
    template_size
):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            "No part contour found."
        )

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
        raise ValueError(
            "Could not identify the part contour."
        )

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

    centered = cv2.warpAffine(
        object_mask,
        translation,
        (
            template_size,
            template_size
        ),
        flags=cv2.INTER_NEAREST
    )

    return centered


def similarity(mask_a, mask_b):

    a = mask_a > 0
    b = mask_b > 0

    intersection = np.logical_and(
        a,
        b
    ).sum()

    union = np.logical_or(
        a,
        b
    ).sum()

    if union == 0:
        return 0.0

    return intersection / union


def find_rotation(
    reference_mask,
    current_mask
):

    size = reference_mask.shape[0]

    center = (
        size // 2,
        size // 2
    )

    best_angle = 0
    best_score = -1

    # Coarse search
    for angle in range(0, 360, 5):

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            reference_mask,
            matrix,
            (
                size,
                size
            ),
            flags=cv2.INTER_NEAREST
        )

        score = similarity(
            rotated,
            current_mask
        )

        if score > best_score:

            best_score = score
            best_angle = angle

    # Fine search
    refined_angle = best_angle
    refined_score = best_score

    for offset in range(-5, 6):

        angle = (
            best_angle + offset
        ) % 360

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            reference_mask,
            matrix,
            (
                size,
                size
            ),
            flags=cv2.INTER_NEAREST
        )

        score = similarity(
            rotated,
            current_mask
        )

        if score > refined_score:

            refined_score = score
            refined_angle = angle

    return (
        float(refined_angle),
        float(refined_score)
    )


def calculate_grasp(
    result,
    mask
):

    grasp_config = load_grasp_config()

    reference_mask = load_reference_mask(
        grasp_config
    )

    pixels_per_mm = (
        grasp_config["pixels_per_mm"]
    )

    template_size = (
        grasp_config["silhouette_size_px"]
    )

    center_x_mm = (
        result["reference_x_mm"]
    )

    center_y_mm = (
        result["reference_y_mm"]
    )

    center_x_px = (
        center_x_mm
        * pixels_per_mm
    )

    center_y_px = (
        center_y_mm
        * pixels_per_mm
    )

    current_mask = center_object_mask(
        mask,
        center_x_px,
        center_y_px,
        template_size
    )

    angle_deg, match_score = find_rotation(
        reference_mask,
        current_mask
    )

    grasp_dx_mm, grasp_dy_mm = (
        grasp_config["grasp_offset_mm"]
    )

    theta = math.radians(
        angle_deg
    )

    rotated_dx = (
        math.cos(theta)
        * grasp_dx_mm
        + math.sin(theta)
        * grasp_dy_mm
    )

    rotated_dy = (
        -math.sin(theta)
        * grasp_dx_mm
        + math.cos(theta)
        * grasp_dy_mm
    )

    grasp_x_mm = (
        center_x_mm
        + rotated_dx
    )

    grasp_y_mm = (
        center_y_mm
        + rotated_dy
    )

    return {
        "part_angle_deg":
            angle_deg,

        "orientation_match":
            match_score,

        "grasp_x_mm":
            grasp_x_mm,

        "grasp_y_mm":
            grasp_y_mm
    }