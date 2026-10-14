"""Visual horseshoe check in Zone 2 using the existing geometry functions.

This tool does not connect to the robot or modify calibration/profiles.
Opening coordinates are visual measurements, not a robot TCP target.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from robot_vision.vision.detection import process_frame
from robot_vision.vision.test_orientation import horseshoe_direction


ROOT = Path(__file__).resolve().parents[3]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path



def interior_parts(frame, calibration):
    """Return complete dark contours inside the fiducials, excluding markers.

    An empty result means no candidate under the current segmentation settings;
    it does not prove that the physical sheet is empty.
    """
    try:
        _, rectified, mask = process_frame(frame, calibration)
    except ValueError as error:
        if str(error) != "No suitable object detected.":
            raise
        return [], None, None
    points = np.asarray(calibration['reference_points_mm'], dtype=float) * 4
    # Use the calibrated marker-centre rectangle. Marker blobs cross its
    # boundaries and are excluded; a complete part near the boundary is valid.
    low = points.min(axis=0)
    high = points.max(axis=0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if (cv2.contourArea(contour) >= 160 and x > low[0] and y > low[1]
                and x + width < high[0] and y + height < high[1]):
            candidates.append(contour)
    return sorted(candidates, key=cv2.contourArea, reverse=True), rectified, mask


def analyze(frame, calibration):
    """Measure the dominant contour; reject ambiguous opening geometry."""
    candidates, rectified, mask = interior_parts(frame, calibration)
    if not candidates:
        raise ValueError("No part detected inside Zone 2 fiducials.")
    if len(candidates) != 1:
        raise ValueError("Multiple separate objects in Zone 2; free opening is ambiguous.")
    selected = np.zeros_like(mask)
    cv2.drawContours(selected, candidates, -1, 255, -1)
    geometry = horseshoe_direction(selected)
    moments = cv2.moments(candidates[0])
    detection = {'status': 'ok', 'message': 'Interior part detected',
                 'reference_x_mm': float(moments['m10'] / moments['m00'] / 4),
                 'reference_y_mm': float(moments['m01'] / moments['m00'] / 4)}
    mouth = np.asarray(geometry['mouth_center_px'])
    bottom = np.asarray(geometry['bottom_px'])
    for first, second, color in (
        (geometry['mouth_start_px'], geometry['mouth_end_px'], (0, 255, 255)),
        (bottom, mouth, (0, 0, 255)),
    ):
        a, b = tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int))
        cv2.line(rectified, a, b, color, 3)
    cv2.drawMarker(rectified, tuple(np.rint(mouth).astype(int)),
                   (255, 0, 255), cv2.MARKER_CROSS, 24, 3)
    labels = [
        'ZONE 2 - OPENING CANDIDATE',
        f"Opening direction: {geometry['direction_image_deg']:.1f} deg",
        f'Opening mouth: X={mouth[0]/4:.2f} Y={mouth[1]/4:.2f} mm',
        'Sheet coordinates - NOT a robot target',
        'Q / ESC: close',
    ]
    header = np.zeros((155, rectified.shape[1], 3), dtype=np.uint8)
    for index, label in enumerate(labels):
        cv2.putText(header, label, (12, 26 + 29*index),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    preview = np.vstack((header, rectified))
    scale = min(900/preview.shape[1], 720/preview.shape[0], 1.)
    preview = cv2.resize(preview, None, fx=scale, fy=scale)
    return preview, detection, geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True,
                        help='Saved directory containing camera.png and calibration.json')
    args = parser.parse_args()
    directory = project_path(args.capture)
    calibration = json.loads((directory/'calibration.json').read_text())
    if calibration.get('zone') != 2:
        raise ValueError('Zone 2 calibration is required')
    frame = cv2.imread(str(directory/'camera.png'))
    if frame is None:
        raise FileNotFoundError(directory/'camera.png')
    preview, detection, geometry = analyze(frame, calibration)
    print('VISUAL CHECK ONLY - no robot connection')
    print(f"Opening direction in rectified sheet: {geometry['direction_image_deg']:.2f} deg")
    print('Opening mouth X,Y [mm]:', (np.asarray(geometry['mouth_center_px'])/4).tolist())
    print('Yellow: opening mouth | Red: outward direction | Magenta: mouth center')
    window = 'Zone 2 horseshoe - visual check only'
    try:
        cv2.imshow(window, preview)
        while True:
            if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

