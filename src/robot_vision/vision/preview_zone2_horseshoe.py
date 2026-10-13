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


def analyze(frame, calibration):
    """Measure one isolated part; reject unsuitable opening geometry."""
    detection, rectified, mask = process_frame(frame, calibration)
    geometry = horseshoe_direction(mask)
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
        'ZONE 2 - ONE ISOLATED PART',
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
