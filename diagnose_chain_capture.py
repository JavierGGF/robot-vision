"""Export portable vision diagnostics from a saved capture. No hardware access."""
import argparse
import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np

from robot_vision.vision import detection
from robot_vision.vision.test_orientation import (
    horseshoe_direction, MIN_DEPTH_RATIO, MAX_SECOND_DEPTH_RATIO,
)


def encode(image, extension, params=None):
    ok, data = cv2.imencode(extension, image, params or [])
    if not ok:
        raise RuntimeError('Image encoding failed')
    return base64.b64encode(data).decode('ascii')


def diagnose(directory):
    directory = Path(directory)
    path = directory / 'vision_diagnostics.json'
    if path.exists():
        raise FileExistsError(f'Diagnostic already exists; original preserved: {path}')
    calibration = json.loads((directory / 'calibration.json').read_text())
    frame = cv2.imread(str(directory / 'camera.png'))
    if frame is None:
        raise FileNotFoundError(directory / 'camera.png')
    if frame.shape[:2] != (calibration['image_height_px'], calibration['image_width_px']):
        raise ValueError('Saved frame resolution differs from calibration')
    matrix = np.diag([4., 4., 1.]) @ np.asarray(calibration['pixel_to_mm_matrix'])
    raw_rectified = cv2.warpPerspective(frame, matrix, (
        round(calibration['reference_width_mm'] * 4),
        round(calibration['reference_height_mm'] * 4)))
    gray = cv2.cvtColor(raw_rectified, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, detection.DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    record = {'schema_version': 1, 'source_capture': str(directory),
              'robot_connected': False, 'camera_opened': False,
              'calibration': calibration, 'dark_threshold': detection.DARK_THRESHOLD,
              'min_depth_ratio': MIN_DEPTH_RATIO,
              'max_second_depth_ratio': MAX_SECOND_DEPTH_RATIO}
    try:
        result, _, actual_mask = detection.process_frame(frame, calibration)
        record['detection'] = result
        record['independent_segmentation_matches'] = bool(np.array_equal(mask, actual_mask))
        mask = actual_mask
    except ValueError as error:
        record['detection_error'] = str(error)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    summaries = []
    for rank, contour in enumerate(sorted(contours, key=cv2.contourArea, reverse=True)):
        area = cv2.contourArea(contour)
        if rank >= 20 and area < 160:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        summary = {'rank': rank, 'area_mm2': float(area / 16),
                   'bounding_box_px': [x, y, w, h],
                   'passes_detection_region_filter': bool(area >= 160 and x > 12 and y > 12
                       and x+w < mask.shape[1]-12 and y+h < mask.shape[0]-12)}
        if moments['m00']:
            summary['center_sheet_mm'] = [moments['m10']/moments['m00']/4,
                                          moments['m01']/moments['m00']/4]
        if len(contour) >= 4:
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull) if hull is not None and len(hull) >= 3 else None
            depths = sorted((float(row[3])/256 for row in np.asarray(defects).reshape(-1, 4)), reverse=True) if defects is not None else []
            summary['defect_depths_px'] = depths[:10]
            summary['largest_depth_ratio'] = depths[0]/math.hypot(w,h) if depths else 0.
            summary['second_to_first_depth_ratio'] = depths[1]/depths[0] if len(depths) > 1 and depths[0] else 0.
        summaries.append(summary)
    record['contours'] = summaries
    try:
        record['geometry'] = horseshoe_direction(mask)
    except ValueError as error:
        record['geometry_error'] = str(error)
    # Text transport permits inspection through connectors that reject binary PNG files.
    record['camera_jpeg_base64'] = encode(frame, '.jpg', [cv2.IMWRITE_JPEG_QUALITY, 82])
    record['rectified_gray_png_base64'] = encode(gray, '.png')
    record['mask_png_base64'] = encode(mask, '.png')
    path.write_text(json.dumps(record, indent=2) + '\n')
    print('SAVED:', path)
    print('GEOMETRY:', record.get('geometry_error', 'accepted'))
    for item in summaries[:5]:
        print('CONTOUR:', {key: item.get(key) for key in ('rank','area_mm2','bounding_box_px','largest_depth_ratio','passes_detection_region_filter')})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    args = parser.parse_args()
    diagnose(args.capture)


if __name__ == '__main__':
    main()
