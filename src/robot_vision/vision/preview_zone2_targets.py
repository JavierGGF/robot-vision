"""Preview locally anchored coupling targets; no robot access or file writes.

Only translations at the taught destination orientation are supported.
Pointer geometry, height effects and physical target accuracy remain unvalidated.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from robot_vision.vision.preview_zone2_horseshoe import analyze

ROOT = Path(__file__).resolve().parents[3]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def matching_calibration(a, b):
    return all(a.get(key) == b.get(key) for key in (
        'zone', 'camera_index', 'image_width_px', 'image_height_px',
        'reference_points_mm', 'pixel_to_mm_matrix'))


def pose_record(folder, prefix):
    paths = sorted(folder.glob(prefix + '*.json'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {prefix} record; found {len(paths)}')
    pose = np.asarray(read(paths[0])['pose_mm_deg'], dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('Invalid taught pose')
    return pose


def displacement_matrix(record, calibration):
    if record.get('zone') != 2 or not matching_calibration(record['camera_calibration'], calibration):
        raise ValueError('Marker record and reference calibration do not match')
    by_name = {item['marker']: item['pose_mm_deg'] for item in record['markers']}
    robot = np.asarray([by_name[key][:2] for key in ('TL', 'TR', 'BL')])
    sheet = np.asarray(calibration['reference_points_mm'])[[0, 1, 3]]
    matrix = np.column_stack((robot[1]-robot[0], robot[2]-robot[0])) @ np.linalg.inv(
        np.column_stack((sheet[1]-sheet[0], sheet[2]-sheet[0])))
    scales = np.linalg.norm(matrix, axis=0)
    cosine = np.dot(matrix[:, 0], matrix[:, 1]) / np.prod(scales)
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 10 or np.any(abs(scales-1) > .1) or abs(cosine) > .15:
        raise ValueError('Invalid marker geometry')
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--current', help='Another saved capture with matching calibration')
    parser.add_argument('--markers', help='Explicit robot_mapping.json; otherwise select newest matching Zone 2 record')
    args = parser.parse_args()
    reference = resolve(args.reference)
    current = resolve(args.current) if args.current else reference
    calibration = read(reference/'calibration.json')
    if not matching_calibration(calibration, read(current/'calibration.json')):
        raise ValueError('Current capture uses a different camera calibration')
    if args.markers:
        marker_path = resolve(args.markers)
    else:
        matches = [p for p in sorted((ROOT/'data/workspace_calibrations').glob('robot1_zone2_*/robot_mapping.json'))
                   if matching_calibration(read(p).get('camera_calibration', {}), calibration)]
        if not matches:
            raise FileNotFoundError('No matching Zone 2 marker record')
        marker_path = matches[-1]
    matrix = displacement_matrix(read(marker_path), calibration)
    entry = pose_record(reference, 'taught_coupling_entry_')
    final = pose_record(reference, 'taught_coupling_final_')
    frames = [cv2.imread(str(folder/'camera.png')) for folder in (reference, current)]
    if any(frame is None for frame in frames):
        raise FileNotFoundError('Reference or current camera.png missing')
    _, _, ref_geometry = analyze(frames[0], calibration)
    preview, _, geometry = analyze(frames[1], calibration)
    delta_sheet = (np.asarray(geometry['mouth_center_px']) - ref_geometry['mouth_center_px']) / 4.
    delta_robot = matrix @ delta_sheet
    rotation = (geometry['direction_image_deg'] - ref_geometry['direction_image_deg'] + 180) % 360 - 180
    if abs(rotation) > 2 or np.linalg.norm(delta_sheet) > 30:
        raise ValueError(f'Local preview rejected: rotation={rotation:.2f} deg; shift={np.linalg.norm(delta_sheet):.2f} mm. Supports <=2 deg and <=30 mm only.')
    lines = ['ESTIMATED TARGETS - NO ROBOT CONNECTION',
             f'Shift robot XY: {delta_robot[0]:.2f}, {delta_robot[1]:.2f} mm',
             f'Opening rotation change: {rotation:.2f} deg (orientation kept)',
             'Pointer / height accuracy NOT physically validated']
    for label, pose in [('ENTRY', entry), ('FINAL', final)]:
        target = pose.copy()
        target[:2] += delta_robot
        lines.append(label + ' XYZ: ' + ', '.join(f'{v:.2f}' for v in target[:3]))
        print(label, target.tolist())
    print('MARKERS:', marker_path)
    print('Reference must correspond to the fixed part used when teaching the poses.')
    panel = np.zeros((190, 1000, 3), dtype=np.uint8)
    for row, line in enumerate(lines):
        cv2.putText(panel, line, (12, 26+29*row), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1)
    scale = min(1000/preview.shape[1], 510/preview.shape[0])
    image = cv2.resize(preview, None, fx=scale, fy=scale)
    canvas = np.zeros((190+image.shape[0], 1000, 3), dtype=np.uint8)
    canvas[:190] = panel
    left = (1000-image.shape[1])//2
    canvas[190:, left:left+image.shape[1]] = image
    window = 'Zone 2 coupling target preview'
    try:
        cv2.imshow(window, canvas)
        while True:
            if cv2.waitKey(30) & 0xFF in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
