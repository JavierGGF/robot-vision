"""Planar opening-to-TCP transforms. No camera or robot access."""
import math
import numpy as np


def matching_calibration(a, b):
    keys = ('zone', 'camera_index', 'image_width_px', 'image_height_px',
            'reference_points_mm', 'pixel_to_mm_matrix')
    return all(a.get(key) == b.get(key) for key in keys)


def displacement_matrix(record, calibration):
    if record.get('zone') != 2 or not matching_calibration(record.get('camera_calibration', {}), calibration):
        raise ValueError('Marker record and reference calibration do not match')
    by_name = {item['marker']: item['pose_mm_deg'] for item in record['markers']}
    robot = np.asarray([by_name[key][:2] for key in ('TL', 'TR', 'BL')], dtype=float)
    sheet = np.asarray(calibration['reference_points_mm'], dtype=float)[[0, 1, 3]]
    if not np.isfinite(robot).all() or not np.isfinite(sheet).all():
        raise ValueError('Invalid marker coordinates')
    matrix = np.column_stack((robot[1]-robot[0], robot[2]-robot[0])) @ np.linalg.inv(
        np.column_stack((sheet[1]-sheet[0], sheet[2]-sheet[0])))
    scales = np.linalg.norm(matrix, axis=0)
    cosine = np.dot(matrix[:, 0], matrix[:, 1]) / np.prod(scales)
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 10 or np.any(abs(scales-1) > .1) or abs(cosine) > .15:
        raise ValueError('Invalid marker geometry')
    return matrix


def compensate(pose, matrix, origin, q0, q1, angle, yaw_change):
    """Rotate a taught TCP about the reference opening, then translate it."""
    pose = np.asarray(pose, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    origin, q0, q1 = [np.asarray(v, dtype=float) for v in (origin, q0, q1)]
    if pose.shape != (6,) or matrix.shape != (2, 2) or any(v.shape != (2,) for v in (origin, q0, q1)):
        raise ValueError('Invalid transform dimensions')
    if not all(np.isfinite(v).all() for v in (pose, matrix, origin, q0, q1)) or not all(math.isfinite(v) for v in (angle, yaw_change)):
        raise ValueError('Non-finite transform input')
    theta = math.radians(angle)
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    target = pose.copy()
    pivot = origin + matrix @ q0
    new_pivot = origin + matrix @ q1
    target[:2] = new_pivot + matrix @ rotation @ np.linalg.solve(matrix, target[:2] - pivot)
    target[5] = (target[5] + yaw_change + 180) % 360 - 180
    return target.tolist()


def opening_targets(reference_geometry, current_geometry, marker_record, calibration, poses):
    """Return estimates and diagnostics; pointer-origin error is not corrected."""
    matrix = displacement_matrix(marker_record, calibration)
    tl = next(item['pose_mm_deg'] for item in marker_record['markers'] if item['marker'] == 'TL')
    origin = np.asarray(tl[:2]) - matrix @ np.asarray(calibration['reference_points_mm'][0])
    q0 = np.asarray(reference_geometry['mouth_center_px'], dtype=float) / 4
    q1 = np.asarray(current_geometry['mouth_center_px'], dtype=float) / 4
    if q0.shape != (2,) or q1.shape != (2,) or not np.isfinite([*q0, *q1]).all():
        raise ValueError('Invalid opening coordinates')
    for q in (q0, q1):
        if not 0 <= q[0] <= calibration['reference_width_mm'] or not 0 <= q[1] <= calibration['reference_height_mm']:
            raise ValueError('Opening outside the calibrated sheet')
    def direction(value):
        a = math.radians(value)
        v = matrix @ np.array([math.cos(a), math.sin(a)])
        return math.degrees(math.atan2(v[1], v[0]))
    angle = (current_geometry['direction_image_deg'] - reference_geometry['direction_image_deg'] + 180) % 360 - 180
    yaw_change = (direction(current_geometry['direction_image_deg']) - direction(reference_geometry['direction_image_deg']) + 180) % 360 - 180
    targets = {label: compensate(pose, matrix, origin, q0, q1, angle, yaw_change)
               for label, pose in poses.items()}
    return targets, {'rotation_sheet_deg': float(angle), 'yaw_change_deg': float(yaw_change),
                     'opening_shift_sheet_mm': float(np.linalg.norm(q1-q0)),
                     'reference_mouth_sheet_mm': q0.tolist(), 'current_mouth_sheet_mm': q1.tolist(),
                     'pointer_offset_corrected': False}
