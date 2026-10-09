"""Metric, offline link-coupling editor. No robot or camera connection.

Select receiving-hole, mating-hole and grasp centers on ONE saved link.
The next link's mating center is aligned with the previous receiving center.
Coordinates are sheet coordinates, NOT robot coordinates. Projection onto the
sheet is approximate for raised parts. A top view cannot establish clearance,
vertical insertion feasibility, retention, or pin insertion success.
"""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE = 'data/demo_trials/20261007_165106_644344/camera.png'
CALIBRATION = 'data/calibration/zone1_camera.json'
WINDOW = 'Link Coupling - Offline Geometry'
WIDTH, HEIGHT = 1100, 740


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def rotation(degrees):
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])


def load_geometry(image_path, calibration_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Cannot read image: {image_path}')
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    if image.shape[1] != calibration['image_width_px'] or image.shape[0] != calibration['image_height_px']:
        raise ValueError('Image resolution differs from calibration.')
    matrix = np.array(calibration['pixel_to_mm_matrix'], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Invalid homography.')
    region = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(np.rint(calibration['image_points']).astype(np.int32)), 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.bitwise_and(cv2.inRange(gray, 0, 110), region)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('No dark part found within calibrated markers.')
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 500:
        raise ValueError('Part contour too small; check image and calibration.')
    # Keep internal holes: metric rectification uses the original threshold.
    isolated = np.zeros_like(mask)
    cv2.drawContours(isolated, [contour], -1, 255, cv2.FILLED)
    mask = cv2.bitwise_and(mask, isolated)
    pixels_per_mm = 4.0
    scale = np.diag([pixels_per_mm, pixels_per_mm, 1.0])
    sheet = cv2.warpPerspective(mask, scale @ matrix,
        (math.ceil(calibration['reference_width_mm'] * pixels_per_mm),
         math.ceil(calibration['reference_height_mm'] * pixels_per_mm)),
        flags=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(sheet, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    crop = sheet[y:y+h, x:x+w]
    ys, xs = np.nonzero(crop)
    if len(xs) == 0:
        raise ValueError('Empty rectified silhouette.')
    # All local coordinates are relative to the cropped rectangle center.
    center = np.array([(w-1)/2, (h-1)/2])
    samples = (np.column_stack([xs, ys]) - center) / pixels_per_mm
    outline = (c[:, 0, :].astype(float) - [x, y] - center) / pixels_per_mm
    return {'crop': crop, 'samples': samples, 'outline': outline,
            'center': center, 'ppm': pixels_per_mm,
            'sheet_center': (np.array([x, y]) + center) / pixels_per_mm,
            'bounds_mm': [w/pixels_per_mm, h/pixels_per_mm]}


def layout(anchors, count, turn, correction):
    """Align the NEW mating center with the PREVIOUS receiving center."""
    receiving, mating = np.array(anchors['receiving']), np.array(anchors['mating'])
    poses = [(np.zeros(2), 0.0)]
    for _ in range(1, count):
        previous, previous_angle = poses[-1]
        angle = previous_angle + turn
        position = (previous + rotation(previous_angle) @ receiving
                    - rotation(angle) @ mating
                    + rotation(previous_angle) @ correction)
        poses.append((position, angle))
    return poses


def label(canvas, text, x, y, color=(230, 230, 230)):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                .48, color, 1, cv2.LINE_AA)


def render(geometry, anchors, count, turn, correction):
    canvas = np.full((HEIGHT, WIDTH, 3), 25, np.uint8)
    label(canvas, 'OFFLINE COUPLING - SHEET COORDINATES ONLY', 20, 25, (80, 220, 255))
    label(canvas, '1: receiving center in horseshoe | 2: mating center at opposite end | 3: grasp center', 20, 50)
    label(canvas, 'Press 1/2/3, then click LEFT part. S: save snapshot + geometry. Q/ESC: exit.', 20, 75)
    label(canvas, 'Top view only: no collision check, robot targets or physical insertion validation.', 20, 100)
    # Left panel preserves holes and enlarges one link for anchor selection.
    crop = geometry['crop']
    zoom = min(370/crop.shape[1], 420/crop.shape[0])
    enlarged = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    origin = np.array([25, 160])
    h, w = enlarged.shape
    left = canvas[160:160+h, 25:25+w]
    left[enlarged > 0] = (170, 175, 185)
    colors = {'receiving': (0, 220, 255), 'mating': (255, 170, 60), 'grasp': (230, 70, 230)}
    for name, point in anchors.items():
        pixel = np.rint(origin + (np.array(point)*geometry['ppm'] + geometry['center'])*zoom).astype(int)
        cv2.drawMarker(canvas, tuple(pixel), colors[name], cv2.MARKER_CROSS, 18, 2)
        label(canvas, name, int(pixel[0])+8, int(pixel[1])-8, colors[name])
    label(canvas, f"Approx. projected size: {geometry['bounds_mm'][0]:.1f} x {geometry['bounds_mm'][1]:.1f} mm", 20, 640)
    label(canvas, '+angle follows sheet X,Y axes; screen Y increases downward.', 20, 665)
    label(canvas, 'Save does not modify existing profiles or calibration.', 20, 690)
    if 'receiving' not in anchors or 'mating' not in anchors:
        label(canvas, 'Mark receiving and mating centers to build the chain.', 450, 200)
        return canvas, (origin, zoom, w, h), []
    poses = layout(anchors, count, turn, correction)
    transformed = [geometry['samples'] @ rotation(a).T + p for p, a in poses]
    all_points = np.concatenate(transformed)
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    scale = min(600/max(high[0]-low[0], 1), 420/max(high[1]-low[1], 1), 5)
    offset = np.array([450., 160.]) - low*scale
    palette = [(70, 190, 90), (230, 140, 60), (150, 100, 220), (80, 220, 220), (200, 160, 160)]
    for i, ((p, a), points) in enumerate(zip(poses, transformed)):
        pixels = np.rint(points*scale+offset).astype(int)
        valid = (pixels[:, 0] >= 440) & (pixels[:, 0] < WIDTH) & (pixels[:, 1] >= 140) & (pixels[:, 1] < 610)
        pixels = pixels[valid]
        layer = np.zeros(canvas.shape[:2], np.uint8)
        layer[pixels[:, 1], pixels[:, 0]] = 255
        if scale > geometry['ppm']:
            layer = cv2.dilate(layer, np.ones((2, 2), np.uint8))
        canvas[layer > 0] = palette[i % len(palette)]
        center = np.rint(p*scale+offset).astype(int)
        label(canvas, f'Link {i+1}', int(center[0]), int(center[1]))
        if i:
            prev, prev_angle = poses[i-1]
            pin = prev + rotation(prev_angle) @ np.array(anchors['receiving'])
            cv2.drawMarker(canvas, tuple(np.rint(pin*scale+offset).astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
    label(canvas, f'Links: {count} | relative turn: {turn:.1f} deg | correction: {correction[0]:.1f}, {correction[1]:.1f} mm', 450, 625)
    label(canvas, 'Yellow crosses: provisional pin centers on previous links.', 450, 650)
    return canvas, (origin, zoom, w, h), poses


def save(geometry, anchors, poses, count, turn, correction, canvas, args):
    if not all(name in anchors for name in ('receiving', 'mating', 'grasp')):
        raise ValueError('Mark receiving, mating AND grasp centers before saving.')
    links, pins, sequence = [], [], []
    for i, (position, angle) in enumerate(poses):
        grasp = position + rotation(angle) @ np.array(anchors['grasp'])
        links.append({'link_index': i+1, 'center_offset_sheet_mm': position.tolist(),
                      'rotation_from_reference_deg': angle,
                      'grasp_offset_from_first_grasp_sheet_mm': (grasp-np.array(anchors['grasp'])).tolist()})
        sequence.extend([{'robot': 'Robot 1', 'action': 'place_first_link' if i == 0 else 'couple_new_mating_end_into_previous_horseshoe', 'link_index': i+1},
                         {'robot': 'Robot 1', 'action': 'retreat_before_robot2', 'link_index': i+1}])
        if i:
            p, a = poses[i-1]
            pin = p + rotation(a) @ np.array(anchors['receiving'])
            pins.append({'joining_links': [i, i+1], 'pin_center_offset_sheet_mm': pin.tolist()})
            sequence.append({'robot': 'Robot 2', 'action': 'insert_pin_pending', 'joining_links': [i, i+1]})
    record = {'profile_version': 1, 'status': 'offline_geometry_only',
              'physical_validation': False, 'robot_execution_enabled': False,
              'coordinate_frame': 'rectified_zone1_sheet_axes_relative_to_reference_link_center',
              'source_image': args.image, 'source_calibration': args.calibration,
              'anchor_offsets_mm': anchors, 'reference_center_sheet_mm': geometry['sheet_center'].tolist(),
              'link_count': count, 'relative_turn_deg': turn,
              'correction_in_previous_link_axes_mm': correction.tolist(),
              'links': links, 'provisional_pin_centers': pins, 'sequence': sequence,
              'pending': ['Verify selected joint centers physically', 'Teach sheet-to-assembly robot transform',
                          'Confirm tool-to-part orientation and grasp offset', 'Teach coupling approach and insertion path',
                          'Check each new route', 'Teach pin grasp, pin axis and insertion depth']}
    output = PROJECT_ROOT / 'data/link_coupling_previews' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    (output/'geometry.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    if not cv2.imwrite(str(output/'preview.png'), canvas):
        raise RuntimeError('Could not write preview image.')
    print('Saved:', output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', default=IMAGE)
    parser.add_argument('--calibration', default=CALIBRATION)
    parser.add_argument('--load', help='Load a previously saved geometry.json for adjustment.')
    parser.add_argument('--headless', action='store_true', help='Validate and save loaded geometry without GUI.')
    args = parser.parse_args()
    geometry = load_geometry(resolve(args.image), resolve(args.calibration))
    anchors, count, turn, correction = {}, 2, 0., np.zeros(2)
    if args.load:
        record = json.loads(resolve(args.load).read_text(encoding='utf-8'))
        if record['source_image'] != args.image or record['source_calibration'] != args.calibration:
            raise ValueError('Saved geometry uses different source files.')
        anchors = record['anchor_offsets_mm']
        count, turn = int(record['link_count']), float(record['relative_turn_deg'])
        correction = np.array(record['correction_in_previous_link_axes_mm'], dtype=float)
        if not 2 <= count <= 5 or not np.isfinite(turn) or correction.shape != (2,) or not np.isfinite(correction).all():
            raise ValueError('Invalid saved layout.')
        for name, point in anchors.items():
            if name not in ('receiving', 'mating', 'grasp') or np.array(point).shape != (2,) or not np.isfinite(point).all():
                raise ValueError('Invalid saved anchor.')
    if args.headless:
        canvas, _, poses = render(geometry, anchors, count, turn, correction)
        save(geometry, anchors, poses, count, turn, correction, canvas, args)
        return
    state = {'selected': 'receiving', 'view': None}
    def click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or state['view'] is None:
            return
        origin, zoom, w, h = state['view']
        relative = np.array([x, y])-origin
        if 0 <= relative[0] < w and 0 <= relative[1] < h:
            point = (relative/zoom-geometry['center'])/geometry['ppm']
            anchors[state['selected']] = point.tolist()
            print(state['selected'], 'offset mm:', point.round(3).tolist())
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Links 2-5', WINDOW, count-2, 3, lambda value: None)
    cv2.createTrackbar('Relative turn +180', WINDOW, round(turn)+180, 360, lambda value: None)
    cv2.createTrackbar('Correction X +20 mm', WINDOW, round(correction[0]*10)+200, 400, lambda value: None)
    cv2.createTrackbar('Correction Y +20 mm', WINDOW, round(correction[1]*10)+200, 400, lambda value: None)
    cv2.setMouseCallback(WINDOW, click)
    print('Mark the three centers on the LEFT link. No robot connection.')
    print('1: receiving center; 2: mating center; 3: grasp. S: save. Q: exit.')
    try:
        while True:
            count = cv2.getTrackbarPos('Links 2-5', WINDOW)+2
            turn = cv2.getTrackbarPos('Relative turn +180', WINDOW)-180
            correction = np.array([(cv2.getTrackbarPos('Correction X +20 mm', WINDOW)-200)/10,
                                   (cv2.getTrackbarPos('Correction Y +20 mm', WINDOW)-200)/10])
            canvas, view, poses = render(geometry, anchors, count, turn, correction)
            state['view'] = view
            label(canvas, 'Selecting: '+state['selected'], 25, 135, (80, 220, 255))
            cv2.imshow(WINDOW, canvas)
            key = cv2.waitKey(30) & 255
            if key in (27, ord('q')) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('1'), ord('2'), ord('3')):
                state['selected'] = ('receiving', 'mating', 'grasp')[key-ord('1')]
            elif key == ord('s'):
                try:
                    save(geometry, anchors, poses, count, turn, correction, canvas, args)
                except ValueError as error:
                    print(error)
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
