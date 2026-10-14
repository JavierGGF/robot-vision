"""Preview or execute two-link coupling with Robot 1.

Default: saved-image preview, without a robot connection. Physical execution
captures one isolated link in Zone 1 before each pick. Camera, sheet, taught
pick/transfer poses and TCP must be physically verified for the current station.
No Robot 2 or pin insertion. Firmware checks do not model external obstacles,
part contact, retention, or displacement of the first link.
"""

import argparse
import copy
import ipaddress
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2

from robot_vision.robot.controller import Lite6Controller
from robot_vision.robot.pick_cycle import (
    build_cycle_plan, load_pick_profile, load_transfer_profile, movement_segments,
)
from robot_vision.robot.simulator_two_link_cycle import build_two_plans
from robot_vision.robot.test_pick_motion import (
    PROJECT_ROOT, capture_frame, load_sheet_calibration, project_path, save_images,
)
from robot_vision.robot.test_rotated_pick import calculate_rotated_target

DEFAULT_IMAGE = 'data/demo_trials/20261007_165106_644344/camera.png'


def save_record(directory, record):
    pending = directory / 'results.pending.json'
    pending.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    pending.replace(directory / 'results.json')


def check_pose(actual, expected):
    position = math.sqrt(sum((a-b)**2 for a, b in zip(actual[:3], expected[:3])))
    angle = max(abs((a-b+180) % 360-180) for a, b in zip(actual[3:], expected[3:]))
    if position > 1.0 or angle > 2.0:
        raise RuntimeError(f'Pose mismatch: {position:.3f} mm, {angle:.3f} deg.')


def confirm(message, word='CONTINUE'):
    if word == 'CONTINUE':
        print(message + ' [automatic]')
        return
    if input(message + f' Type {word}: ').strip() != word:
        raise KeyboardInterrupt('Cancelled by operator.')


class PhysicalCouplingController(Lite6Controller):
    """Use joint-seeded virtual chains without interleaved endpoint IK queries."""

    def __init__(self):
        super().__init__()
        if self.mode == 'simulation':
            raise RuntimeError('Physical execution rejects simulation configuration.')
        try:
            address = ipaddress.ip_address(self.robot_ip)
        except ValueError as error:
            raise ValueError('Use an explicit physical robot IP address.') from error
        if address.is_loopback or address.is_unspecified or address.is_multicast:
            raise ValueError('Physical execution rejects loopback or non-unicast IPs.')
        self.last_stationary_check = None

    def connect(self):
        super().connect()
        if self.arm.axis != 6 or self.arm.device_type != 9:
            raise RuntimeError('This program requires the Lite6 and its Lite6 gripper.')
        if self.arm.mode != 0:
            raise RuntimeError("Disable manual mode; position mode 0 is required.")
        code, diagnostics = self.arm.get_err_warn_code()
        if code != 0 or any(diagnostics):
            raise RuntimeError(f"Resolve diagnostics first: {code}, {diagnostics}")
        code, state = self.arm.get_state()
        if code != 0:
            raise RuntimeError(f"State read failed: {code}")
        if state == 4:
            print("Enabling stopped controller for this new execution.")
            code = self.arm.set_state(0)
            if code != 0:
                raise RuntimeError(f"Enable rejected: {code}")
            time.sleep(1.0)
        self._require_motion_ready()

    def wait_stationary(self, timeout=120.0, stable_seconds=0.35):
        deadline = time.monotonic() + timeout
        anchor = None
        stable_since = None
        last = {}
        while time.monotonic() < deadline:
            self._require_connection()
            code, diagnostics = self.arm.get_err_warn_code()
            if code != 0 or diagnostics != [0, 0]:
                raise RuntimeError(f'Diagnostics rejected: {code}, {diagnostics}')
            code, state = self.arm.get_state()
            if code != 0 or state not in (0, 1, 2):
                raise RuntimeError(f'State rejected: {code}, {state}')
            if self.arm.mode != 0:
                raise RuntimeError('Controller mode changed during movement.')
            joints = self.get_joints()
            now = time.monotonic()
            last = {'state': state, 'joints_deg': joints}
            if state != 2:
                anchor = None
                stable_since = None
            elif anchor is None or max(abs(a-b) for a, b in zip(joints, anchor)) > 0.01:
                anchor = list(joints)
                stable_since = now
            elif now - stable_since >= stable_seconds:
                self.last_stationary_check = last
                return last
            time.sleep(0.1)
        raise RuntimeError(f'Motion settling timeout: {last}')

    def _send_linear(self, pose, speed, mvacc, check_type, wait):
        return self.arm.set_position(
            x=pose[0], y=pose[1], z=pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5],
            radius=-1.0, motion_type=0, relative=False, is_radian=False,
            speed=speed, mvacc=mvacc, only_check_type=check_type,
            wait=wait, timeout=120.0,
        )

    def check_linear_route(self, segments):
        if not segments:
            raise ValueError('A route needs at least one movement.')
        prepared = [dict(label=str(s['label']), pose=self._validate_pose(s['pose']),
                         speed=self._validate_positive(s['speed'], 'Speed'),
                         mvacc=self._validate_positive(s['mvacc'], 'Acceleration'))
                    for s in segments]
        self.wait_stationary()
        self._require_motion_ready()
        report = {'status': 'started', 'method': 'actual_joint_seed_cartesian_chain',
                  'start_pose': self.get_pose(), 'start_joints': self.get_joints(),
                  'segments': [], 'external_obstacles_checked': False}
        self.last_path_check = report
        try:
            code = self.arm.set_servo_angle(
                angle=report['start_joints'], speed=5.0, mvacc=5.0,
                relative=False, is_radian=False, wait=False, only_check_type=1,
            )
            result = self.arm.only_check_result
            report['seed'] = {'return_code': code, 'planner_result': result}
            if code != 0 or result != 0:
                raise RuntimeError(f'Joint seed rejected: {code}, {result}')
            for i, step in enumerate(prepared):
                self._require_motion_ready()
                entry = dict(step)
                report['segments'].append(entry)
                kind = 2 if i == len(prepared)-1 else 3
                code = self._send_linear(step['pose'], step['speed'], step['mvacc'], kind, False)
                result = self.arm.only_check_result
                entry.update(return_code=code, planner_result=result, check_type=kind)
                if code != 0 or result != 0:
                    raise RuntimeError(f"Route rejected at {step['label']}: {code}, {result}")
            if max(abs(a-b) for a, b in zip(self.get_joints(), report['start_joints'])) > 0.01:
                raise RuntimeError('Joints changed during non-moving route check.')
            report['status'] = 'passed'
            return report
        except BaseException as error:
            report.update(status='failed', error=str(error))
            raise

    def open_gripper(self):
        code = super().open_gripper()
        time.sleep(0.3)
        code = super().open_gripper()
        time.sleep(2.0)
        print("Opening commands sent; physical opening not verified")
        return code

    def close_gripper(self):
        code = super().close_gripper()
        time.sleep(0.3)
        code = super().close_gripper()
        time.sleep(2.0)
        print("Closing commands sent; physical grasp not verified")
        return code

    def stop_gripper_after_completion(self):
        """Stop Lite6 gripper drive after normal release and retreat."""
        self.wait_stationary()
        self._require_motion_ready()
        code = self.arm.stop_lite6_gripper(sync=False)
        self._check_code(code, 'Gripper drive stop')
        print('GRIPPER DRIVE STOPPED')
        return code

    def move_linear(self, target, speed=10.0, mvacc=20.0, wait=True):
        code = super().move_linear(target, speed=speed, mvacc=mvacc, wait=wait)
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            self._require_motion_ready()
            actual = self.get_pose()
            try:
                check_pose(actual, target)
            except RuntimeError:
                time.sleep(0.1)
                continue
            self.wait_stationary()
            check_pose(self.get_pose(), target)
            return code
        raise RuntimeError(
            f"Arrival timeout: target={target}, actual={self.get_pose()}"
        )


def inspect_image(frame, calibration, reference, pick, directory, source):
    if frame is None:
        raise ValueError(f'Image is missing: {source}')
    expected = (int(calibration['image_height_px']), int(calibration['image_width_px']))
    if frame.shape[:2] != expected:
        raise ValueError(f'Camera resolution {frame.shape[:2]} differs from calibration {expected}.')
    save_images(directory, {'camera.png': frame})
    analysis, images = calculate_rotated_target(frame, calibration, reference, pick)
    save_images(directory, images)
    runtime = copy.deepcopy(pick)
    runtime['motion']['tool_orientation_deg'] = analysis['target_orientation_deg']
    return analysis, runtime


def run_block(controller, name, plan, expected, record, output, contact=False):
    block_started = time.monotonic()
    block = {'name': name, 'status': 'started', 'execution': [], 'plan': plan,
             'started_at':datetime.now().astimezone().isoformat()}
    record['blocks'].append(block)
    save_record(output, record)
    active = None
    try:
        controller.wait_stationary()
        check_pose(controller.get_pose(), expected)
        route_started = time.monotonic()
        block['route_check'] = controller.check_linear_route(movement_segments(plan))
        block['route_check_seconds'] = time.monotonic() - route_started
        save_record(output, record)
        if contact:
            confirm('Verify entry alignment, first-link restraint and clearance before coupling.')
        else:
            confirm(f'Inspect {name} plan, grasp overlays and all station clearances.')
        check_pose(controller.get_pose(), expected)
        for step in plan:
            if step['operation'] == 'move':
                try:
                    check_pose(controller.get_pose(), step['pose'])
                except RuntimeError:
                    pass
                else:
                    print(name + ': ' + step['label'] + ' [already reached]')
                    expected = list(step['pose'])
                    continue
            step_started = time.monotonic()
            active = dict(step, status='started',
                          started_at=datetime.now().astimezone().isoformat())
            block['execution'].append(active)
            save_record(output, record)
            controller._require_motion_ready()
            check_pose(controller.get_pose(), expected)
            print(name + ': ' + step['label'])
            if step['operation'] == 'move':
                active['return_code'] = controller.move_linear(
                    step['pose'], speed=step['speed'], mvacc=step['mvacc'], wait=True,
                )
                active['actual_pose'] = controller.get_pose()
                active['actual_joints'] = controller.get_joints()
                check_pose(active['actual_pose'], step['pose'])
                expected = list(step['pose'])
            elif step['operation'] == 'gripper':
                if contact and step['label'] == 'release_in_zone2':
                    confirm('Verify coupling and support before opening the gripper.')
                    controller._require_motion_ready()
                    check_pose(controller.get_pose(), expected)
                active['return_code'] = (controller.open_gripper() if step['opening']
                                         else controller.close_gripper())
                time.sleep(step['delay_s'])
                active['retention_verified'] = False
                if step['label'] == 'close_gripper':
                    confirm('Visually verify the grasp before lifting.')
            else:
                raise ValueError('Unknown operation.')
            active['status'] = 'completed'
            active['duration_seconds'] = time.monotonic() - step_started
            active['finished_at'] = datetime.now().astimezone().isoformat()
            save_record(output, record)
        block['status'] = 'completed'
        return expected
    except BaseException as error:
        block.update(status='failed', error=str(error) or 'Interrupted')
        if active is not None and active['status'] != 'completed':
            active.update(status='failed', error=block['error'])
        # Stop motion on failures or cancellation. Do not release or retry.
        try:
            block['stop_return_code'] = controller.arm.set_state(4)
        except Exception as stop_error:
            block['stop_error'] = str(stop_error)
        raise
    finally:
        block['duration_seconds'] = time.monotonic() - block_started
        block['finished_at'] = datetime.now().astimezone().isoformat()
        block['latest_path_check'] = controller.last_path_check
        save_record(output, record)


def taught_second_plans(start, target, pick, transfer, station, speed):
    """Use the taught full entry and final poses."""
    entry = list(station['poses']['second_entry'])
    final = list(station['poses']['second_final'])
    second_transfer = copy.deepcopy(transfer)
    second_transfer['placement_pose_mm_deg'] = entry
    plan = build_cycle_plan(start, target, pick, second_transfer)
    split = next(i for i,s in enumerate(plan) if s['label']=='descend_to_place')
    plan[split]['label'] = 'approach_coupling_entry'
    coupling = [{'operation':'move', 'label':'couple_second_link', 'pose':final,
                 'speed':speed, 'mvacc':transfer['motion']['placement_acceleration']}] + copy.deepcopy(plan[split+1:])
    height = next(s['pose'][2] for s in plan if s['label']=='raise')
    for step in coupling:
        if step['label']=='clear_zone2':
            step['pose'] = [*final[:2],height,*final[3:]]
    return plan[:split+1], coupling


def move_to_start_retreat(controller, transfer, expected, record, output):
    """Park before vision, retaining orientation during translation."""
    retreat = list(transfer['retreat_pose_mm_deg'])
    if len(retreat) != 6 or not all(math.isfinite(float(v)) for v in retreat):
        raise ValueError('Invalid startup retreat pose')
    clearance = float(transfer['motion']['transfer_z_mm'])
    if not retreat[2] <= clearance <= 190.0:
        raise ValueError('Startup clearance must cover retreat and be at most 190 mm')
    current = list(expected)
    vertical = list(current)
    vertical[2] = clearance
    travel = list(retreat)
    travel[2] = clearance
    travel[3:6] = current[3:6]
    orient = list(retreat)
    orient[2] = clearance
    plan = [
        {'operation':'move', 'label':'startup_vertical_clearance',
         'pose':vertical, 'speed':20.0, 'mvacc':20.0},
        {'operation':'move', 'label':'startup_travel_to_retreat',
         'pose':travel, 'speed':20.0, 'mvacc':20.0},
        {'operation':'move', 'label':'startup_orient_at_retreat',
         'pose':orient, 'speed':10.0, 'mvacc':20.0},
        {'operation':'move', 'label':'startup_finish_retreat',
         'pose':retreat, 'speed':10.0, 'mvacc':20.0},
    ]
    print('STARTUP RETREAT: route checked before movement; no gripper command.')
    return run_block(controller, 'startup_retreat', plan, current, record, output)



def first_chain_plan(expected, analysis, runtime, transfer, settings):
    """Keep the taught first-placement pose and the proven retreat sequence."""
    runtime = copy.deepcopy(runtime)
    runtime['motion']['approach_z_mm'] = settings['transfer_z_mm']
    runtime['motion']['travel_speed_mm_s'] = 20.0
    transfer = copy.deepcopy(transfer)
    transfer['motion']['transfer_z_mm'] = settings['transfer_z_mm']
    transfer['motion']['travel_speed_mm_s'] = 20.0
    original = build_cycle_plan(expected, analysis['target_xy_mm'], runtime, transfer)
    adjusted = []
    for index, item in enumerate(original):
        step = copy.deepcopy(item)
        if step['label'] == 'intermediate_orientation':
            previous = next(v for v in reversed(original[:index]) if v['operation'] == 'move')
            following = next(v for v in original[index+1:] if v['operation'] == 'move')
            turn = (following['pose'][5] - previous['pose'][5] + 180) % 360 - 180
            step['pose'][5] = (previous['pose'][5] + turn / 2 + 180) % 360 - 180
        if step['operation'] == 'move' and step['pose'][2] > settings['transfer_z_mm'] + .5:
            raise ValueError('First-link plan exceeds configured clearance')
        if step['operation'] == 'move':
            step['pose'][2] = min(step['pose'][2], settings['transfer_z_mm'])
        if step['label'] == 'approach_retreat_and_orient':
            previous = next(v for v in reversed(adjusted) if v['operation'] == 'move')
            orientation = copy.deepcopy(step)
            orientation.update(label='orient_at_retreat', speed=10.0)
            step.update(label='travel_to_retreat', speed=20.0)
            step['pose'][3:] = previous['pose'][3:]
            adjusted.extend([step, orientation])
        else:
            adjusted.append(step)
    return adjusted


def require_chain_target_range(diagnostic, settings):
    """Reject an unvalidated insertion before issuing a pick or gripper command."""
    angle = diagnostic['rotation_sheet_deg']
    distance = diagnostic['opening_shift_sheet_mm']
    if (not math.isfinite(angle) or not math.isfinite(distance)
            or abs(angle) > settings['max_insertion_rotation_deg']
            or distance > settings['max_opening_shift_mm']):
        raise ValueError(
            f'Insertion blocked before pick: opening rotation={angle:.2f} deg, '
            f'shift={distance:.2f} mm; allowed '
            f'{settings["max_insertion_rotation_deg"]:.1f} deg / '
            f'{settings["max_opening_shift_mm"]:.1f} mm. '
            'No global fitted-pose correction is applied.')



def load_chain_zone2_calibration(value):
    """Validate Zone 2 without invoking the Zone 1-only pick loader."""
    import numpy as np
    calibration = json.loads(project_path(value).read_text(encoding='utf-8'))
    if calibration.get('zone') != 2:
        raise ValueError('Chain destination requires the Zone 2 calibration')
    if calibration.get('reference_type') != 'printed_fiducial_centers':
        raise ValueError('A fiducial-center calibration is required')
    if calibration.get('point_order') != ['TL', 'TR', 'BR', 'BL']:
        raise ValueError('Invalid physical fiducial order')
    matrix = np.asarray(calibration['pixel_to_mm_matrix'], dtype=float)
    points = np.asarray(calibration['reference_points_mm'], dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError('Invalid Zone 2 sheet transformation')
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Invalid Zone 2 fiducial coordinates')
    for key in ('image_width_px', 'image_height_px', 'reference_width_mm', 'reference_height_mm'):
        value = calibration[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError('Invalid Zone 2 image or sheet dimensions')
    return calibration


def run_vision_chain(args):
    """Run the existing first placement, then inspect each newly exposed opening.

    This is an experimental integration of existing operations. The camera does
    not verify grip retention, physical contact, or successful assembly.
    """
    from robot_vision.robot.chain_settings import load_settings, resolve
    from robot_vision.robot.chain_assembly import assembly_plan
    from robot_vision.vision.preview_zone2_horseshoe import analyze, interior_parts
    from robot_vision.vision.chain_targets import (
        matching_calibration, displacement_matrix, opening_targets,
    )

    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    output = PROJECT_ROOT / 'data/robot1_two_link_trials' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    record = {'status': 'started', 'mode': 'physical_execution' if args.execute else 'camera_preview',
              'started_at': datetime.now().astimezone().isoformat(),
              'requested_links': args.links, 'completed_links': 0, 'blocks': [], 'vision': [],
              'physical_assembly_verified': False, 'external_obstacles_checked': False,
              'pointer_offset_corrected': False, 'global_fit_correction_applied': False}
    controller = None
    exit_code = 0
    released_and_parked = False
    started = time.monotonic()
    try:
        settings = load_settings(args.chain_config)
        record['settings'] = settings
        pick_folder = resolve(settings['pick_reference'])
        destination = resolve(settings['coupling_reference'])
        pick = load_pick_profile(pick_folder / 'pick_profile.json')
        reference = load_sheet_calibration(pick_folder / 'reference_camera.json')
        transfer = load_transfer_profile(project_path(args.station) / 'transfer_profile.json')
        transfer['motion']['transfer_z_mm'] = settings['transfer_z_mm']
        cal1 = load_sheet_calibration('data/calibration/zone1_camera.json')
        cal2 = load_chain_zone2_calibration('data/calibration/zone2_camera.json')
        if any(cal1[key] != cal2[key] for key in ('camera_index', 'image_width_px', 'image_height_px')):
            raise ValueError('Both zones must use the same camera and resolution')
        if not matching_calibration(cal2, read(destination / 'calibration.json')):
            raise ValueError('Zone 2 reference and current camera calibration differ')
        markers = read(resolve(settings['zone2_markers']))
        displacement_matrix(markers, cal2)
        reference_frame = cv2.imread(str(destination / 'camera.png'))
        if reference_frame is None:
            raise FileNotFoundError(destination / 'camera.png')
        _, _, reference_geometry = analyze(reference_frame, cal2)
        poses = {name: read(destination / f'taught_coupling_{name}.json')['pose_mm_deg']
                 for name in ('entry', 'final')}
        # Validate saved poses without altering the user's station or calibration.
        opening_targets(reference_geometry, reference_geometry, markers, cal2, poses)
        record.update(pick_profile=pick, transfer_profile=transfer, calibration=cal1,
                      reference_calibration=reference, zone2_calibration=cal2)
        save_record(output, record)
        if args.execute:
            print('EXPERIMENTAL CHAIN RUN: taught first placement, then vision-based coupling.')
            print('Start with EMPTY Zone 2, EMPTY gripper, and ONE loose link in Zone 1.')
            print('No OPEN/HELD prompts. Feed each next link only while the robot is parked.')
            print('The previous 4 mm local error is NOT corrected automatically.')
            confirm('Verify current setup, supported chain, empty gripper and clear startup paths.', 'VERIFIED')
        initial = capture_frame(cal1)
        save_images(output, {'initial_camera.png': initial})
        if interior_parts(initial, cal2)[0]:
            raise ValueError('Start-from-first requires empty Zone 2. Existing-chain extension remains a separate menu activity.')
        if args.execute:
            controller = PhysicalCouplingController()
            if controller.robot_ip != settings['robot_ip']:
                raise ValueError('Controller IP and chain settings differ')
            controller.connect()
            controller.wait_stationary()
            expected = controller.get_pose()
            if expected[2] > settings['transfer_z_mm'] + .5:
                raise ValueError('Current pose is above configured clearance; startup route needs inspection')
            expected = move_to_start_retreat(controller, transfer, expected, record, output)
            released_and_parked = True
        else:
            expected = list(transfer['retreat_pose_mm_deg'])
            print('CAMERA PREVIEW ONLY: first placement plan; no robot connection.')
        for index in range(1, args.links + 1):
            if index > 1:
                if not args.execute:
                    record['status'] = 'first_plan_preview_completed_future_scene_required'
                    print('Later targets require fresh images after physical placement; no invented chain scene.')
                    break
                print(f'Robot parked. Load ONE loose link for link {index}; keep the chain supported.')
                if input('Press ENTER when clear, or Q to end this run: ').strip().lower() == 'q':
                    record['status'] = 'stopped_by_operator_at_feed'
                    break
            if controller:
                controller.wait_stationary()
                check_pose(controller.get_pose(), expected)
            directory = output / f'link{index}_vision'
            directory.mkdir()
            frame = capture_frame(cal1)
            save_images(directory, {'camera.png': frame})
            candidates, _, _ = interior_parts(frame, cal1)
            if not candidates:
                record.update(status='stopped_no_pick_candidate', stopped_before_link_index=index)
                print('ZONE 1: no interior pick candidate. No pick issued.')
                break
            if len(candidates) != 1:
                raise ValueError('Zone 1 requires ONE separated interior part; no pick issued')
            item = {'link_index': index, 'source': 'fresh_live_capture', 'analysis': {}}
            record['vision'].append(item)
            if index == 1:
                if interior_parts(frame, cal2)[0]:
                    raise ValueError('Zone 2 changed after startup; no pick issued')
            else:
                preview, _, geometry = analyze(frame, cal2)
                save_images(directory, {'zone2_opening.png': preview})
                targets, diagnostic = opening_targets(reference_geometry, geometry, markers, cal2, poses)
                item.update(zone2_geometry=geometry, targets=targets, destination_diagnostics=diagnostic)
                record['targets'] = targets
                save_record(output, record)
                require_chain_target_range(diagnostic, settings)
                print(f'Link {index} opening: {diagnostic["rotation_sheet_deg"]:.2f} deg; '
                      f'shift {diagnostic["opening_shift_sheet_mm"]:.2f} mm')
            analysis, runtime = inspect_image(frame, cal1, reference, pick, directory, 'fresh_live_capture')
            item['analysis'] = analysis
            if index == 1:
                plan = first_chain_plan(expected, analysis, runtime, transfer, settings)
                name = 'first_link_cycle'
                record.pop('targets', None)
            else:
                plan = assembly_plan(expected, analysis['target_xy_mm'], runtime,
                                     targets['entry'], targets['final'], settings)
                name = f'link{index}_cycle'
            save_record(output, record)
            for step in plan:
                print(step['label'], step.get('pose', 'gripper operation'))
            if args.execute:
                released_and_parked = False
                expected = run_block(controller, name, plan, expected, record, output, contact=index > 1)
                released_and_parked = True
                record['completed_links'] = index  # Motion count, not physical fit verification.
            else:
                record['blocks'].append({'name': name, 'status': 'preview_only', 'plan': plan})
            save_record(output, record)
        else:
            record['status'] = ('physical_motion_completed_assembly_not_verified' if args.execute
                                else 'first_plan_preview_completed_future_scene_required')
    except (Exception, KeyboardInterrupt) as error:
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
        record.update(status='failed', error=str(error) or 'Interrupted')
        if controller is not None and controller.arm is not None:
            try:
                record['stop_return_code'] = controller.arm.set_state(4)
            except Exception as stop_error:
                record['stop_error'] = str(stop_error)
        print('Stopped:', record['error'])
    finally:
        if controller is not None:
            try:
                if released_and_parked and record['status'] != 'failed':
                    record['gripper_shutdown_return_code'] = controller.stop_gripper_after_completion()
                else:
                    record['gripper_shutdown'] = 'preserved_after_failure_or_interruption'
            except Exception as error:
                record['gripper_shutdown_error'] = str(error)
                exit_code = 1
                print('Gripper drive stop failed:', error)
            finally:
                try:
                    controller.disconnect()
                except Exception as error:
                    record['disconnect_error'] = str(error)
                    exit_code = 1
                    print('Disconnect failed:', error)
        record['elapsed_seconds'] = time.monotonic() - started
        record['finished_at'] = datetime.now().astimezone().isoformat()
        save_record(output, record)
        print('Saved:', output)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Physical Robot 1; live capture and operator pauses.')
    parser.add_argument('--first-image', type=Path)
    parser.add_argument('--second-image', type=Path)
    parser.add_argument('--pick-profile', default='config/robot1_pick.json')
    parser.add_argument('--transfer-profile', default='config/robot1_transfer.json')
    parser.add_argument('--offset-robot', nargs=3, type=float, required=False, metavar=('DX','DY','DZ'),
                        help='Second TCP minus taught first placement, in robot-base mm.')
    parser.add_argument('--entry-offset', nargs=3, type=float, required=False, metavar=('DX','DY','DZ'),
                        help='Entry TCP minus final second TCP, in robot-base mm.')
    parser.add_argument('--coupling-speed', type=float, default=1.0)
    parser.add_argument('--station', type=Path, help='Current station directory with taught poses and profiles.')
    parser.add_argument('--reference-calibration', default='data/calibration/zone1_reference_camera.json')
    parser.add_argument('--second-only', action='store_true')
    parser.add_argument('--chain-config', help='Use current references for first placement and visual chain extension')
    parser.add_argument('--links', type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args(argv)
    if args.chain_config:
        if not args.station or args.second_only or args.first_image or args.second_image:
            parser.error('Chain mode requires --station and fresh camera images; no --second-only')
        return run_vision_chain(args)
    if args.links != 2:
        parser.error('--links requires --chain-config')
    station = None
    if args.station:
        folder = project_path(args.station)
        station = json.loads((folder / 'station.json').read_text())
        coupling = json.loads((folder / 'coupling.json').read_text())
        args.pick_profile = str(folder / 'pick_profile.json')
        args.transfer_profile = str(folder / 'transfer_profile.json')
        capture = project_path(station['capture_directory'])
        args.reference_calibration = str(capture / 'reference_camera.json')
        args.offset_robot = coupling['offset_robot_mm']
        args.entry_offset = coupling['entry_offset_mm']
        if not args.execute:
            args.first_image = args.first_image or capture / 'camera.png'
            args.second_image = args.second_image or capture / 'camera.png'
    if args.offset_robot is None or args.entry_offset is None:
        parser.error('Use --station or supply both offset arguments.')
    if args.execute and (args.first_image or args.second_image):
        parser.error('Physical execution requires fresh live captures, not saved images.')
    if any(not math.isfinite(v) for v in args.offset_robot + args.entry_offset):
        parser.error('Offsets must be finite.')
    if math.sqrt(sum(v*v for v in args.offset_robot)) > 150:
        parser.error('Placement offset exceeds 150 mm.')
    if math.sqrt(sum(v*v for v in args.entry_offset)) > 50:
        parser.error('Entry offset exceeds 50 mm.')
    if not 0 < args.coupling_speed <= 2:
        parser.error('Coupling speed must be in (0, 2] mm/s.')
    if args.entry_offset[2] < 0 or sum(v*v for v in args.entry_offset) < 1e-12:
        parser.error('Entry needs a nonzero offset and nonnegative DZ.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = PROJECT_ROOT / 'data/robot1_two_link_trials' / stamp
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'started', 'started_at': datetime.now().astimezone().isoformat(),
              'mode': 'physical_execution' if args.execute else 'offline_preview',
              'arguments': {k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'physical_assembly_verified': False, 'pin_insertion': 'not_included',
              'external_obstacles_checked': False, 'blocks': [], 'vision': []}
    controller = None
    exit_code = 0
    try:
        pick = load_pick_profile(args.pick_profile)
        transfer = load_transfer_profile(args.transfer_profile)
        calibration = load_sheet_calibration('data/calibration/zone1_camera.json')
        reference = load_sheet_calibration(args.reference_calibration)
        record.update(pick_profile=pick, transfer_profile=transfer,
                      calibration=calibration, reference_calibration=reference)
        # Validate both placement plans before any robot connection or capture.
        build_two_plans(transfer['retreat_pose_mm_deg'], [0,0], pick, transfer,
                        args.offset_robot, args.entry_offset, args.coupling_speed)
        if args.execute:
            print('PHYSICAL ROBOT EXECUTION. Startup retreat before vision; no movement retry or automatic recovery.')
            print('Verify camera/sheet registration, taught poses, TCP, gripper and first-link support.')
            confirm('Confirm calibration, EMPTY gripper, and clear vertical/retreat paths. Restart only after freeing any held or jammed part.', word='VERIFIED')
            controller = PhysicalCouplingController()
            connected_started = time.monotonic()
            controller.connect()
            record.setdefault('timings', {})['connect_seconds'] = time.monotonic() - connected_started
            controller.wait_stationary()
            expected = controller.get_pose()
            # Start from the actual stationary pose; routes remain checked.
            record['start_joints'] = controller.get_joints()
            record['start_pose_before_retreat'] = list(expected)
            expected = move_to_start_retreat(controller, transfer, expected, record, output)
        else:
            expected = list(transfer['retreat_pose_mm_deg'])
            print('OFFLINE PREVIEW: no camera or robot connection.')
            print('Saved source images must use the current Zone 1 calibration.')
        for index in ((2,) if args.second_only else (1,2)):
            directory = output / f'link{index}_vision'
            directory.mkdir()
            if args.execute:
                controller.wait_stationary()
                check_pose(controller.get_pose(), expected)
                confirm(f'Place ONE link in Zone 1 for pick {index}; keep robot outside camera view.')
                capture_started = time.monotonic()
                frame = capture_frame(calibration)
                record.setdefault('capture_timings', []).append({
                    'link_index':index, 'seconds':time.monotonic()-capture_started})
                source = 'fresh_live_capture'
            else:
                value = args.first_image if index == 1 else args.second_image
                source = str(project_path(value or DEFAULT_IMAGE))
                frame = cv2.imread(source)
            if station is not None:
                import numpy as np
                empty_dir = project_path(
                    "data/orientation_tests/20261010_094517_994548/"
                    "station_20261010_100335_354623/"
                    "empty_zone1_20261010_113434_491371"
                )
                empty_calibration = json.loads(
                    (empty_dir / "calibration.json").read_text()
                )
                if empty_calibration != calibration:
                    raise RuntimeError("Empty-sheet reference needs recalibration")
                empty_frame = cv2.imread(str(empty_dir / "camera.png"))
                if empty_frame is None or empty_frame.shape != frame.shape:
                    raise RuntimeError("Invalid empty-sheet reference")
                matrix = np.diag([4.0, 4.0, 1.0]) @ np.asarray(
                    calibration["pixel_to_mm_matrix"], dtype=float
                )
                size = (
                    round(calibration["reference_width_mm"] * 4),
                    round(calibration["reference_height_mm"] * 4),
                )
                current = cv2.cvtColor(
                    cv2.warpPerspective(frame, matrix, size), cv2.COLOR_BGR2GRAY
                )
                baseline = cv2.cvtColor(
                    cv2.warpPerspective(empty_frame, matrix, size), cv2.COLOR_BGR2GRAY
                )
                points = np.asarray(calibration["reference_points_mm"]) * 4
                left, top = np.floor(points.min(axis=0)).astype(int)
                right, bottom = np.ceil(points.max(axis=0)).astype(int)
                current = current[top:bottom, left:right]
                baseline = baseline[top:bottom, left:right]
                brightness_change = float(np.median(
                    current.astype(float) - baseline.astype(float)
                ))
                if abs(brightness_change) > 15:
                    raise RuntimeError("Lighting changed; empty check uncertain")
                corrected = np.clip(
                    current.astype(float) - brightness_change, 0, 255
                ).astype(np.uint8)
                difference = cv2.absdiff(corrected, baseline)
                changed = (difference > 25).astype(np.uint8) * 255
                changed = cv2.medianBlur(changed, 5)
                contours, _ = cv2.findContours(
                    changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                largest_mm2 = max(
                    (cv2.contourArea(c) / 16.0 for c in contours), default=0.0
                )
                save_images(directory, {
                    "camera.png": frame,
                    "empty_reference_difference.png": changed,
                })
                record.setdefault("empty_checks", []).append({
                    "link_index": index,
                    "largest_changed_area_mm2": largest_mm2,
                    "brightness_change": brightness_change,
                    "reference": str(empty_dir),
                })
                if largest_mm2 < 10.0:
                    record["status"] = "stopped_zone1_matches_empty_reference"
                    record["stopped_before_link_index"] = index
                    save_record(output, record)
                    print("ZONE 1 matches empty reference. No further movement.")
                    return 0
            try:
                vision_started = time.monotonic()
                analysis, runtime = inspect_image(
                    frame, calibration, reference, pick, directory, source
                )
            except ValueError as error:
                if str(error) != "No suitable object detected.":
                    raise
                record["status"] = "stopped_no_pick_candidate"
                record["stop_reason"] = "No suitable object detected in Zone 1"
                record["stopped_before_link_index"] = index
                record["physical_assembly_verified"] = False
                save_record(output, record)
                print("ZONE 1: no pick candidate. Stopping without further movement.")
                return 0
            record['vision'].append({'link_index': index, 'source': source, 'analysis': analysis,
                                    'duration_seconds':time.monotonic()-vision_started})
            if not args.execute and index == 2 and args.second_image is None:
                print('WARNING: second pick uses the same default saved image; not independent vision.')
            print(f'Link {index} grasp X,Y: {analysis["target_xy_mm"]}; tool: {analysis["target_orientation_deg"]}')
            if index == 1:
                plans = [('first_link_cycle', build_cycle_plan(expected, analysis['target_xy_mm'], runtime, transfer))]
            else:
                _, approach, coupling = build_two_plans(
                    expected, analysis['target_xy_mm'], runtime, transfer,
                    args.offset_robot, args.entry_offset, args.coupling_speed,
                )
                if station is not None:
                    approach, coupling = taught_second_plans(
                        expected, analysis['target_xy_mm'], runtime, transfer,
                        station, args.coupling_speed,
                    )
                plans = [('second_link_approach',approach),('second_link_coupling_and_retreat',coupling)]
            adjusted_plans = []
            for block_name, original_plan in plans:
                adjusted = []
                for step in original_plan:
                    original_index = original_plan.index(step)
                    step = copy.deepcopy(step)
                    if step["label"] == "intermediate_orientation":
                        previous = next(
                            item for item in reversed(original_plan[:original_index])
                            if item["operation"] == "move"
                        )
                        following = next(
                            item for item in original_plan[original_index + 1:]
                            if item["operation"] == "move"
                        )
                        start_yaw = previous["pose"][5]
                        end_yaw = following["pose"][5]
                        turn = (end_yaw - start_yaw + 180) % 360 - 180
                        step["pose"][5] = (start_yaw + turn / 2 + 180) % 360 - 180
                    if step['operation'] == 'move':
                        step['pose'][2] = min(step['pose'][2], 190.0)
                    if step["operation"] == "move" and step["label"] in (
                        "descend_to_pick", "descend_to_place", "approach_coupling_entry"
                    ):
                        approach = copy.deepcopy(step)
                        approach["label"] = "fast_approach_" + step["label"]
                        approach["pose"][2] += 20.0
                        approach["speed"] = 30.0
                        adjusted.append(approach)
                    if step["label"] == "approach_retreat_and_orient":
                        previous = next(
                            item for item in reversed(original_plan[:original_index])
                            if item["operation"] == "move"
                        )
                        orientation_step = copy.deepcopy(step)
                        orientation_step["label"] = "orient_at_retreat"
                        orientation_step["speed"] = 10.0
                        step["label"] = "travel_to_retreat"
                        step["pose"][3:6] = previous["pose"][3:6]
                        step["speed"] = 20.0
                        adjusted.append(step)
                        adjusted.append(orientation_step)
                    else:
                        adjusted.append(step)
                adjusted_plans.append((block_name, adjusted))
            plans = adjusted_plans
            for name, plan in plans:
                print('\n' + name)
                for step in plan:
                    print(step['label'], step.get('pose', 'physical gripper operation'))
                if args.execute:
                    expected = run_block(controller, name, plan, expected, record, output,
                                         contact=name=='second_link_coupling_and_retreat')
                else:
                    record['blocks'].append({'name':name,'status':'preview_only','plan':plan})
                    moves = [s for s in plan if s['operation']=='move']
                    expected = list(moves[-1]['pose'])
                save_record(output, record)
            if args.execute and index == 1:
                confirm('Verify first-link placement and support before the second capture.')
        record['status'] = ('physical_motion_completed_assembly_not_verified' if args.execute
                            else 'offline_preview_completed')
    except (Exception, KeyboardInterrupt) as error:
        exit_code = 130 if isinstance(error,KeyboardInterrupt) else 1
        record.update(status='failed', error=str(error) or 'Interrupted')
        if controller is not None and controller.arm is not None:
            try:
                record['stop_return_code'] = controller.arm.set_state(4)
            except Exception as stop_error:
                record['stop_error'] = str(stop_error)
            for key, read in (('stopped_pose',controller.get_pose),('stopped_joints',controller.get_joints),
                              ('diagnostics',controller.arm.get_err_warn_code)):
                try:
                    record[key] = read()
                except Exception as diagnostic_error:
                    record[key+'_error'] = str(diagnostic_error)
        print('Stopped:',record['error'])
    finally:
        try:
            if controller is not None:
                try:
                    if record.get('status') in (
                        'physical_motion_completed_assembly_not_verified',
                        'stopped_zone1_matches_empty_reference', 'stopped_no_pick_candidate'
                    ):
                        record['gripper_shutdown_return_code'] = controller.stop_gripper_after_completion()
                    else:
                        record['gripper_shutdown'] = 'preserved_after_failure_or_interruption'
                except Exception as shutdown_error:
                    record['gripper_shutdown_error'] = str(shutdown_error)
                    exit_code = 1
                    print('Gripper drive stop failed:', shutdown_error)
                finally:
                    controller.disconnect()
        finally:
            record['finished_at'] = datetime.now().astimezone().isoformat()
            save_record(output,record)
            print('Saved:',output)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())


