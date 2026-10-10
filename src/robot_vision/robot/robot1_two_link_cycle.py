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
        self._require_motion_ready()
        if self.arm.mode != 0:
            raise RuntimeError('Position mode 0 is required; no automatic mode reset.')

    def wait_stationary(self, timeout=120.0, stable_seconds=1.0):
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
        if input("Verify jaws physically OPEN. Type OPEN: ").strip() != "OPEN":
            raise RuntimeError("Physical opening not confirmed; movement cancelled")
        return code

    def close_gripper(self):
        code = super().close_gripper()
        time.sleep(0.3)
        code = super().close_gripper()
        time.sleep(2.0)
        if input("Verify jaws CLOSED and part held. Type HELD: ").strip() != "HELD":
            raise RuntimeError("Grasp not confirmed; lift cancelled")
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
    block = {'name': name, 'status': 'started', 'execution': [], 'plan': plan}
    record['blocks'].append(block)
    save_record(output, record)
    active = None
    try:
        controller.wait_stationary()
        check_pose(controller.get_pose(), expected)
        block['route_check'] = controller.check_linear_route(movement_segments(plan))
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
            active = dict(step, status='started')
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
    args = parser.parse_args(argv)
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
            print('PHYSICAL ROBOT EXECUTION. No automatic return, retry or gripper release.')
            print('Verify camera/sheet registration, taught poses, TCP, gripper and first-link support.')
            confirm('Confirm current physical station calibration and tooling.', word='VERIFIED')
            controller = PhysicalCouplingController()
            controller.connect()
            controller.wait_stationary()
            expected = controller.get_pose()
            # Start from the actual stationary pose; routes remain checked.
            record['start_joints'] = controller.get_joints()
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
                frame = capture_frame(calibration)
                source = 'fresh_live_capture'
            else:
                value = args.first_image if index == 1 else args.second_image
                source = str(project_path(value or DEFAULT_IMAGE))
                frame = cv2.imread(source)
            analysis, runtime = inspect_image(frame, calibration, reference, pick, directory, source)
            record['vision'].append({'link_index': index, 'source': source, 'analysis': analysis})
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
                    step = copy.deepcopy(step)
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
                controller.disconnect()
        finally:
            record['finished_at'] = datetime.now().astimezone().isoformat()
            save_record(output,record)
            print('Saved:',output)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())

