"""Preview or execute two link placements in the verified LOCAL simulator.

Reuse the successful saved-image cycle and virtual gripper events. The second
placement has an adjustable robot-base offset and an adjustable entry offset.
A pause separates approach from the final coupling movement. This does not
model contact, obstacles, link retention, a real gripper, or the pin robot.
No physical-robot execution is provided.
"""
import argparse
import copy
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from robot_vision.robot import simulator_demo_cycle as demo
from robot_vision.robot.test_rotated_pick import sheet_to_robot_matrix

DEFAULT_GEOMETRY = 'data/link_coupling_previews/20261009_123745_720054/geometry.json'


class TwoLinkLocalSimulator(demo.LocalSimulator):
    """Allow slow local motions to settle without the planner's 10 s cutoff."""

    def wait_stationary(self, timeout=120.0, stable_seconds=1.0):
        # Preserve idle state, joint stability, diagnostics and connection checks.
        # No motion retry, warning reset or controller protection change.
        return super().wait_stationary(
            timeout=timeout, stable_seconds=stable_seconds
        )


def rotation(angle):
    radians = math.radians(angle)
    return np.array([[math.cos(radians), -math.sin(radians)],
                     [math.sin(radians), math.cos(radians)]])


def vector(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f'{name} requires {length} finite numbers.')
    return result


def build_two_plans(start, target, pick, transfer, offset, entry_offset, coupling_speed):
    """Build ordinary first placement and staged second placement."""
    offset = vector(offset, 3, 'Second placement offset')
    entry_offset = vector(entry_offset, 3, 'Entry offset')
    if entry_offset[2] < 0:
        raise ValueError('Entry Z offset must be nonnegative.')
    if np.linalg.norm(entry_offset) < 1e-6:
        raise ValueError('Entry offset must define a nonzero coupling movement.')
    if not math.isfinite(coupling_speed) or coupling_speed <= 0 or coupling_speed > 5:
        raise ValueError('Coupling speed must be in (0, 5] mm/s.')
    second_transfer = copy.deepcopy(transfer)
    second_pose = vector(transfer['placement_pose_mm_deg'], 6, 'First placement').copy()
    second_pose[:3] += offset
    second_transfer['placement_pose_mm_deg'] = second_pose.tolist()
    entry = second_pose.copy()
    entry[:3] += entry_offset
    if max(entry[2], second_pose[2]) >= transfer['motion']['transfer_z_mm']:
        raise ValueError('Coupling entry must be below transfer height.')
    first = demo.build_cycle_plan(start, target, pick, transfer)
    second = demo.build_cycle_plan(transfer['retreat_pose_mm_deg'], target, pick, second_transfer)
    placement_index = next(i for i, step in enumerate(second) if step['label'] == 'descend_to_place')
    step = copy.deepcopy(second[placement_index])
    # Keep lateral approach above the requested entry, then descend vertically.
    elevated = copy.deepcopy(second[placement_index-1])
    elevated['label'] = 'above_coupling_entry'
    elevated['pose'] = [float(entry[0]), float(entry[1]), elevated['pose'][2], *second_pose[3:].tolist()]
    entry_step = copy.deepcopy(step)
    entry_step.update(label='approach_coupling_entry', pose=entry.tolist())
    step.update(label='couple_second_link', speed=coupling_speed)
    second[placement_index:placement_index+1] = [elevated, entry_step, step]
    split = next(i for i, step in enumerate(second) if step['label'] == 'couple_second_link')
    return first, second[:split], second[split:]


def ask(message):
    if input(message+' Type RUN to continue: ').strip() != 'RUN':
        raise KeyboardInterrupt('Cancelled at operator pause.')


def execute_blocks(controller, blocks, start, record, output, automatic=False):
    """Keep each block's measured arrivals and existing failure handling."""
    expected = list(start)
    record['blocks'] = []
    for name, plan in blocks:
        if not automatic:
            ask('\nLOCAL SIMULATOR ONLY - '+name+'.')
        else:
            print('\nAUTOMATIC LOCAL SIMULATOR - '+name)
        demo.require_ready(controller)
        demo.require_pose(controller.get_pose(), expected)
        block = {'name': name, 'status': 'started'}
        record['blocks'].append(block)
        demo.save_record(output, record)
        # The existing runner saves a separate journal for each block.
        block_output = output / name
        block_output.mkdir()
        try:
            demo.run_cycle(controller, plan, expected, block, block_output)
            block['status'] = 'completed'
        except BaseException as error:
            block['status'] = 'failed'
            block['error'] = str(error) or 'Interrupted'
            execution = block.get('execution', [])
            if execution and execution[-1].get('status') != 'completed':
                failed = execution[-1]
                failed['status'] = 'failed'
                failed['error'] = block['error']
                failed['failure_phase'] = (
                    'after_command_return' if 'return_code' in failed
                    else 'command_call_or_before_return'
                )
            # The shared runner may already have stopped the simulator.
            snapshot = {'timing': 'after_shared_runner_failure_handling'}
            for key, read in (
                ('state', controller.arm.get_state),
                ('diagnostics', controller.arm.get_err_warn_code),
                ('pose', controller.get_pose),
                ('joints', controller.get_joints),
            ):
                try:
                    snapshot[key] = read()
                except Exception as diagnostic_error:
                    snapshot[key + '_error'] = str(diagnostic_error)
            block['failure_snapshot'] = snapshot
            raise
        finally:
            # Persist in-memory command return and stop results on failure too.
            demo.save_record(block_output, block)
            demo.save_record(output, record)
        moves = [step for step in plan if step['operation'] == 'move']
        if moves:
            expected = list(moves[-1]['pose'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', default=DEFAULT_GEOMETRY)
    parser.add_argument('--offset-robot', nargs=3, type=float, metavar=('DX', 'DY', 'DZ'),
                        help='Override second TCP placement offset in ROBOT BASE millimeters.')
    parser.add_argument('--entry-offset', nargs=3, type=float, default=[0, 0, 10],
                        metavar=('DX', 'DY', 'DZ'), help='Entry minus final TCP pose in robot-base mm; DZ=0 allows horizontal coupling.')
    parser.add_argument('--coupling-speed', type=float, default=2.0)
    parser.add_argument('--execute', action='store_true', help='Verified LOCAL simulator only.')
    parser.add_argument('--auto', action='store_true', help='Skip operator pauses in LOCAL simulator execution.')
    args = parser.parse_args()
    if args.auto and not args.execute:
        parser.error('--auto requires --execute.')
    output = demo.PROJECT_ROOT / 'data/two_link_simulator_cycles' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'started', 'mode': 'local_simulator' if args.execute else 'offline_preview',
              'started_at': datetime.now().astimezone().isoformat(),
              'physical_validation': False, 'gripper_mode': 'virtual_events_only',
              'pin_insertion': 'pending_not_executed', 'arguments': vars(args)}
    controller = None
    exit_code = 0
    try:
        path = Path(args.geometry)
        if not path.is_absolute():
            path = demo.PROJECT_ROOT / path
        geometry = json.loads(path.read_text(encoding='utf-8'))
        if geometry.get('profile_version') != 1 or geometry.get('link_count') != 2:
            raise ValueError('Use a saved two-link geometry profile, version 1.')
        if geometry.get('coordinate_frame') != 'rectified_zone1_sheet_axes_relative_to_reference_link_center':
            raise ValueError('Unsupported geometry coordinate frame.')
        if geometry.get('source_image') != str(demo.IMAGE.relative_to(demo.PROJECT_ROOT)) or geometry.get('source_calibration') != 'data/calibration/zone1_camera.json':
            raise ValueError('Geometry must correspond to the successful cycle source image and calibration.')
        link = geometry['links'][1]
        if int(link['link_index']) != 2 or abs(float(link['rotation_from_reference_deg'])) > 1e-6:
            raise ValueError('This initial cycle requires two links with the SAME orientation.')
        delta_sheet = vector(link['grasp_offset_from_first_grasp_sheet_mm'], 2, 'Sheet offset')
        start, reference_joints = demo.load_reference()
        pick, transfer = demo.load_pick_profile(), demo.load_transfer_profile()
        calibration = demo.load_sheet_calibration('data/calibration/zone1_camera.json')
        reference = demo.load_sheet_calibration('data/calibration/zone1_reference_camera.json')
        frame = cv2.imread(str(demo.IMAGE))
        if frame is None:
            raise ValueError('Successful cycle image is missing.')
        analysis, images = demo.calculate_rotated_target(frame, calibration, reference, pick)
        demo.save_images(output, {'camera.png': frame, **images})
        runtime = copy.deepcopy(pick)
        runtime['motion']['tool_orientation_deg'] = analysis['target_orientation_deg']
        # Approximate rigid in-plane transfer. Requires physical confirmation:
        # roll/pitch tilt and part slip are not represented by this yaw model.
        mapping = sheet_to_robot_matrix(pick, reference)
        yaw_change = transfer['placement_pose_mm_deg'][5] - analysis['target_orientation_deg'][2]
        estimated_xy = rotation(yaw_change) @ mapping @ delta_sheet
        estimated = np.array([*estimated_xy, 0.0])
        offset = estimated if args.offset_robot is None else vector(args.offset_robot, 3, 'Robot offset')
        if np.linalg.norm(offset) > 150:
            raise ValueError('Second placement offset exceeds this initial demo limit of 150 mm.')
        plans = build_two_plans(start, analysis['target_xy_mm'], runtime, transfer,
                                offset, args.entry_offset, args.coupling_speed)
        blocks = list(zip(('first_link_cycle', 'second_link_approach', 'second_link_coupling_and_retreat'), plans))
        record.update(geometry=geometry, analysis=analysis, pick_profile=runtime,
                      transfer_profile=transfer, estimated_offset_robot_mm=estimated.tolist(),
                      selected_offset_robot_mm=offset.tolist(), offset_estimate_physically_verified=False,
                      mapping_assumption='rigid_planar_transfer_using_tool_yaw_change',
                      plan_blocks=[{'name': name, 'plan': plan} for name, plan in blocks])
        print('\nTWO-LINK LOCAL SIMULATOR CYCLE')
        print('Estimated second TCP offset in robot base [mm]:', estimated.round(3).tolist())
        print('Selected second TCP offset [mm]:', offset.round(3).tolist())
        print('Entry offset [mm]:', args.entry_offset)
        print('SAME source image reused for BOTH virtual picks; no two-part detection.')
        print('Offset is provisional. Contact, obstacles and insertion are NOT simulated.')
        for name, plan in blocks:
            print('\n'+name)
            for step in plan:
                print(step['label'], step.get('pose', 'virtual gripper event'))
        demo.save_record(output, record)
        if args.execute:
            controller = TwoLinkLocalSimulator()
            controller.connect()
            demo.require_ready(controller)
            demo.require_pose(controller.get_pose(), start)
            if np.max(np.abs(vector(controller.get_joints(), 6, 'Joints')-vector(reference_joints, 6, 'Reference joints'))) > 2:
                raise ValueError('Simulator must start at the saved retreat joint configuration.')
            execute_blocks(controller, blocks, start, record, output, automatic=args.auto)
            record['status'] = 'two_link_simulator_motion_completed_not_physical_assembly'
        else:
            record['status'] = 'two_link_offline_preview_not_executed'
            print('\nPREVIEW ONLY - no controller connection.')
    except (Exception, KeyboardInterrupt) as error:
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
        record.update(status='failed', error=str(error) or 'Interrupted')
        print('Stopped:', record['error'])
    finally:
        try:
            if controller is not None:
                controller.disconnect()
        finally:
            record['finished_at'] = datetime.now().astimezone().isoformat()
            demo.save_record(output, record)
            print('Saved:', output)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
