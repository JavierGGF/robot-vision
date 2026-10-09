"""Preview or run the saved-image pick-and-place demonstration locally.

The execution uses direct Cartesian commands verified in the basic simulator
pick and transfer tests. It does not run the experimental virtual planner.
Controller protections remain enabled, and every arrival is measured.
No physical robot, camera, gripper contact, or insertion is controlled here.
"""

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from robot_vision.robot.simulate_pick_place import LocalSimulator
from robot_vision.robot.pick_cycle import build_cycle_plan, load_pick_profile, load_transfer_profile
from robot_vision.robot.test_pick_motion import PROJECT_ROOT, load_sheet_calibration, save_images
from robot_vision.robot.test_rotated_pick import calculate_rotated_target

IMAGE = PROJECT_ROOT / 'data/demo_trials/20261007_165106_644344/camera.png'
REFERENCE = PROJECT_ROOT / 'data/simulation_diagnostics/basic_transfer_20261009_095320_836618.json'


def pose_errors(actual, target):
    """Compare position and wrapped orientation with the taught target."""
    return (max(abs(a-b) for a,b in zip(actual[:3],target[:3])),
            max(abs((a-b+180)%360-180) for a,b in zip(actual[3:],target[3:])))


def require_ready(controller):
    """Accept ready or idle position mode only, with clear diagnostics."""
    controller._require_motion_ready()
    if controller.arm.mode != 0:
        raise RuntimeError('Simulator must be in position mode 0.')
    code, diagnostic = controller.arm.get_err_warn_code()
    if code != 0 or diagnostic != [0,0]:
        raise RuntimeError(f'Simulator diagnostics require attention: {code}, {diagnostic}')


def require_pose(actual, expected):
    """Reject unexpected drift or an incomplete movement."""
    distance, angle = pose_errors(actual, expected)
    if distance > 1.0 or angle > 2.0:
        raise RuntimeError(f'Position mismatch: {distance:.3f} mm, {angle:.3f} deg.')


def load_reference():
    """Use the successful transfer endpoint as the repeatable start."""
    data = json.loads(REFERENCE.read_text(encoding='utf-8'))
    if data.get('status') != 'basic_transfer_motion_passed':
        raise RuntimeError('A successful transfer reference is required.')
    last = data['execution'][-1]
    if last.get('label') != 'finish_retreat' or last.get('status') != 'completed':
        raise RuntimeError('The saved transfer did not finish its retreat.')
    return last['actual_pose'], last['actual_joints']


def save_record(output, record):
    """Retain progress without changing any calibration or profile."""
    temporary = output / 'results.pending.json'
    temporary.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    temporary.replace(output/'results.json')


def run_cycle(controller, plan, start, record, output):
    """Execute the shared plan using the proven direct simulator method."""
    expected = list(start)
    outstanding = False
    record['execution'] = []
    try:
        # Set only the SDK command mode; no controller protection is disabled.
        controller.arm.set_only_check_type(0)
        for index, step in enumerate(plan, 1):
            require_ready(controller)
            require_pose(controller.get_pose(), expected)
            entry = dict(step, status='started')
            record['execution'].append(entry)
            save_record(output, record)
            print(f"{index:02d}/{len(plan):02d}  {step['label']}")
            if step['operation'] == 'gripper':
                entry['virtual_event'] = 'OPEN' if step['opening'] else 'CLOSE'
                print('   Virtual gripper:', entry['virtual_event'])
                time.sleep(step['delay_s'])
            elif step['operation'] == 'move':
                target = step['pose']
                print(f"   Speed: {step['speed']} mm/s; target Z: {target[2]:.3f} mm")
                outstanding = True
                entry['motion_requested'] = True
                save_record(output, record)
                # Send once, wait with a deadline, and never retry a failed command.
                code = controller.arm.set_position(
                    x=target[0], y=target[1], z=target[2], roll=target[3],
                    pitch=target[4], yaw=target[5], radius=-1.0, motion_type=0,
                    relative=False, is_radian=False, speed=step['speed'],
                    mvacc=step['mvacc'], wait=True, timeout=120.0,
                )
                entry['return_code'] = code
                if code != 0:
                    raise RuntimeError(f'Linear movement failed: {code}')
                controller.wait_stationary()
                require_ready(controller)
                actual = controller.get_pose()
                entry.update(actual_pose=actual, actual_joints=controller.get_joints())
                require_pose(actual, target)
                outstanding = False
                expected = list(target)
                print('   Arrival verified')
            else:
                raise RuntimeError('Unknown operation in the plan.')
            entry['status'] = 'completed'
            save_record(output, record)
    except BaseException:
        if outstanding:
            # Stop only the verified local simulator on unconfirmed completion.
            try:
                record['stop_return_code'] = controller.arm.set_state(4)
            except Exception as error:
                record['stop_error'] = str(error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='Move the verified local simulator; default is offline preview.')
    parser.add_argument('--output-dir', type=Path, help='Optional new folder for this run.')
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = args.output_dir or PROJECT_ROOT/'data/simulator_demo_cycles'/stamp
    output.mkdir(parents=True, exist_ok=False)
    record = {'started_at': datetime.now().astimezone().isoformat(),
              'mode': 'local_simulator_execution' if args.execute else 'offline_preview',
              'virtual_preflight_used': False, 'physical_validation': False,
              'gripper_mode': 'virtual_events_only', 'source_image': str(IMAGE),
              'start_reference': str(REFERENCE)}
    controller = None
    exit_code = 0
    try:
        expected_pose, expected_joints = load_reference()
        pick, transfer = load_pick_profile(), load_transfer_profile()
        calibration = load_sheet_calibration('data/calibration/zone1_camera.json')
        reference = load_sheet_calibration('data/calibration/zone1_reference_camera.json')
        frame = cv2.imread(str(IMAGE))
        if frame is None:
            raise RuntimeError(f'Saved camera image is missing: {IMAGE}')
        analysis, images = calculate_rotated_target(frame, calibration, reference, pick)
        save_images(output, {'camera.png': frame, **images})
        runtime = copy.deepcopy(pick)
        runtime['motion']['tool_orientation_deg'] = analysis['target_orientation_deg']
        record.update(analysis=analysis, pick_profile=runtime, transfer_profile=transfer,
                      camera_calibration=calibration, reference_calibration=reference)
        start = expected_pose
        if args.execute:
            controller = LocalSimulator()
            controller.connect()
            require_ready(controller)
            start = controller.get_pose()
            joints = controller.get_joints()
            record.update(start_pose=start, start_joints=joints, firmware=controller.arm.version)
            try:
                require_pose(start, expected_pose)
                if max(abs(a-b) for a,b in zip(joints, expected_joints)) > 2.0:
                    raise RuntimeError('Joint configuration differs from the saved retreat.')
            except RuntimeError as error:
                raise RuntimeError(
                    'Simulator must start at the verified retreat pose. '
                    'No automatic return from an arbitrary position is attempted. '
                    + str(error)
                ) from error
        plan = build_cycle_plan(start, analysis['target_xy_mm'], runtime, transfer)
        record.update(start_pose=start, plan=plan)
        print('\nLOCAL SIMULATOR: PICK, PLACE, AND RETREAT')
        print('Vision source: saved image. Gripper: virtual events.')
        print('This activity does not simulate part retention or insertion.')
        for index, step in enumerate(plan, 1):
            if step['operation']=='move':
                print(f"{index:02d}. {step['label']} | Z={step['pose'][2]:.3f} mm | {step['speed']} mm/s")
            else:
                print(f"{index:02d}. {step['label']} | virtual gripper event")
        save_record(output, record)
        if args.execute:
            run_cycle(controller, plan, start, record, output)
            record['status'] = 'simulator_demo_completed'
            print('\nSIMULATOR CYCLE COMPLETED: all movement endpoints verified.')
        else:
            record['status'] = 'offline_preview_not_executed'
            print('\nPREVIEW ONLY. No simulator connection or movement.')
    except (Exception, KeyboardInterrupt) as error:
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
        record.update(status='failed', error=str(error) or 'Interrupted by user')
        print('\nActivity stopped:', record['error'])
    finally:
        try:
            if controller is not None:
                controller.disconnect()
        finally:
            record['finished_at'] = datetime.now().astimezone().isoformat()
            save_record(output, record)
            print('Run saved:', output)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())