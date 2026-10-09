"""Run the basic pick motion in the verified local Docker simulator.

Reproduce direct Cartesian SDK motion without virtual planning queries.
Gripper actions are recorded events only; no contact physics is simulated.
Real-robot configuration and motion code are never modified.
"""

import json
import time
from datetime import datetime

from robot_vision.robot.simulate_pick_place import LocalSimulator
from robot_vision.robot.pick_cycle import load_pick_profile
from robot_vision.robot.test_pick_motion import PROJECT_ROOT

SOURCE = PROJECT_ROOT / 'data/simulation_trials/20261008_213316_604881/results.json'


def pose_errors(actual, target):
    """Compare position and wrapped tool angles in millimeters and degrees."""
    return (
        max(abs(a-b) for a, b in zip(actual[:3], target[:3])),
        max(abs((a-b+180.0)%360.0-180.0) for a,b in zip(actual[3:],target[3:])),
    )


def require_clear(controller):
    """Require position mode, readiness, and enabled collision protection."""
    controller._require_motion_ready()
    if controller.arm.mode != 0:
        raise RuntimeError('Expected position mode 0; no mode change attempted.')
    code, diagnostics = controller.arm.get_err_warn_code()
    if code != 0 or diagnostics != [0, 0]:
        raise RuntimeError(f'Controller diagnostics not clear: {code}, {diagnostics}')


def main():
    controller = LocalSimulator()
    record = {'simulation_only': True, 'virtual_preflight_used': False,
              'gripper_mode': 'virtual_events_only', 'part_retention_verified': False,
              'execution': []}
    outstanding = False
    try:
        # Read preserved targets and the current pick profile without editing them.
        saved = json.loads(SOURCE.read_text(encoding='utf-8'))
        above = next(s['pose'] for s in saved['plan']
                     if s['label'] == 'above_zone1_and_orient')
        profile = load_pick_profile()
        motion = profile['motion']
        record.update(source=str(SOURCE), pick_profile=profile)
        controller.connect()
        require_clear(controller)
        start = controller.get_pose()
        record.update(start_pose=start, start_joints=controller.get_joints())
        pe, ae = pose_errors(start, above)
        if pe > 1.0 or ae > 2.0:
            raise RuntimeError('Start must match the saved above-pick pose at Z=200 mm.')

        # Keep the measured X,Y and orientation for both vertical segments.
        down = list(start)
        down[2] = float(motion['pick_z_mm'])
        lift = list(down)
        lift[2] += float(motion['lift_mm'])
        if not (0 < down[2] < lift[2] <= start[2]):
            raise RuntimeError('Unexpected pick/lift heights; no movement sent.')
        controller.arm.set_only_check_type(0)
        expected = start
        plan = [
            {'label': 'open_before_pick', 'event': 'OPEN', 'delay': motion['open_delay_s']},
            {'label': 'descend_to_pick', 'pose': down, 'speed': motion['descent_speed_mm_s'],
             'acceleration': motion['descent_acceleration']},
            {'label': 'close_at_pick', 'event': 'CLOSE', 'delay': motion['close_delay_s']},
            {'label': 'lift_40_mm', 'pose': lift, 'speed': motion['lift_speed_mm_s'],
             'acceleration': motion['travel_acceleration']},
        ]
        for step in plan:
            require_clear(controller)
            pe, ae = pose_errors(controller.get_pose(), expected)
            if pe > 1.0 or ae > 2.0:
                raise RuntimeError('Simulator position changed between steps.')
            entry = dict(step, status='started')
            record['execution'].append(entry)
            if 'event' in step:
                print('VIRTUAL GRIPPER:', step['event'])
                time.sleep(float(step['delay']))
            else:
                target = step['pose']
                speed = controller._validate_positive(step['speed'], 'Speed')
                acceleration = controller._validate_positive(step['acceleration'], 'Acceleration')
                print(step['label'], ': Z=', target[2], 'speed=', speed)
                # Exactly one direct linear command, as in the successful baseline.
                outstanding = True
                code = controller.arm.set_position(
                    x=target[0], y=target[1], z=target[2],
                    roll=target[3], pitch=target[4], yaw=target[5],
                    radius=-1.0, motion_type=0, relative=False, is_radian=False,
                    speed=speed, mvacc=acceleration, wait=True, timeout=45.0,
                )
                entry['return_code'] = code
                if code != 0:
                    raise RuntimeError(f'Linear command failed: {code}')
                controller.wait_stationary()
                require_clear(controller)
                actual = controller.get_pose()
                entry.update(actual_pose=actual, actual_joints=controller.get_joints())
                pe, ae = pose_errors(actual, target)
                if pe > 1.0 or ae > 2.0:
                    raise RuntimeError(f'Arrival mismatch: {pe:.3f} mm, {ae:.3f} deg')
                outstanding = False
                expected = target
                print('VERIFIED Z:', actual[2])
            entry['status'] = 'completed'
        record['status'] = 'basic_pick_motion_passed'
        print('BASIC PICK MOTION PASSED. Gripper events were virtual.')
    except BaseException as error:
        record.update(status='failed', error=str(error))
        print('Test stopped:', error)
        if outstanding and controller.arm is not None:
            # Stop only this local simulator if a submitted move is unverified.
            try:
                record['stop_return_code'] = controller.arm.set_state(4)
            except Exception as stop_error:
                record['stop_error'] = str(stop_error)
    finally:
        try:
            controller.disconnect()
        finally:
            folder = PROJECT_ROOT / 'data/simulation_diagnostics'
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output = folder / f'basic_pick_{stamp}.json'
            output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
            print('Report saved:', output)


if __name__ == '__main__':
    main()