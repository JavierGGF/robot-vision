"""Compare basic Cartesian motion with the previously used SDK command.

Local Docker simulator only. No gripper, vision, or virtual planning queries.
Controller self-collision protection stays enabled. No errors are reset.
"""

import json
from datetime import datetime
from robot_vision.robot.simulate_pick_place import LocalSimulator
from robot_vision.robot.test_pick_motion import PROJECT_ROOT

controller = LocalSimulator()
record = {'simulation_only': True, 'baseline_commit': 'f936e3b',
          'virtual_preflight_used': False, 'movements': []}
submitted = False

try:
    # Reuse local-container verification, not real-robot connection settings.
    controller.connect()
    # A newly enabled simulator reports ready state 0 before its first move.
    # The shared readiness check accepts 0 or 2 and rejects moving/stopped states.
    controller._require_motion_ready()
    arm = controller.arm
    if arm.mode != 0:
        raise RuntimeError(f'Expected position mode 0; received {arm.mode}.')
    code, diagnostics = arm.get_err_warn_code()
    if code != 0 or diagnostics != [0, 0]:
        raise RuntimeError(f'Controller diagnostics must be clear: {code}, {diagnostics}')
    start = controller.get_pose()
    record['start_pose'] = start
    record['start_joints'] = controller.get_joints()
    if abs(start[2]-200.0) > 1.0:
        raise RuntimeError('Expected the stopped simulator at Z=200 mm.')

    # This changes only the SDK's command mode; it disables no protection.
    arm.set_only_check_type(0)
    down = list(start)
    down[2] -= 20.0
    for label, target in [('down_20_mm', down), ('return_to_start', start)]:
        controller._require_motion_ready()
        print(label, ':', target)
        entry = {'label': label, 'target': target, 'status': 'submitted'}
        record['movements'].append(entry)
        submitted = True
        # Match the original Cartesian command, with a bounded wait.
        code = arm.set_position(
            x=target[0], y=target[1], z=target[2],
            roll=target[3], pitch=target[4], yaw=target[5],
            radius=-1.0, motion_type=0, relative=False, is_radian=False,
            speed=5.0, mvacc=20.0, wait=True, timeout=20.0,
        )
        entry['return_code'] = code
        if code != 0:
            raise RuntimeError(f'Linear command failed: {code}')
        controller.wait_stationary()
        actual = controller.get_pose()
        entry['actual_pose'] = actual
        entry['actual_joints'] = controller.get_joints()
        print('Measured Z:', actual[2])
        pe = max(abs(a-b) for a,b in zip(actual[:3], target[:3]))
        ae = max(abs((a-b+180)%360-180) for a,b in zip(actual[3:], target[3:]))
        code, diagnostics = arm.get_err_warn_code()
        entry['diagnostics'] = diagnostics
        if code != 0 or diagnostics != [0,0] or pe > 1.0 or ae > 2.0:
            raise RuntimeError(f'Arrival not verified: {pe:.3f} mm, {ae:.3f} deg, {diagnostics}')
        entry['status'] = 'completed'
        submitted = False
    record['status'] = 'baseline_passed'
    print('BASELINE PASSED: down 20 mm and back, positions verified.')
except Exception as error:
    record['status'] = 'failed'
    record['error'] = str(error)
    print('Baseline stopped:', error)
    if submitted:
        # Stop only the verified local simulator if arrival cannot be confirmed.
        record['stop_return_code'] = controller.arm.set_state(4)
finally:
    try:
        controller.disconnect()
    finally:
        folder = PROJECT_ROOT / 'data/simulation_diagnostics'
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = folder / f'baseline_{stamp}.json'
        output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print('Report saved:', output)
