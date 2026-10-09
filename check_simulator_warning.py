"""Isolate simulator warning 14 with planning queries only."""

import json
from datetime import datetime
from robot_vision.robot.simulate_pick_place import LocalSimulator
from robot_vision.robot.test_pick_motion import PROJECT_ROOT

SOURCE = PROJECT_ROOT / 'data/simulation_trials/20261008_211642_431626/results.json'
controller = LocalSimulator()
report = {'motion_executed': False, 'warning_clear_requested': False, 'stages': []}

def snapshot(label):
    code, values = controller.arm.get_err_warn_code()
    entry = {'stage': label, 'return_code': code, 'error_warning': values,
             'pose': controller.get_pose(), 'joints': controller.get_joints()}
    report['stages'].append(entry)
    print(json.dumps(entry, indent=2))
    if code != 0 or values[0] != 0:
        raise RuntimeError('Controller error or unreadable diagnostics; no reset attempted.')
    return values

try:
    saved = json.loads(SOURCE.read_text(encoding='utf-8'))
    step = next(s for s in saved['plan'] if s['label'] == 'descend_to_pick')
    report['target'] = step
    controller.connect()
    controller.wait_stationary()
    snapshot('before_warning_clear')

    # Clear only the recorded warning, once, on the verified local simulator.
    # Never clear errors, change mode/state, or disable any protection.
    report['warning_clear_requested'] = True
    code = controller.arm.clean_warn()
    if code != 0:
        raise RuntimeError(f'Warning clear failed: {code}')
    values = snapshot('after_warning_clear')
    if values[1] != 0:
        raise RuntimeError('Warning remains active; stopping this diagnostic.')

    joints = controller.get_joints()
    code = controller.arm.set_servo_angle(
        angle=joints, speed=5.0, mvacc=5.0, is_radian=False,
        relative=False, wait=False, only_check_type=1,
    )
    report['seed'] = {'code': code, 'planner_result': controller.arm.only_check_result}
    values = snapshot('after_virtual_joint_seed')
    if code != 0 or report['seed']['planner_result'] != 0 or values[1] != 0:
        raise RuntimeError('Seed rejected or generated a warning.')

    # Check the descent but never send an execution command (type zero).
    code = controller._send_linear(
        step['pose'], step['speed'], step['mvacc'], check_type=2, wait=False,
    )
    report['descent_check'] = {'code': code, 'planner_result': controller.arm.only_check_result}
    snapshot('after_descent_planning')
    print('PLANNING ONLY. No movement or gripper command was executed.')
except Exception as error:
    report['error'] = str(error)
    print('Diagnostic stopped:', error)
finally:
    try:
        report['command_trace'] = controller.command_trace
        controller.disconnect()
    finally:
        folder = PROJECT_ROOT / 'data/simulation_diagnostics'
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = folder / f'warning_source_{stamp}.json'
        output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print('Report saved:', output)