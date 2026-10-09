"""Inspect the failed simulator segment without executing any movement."""

import json
from datetime import datetime
from pathlib import Path

from robot_vision.robot.simulate_pick_place import LocalSimulator
from robot_vision.robot.test_pick_motion import PROJECT_ROOT

# Reuse the exact target from the failed execution, not a new vision estimate.
SOURCE = PROJECT_ROOT / 'data/simulation_trials/20261008_202934_194228/results.json'
controller = LocalSimulator()
report = {'planning_only': True, 'motion_executed': False, 'checks': []}

try:
    saved = json.loads(SOURCE.read_text(encoding='utf-8'))
    step = next(s for s in saved['plan'] if s['label'] == 'above_zone1_and_orient')
    report['source'] = str(SOURCE)
    report['segment'] = step

    # The inherited connection verifies loopback, Docker ports, and Lite6.
    controller.connect()
    controller._require_motion_ready()
    start_joints = controller.get_joints()
    report['start_pose'] = controller.get_pose()
    report['start_joints'] = start_joints
    report['diagnostics_before'] = controller.arm.get_err_warn_code()
    print('ACTUAL START POSE:', report['start_pose'])
    print('ACTUAL START JOINTS:', start_joints)
    print('TARGET:', step['pose'])

    # Endpoint IK is recorded for comparison, not treated as the actual path.
    report['endpoint_ik'] = controller.calculate_ik(step['pose'])
    for speed, acceleration in [(30.0, 20.0), (10.0, 10.0), (2.0, 2.0)]:
        controller._require_motion_ready()
        joints = controller.get_joints()
        if max(abs(a-b) for a, b in zip(joints, start_joints)) > 0.1:
            raise RuntimeError('Simulator configuration changed during the check.')

        # Type 1 starts each independent query at the actual configuration.
        code = controller._send_linear(
            step['pose'], speed, acceleration, check_type=1, wait=False
        )
        result = controller.arm.only_check_result
        entry = {'speed_mm_s': speed, 'acceleration_mm_s2': acceleration,
                 'return_code': code, 'planner_result': result,
                 'passed': code == 0 and result == 0}
        report['checks'].append(entry)
        print(json.dumps(entry, indent=2))

    report['end_joints'] = controller.get_joints()
    report['diagnostics_after'] = controller.arm.get_err_warn_code()
    print('No movement or gripper operation was executed.')

except Exception as error:
    report['error'] = str(error)
    print('Diagnostic stopped:', error)

finally:
    try:
        controller.disconnect()
    finally:
        folder = PROJECT_ROOT / 'data/simulation_diagnostics'
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = folder / f'zone1_segment_{stamp}.json'
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print('Report saved:', output)
