"""Compare local simulator planner initialization without executing motion."""

import json
from datetime import datetime
from robot_vision.robot.simulate_pick_place import LocalSimulator
from robot_vision.robot.test_pick_motion import PROJECT_ROOT

SOURCE = PROJECT_ROOT / 'data/simulation_trials/20261008_204306_555739/results.json'
controller = LocalSimulator()
report = {'planning_only': True, 'motion_executed': False, 'checks': []}

try:
    saved = json.loads(SOURCE.read_text(encoding='utf-8'))
    step = next(s for s in saved['plan'] if s['label'] == 'lift_after_pick')
    controller.connect()
    controller.wait_stationary()
    joints = controller.get_joints()
    report.update(source=str(SOURCE), target=step,
                  start_pose=controller.get_pose(), start_joints=joints,
                  firmware=controller.arm.version)
    print('ACTUAL JOINTS:', joints)
    print('ACTUAL POSE:', report['start_pose'])
    # Keep ordinary endpoint checks active before comparing initialization.
    report['endpoint_ik'] = controller.calculate_ik(step['pose'])

    def unchanged():
        controller._require_motion_ready()
        actual = controller.get_joints()
        if max(abs(a-b) for a,b in zip(actual,joints)) > 0.01:
            raise RuntimeError('Actual simulator joints changed; stopping.')

    # Repeat the pair for diagnostic consistency only. Never execute or retry motion.
    for run in (1, 2):
        for method in ('direct_actual_state', 'explicit_joint_seed'):
            unchanged()
            entry = {'run': run, 'method': method}
            report['checks'].append(entry)
            if method == 'explicit_joint_seed':
                # Type 1 checks a zero-distance joint path and seeds virtual state.
                code = controller.arm.set_servo_angle(
                    angle=joints, speed=5.0, mvacc=5.0, is_radian=False,
                    relative=False, wait=False, only_check_type=1,
                )
                result = controller.arm.only_check_result
                entry['seed'] = {'code': code, 'planner_result': result}
                if code != 0 or result != 0:
                    print(json.dumps(entry, indent=2))
                    continue
                check_type = 2  # Check from the seeded state and finish the chain.
            else:
                check_type = 1  # Check from the actual state directly.
            code = controller._send_linear(
                step['pose'], step['speed'], step['mvacc'],
                check_type=check_type, wait=False,
            )
            result = controller.arm.only_check_result
            entry.update(code=code, planner_result=result,
                         passed=(code == 0 and result == 0))
            unchanged()
            print(json.dumps(entry, indent=2))
    report['final_diagnostics'] = controller.arm.get_err_warn_code()
    print('No movement, gripper command, or error reset was executed.')
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
        output = folder / f'planner_initialization_{stamp}.json'
        output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print('Report saved:', output)
