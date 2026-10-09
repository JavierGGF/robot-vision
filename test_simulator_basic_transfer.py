"""Continue the successful basic pick in the local Docker simulator only.

Use the shared transfer plan and direct Cartesian motion for comparison with
virtual-planning runs. No virtual preflight is performed in this diagnostic.
Firmware collision protection remains enabled. Gripper events are virtual.
"""

import copy
import json
import time
from datetime import datetime

from robot_vision.robot.simulate_pick_place import LocalSimulator
from robot_vision.robot.pick_cycle import build_cycle_plan, load_transfer_profile
from robot_vision.robot.test_pick_motion import PROJECT_ROOT

SOURCE = PROJECT_ROOT / 'data/simulation_diagnostics/basic_pick_20261009_094942_818129.json'


def errors(actual, target):
    """Compare Cartesian positions and wrapped tool orientations."""
    return (max(abs(a-b) for a,b in zip(actual[:3],target[:3])),
            max(abs((a-b+180)%360-180) for a,b in zip(actual[3:],target[3:])))


def ready(controller):
    """Retain the controller's readiness and self-collision requirements."""
    controller._require_motion_ready()
    if controller.arm.mode != 0:
        raise RuntimeError('Position mode 0 is required; no mode change attempted.')
    code, values = controller.arm.get_err_warn_code()
    if code != 0 or values != [0,0]:
        raise RuntimeError(f'Controller diagnostics not clear: {code}, {values}')


def make_plan(saved, transfer):
    """Continue the shared plan exactly after the successful pick lift."""
    if saved.get('status') != 'basic_pick_motion_passed':
        raise RuntimeError('A successful basic-pick record is required.')
    last = saved['execution'][-1]
    if last.get('status') != 'completed' or last.get('label') != 'lift_40_mm':
        raise RuntimeError('The saved pick did not finish its lift.')
    start = last['actual_pose']
    profile = copy.deepcopy(saved['pick_profile'])
    profile['motion']['tool_orientation_deg'] = start[3:6]
    complete = build_cycle_plan(start, start[:2], profile, transfer)
    index = next(i for i,s in enumerate(complete) if s['label']=='raise_for_transfer')
    return start, last['actual_joints'], complete[index:]


def main():
    controller = LocalSimulator()
    record = {'simulation_only': True, 'virtual_preflight_used': False,
              'gripper_mode': 'virtual_events_only', 'physical_validation': False,
              'execution': []}
    outstanding = False
    output = None
    try:
        saved = json.loads(SOURCE.read_text(encoding='utf-8'))
        transfer = load_transfer_profile()
        start, joints, plan = make_plan(saved, transfer)
        record.update(source=str(SOURCE), expected_start=start,
                      transfer_profile=transfer, plan=plan)
        folder = PROJECT_ROOT/'data/simulation_diagnostics'
        folder.mkdir(parents=True,exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output = folder/f'basic_transfer_{stamp}.json'
        controller.connect()
        ready(controller)
        pe,ae=errors(controller.get_pose(),start)
        if pe>1 or ae>2 or max(abs(a-b) for a,b in zip(controller.get_joints(),joints))>1:
            raise RuntimeError('Start differs from the successful pick endpoint; no motion sent.')
        controller.arm.set_only_check_type(0)
        expected=start
        for step in plan:
            ready(controller)
            pe,ae=errors(controller.get_pose(),expected)
            if pe>1 or ae>2:
                raise RuntimeError('Simulator pose changed between steps.')
            entry=dict(step,status='started')
            record['execution'].append(entry)
            output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
            if step['operation']=='gripper':
                print('VIRTUAL GRIPPER:', 'OPEN' if step['opening'] else 'CLOSE')
                time.sleep(step['delay_s'])
            elif step['operation']=='move':
                target=step['pose']
                print(step['label'], ':', [round(v,3) for v in target])
                print('Speed:',step['speed'],'mm/s')
                outstanding=True
                # Direct Cartesian command, with no joint fallback or automatic retry.
                code=controller.arm.set_position(
                    x=target[0],y=target[1],z=target[2],roll=target[3],
                    pitch=target[4],yaw=target[5],radius=-1.0,motion_type=0,
                    relative=False,is_radian=False,speed=step['speed'],
                    mvacc=step['mvacc'],wait=True,timeout=120.0,
                )
                entry['return_code']=code
                if code!=0:
                    raise RuntimeError(f'Linear command failed: {code}')
                controller.wait_stationary()
                ready(controller)
                actual=controller.get_pose()
                entry.update(actual_pose=actual,actual_joints=controller.get_joints())
                pe,ae=errors(actual,target)
                if pe>1 or ae>2:
                    raise RuntimeError(f'Arrival mismatch: {pe:.3f} mm, {ae:.3f} deg')
                outstanding=False
                expected=target
                print('ARRIVAL VERIFIED')
            else:
                raise RuntimeError('Unknown operation.')
            entry['status']='completed'
        record['status']='basic_transfer_motion_passed'
        print('BASIC TRANSFER PASSED: placement and retreat poses verified.')
        print('Gripper events were virtual; part retention was not simulated.')
    except BaseException as error:
        record.update(status='failed',error=str(error))
        print('Transfer stopped:',error)
        if outstanding and controller.arm is not None:
            try:
                record['stop_return_code']=controller.arm.set_state(4)
            except Exception as stop_error:
                record['stop_error']=str(stop_error)
    finally:
        try:
            controller.disconnect()
        finally:
            if output is not None:
                output.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
                print('Report saved:',output)


if __name__=='__main__':
    main()