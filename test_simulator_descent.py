"""Send one checked local-simulator descent and observe actual completion."""

import json
import time
from datetime import datetime
from robot_vision.robot.simulate_pick_place import ExecutableLocalSimulator
from robot_vision.robot.test_pick_motion import PROJECT_ROOT

SOURCE = PROJECT_ROOT / 'data/simulation_trials/20261008_213316_604881/results.json'
controller = ExecutableLocalSimulator()
report = {'simulation_only': True, 'command_sent': False, 'samples': []}

try:
    saved = json.loads(SOURCE.read_text(encoding='utf-8'))
    step = next(s for s in saved['plan'] if s['label'] == 'descend_to_pick')
    controller.connect()
    controller.wait_stationary()
    controller._require_motion_ready()
    start = controller.get_pose()
    report.update(start_pose=start, mode=controller.arm.mode, target=step)
    print('CONTROLLER MODE:', controller.arm.mode)
    if controller.arm.mode != 0:
        raise RuntimeError('Expected position mode 0; no mode change attempted.')
    if max(abs(start[i]-step['pose'][i]) for i in (0,1)) > 1.0 or abs(start[2]-200.0)>1.0:
        raise RuntimeError('Simulator is not at the expected above-pick position.')
    if max(abs((start[i]-step['pose'][i]+180)%360-180) for i in (3,4,5)) > 2.0:
        raise RuntimeError('Tool orientation differs from the saved test.')

    report['path_check'] = controller.check_linear_route([step])
    controller._require_motion_ready()
    code, diagnostic = controller.arm.get_err_warn_code()
    if code != 0 or diagnostic != [0,0]:
        raise RuntimeError(f'Controller diagnostic is not clear: {code}, {diagnostic}')
    if max(abs(a-b) for a,b in zip(controller.get_joints(),report['path_check']['start_joints_deg'])) > 0.01:
        raise RuntimeError('Start joints changed after checking.')

    # Send exactly one Cartesian command; never resend on timeout or rejection.
    pose=step['pose']
    print('Sending ONE checked descent to Z=',pose[2])
    report['command_sent']=True
    code=controller.arm.set_position(
        x=pose[0],y=pose[1],z=pose[2],roll=pose[3],pitch=pose[4],yaw=pose[5],
        speed=step['speed'],mvacc=step['mvacc'],mvtime=0,
        radius=-1.0,motion_type=0,relative=False,is_radian=False,
        only_check_type=0,wait=False,
    )
    report['send_return_code']=code
    if code != 0:
        raise RuntimeError(f'Command rejected: {code}')

    # Observe independently of the SDK's wait=True completion mechanism.
    began=time.monotonic()
    stable_since=None
    while time.monotonic()-began < 45.0:
        elapsed=time.monotonic()-began
        code,state=controller.arm.get_state()
        ecode,diagnostic=controller.arm.get_err_warn_code()
        actual=controller.get_pose()
        sample={'elapsed_s':round(elapsed,2),'state':state,'state_code':code,
                'diagnostics':diagnostic,'pose':actual}
        report['samples'].append(sample)
        print(f"{elapsed:5.1f}s  state={state}  Z={actual[2]:.3f}  diagnostics={diagnostic}")
        if code != 0 or ecode != 0 or state not in (0,1,2) or diagnostic != [0,0]:
            raise RuntimeError('Controller status changed; stopping observation.')
        pe=max(abs(a-b) for a,b in zip(actual[:3],pose[:3]))
        ae=max(abs((a-b+180)%360-180) for a,b in zip(actual[3:],pose[3:]))
        if state==2 and pe<=1.0 and ae<=2.0:
            stable_since=elapsed if stable_since is None else stable_since
            if elapsed-stable_since>=1.0:
                report['status']='target_reached'
                print('DESCENT COMPLETED AND POSITION VERIFIED.')
                break
        else:
            stable_since=None
        time.sleep(0.5)
    else:
        report['status']='arrival_timeout'
        # A timeout can leave a command outstanding. Stop only this local simulator.
        report['stop_return_code']=controller.arm.set_state(4)
        raise RuntimeError('Target not reached within 45 seconds; simulator stop requested.')
except Exception as error:
    report['error']=str(error)
    print('Test stopped:',error)
    # Do not leave a submitted movement unobserved on an abnormal exit.
    if report['command_sent'] and report.get('status')!='target_reached':
        if 'stop_return_code' not in report:
            report['stop_return_code']=controller.arm.set_state(4)
finally:
    try:
        controller.disconnect()
    finally:
        folder=PROJECT_ROOT/'data/simulation_diagnostics'
        folder.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output=folder/f'descent_observation_{stamp}.json'
        output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('Report saved:',output)
