"""Save an operator-confirmed local fit. Never modify calibration or move an arm."""
import argparse
import json
import math
from datetime import datetime

from robot_vision.robot.chain_settings import load_settings, resolve, ROOT, DEFAULT_CONFIG


def record_fitted_pose(trial, actual, robot_ip):
    trial = resolve(trial)
    source = json.loads((trial / 'results.json').read_text())
    if source.get('status') != 'elevated_approach_completed_part_held':
        raise ValueError('Choose an elevated approach trial, not a completed release or rejected run')
    planned = source['targets']['final']
    if len(actual) != 6 or not all(math.isfinite(v) for v in actual):
        raise ValueError('Invalid measured pose')
    xyz = [actual[i] - planned[i] for i in range(3)]
    angles = [(actual[i] - planned[i] + 180) % 360 - 180 for i in range(3, 6)]
    if math.sqrt(sum(v*v for v in xyz)) > 10 or max(abs(v) for v in angles) > 5:
        raise ValueError('Fit differs by over 10 mm / 5 deg; check selected trial and grasp')
    record = {'created_at': datetime.now().astimezone().isoformat(), 'robot_ip': robot_ip,
              'planned_final_pose_mm_deg': planned, 'fitted_pose_mm_deg': list(actual),
              'correction_xyz_mm': xyz, 'correction_orientation_deg': angles,
              'physical_fit_verified_by_operator': True,
              'scope': 'current_destination_and_current_grasp_only',
              'automatic_corrected_insertion_validated': False,
              'source_trial': str(trial), 'destination_geometry': source.get('destination_geometry'),
              'pick_analysis': source.get('pick_analysis')}
    path = trial / f'operator_fitted_pose_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    print('FITTED POSE:', actual)
    print('CORRECTION XYZ [mm]:', xyz)
    print('CORRECTION ORIENTATION [deg]:', angles)
    print('SAVED:', path)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', help='Approach trial directory; defaults to latest elevated approach')
    parser.add_argument('--fitted', action='store_true', help='Operator asserts the held part is currently fitted')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if not args.fitted:
        parser.error('Use --fitted only while the held part is physically fitted; no automatic fit sensing')
    trial = resolve(args.trial) if args.trial else None
    if trial is None:
        for path in sorted((ROOT / 'data/chain_extension_trials').glob('*/results.json'), reverse=True):
            if json.loads(path.read_text()).get('status') == 'elevated_approach_completed_part_held':
                trial = path.parent
                break
    if trial is None:
        raise FileNotFoundError('No elevated approach trial')
    from xarm.wrapper import XArmAPI
    from robot_vision.robot.capture_workspace_robot import read_stationary_pose
    arm = XArmAPI(settings['robot_ip'])
    try:
        actual = read_stationary_pose(arm)
    finally:
        arm.disconnect()
    record_fitted_pose(trial, actual, settings['robot_ip'])


if __name__ == '__main__':
    main()
