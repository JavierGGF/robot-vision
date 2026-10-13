"""Load explicit chain-trial references without editing saved calibration."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = 'config/chain_assembly.json'


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_settings(path=DEFAULT_CONFIG):
    settings = json.loads(resolve(path).read_text())
    if settings.get('schema_version') != 1:
        raise ValueError('Unsupported chain settings schema')
    for key in ('robot_ip', 'pick_reference', 'coupling_reference', 'zone2_markers'):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ValueError(f'Missing chain setting: {key}')
    for key in ('transfer_z_mm', 'max_insertion_rotation_deg', 'max_approach_rotation_deg',
                'max_opening_shift_mm', 'preview_match_shift_mm', 'preview_match_rotation_deg', 'coupling_speed_mm_s'):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid chain setting: {key}')
    if settings['transfer_z_mm'] > 190 or not settings['max_insertion_rotation_deg'] <= settings['max_approach_rotation_deg'] <= 20:
        raise ValueError('Chain settings exceed this trial clearance/rotation range')
    if settings['max_opening_shift_mm'] > 100 or settings['coupling_speed_mm_s'] > 2:
        raise ValueError('Chain settings exceed this trial shift/speed range')
    retreat = settings.get('retreat_xy_mm')
    if not isinstance(retreat, list) or len(retreat) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in retreat):
        raise ValueError('Invalid retreat XY')
    if settings.get('apply_local_fit_automatically') is not False:
        raise ValueError('Local operator fit cannot be applied as a global correction')
    return settings
