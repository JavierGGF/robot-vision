"""Shared operation plans; no robot connection or SDK calls."""
import math


def build_release_plan(start, retreat_xy=(-120.758293, -155.405518), clearance=190.0):
    """Include release explicitly before lift and retreat, preserving orientation."""
    if len(start) != 6 or len(retreat_xy) != 2 or not all(math.isfinite(v) for v in [*start, *retreat_xy, clearance]):
        raise ValueError('Invalid release/retreat geometry')
    if not 0 < clearance <= 190 or not 0 < start[2] <= clearance + 0.5:
        raise ValueError('Clearance must be at least current Z and at most 190 mm')
    vertical = list(start)
    vertical[2] = clearance
    retreat = list(vertical)
    retreat[:2] = retreat_xy
    return [
        {'operation': 'gripper', 'label': 'release_in_zone2', 'opening': True, 'delay_s': 1.0},
        {'operation': 'move', 'label': 'clear_zone2', 'pose': vertical, 'speed': 10.0, 'mvacc': 20.0},
        {'operation': 'move', 'label': 'travel_to_retreat', 'pose': retreat, 'speed': 20.0, 'mvacc': 20.0},
    ]
