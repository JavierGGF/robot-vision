"""Offline checks only. Never import the xArm SDK or connect to hardware."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from datetime import datetime

import numpy as np

ROOT = Path(os.environ.get('ROBOT_TEST_ROOT', Path(__file__).resolve().parents[1]))
SRC = ROOT / 'src/robot_vision'
sys.path.insert(0, str(ROOT / 'src'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load('chain_geometry_test', SRC / 'vision/chain_targets.py')
operations = load('chain_operations_test', SRC / 'robot/assembly_operations.py')
pick_cycle = load('pick_cycle_test', SRC / 'robot/pick_cycle.py')
settings = json.loads((ROOT / 'config/chain_assembly.json').read_text())
fixture = json.loads((Path(__file__).parent / 'fixtures/chain_oct13.json').read_text())

chain_ast = ast.parse((SRC / 'robot/chain_assembly.py').read_text())
planner = next(n for n in chain_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'assembly_plan')
planner_globals = {'copy': copy, 'build_cycle_plan': pick_cycle.build_cycle_plan,
                   'build_release_plan': operations.build_release_plan}
exec(compile(ast.Module(body=[planner], type_ignores=[]), '<planner-test>', 'exec'), planner_globals)


class BaseFake:
    def open_gripper(self):
        self.opens += 1
        self.events.append('open')
        if self.opens == self.fail_open:
            raise RuntimeError('Opening rejected')
        return 0

    def close_gripper(self):
        self.closes += 1
        self.events.append('close')
        if self.closes == self.fail_close:
            raise RuntimeError('Closing rejected')
        return 0


runner_ast = ast.parse((SRC / 'robot/robot1_two_link_cycle.py').read_text())
controller_class = next(n for n in runner_ast.body if isinstance(n, ast.ClassDef) and n.name == 'PhysicalCouplingController')
methods = [n for n in controller_class.body if isinstance(n, ast.FunctionDef) and n.name in ('open_gripper', 'close_gripper')]
fake_class = ast.ClassDef(name='RedundantController', bases=[ast.Name(id='BaseFake', ctx=ast.Load())], keywords=[], body=methods, decorator_list=[])
clock = types.SimpleNamespace(sleep=lambda _: None, monotonic=lambda: 1.0)
runner_globals = {'BaseFake': BaseFake, 'time': clock, 'datetime': datetime, 'math': math,
                  'save_record': lambda *a: None, 'confirm': lambda *a: None,
                  'movement_segments': pick_cycle.movement_segments}
run_functions = [n for n in runner_ast.body if isinstance(n, ast.FunctionDef) and n.name in ('check_pose', 'run_block')]
exec(compile(ast.fix_missing_locations(ast.Module(body=[fake_class, *run_functions], type_ignores=[])), '<runner-test>', 'exec'), runner_globals)


class Fake(runner_globals['RedundantController']):
    def __init__(self, fail_open=0, fail_close=0, reject_route=False):
        self.pose = [250., -100., 86., 180., -5., 10.]
        self.opens = self.closes = 0
        self.fail_open, self.fail_close = fail_open, fail_close
        self.reject_route = reject_route
        self.events = []
        self.last_path_check = {}
        self.arm = types.SimpleNamespace(set_state=lambda state: self.events.append('stop') or 0)

    def wait_stationary(self): pass
    def _require_motion_ready(self): pass
    def get_pose(self): return list(self.pose)
    def get_joints(self): return [0.] * 6
    def check_linear_route(self, segments):
        if self.reject_route: raise RuntimeError('Route rejected')
        return {'status': 'passed'}
    def move_linear(self, pose, **kwargs):
        self.events.append('move')
        self.pose = list(pose)
        return 0


class ChainTests(unittest.TestCase):
    def test_zero_change_and_translation(self):
        A = np.diag([-1., 1.]); origin = np.array([350., -215.]); q = np.array([100., 100.])
        p = [247., -87., 86., -177., -5., -5.]
        self.assertTrue(np.allclose(geometry.compensate(p, A, origin, q, q, 0, 0), p))
        q1 = q + [5., -3.]
        transformed = geometry.compensate(p, A, origin, q, q1, 0, 0)
        self.assertTrue(np.allclose(transformed[:2], np.asarray(p[:2]) + A @ (q1-q)))

    def test_rotation_independently_in_robot_plane(self):
        A = np.diag([-1., 1.]); origin = np.array([350., -215.]); q = np.array([100., 100.])
        p = [247., -87., 86., -177., -5., -5.]
        result = geometry.compensate(p, A, origin, q, q, 90, -90)
        pivot = origin + A @ q
        expected = pivot + np.array([[0., 1.], [-1., 0.]]) @ (np.asarray(p[:2])-pivot)
        self.assertTrue(np.allclose(result[:2], expected))
        self.assertAlmostEqual(result[5], -95.)

    def test_published_oct13_target_regression(self):
        targets, diagnostic = geometry.opening_targets(fixture['reference_geometry'], fixture['current_geometry'],
                                                       fixture['markers'], fixture['calibration'], fixture['poses'])
        for name in targets:
            self.assertTrue(np.allclose(targets[name], fixture['expected_targets'][name], atol=1e-8))
        self.assertAlmostEqual(diagnostic['rotation_sheet_deg'], -15.401, places=2)

    def test_invalid_calibration_rejected(self):
        bad = copy.deepcopy(fixture['calibration']); bad['camera_index'] += 1
        with self.assertRaises(ValueError): geometry.displacement_matrix(fixture['markers'], bad)

    def test_nonfinite_pose_rejected(self):
        with self.assertRaises(ValueError): geometry.compensate([float('nan'), 0, 0, 0, 0, 0], np.eye(2), [0,0], [0,0], [0,0], 0, 0)

    def test_plan_release_and_approach_only(self):
        profile = json.loads((ROOT / settings['pick_reference'] / 'pick_profile.json').read_text())
        plan = planner_globals['assembly_plan']([0,0,190.00001,180,0,0], [33,-269], profile,
                                                fixture['expected_targets']['entry'], fixture['expected_targets']['final'], settings)
        labels = [step['label'] for step in plan]
        release = labels.index('release_in_zone2')
        self.assertEqual(labels[release-1:release+3], ['couple_next_link','release_in_zone2','clear_zone2','travel_to_retreat'])
        self.assertTrue(all(step['pose'][2] <= 190 for step in plan if step['operation'] == 'move'))
        stop = labels.index('above_entry')
        self.assertNotIn('release_in_zone2', labels[:stop+1])
        self.assertNotIn('couple_next_link', labels[:stop+1])

    def execute(self, controller, plan):
        record = {'blocks': []}
        with contextlib.redirect_stdout(io.StringIO()):
            runner_globals['run_block'](controller, 'test', plan, controller.get_pose(), record, Path('.'))

    def test_two_open_commands_before_retreat(self):
        c = Fake(); self.execute(c, operations.build_release_plan(c.get_pose()))
        self.assertEqual(c.events, ['open','open','move','move'])

    def test_first_or_second_open_failure_prevents_movement(self):
        for ordinal in (1, 2):
            c = Fake(fail_open=ordinal)
            with self.assertRaises(RuntimeError): self.execute(c, operations.build_release_plan(c.get_pose()))
            self.assertNotIn('move', c.events)
            self.assertEqual(c.events[-1], 'stop')

    def test_route_failure_prevents_release(self):
        c = Fake(reject_route=True)
        with self.assertRaises(RuntimeError): self.execute(c, operations.build_release_plan(c.get_pose()))
        self.assertEqual(c.events, ['stop'])

    def test_close_failure_preserves_grip_and_prevents_lift(self):
        for ordinal in (1, 2):
            c = Fake(fail_close=ordinal)
            plan = [{'operation':'gripper','label':'close_gripper','opening':False,'delay_s':1},
                    {'operation':'move','label':'lift','pose':[250,-100,126,180,-5,10],'speed':10,'mvacc':20}]
            with self.assertRaises(RuntimeError): self.execute(c, plan)
            self.assertNotIn('open', c.events)
            self.assertNotIn('move', c.events)


if __name__ == '__main__':
    unittest.main()
