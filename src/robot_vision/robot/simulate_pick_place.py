"""Replay saved vision, check a route, or execute it in the local simulator.

Default: offline plan starting from the recorded retreat pose.
--check-route: ask only the local Docker Lite6 to check a plan from its
actual current pose.
--execute: execute checked Cartesian moves locally; gripper events are virtual.
Never read the real-robot connection configuration.
"""

import argparse
import copy
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2

from robot_vision.robot.controller import Lite6Controller
from robot_vision.robot.pick_cycle import (
    build_cycle_plan, load_pick_profile, load_transfer_profile,
    movement_segments,
)
from robot_vision.robot.start_simulator import (
    CONTAINER, REQUIRED_PORTS, docker_command, internal_ports,
    require_local_docker,
)
from robot_vision.robot.test_pick_motion import (
    PROJECT_ROOT, load_sheet_calibration, project_path, save_images,
)
from robot_vision.robot.test_rotated_pick import calculate_rotated_target


class LocalSimulator(Lite6Controller):
    """Reuse guarded planning with an explicit loopback-only connection."""

    def __init__(self):
        # Do not call the parent constructor: it reads the real profile.
        self.arm = None
        self.config = {"mode": "simulation", "controller_ip": "127.0.0.1"}
        self.mode = "simulation"
        self.robot_ip = "127.0.0.1"
        self.last_path_check = None
        self.command_trace = []

    def connect(self):
        """Require the expected local container before connecting."""
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("Start the local Lite6 simulator first.")
        require_local_docker(docker)
        data = json.loads(docker_command(docker, "inspect", CONTAINER))[0]
        if not data["State"]["Running"]:
            raise RuntimeError("The simulator container is stopped.")
        if data["HostConfig"].get("NetworkMode") == "host":
            raise RuntimeError("Published container ports are required.")
        bindings = data["NetworkSettings"].get("Ports") or {}
        for port in REQUIRED_PORTS:
            if not any(
                item.get("HostPort") == str(port)
                and item.get("HostIp", "") in ("", "127.0.0.1", "0.0.0.0")
                for item in (bindings.get(f"{port}/tcp") or [])
            ):
                raise RuntimeError(f"Simulator port mapping missing: {port}")
        if not REQUIRED_PORTS.issubset(internal_ports(docker)):
            raise RuntimeError("Simulator services are not ready.")
        super().connect()
        if self.arm.axis != 6 or self.arm.device_type != 9:
            raise RuntimeError("Local controller does not identify as Lite6.")

        # SDK 1.18.4 parses serial characters as digits in joint checks.
        # This simulator reports a whitespace-only serial. Represent that
        # missing value as None in this connection's SDK cache, so the SDK
        # uses the verified Lite6 device type and retains its joint limits.
        # This does not write a serial number to the controller.
        serial = self.arm.sn
        if isinstance(serial, str) and serial and not serial.strip():
            self.arm._arm._robot_sn = None
            print("Simulator compatibility: blank SDK serial normalized.")

    def send_traced_linear(self, pose, speed, mvacc, check_type, wait):
        """Observe the exact SDK transport arguments without changing them."""
        transport = self.arm._arm.arm_cmd
        original = transport.move_line_common
        entry = {
            "requested": {"pose_mm_deg": list(pose), "speed": speed,
                          "acceleration": mvacc, "check_type": check_type,
                          "wait": wait},
            "before_pose": self.get_pose(),
            "before_joints": self.get_joints(),
            "before_state": self.arm.get_state(),
            "sdk_cached_mvtime": self.arm._arm._mvtime,
            "transport_calls": [],
        }
        self.command_trace.append(entry)

        def observe(*args, **kwargs):
            # Copy only numeric motion arguments; preserve SDK behavior exactly.
            call = {"arguments": list(args),
                    "keywords": {k: v for k, v in kwargs.items()
                                 if k != "feedback_key"}}
            entry["transport_calls"].append(call)
            reply = original(*args, **kwargs)
            call["raw_response"] = list(reply)
            return reply

        transport.move_line_common = observe
        try:
            code = Lite6Controller._send_linear(
                self, pose, speed, mvacc, check_type, wait
            )
            entry["api_return_code"] = code
            entry["planner_result"] = self.arm.only_check_result
            return code
        except Exception as error:
            entry["exception"] = str(error)
            raise
        finally:
            # Restore the original method on every exit, including exceptions.
            transport.move_line_common = original
            try:
                entry["after_pose"] = self.get_pose()
                entry["after_joints"] = self.get_joints()
                entry["after_diagnostics"] = self.arm.get_err_warn_code()
            except Exception as error:
                entry["diagnostic_error"] = str(error)

    def wait_stationary(self, timeout=10.0, stable_seconds=1.0):
        """Require idle state and stable joint readings before planning.

        This is a simulator synchronization guard, not a retry of a failed
        trajectory. It never clears warnings, enables motion, or resets state.
        """
        deadline = time.monotonic() + timeout
        anchor = None
        stable_since = None
        while time.monotonic() < deadline:
            self._require_connection()
            code, diagnostics = self.arm.get_err_warn_code()
            if code != 0 or diagnostics[0] != 0:
                raise RuntimeError(f"Simulator diagnostics rejected: {code}, {diagnostics}")
            code, state = self.arm.get_state()
            if code != 0 or state not in (0, 1, 2):
                raise RuntimeError(f"Simulator state rejected: {code}, {state}")
            joints = self.get_joints()
            now = time.monotonic()
            if state != 2:
                anchor = None
                stable_since = None
            elif anchor is None or max(abs(a-b) for a, b in zip(joints, anchor)) > 0.01:
                anchor = list(joints)
                stable_since = now
            elif now - stable_since >= stable_seconds:
                return {"joints_deg": joints, "state": state,
                        "stable_seconds": stable_seconds}
            time.sleep(0.1)
        raise RuntimeError("Simulator did not settle before the planning timeout.")

    def check_linear_route(self, segments):
        """Check a continuous virtual route with the firmware path checker.

        The SDK documents only_check_type as checking self-collision, joint
        limits, Cartesian limits, and overspeed. Do not interleave independent
        endpoint IK queries with this virtual chain: their returned solution
        is not the trajectory's joint configuration. No protection is disabled.
        """
        if not segments:
            raise ValueError("A route requires at least one segment.")
        prepared = [
            {"label": str(step.get("label", f"segment_{i+1}")),
             "pose": self._validate_pose(step["pose"]),
             "speed": self._validate_positive(step["speed"], "Speed"),
             "mvacc": self._validate_positive(step["mvacc"], "Acceleration")}
            for i, step in enumerate(segments)
        ]
        self.last_stationary_check = self.wait_stationary()
        self._require_motion_ready()
        report = {
            "status": "checking",
            "method": "actual_joint_seed_then_cartesian_path",
            "start_pose_mm_deg": self.get_pose(),
            "start_joints_deg": self.get_joints(),
            "segments": [],
            "external_obstacles_checked": False,
        }
        self.last_path_check = report
        try:
            # Start the virtual chain at the actual six joint angles, including
            # their winding (do not normalize 233 degrees to -127 degrees).
            # This is a zero-distance planning query, never a joint movement.
            code = self.arm.set_servo_angle(
                angle=report["start_joints_deg"], speed=5.0, mvacc=5.0,
                is_radian=False, relative=False, wait=False, only_check_type=1,
            )
            result = self.arm.only_check_result
            report["actual_joint_seed"] = {
                "joints_deg": report["start_joints_deg"],
                "return_code": code, "planner_result": result,
                "motion_executed": False,
            }
            if code != 0 or result != 0:
                raise RuntimeError(f"Actual-joint seed rejected: {code}, {result}")
            for i, step in enumerate(prepared):
                self._require_motion_ready()
                kind = 2 if i == len(prepared)-1 else 3
                entry = dict(step, check_type=kind)
                report["segments"].append(entry)
                code = self._send_linear(
                    step["pose"], step["speed"], step["mvacc"],
                    check_type=kind, wait=False,
                )
                result = self.arm.only_check_result
                entry.update(return_code=code, planner_result=result)
                if code != 0 or result != 0:
                    raise RuntimeError(
                        f"Path rejected at '{step['label']}': "
                        f"code={code}, planner_result={result}."
                    )
                print("PATH CHECK OK:", step["label"])

            # Planning must leave the actual robot stationary and unchanged.
            actual = self.get_joints()
            if max(abs(a-b) for a,b in zip(actual, report["start_joints_deg"])) > 0.01:
                raise RuntimeError("Actual joints changed during planning.")
            report["status"] = "passed"
            return report
        except Exception as error:
            report["status"] = "failed"
            report["error"] = str(error)
            raise

    # This program has no movement interface, even if called accidentally.
    def move_linear(self, *args, **kwargs):
        raise RuntimeError("This activity performs planning only.")

    def move_to_target(self, *args, **kwargs):
        raise RuntimeError("This activity performs planning only.")

    def open_gripper(self):
        raise RuntimeError("Gripper commands are disabled in this activity.")

    def close_gripper(self):
        raise RuntimeError("Gripper commands are disabled in this activity.")

    def _send_linear(self, pose, speed, mvacc, check_type, wait):
        if check_type not in (1, 2, 3):
            raise RuntimeError("Only non-moving planner queries are allowed.")
        return self.send_traced_linear(
            pose, speed, mvacc, check_type=check_type, wait=False
        )



class ExecutableLocalSimulator(LocalSimulator):
    """Explicit execution opt-in; preserve checks and forbid joint fallback."""

    def move_linear(self, *args, **kwargs):
        # The shared controller checks each segment immediately before motion.
        result = Lite6Controller.move_linear(self, *args, **kwargs)
        # Confirm a stable idle state after completion before the next operation.
        self.last_stationary_check = self.wait_stationary()
        return result

    def _send_linear(self, pose, speed, mvacc, check_type, wait):
        # Connection validation remains inherited from LocalSimulator.
        if self.robot_ip != "127.0.0.1" or self.mode != "simulation":
            raise RuntimeError("Execution requires the local simulator.")
        if check_type not in (0, 1, 2, 3):
            raise ValueError("Unsupported planner mode.")
        if check_type == 0 and wait is not True:
            raise ValueError("Execution must wait for completion.")
        return self.send_traced_linear(
            pose, speed, mvacc, check_type, wait
        )


def verify_pose(controller, expected):
    """Stop on position drift or failure to reach the intended endpoint."""
    actual = controller.get_pose()
    position_error = max(abs(a - b) for a, b in zip(actual[:3], expected[:3]))
    angle_error = max(
        abs((a - b + 180.0) % 360.0 - 180.0)
        for a, b in zip(actual[3:6], expected[3:6])
    )
    if position_error > 1.0 or angle_error > 2.0:
        raise RuntimeError(
            f"Pose mismatch: {position_error:.3f} mm, {angle_error:.3f} deg"
        )
    return actual


def execute_plan(controller, plan, start, record, output):
    """Execute checked moves and record virtual gripper events separately."""
    expected = list(start)
    record["execution"] = []
    record["gripper_mode"] = "virtual_events_only"
    record["part_retention_simulated"] = False
    for step in plan:
        controller._require_motion_ready()
        verify_pose(controller, expected)
        entry = {"label": step["label"], "status": "started"}
        record["execution"].append(entry)
        # Persist progress before a potentially blocking movement.
        (output / "results.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        if step["operation"] == "move":
            print("SIMULATOR MOVE:", step["label"])
            record["motion_requested"] = True
            controller.move_linear(
                step["pose"], speed=step["speed"],
                mvacc=step["mvacc"], wait=True,
            )
            record["motion_executed"] = True
            entry["actual_pose"] = verify_pose(controller, step["pose"])
            entry["actual_joints"] = controller.get_joints()
            entry["stationary_check"] = controller.last_stationary_check
            expected = list(step["pose"])
        elif step["operation"] == "gripper":
            # Docker motion simulation does not validate contact or retention.
            entry["virtual_gripper"] = "OPEN" if step["opening"] else "CLOSE"
            print("VIRTUAL GRIPPER:", entry["virtual_gripper"])
            time.sleep(step["delay_s"])
        else:
            raise ValueError(f"Unknown operation: {step['operation']}")
        entry["status"] = "completed"


def main():
    """Reprocess evidence, build the shared plan, and optionally check it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default="data/demo_trials/20261007_165106_644344/camera.png",
        help="Saved image from the calibrated Zone 1 camera view.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-route", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--above-only", action="store_true")
    parser.add_argument(
        "--output-dir", help="Optional destination for this trial's evidence."
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = (
        project_path(args.output_dir) if args.output_dir
        else PROJECT_ROOT / "data/simulation_trials" / stamp
    )
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "started_at": datetime.now().astimezone().isoformat(),
        "mode": ("local_simulator_execute" if args.execute else
                 "local_simulator_check" if args.check_route else "offline_plan"),
        "motion_executed": False,
        "above_only": args.above_only,
        "physical_validation": False,
    }
    controller = None
    try:
        pick = load_pick_profile()
        transfer = load_transfer_profile()
        calibration = load_sheet_calibration("data/calibration/zone1_camera.json")
        reference = load_sheet_calibration("data/calibration/zone1_reference_camera.json")
        image_path = project_path(args.image)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        record.update({
            "source_image": str(image_path), "pick_profile": pick,
            "transfer_profile": transfer, "camera_calibration": calibration,
            "reference_calibration": reference,
        })
        save_images(output, {"camera.png": frame})
        analysis, images = calculate_rotated_target(frame, calibration, reference, pick)
        record["analysis"] = analysis
        save_images(output, images)
        runtime = copy.deepcopy(pick)
        runtime["motion"]["tool_orientation_deg"] = analysis["target_orientation_deg"]

        # Offline preview assumes Home. A simulator check uses its actual pose.
        start = list(transfer["retreat_pose_mm_deg"])
        record["start_source"] = "recorded_retreat_assumption"
        if args.check_route or args.execute:
            controller = ExecutableLocalSimulator() if args.execute else LocalSimulator()
            controller.connect()
            start = controller.get_pose()
            record["start_source"] = "actual_simulator_pose"
            record["start_joints_deg"] = controller.get_joints()
            record["firmware"] = controller.arm.version

        plan = build_cycle_plan(
            start, analysis["target_xy_mm"], runtime, transfer,
            above_only=args.above_only,
        )
        record.update({"start_pose": start, "plan": plan})
        print("\nLOCAL SIMULATOR EXECUTION PLAN" if args.execute
              else "\nSIMULATION PLAN - NO MOVEMENT")
        print("Starting pose source:", record["start_source"])
        print("Detected target X,Y:", analysis["target_xy_mm"])
        print("Target orientation:", analysis["target_orientation_deg"])
        for index, step in enumerate(plan, 1):
            if step["operation"] == "move":
                pose = [round(v, 3) for v in step["pose"]]
                print(f"{index:02d}. {step['label']}: {pose}; speed={step['speed']}")
            else:
                action = "OPEN" if step["opening"] else "CLOSE"
                print(f"{index:02d}. {step['label']}: {action} (virtual event)")

        # Retain the intended plan even if the controller rejects it.
        (output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
        if controller:
            record["path_check"] = controller.check_linear_route(movement_segments(plan))
            record["status"] = "simulator_route_check_passed"
            if args.execute:
                print("\nROUTE CHECK PASSED. Executing in LOCAL SIMULATOR only.")
                execute_plan(controller, plan, start, record, output)
                record["status"] = "simulator_cycle_completed"
                print("SIMULATOR CYCLE COMPLETED. Physical grasp is not validated.")
            else:
                print("\nSIMULATOR ROUTE CHECK PASSED. No motion was executed.")
        else:
            record["status"] = "offline_plan_generated_not_validated"
            print("\nOffline plan generated. Kinematics and collisions were not checked.")
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        # Preserve the actual stopped configuration for comparison with planning.
        if controller and controller.arm is not None:
            try:
                record["stopped_pose"] = controller.get_pose()
                record["stopped_joints"] = controller.get_joints()
                record["stopped_diagnostics"] = controller.arm.get_err_warn_code()
            except Exception as diagnostic_error:
                record["diagnostic_error"] = str(diagnostic_error)
        if controller:
            record["path_check"] = controller.last_path_check
        print("\nSimulation stopped:", error)
        raise
    finally:
        try:
            if controller:
                record["command_trace"] = controller.command_trace
                controller.disconnect()
        finally:
            record["finished_at"] = datetime.now().astimezone().isoformat()
            (output / "results.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            print("Trial saved:", output)


if __name__ == "__main__":
    main()