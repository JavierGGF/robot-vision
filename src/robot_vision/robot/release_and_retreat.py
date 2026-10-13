"""Release an already supported assembly, then retreat using existing controls."""
import argparse
import json
from datetime import datetime

from robot_vision.robot.robot1_two_link_cycle import (
    PhysicalCouplingController, check_pose, run_block,
)
from robot_vision.robot.test_pick_motion import PROJECT_ROOT
from robot_vision.robot.assembly_operations import build_release_plan
from robot_vision.robot.chain_settings import load_settings, DEFAULT_CONFIG


def release_and_retreat(controller, output, record, settings=None):
    settings = load_settings() if settings is None else settings
    """Opening (two SDK commands) is mandatory before either movement."""
    controller.wait_stationary()
    start = controller.get_pose()
    plan = build_release_plan(start, settings["retreat_xy_mm"], settings["transfer_z_mm"])
    # PhysicalCouplingController.open_gripper sends two immediate commands,
    # checks both returns and waits two seconds. Failure prevents movement.
    actual = run_block(controller, "release_and_retreat", plan, start,
                       record, output)
    check_pose(controller.get_pose(), actual)
    controller.stop_gripper_after_completion()
    return actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if not args.execute:
        print("PLAN: open twice -> wait -> vertical Z190 -> retreat XY")
        print("Orientation preserved. No robot connection.")
        return
    output = PROJECT_ROOT / "data/release_retreat_trials" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    record = {"status": "started", "blocks": [],
              "physical_opening_verified": False,
              "external_obstacles_checked": False}
    controller = PhysicalCouplingController()
    if controller.robot_ip != settings["robot_ip"]:
        raise RuntimeError("Unexpected Robot 1 IP; no connection issued")
    try:
        controller.connect()
        record["final_pose"] = release_and_retreat(controller, output, record, settings)
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="stopped", error=str(error) or "Interrupted")
        if controller.arm is not None:
            controller.arm.set_state(4)
        raise
    finally:
        try:
            controller.disconnect()
        finally:
            (output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
            print("SAVED:", output)


if __name__ == "__main__":
    main()
