"""Release an already supported assembly, then retreat using existing controls."""
import argparse
import json
from datetime import datetime

from robot_vision.robot.robot1_two_link_cycle import (
    PhysicalCouplingController, check_pose, run_block,
)
from robot_vision.robot.test_pick_motion import PROJECT_ROOT


def release_and_retreat(controller, output, record):
    """Opening (two SDK commands) is mandatory before either movement."""
    controller.wait_stationary()
    start = controller.get_pose()
    if start[2] > 190.0 + 1.0:
        raise ValueError("Current Z exceeds the validated 190 mm clearance")
    vertical = list(start)
    vertical[2] = 190.0
    retreat = list(vertical)
    retreat[:2] = [-120.758293, -155.405518]
    plan = [
        {"operation": "gripper", "label": "release_before_retreat",
         "opening": True, "delay_s": 1.0},
        {"operation": "move", "label": "clear_zone2", "pose": vertical,
         "speed": 10.0, "mvacc": 20.0},
        {"operation": "move", "label": "travel_to_retreat", "pose": retreat,
         "speed": 20.0, "mvacc": 20.0},
    ]
    # PhysicalCouplingController.open_gripper sends two immediate commands,
    # checks both returns and waits two seconds. Failure prevents movement.
    actual = run_block(controller, "release_and_retreat", plan, start,
                       record, output)
    check_pose(controller.get_pose(), actual)
    controller.stop_gripper_after_completion()
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
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
    if controller.robot_ip != "192.168.1.185":
        raise RuntimeError("Unexpected Robot 1 IP; no connection issued")
    try:
        controller.connect()
        record["final_pose"] = release_and_retreat(controller, output, record)
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="stopped", error=str(error) or "Interrupted")
        if controller.arm is not None:
            controller.arm.set_state(4)
        raise
    finally:
        controller.disconnect()
        (output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
        print("SAVED:", output)


if __name__ == "__main__":
    main()
