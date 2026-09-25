"""
Configure team-specific grasp height.

Author: Javier G. Fontanet
"""

import json
import math
from pathlib import Path


def main():

    project_root = Path(__file__).resolve().parents[3]

    config_path = (
        project_root
        / "data"
        / "calibration"
        / "grasp_calibration.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Grasp calibration does not exist.\n"
            "Run select_grasp.py first."
        )

    grasp_config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    print()
    print("======================================")
    print("      GRIPPER HEIGHT CALIBRATION")
    print("======================================")
    print()

    print("What is PICK_Z?")
    print()
    print(
        "PICK_Z is the robot height used when the gripper "
        "actually grabs the part."
    )
    print()
    print(
        "Different teams may have different gripper extensions, "
        "so each team must find its own PICK_Z."
    )
    print()
    print("To find PICK_Z:")
    print()
    print("1. Place the part on the work surface.")
    print(
        "2. Use UFACTORY to slowly move the gripper above the part."
    )
    print(
        "3. Lower the gripper until your gripper extension is "
        "at the correct height to grab the part."
    )
    print(
        "4. Read the Z coordinate shown in UFACTORY."
    )
    print(
        "5. Enter that Z value below."
    )
    print()

    print(
        f"Part height: "
        f"{grasp_config['part_height_mm']:.1f} mm"
    )

    current_pick_z = grasp_config.get("pick_z_mm")

    if current_pick_z is None:
        print("Current PICK_Z: not calibrated")
    else:
        print(
            f"Current PICK_Z: "
            f"{current_pick_z:.2f} mm"
        )

    print()

    while True:

        value = input(
            "Enter your PICK_Z value in mm: "
        ).strip()

        try:

            pick_z = float(value)

            if not math.isfinite(pick_z):
                raise ValueError

            break

        except ValueError:

            print(
                "Please enter a valid number."
            )

    grasp_config["pick_z_mm"] = pick_z

    config_path.write_text(
        json.dumps(
            grasp_config,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("======================================")
    print("        PICK_Z CALIBRATION SAVED")
    print("======================================")
    print()
    print(
        f"Your PICK_Z = {pick_z:.2f} mm"
    )
    print()
    print(
        "The robot will use this height when it "
        "moves down to grab the part."
    )
    print()


if __name__ == "__main__":
    main()