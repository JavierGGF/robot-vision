"""
Calibrate the position of the vision workspace relative to the xArm base.

The student teaches two corners of the US Letter reference sheet:

1. Top-left
2. Top-right

From these two robot coordinates, the program calculates:
- the location of the sheet relative to the robot base
- the rotation of the sheet relative to the robot

Author: Javier G. Fontanet
"""

import json
import math
from pathlib import Path


def read_coordinate(name):

    while True:

        value = input(
            f"Enter robot {name} coordinate in mm: "
        ).strip()

        try:
            value = float(value)

            if not math.isfinite(value):
                raise ValueError

            return value

        except ValueError:

            print(
                "Please enter a valid number."
            )


def main():

    project_root = Path(__file__).resolve().parents[3]

    config_path = (
        project_root
        / "config"
        / "simulation.json"
    )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    width_mm = config[
        "reference_width_mm"
    ]

    print()
    print("======================================")
    print("     WORKSPACE / ROBOT CALIBRATION")
    print("======================================")
    print()
    print(
        "This step tells the vision system where the "
        "US Letter workspace is located relative to the robot."
    )
    print()
    print("DO NOT move the paper during this calibration.")
    print()
    print("STEP 1")
    print(
        "Use UFACTORY to move the center of the robot tool "
        "directly above the TOP-LEFT corner of the paper."
    )
    print()
    print(
        "You only need to record X and Y. "
        "Keep the robot at a safe Z height."
    )
    print()

    input(
        "Press ENTER when the robot is above the TOP-LEFT corner..."
    )

    print()

    x1 = read_coordinate("X")
    y1 = read_coordinate("Y")

    print()
    print("--------------------------------------")
    print()
    print("STEP 2")
    print(
        "Now move the center of the robot tool directly above "
        "the TOP-RIGHT corner of the paper."
    )
    print()

    input(
        "Press ENTER when the robot is above the TOP-RIGHT corner..."
    )

    print()

    x2 = read_coordinate("X")
    y2 = read_coordinate("Y")

    # ---------------------------------------------------------
    # Calculate sheet orientation
    # ---------------------------------------------------------

    dx = x2 - x1
    dy = y2 - y1

    measured_width = math.hypot(
        dx,
        dy
    )

    rotation_deg = math.degrees(
        math.atan2(
            dy,
            dx
        )
    )

    print()
    print("======================================")
    print("       CALIBRATION RESULT")
    print("======================================")
    print()
    print(
        f"Paper origin in robot coordinates:"
    )
    print(
        f"X = {x1:.2f} mm"
    )
    print(
        f"Y = {y1:.2f} mm"
    )
    print()
    print(
        f"Paper rotation = "
        f"{rotation_deg:.2f} deg"
    )
    print()
    print(
        f"Expected paper width = "
        f"{width_mm:.1f} mm"
    )
    print(
        f"Measured robot distance = "
        f"{measured_width:.1f} mm"
    )
    print()

    # ---------------------------------------------------------
    # Basic calibration check
    # ---------------------------------------------------------

    width_error = abs(
        measured_width - width_mm
    )

    if width_error > 15:

        print(
            "WARNING:"
        )
        print(
            "The measured distance between the two corners "
            "is significantly different from the paper width."
        )
        print(
            "Check that the correct two corners were selected."
        )
        print()

    # ---------------------------------------------------------
    # Save configuration
    # ---------------------------------------------------------

    config[
        "reference_origin_robot_xy_mm"
    ] = [
        x1,
        y1
    ]

    config[
        "reference_rotation_deg"
    ] = rotation_deg

    config_path.write_text(
        json.dumps(
            config,
            indent=2
        ),
        encoding="utf-8"
    )

    print("======================================")
    print("      WORKSPACE CALIBRATION SAVED")
    print("======================================")
    print()
    print(
        "The vision system can now convert positions "
        "on the paper into robot coordinates."
    )
    print()


if __name__ == "__main__":
    main()