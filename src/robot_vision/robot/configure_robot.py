"""
Configure the xArm connection for each student station.

Author: Javier G. Fontanet
"""

import json
from pathlib import Path


def main():

    project_root = Path(__file__).resolve().parents[3]

    config_path = (
        project_root
        / "config"
        / "simulation.json"
    )

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    print()
    print("======================================")
    print("        ROBOT CONNECTION SETUP")
    print("======================================")
    print()
    print("Choose the robot connection:")
    print()
    print("1. Simulator / Docker")
    print("2. Real xArm")
    print()

    while True:

        option = input(
            "Enter 1 or 2: "
        ).strip()

        if option in ("1", "2"):
            break

        print("Please enter 1 or 2.")

    if option == "1":

        config["mode"] = "simulation"
        config["controller_ip"] = "127.0.0.1"

        print()
        print("Simulator selected.")

    else:

        print()
        print("Enter the IP address shown for your xArm.")
        print("Example: 192.168.1.215")
        print()

        robot_ip = input(
            "Robot IP: "
        ).strip()

        if not robot_ip:
            raise ValueError(
                "Robot IP cannot be empty."
            )

        config["mode"] = "real"
        config["controller_ip"] = robot_ip

        print()
        print(
            f"Real robot selected: {robot_ip}"
        )

    config_path.write_text(
        json.dumps(
            config,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("======================================")
    print("       ROBOT SETUP SAVED")
    print("======================================")
    print()
    print(
        "You do not need to enter the robot IP again "
        "unless your station changes."
    )
    print()


if __name__ == "__main__":
    main()