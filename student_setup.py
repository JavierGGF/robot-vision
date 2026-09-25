"""
Student setup menu for the xArm Vision Project.

Students do not need to edit any code.
Use this menu to configure and run the vision system.

Author: Javier G. Fontanet
"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_script(relative_path):

    script_path = PROJECT_ROOT / relative_path

    if not script_path.exists():
        print()
        print(f"ERROR: File not found:")
        print(script_path)
        input("\nPress ENTER to continue...")
        return

    print()
    print("--------------------------------------")
    print(f"Starting: {script_path.name}")
    print("--------------------------------------")
    print()

    subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        cwd=PROJECT_ROOT
    )

    print()
    input("Press ENTER to return to the setup menu...")


def start_live_vision():

    script_path = (
        PROJECT_ROOT
        / "src"
        / "robot_vision"
        / "app.py"
    )

    print()
    print("======================================")
    print("        STARTING LIVE VISION")
    print("======================================")
    print()
    print("The camera window will remain open.")
    print()
    print("When the part is detected correctly,")
    print("the screen should display READY.")
    print()
    print("Keep this program running while")
    print("using UFACTORY.")
    print()
    print("Press Q in the camera window to stop.")
    print()

    subprocess.run(
        [
            sys.executable,
            str(script_path)
        ],
        cwd=PROJECT_ROOT
    )


def show_menu():

    print()
    print("======================================")
    print("       XARM VISION STUDENT SETUP")
    print("======================================")
    print()
    print("Complete the setup in this order:")
    print()
    print("1. Configure Robot / IP")
    print("2. Calibrate Camera + Letter Workspace")
    print("3. Teach Grasp Point")
    print("4. Calibrate PICK_Z")
    print("5. Calibrate Workspace to Robot")
    print("6. Start Live Vision")
    print()
    print("0. Exit")
    print()


def main():

    while True:

        show_menu()

        choice = input(
            "Select an option (0-6): "
        ).strip()

        if choice == "1":

            run_script(
                "src/robot_vision/robot/configure_robot.py"
            )

        elif choice == "2":

            run_script(
                "src/robot_vision/vision/select_reference.py"
            )

        elif choice == "3":

            run_script(
                "src/robot_vision/vision/select_grasp.py"
            )

        elif choice == "4":

            run_script(
                "src/robot_vision/robot/configure_grasp.py"
            )

        elif choice == "5":

            run_script(
                "src/robot_vision/robot/configure_workspace.py"
            )

        elif choice == "6":

            start_live_vision()

        elif choice == "0":

            print()
            print("Setup closed.")
            break

        else:

            print()
            print("Please select a number from 0 to 6.")


if __name__ == "__main__":
    main()