"""
Guided student menu for the robotics assembly demonstration.

Select simulation or real-robot preparation before choosing an activity.
Keep offline practice separate from existing hardware setup tools.

Selecting a mode does not change robot configuration or start motion.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"


def pause():
    """Keep the result visible until the student continues."""

    input("\nPress ENTER to continue...")


def heading(title):
    """Display a consistent screen heading."""

    print("\n" + "=" * 54)
    print(title)
    print("=" * 54)


def run_module(module, *arguments):
    """Run a project activity using the current Python environment."""

    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SOURCE_ROOT)

    if existing:
        environment["PYTHONPATH"] += os.pathsep + existing

    try:
        result = subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )

        if result.returncode:
            print("\nThe activity ended with an error.")
            print("Review the message above before continuing.")

    except KeyboardInterrupt:
        print("\nActivity interrupted.")

    except OSError as error:
        print(f"\nCould not open the activity: {error}")

    pause()


def show_workflow():
    """Explain the complete assembly workflow without operating hardware."""

    heading("HOW THE SYSTEM WORKS")

    steps = [
        (
            "1. Camera Setup",
            "Choose a camera and confirm that both workspaces are visible.",
        ),
        (
            "2. Robot Connections",
            "Identify Robot 1 and Robot 2 and check their connections.",
        ),
        (
            "3. Workspace Calibration",
            "Use four black marker centers to convert pixels to millimeters.",
        ),
        (
            "4. Robot Calibration",
            "Relate positions on each sheet to its robot coordinates.",
        ),
        (
            "5. Part Detection and Orientation",
            "Find the contour, center, and horseshoe opening direction.",
        ),
        (
            "6. Grasp Point",
            "Choose where to hold the part and rotate that point with it.",
        ),
        (
            "7. Gripper and Grasp Height",
            "Teach the final grasp height for the installed fingers.",
        ),
        (
            "8. Home and Placement",
            "Use recorded poses and check the route from the actual start.",
        ),
        (
            "9. Assembly Preview",
            "Place one link, insert the next link, then insert the pin.",
        ),
        (
            "10. Step-by-Step Demonstration",
            "Check each operation before running the complete sequence.",
        ),
    ]

    for title, explanation in steps:
        print(f"\n{title}")
        print(f"   {explanation}")

    print("\nA saved setup can be inspected without recalibrating.")
    print("Simulation does not establish physical grasp or insertion success.")
    pause()


def explore_orientation():
    """Open the working offline activity in either menu."""

    heading("EXPLORE PART ORIENTATION")
    print("This activity uses a saved silhouette.")
    print("It does not require a camera, Docker, or a robot.")
    print()
    print("Click the image window before using its keyboard controls.")
    print("Slider / A / D: rotate")
    print("Z: set practice zero")
    print("Mouse click: select a practice grasp point")
    print("R: reset")
    print("Q / ESC: close")
    print()
    print("Practice zero and grasp point do not change robot calibration.")

    run_module("robot_vision.vision.explore_orientation")


def inspect_saved_setup():
    """Show saved project data without treating file presence as validation."""

    heading("VIEW SAVED SETUP")
    print("Saved values are references, not confirmation of readiness.")

    profiles = [
        ("Robot connection", "config/simulation.json"),
        ("Robot 1 pick", "config/robot1_pick.json"),
        ("Robot 1 transfer", "config/robot1_transfer.json"),
        ("Zone 1 camera", "data/calibration/zone1_camera.json"),
        (
            "Zone 1 reference camera",
            "data/calibration/zone1_reference_camera.json",
        ),
    ]

    while True:
        print()
        for index, (label, relative_path) in enumerate(profiles, 1):
            exists = (PROJECT_ROOT / relative_path).is_file()
            state = "Saved" if exists else "Not found"
            print(f"{index}. {label} [{state}]")

        print("0. Back")
        choice = input("\nSelect a profile to inspect: ").strip()

        if choice == "0":
            return

        if not choice:
            continue

        if not choice.isdigit() or not 1 <= int(choice) <= len(profiles):
            print("Please select an available option.")
            continue

        label, relative_path = profiles[int(choice) - 1]
        path = PROJECT_ROOT / relative_path

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            heading(label.upper())
            print(json.dumps(data, indent=2))
        except (OSError, ValueError) as error:
            print(f"Could not read this profile: {error}")

        pause()


def simulator_status():
    """Check Docker and the named container without starting services."""

    heading("CHECK SIMULATOR SETUP")
    print("Required model: UFACTORY Lite6")
    print("Expected container: uf_software")
    print("This operation only reads Docker status.")

    docker = shutil.which("docker")

    if docker is None:
        print("\nDocker was not found.")
        print("Docker Desktop must be installed for robot simulation.")
        pause()
        return

    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )

        if result.returncode:
            print("\nDocker Engine is not available.")
            print("Open Docker Desktop and wait until it is ready.")
            print(result.stderr.strip()[-1000:])
            pause()
            return

        print("\nDocker Engine:", result.stdout.strip())

        result = subprocess.run(
            [
                docker,
                "container",
                "inspect",
                "--format",
                "{{.State.Status}}",
                "uf_software",
            ],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )

        if result.returncode:
            print("The simulator container could not be inspected.")
            print(result.stderr.strip()[-1000:])
        else:
            print("Container:", result.stdout.strip())
            print("Robot model and simulator readiness are not yet verified.")

    except subprocess.TimeoutExpired:
        print("Docker did not respond in time.")
    except OSError as error:
        print(f"Could not query Docker: {error}")

    pause()


def instructor_tools():
    """Preserve the original setup tools with their existing behavior."""

    tools = [
        (
            "Configure Existing Robot Connection",
            "robot_vision.robot.configure_robot",
        ),
        (
            "Legacy Camera / Workspace Calibration",
            "robot_vision.vision.select_reference",
        ),
        (
            "Teach Grasp Point",
            "robot_vision.vision.select_grasp",
        ),
        (
            "Teach Grasp Height",
            "robot_vision.robot.configure_grasp",
        ),
        (
            "Calibrate Workspace to Robot",
            "robot_vision.robot.configure_workspace",
        ),
        (
            "Start Legacy Live Vision",
            "robot_vision.app",
        ),
    ]

    while True:
        heading("EXISTING INSTRUCTOR TOOLS")
        print("These tools retain their original configuration behavior.")
        print("They may access hardware or overwrite calibration files.")
        print("This submenu is not a password-protected access system.")
        print()

        for index, (label, _) in enumerate(tools, 1):
            print(f"{index}. {label}")

        print("0. Back")
        choice = input("\nSelect an option: ").strip()

        if choice == "0":
            return

        if not choice:
            continue

        if not choice.isdigit() or not 1 <= int(choice) <= len(tools):
            print("Please select an available option.")
            continue

        label, module = tools[int(choice) - 1]
        answer = input(
            f"\nOpen '{label}' with the existing configuration? [y/N]: "
        ).strip().lower()

        if answer == "y":
            run_module(module)


def simulation_menu():
    """Keep offline activities separate from robot simulation readiness."""

    while True:
        heading("SIMULATION")
        print("1. How the System Works")
        print("2. Explore Part Orientation - Offline Practice")
        print("3. View Saved Setup - Read Only")
        print("4. Check Docker Simulator Setup")
        print("5. Start Robot Simulator - Lite6")
        print("6. Preview Pick and Place - Saved Image")
        print("7. Run Pick and Place - Local Simulator Movement")
        print("0. Back")
        print()
        print("Start Robot Simulator opens the local simulator in your browser.")
        print("Pick and place uses a saved image and virtual gripper events.")
        print("Run starts from the verified retreat pose; start the simulator first.")
        print("Link insertion and Robot 2 are not included yet.")
        print("No real-robot execution is available in this menu.")

        choice = input("\nSelect an option: ").strip()

        if choice == "0":
            return
        elif choice == "1":
            show_workflow()
        elif choice == "2":
            explore_orientation()
        elif choice == "3":
            inspect_saved_setup()
        elif choice == "4":
            simulator_status()
        elif choice == "5":
            run_module("robot_vision.robot.start_simulator")
        elif choice == "6":
            run_module("robot_vision.robot.simulator_demo_cycle")
        elif choice == "7":
            run_module("robot_vision.robot.simulator_demo_cycle", "--execute")
        elif choice:
            print("Please select an available option.")


def real_menu():
    """Present preparation tools without enabling an assembly cycle."""

    while True:
        heading("REAL ROBOTS - PREPARATION")
        print("Selecting this menu does not connect or move a robot.")
        print()
        print("1. How the System Works")
        print("2. View Saved Setup - Read Only")
        print("3. Explore Part Orientation - Offline Practice")
        print("4. Existing Instructor Setup Tools")
        print("0. Back")
        print()
        print("Guided two-robot setup: integration pending.")
        print("Assembly execution is not enabled in this menu.")

        choice = input("\nSelect an option: ").strip()

        if choice == "0":
            return
        elif choice == "1":
            show_workflow()
        elif choice == "2":
            inspect_saved_setup()
        elif choice == "3":
            explore_orientation()
        elif choice == "4":
            instructor_tools()
        elif choice:
            print("Please select an available option.")


def main():
    """Select the work area without rewriting any configuration."""

    try:
        while True:
            heading("ROBOTICS ASSEMBLY DEMO")
            print("1. Simulation")
            print("2. Real Robots")
            print("3. How the System Works")
            print("0. Exit")

            choice = input("\nSelect an option: ").strip()

            if choice == "0":
                print("\nDemonstration menu closed.")
                return
            elif choice == "1":
                simulation_menu()
            elif choice == "2":
                real_menu()
            elif choice == "3":
                show_workflow()
            elif choice:
                print("Please select an available option.")

    except (KeyboardInterrupt, EOFError):
        print("\nDemonstration menu closed.")


if __name__ == "__main__":
    main()