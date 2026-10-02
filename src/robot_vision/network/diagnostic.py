"""
Network and robot communication diagnostic.

This module checks the basic communication path between the
vision computer and the xArm robot.

It does NOT modify network settings and does NOT move the robot.
"""

import json
import platform
import socket
import subprocess
from pathlib import Path

from xarm.wrapper import XArmAPI


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.json"


def load_config():
    """Load the robot configuration file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def run_command(command):
    """Run a system command and return its output."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_default_route():
    """Return the interface used for the default network route."""
    if platform.system() != "Darwin":
        return "unknown"

    output = run_command(["route", "-n", "get", "default"])

    for line in output.splitlines():
        line = line.strip()

        if line.startswith("interface:"):
            return line.split(":", 1)[1].strip()

    return "unknown"


def get_interface_ip(interface):
    """Return the IPv4 address assigned to a macOS interface."""
    if platform.system() != "Darwin":
        return None

    output = run_command(["ipconfig", "getifaddr", interface])

    return output if output else None


def find_robot_interface(robot_ip):
    """
    Determine which local interface macOS would use
    to communicate with the robot.
    """
    if platform.system() != "Darwin":
        return None

    output = run_command(["route", "-n", "get", robot_ip])

    for line in output.splitlines():
        line = line.strip()

        if line.startswith("interface:"):
            return line.split(":", 1)[1].strip()

    return None


def ping_robot(robot_ip):
    """Check whether the robot responds to a single ping."""
    if platform.system() == "Darwin":
        command = ["ping", "-c", "1", "-W", "1000", robot_ip]
    else:
        command = ["ping", "-c", "1", robot_ip]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_internet():
    """
    Perform a simple DNS/network connectivity check.

    No data are sent to the robot and no configuration is changed.
    """
    try:
        socket.gethostbyname("google.com")
        return True
    except OSError:
        return False


def check_robot_sdk(robot_ip):
    """
    Connect to the xArm controller and read basic information.

    This function does NOT enable motion and does NOT move the robot.
    """
    arm = None

    try:
        arm = XArmAPI(robot_ip, is_radian=False)

        if not arm.connected:
            return False, None, None, None

        state = arm.get_state()
        errors = arm.get_err_warn_code()
        joints = arm.get_servo_angle(is_radian=False)

        return True, state, errors, joints

    except Exception as exc:
        return False, None, None, str(exc)

    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass


def main():
    config = load_config()

    mode = config.get("mode", "unknown")
    robot_ip = config.get("controller_ip")

    print()
    print("Robot Network Diagnostic")
    print("------------------------")
    print(f"Mode                : {mode}")
    print(f"Configured robot IP : {robot_ip}")

    if not robot_ip:
        print("RESULT              : FAIL - robot IP not configured")
        return

    robot_interface = find_robot_interface(robot_ip)

    if robot_interface:
        ethernet_ip = get_interface_ip(robot_interface)
    else:
        ethernet_ip = None

    print(f"Robot interface     : {robot_interface or 'not found'}")
    print(f"Computer robot IP   : {ethernet_ip or 'not found'}")

    default_interface = get_default_route()

    print(f"Default route       : {default_interface}")

    internet_ok = check_internet()
    print(f"Internet/DNS        : {'OK' if internet_ok else 'NOT AVAILABLE'}")

    ping_ok = ping_robot(robot_ip)
    print(f"Robot ping          : {'OK' if ping_ok else 'FAILED'}")

    if not ping_ok:
        print("xArm SDK connection : NOT TESTED")
        print("RESULT              : FAIL")
        return

    sdk_ok, state, errors, joints = check_robot_sdk(robot_ip)

    print(f"xArm SDK connection : {'OK' if sdk_ok else 'FAILED'}")

    if sdk_ok:
        print(f"Robot state         : {state}")
        print(f"Robot error/warn    : {errors}")

        if isinstance(joints, tuple) and joints[0] == 0:
            print(
                "Robot joints (deg)  : "
                + str([round(value, 2) for value in joints[1][:6]])
            )

    if ping_ok and sdk_ok:
        print("RESULT              : READY")
    else:
        print("RESULT              : FAIL")


if __name__ == "__main__":
    main()