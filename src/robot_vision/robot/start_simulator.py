"""
Start and verify the local UFACTORY Lite6 simulator.

Inspect internal listening ports before connecting.
Never modify real-robot configuration or request robot motion.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


CONTAINER = "uf_software"
HOST = "127.0.0.1"
REQUIRED_PORTS = {502, 18333, 30001, 30002, 30003}


def docker_command(docker, *arguments, timeout=15):
    """Execute one bounded Docker operation."""

    result = subprocess.run(
        [docker, *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or "Docker command failed."
        )
    return result.stdout.strip()


def require_local_docker(docker):
    """Reject remote Docker contexts."""

    endpoint = os.environ.get("DOCKER_HOST", "")
    if os.environ.get("DOCKER_CONTEXT") or not endpoint:
        context = docker_command(docker, "context", "show")
        data = json.loads(
            docker_command(docker, "context", "inspect", context)
        )
        endpoint = data[0]["Endpoints"]["docker"]["Host"]

    if not endpoint.startswith(("unix://", "npipe://")):
        raise RuntimeError("Select a local Docker Desktop context.")


def engine_ready(docker):
    """Read Docker Engine status without changing it."""

    try:
        docker_command(
            docker, "info", "--format", "{{.ServerVersion}}", timeout=5
        )
        return True
    except (RuntimeError, subprocess.TimeoutExpired):
        return False


def open_docker():
    """Open an installed Docker Desktop application."""

    if sys.platform == "darwin":
        subprocess.run(["open", "-a", "Docker"], check=True)
        return

    if sys.platform == "win32":
        candidates = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Docker/Docker/Docker Desktop.exe",
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Programs/DockerDesktop/Docker Desktop.exe",
        ]
        for path in candidates:
            if path.is_file():
                subprocess.Popen([str(path)])
                return

    raise RuntimeError("Open Docker Desktop manually, then try again.")


def internal_ports(docker):
    """Read listening TCP ports inside the Linux container."""

    output = docker_command(
        docker,
        "exec",
        CONTAINER,
        "sh",
        "-c",
        "cat /proc/net/tcp /proc/net/tcp6",
    )

    ports = set()
    for line in output.splitlines():
        fields = line.split()
        # Linux TCP state 0A means LISTEN.
        if len(fields) >= 4 and fields[3] == "0A":
            ports.add(int(fields[1].rsplit(":", 1)[1], 16))
    return ports


def active_sessions(docker):
    """Ignore dead screen entries from earlier container runs."""

    output = docker_command(
        docker,
        "exec",
        CONTAINER,
        "sh",
        "-c",
        "screen -ls 2>/dev/null || true",
    )
    return [
        line.strip()
        for line in output.splitlines()
        if (
            "xarm_controller_screen" in line
            or "xarm_studio_screen" in line
        )
        and ("(Detached)" in line or "(Attached)" in line)
    ]


def verify_lite6():
    """Read controller identity with a bounded child process."""

    probe = """
import json
from xarm.wrapper import XArmAPI

arm = None
try:
    arm = XArmAPI("127.0.0.1", is_radian=False)
    if not arm.connected:
        raise RuntimeError("Local controller is not connected.")

    # Lite6 identifies itself as axis=6 and device_type=9.
    # TYPE1300 in the SDK connection message is a different field.
    if arm.axis != 6 or arm.device_type != 9:
        raise RuntimeError(
            f"Expected Lite6; received axis={arm.axis}, "
            f"device_type={arm.device_type}."
        )

    code, pose = arm.get_position(is_radian=False)
    if code != 0:
        raise RuntimeError(f"Position read failed: {code}")

    code, diagnostics = arm.get_err_warn_code()
    if code != 0 or diagnostics[0] != 0:
        raise RuntimeError(
            f"Controller diagnostic: code={code}, values={diagnostics}"
        )

    print(json.dumps({
        "model": "Lite6",
        "axis": arm.axis,
        "device_type": arm.device_type,
        "firmware": arm.version,
        "pose": pose,
        "diagnostics": diagnostics
    }, indent=2))
finally:
    if arm is not None:
        arm.disconnect()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Controller verification exceeded 20 seconds. "
            "The verification process was stopped."
        ) from None

    if result.returncode:
        details = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(
            "Controller verification failed:\n" + details[-1800:]
        )

    print(result.stdout.strip())


def main():
    """Start simulator services without sending movement commands."""

    print("\nSTART LOCAL LITE6 SIMULATOR")
    print("Real-robot configuration will not be changed.")

    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Install Docker Desktop before continuing.")

    require_local_docker(docker)

    if not engine_ready(docker):
        print("Opening Docker Desktop...")
        open_docker()
        deadline = time.monotonic() + 60

        while not engine_ready(docker):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Docker is still starting. Try again when it is ready."
                )
            print("Waiting for Docker Engine...")
            time.sleep(3)

    data = json.loads(
        docker_command(docker, "inspect", CONTAINER)
    )[0]

    if data["HostConfig"].get("NetworkMode") == "host":
        raise RuntimeError("Published container ports are required.")

    bindings = data["HostConfig"].get("PortBindings") or {}
    for port in REQUIRED_PORTS:
        entries = bindings.get(f"{port}/tcp") or []
        if not any(
            item.get("HostPort") == str(port)
            and item.get("HostIp", "") in ("", "0.0.0.0", "127.0.0.1")
            for item in entries
        ):
            raise RuntimeError(f"Missing local port mapping: {port}")

    state = data["State"]
    if state.get("Paused") or state.get("Restarting"):
        raise RuntimeError("The container is paused or restarting.")

    if not state["Running"]:
        print("Starting container...")
        docker_command(docker, "start", CONTAINER, timeout=30)

    ports = internal_ports(docker)
    sessions = active_sessions(docker)
    print("Internal listening ports:", sorted(ports))

    if not sessions and not (ports & REQUIRED_PORTS):
        print("Starting Lite6 services...")
        # The vendor script removes dead sessions before starting.
        docker_command(
            docker,
            "exec",
            "-d",
            CONTAINER,
            "/bin/bash",
            "/xarm_scripts/xarm_start.sh",
            "6",
            "9",
        )
    else:
        print("Existing activity detected; checking without restarting.")

    deadline = time.monotonic() + 60
    while True:
        ports = internal_ports(docker)
        missing = REQUIRED_PORTS - ports

        if not missing:
            break

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Services did not start. Missing internal ports: "
                f"{sorted(missing)}. No automatic restart was attempted."
            )

        print("Waiting for internal services:", sorted(missing))
        time.sleep(3)

    print("Verifying Lite6 controller...")
    verify_lite6()

    print("\nLITE6 SIMULATOR CONNECTED")
    print("No motion or gripper command was issued.")
    print("Simulator firmware may differ from the physical robot.")
    print("Physical trajectory validation remains a separate task.")

    url = f"http://{HOST}:18333"
    print("Opening:", url)
    webbrowser.open(url)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nSimulator setup stopped: {error}")
        sys.exit(1)