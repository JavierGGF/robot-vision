"""
Convert reference-plane coordinates to simulated robot coordinates.

Author: Javier G. Fontanet
"""

import json
import math
from pathlib import Path


def load_config():
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "config" / "simulation.json"

    return json.loads(config_path.read_text(encoding="utf-8"))


def reference_to_robot(x_mm, y_mm, config):
    """Transform reference XY coordinates into robot-base XY coordinates."""
    if not all(math.isfinite(value) for value in (x_mm, y_mm)):
        raise ValueError("Coordinates must be finite.")

    width = config["reference_width_mm"]
    height = config["reference_height_mm"]

    if not (0 <= x_mm <= width and 0 <= y_mm <= height):
        raise ValueError("Point is outside the reference rectangle.")

    origin_x, origin_y = config["reference_origin_robot_xy_mm"]
    angle = math.radians(config["reference_rotation_deg"])

    x_robot = (
        origin_x
        + x_mm * math.cos(angle)
        - y_mm * math.sin(angle)
    )
    y_robot = (
        origin_y
        + x_mm * math.sin(angle)
        + y_mm * math.cos(angle)
    )

    return x_robot, y_robot


def main():
    config = load_config()

    # Temporary test values; vision will supply these later.
    x_robot, y_robot = reference_to_robot(54.88, 72.82, config)
    z_robot = config["approach_z_mm"]
    roll, pitch, yaw = config["tool_orientation_deg"]

    print("SIMULATION ONLY — assumed reference placement")
    print(
        f"Target position: X={x_robot:.2f}, "
        f"Y={y_robot:.2f}, Z={z_robot:.2f} mm"
    )
    print(f"Tool orientation: roll={roll}, pitch={pitch}, yaw={yaw} deg")
    print("No movement command sent.")


if __name__ == "__main__":
    main()