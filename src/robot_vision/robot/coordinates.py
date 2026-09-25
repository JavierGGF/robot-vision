from pathlib import Path
import json
import math


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.json"


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def reference_to_robot(x_mm, y_mm, config=None):
    """
    Convert coordinates on the Letter sheet to robot X,Y coordinates.

    Paper coordinate system:
        (0, 0) = TOP-LEFT
        +X      = toward TOP-RIGHT
        +Y      = toward BOTTOM-LEFT
    """

    if config is None:
        config = load_config()

    origin = config.get("reference_origin_robot_xy_mm")

    if origin is None or len(origin) != 2:
        raise RuntimeError(
            "Workspace origin is not configured. "
            "Run configure_workspace.py first."
        )

    origin_x = float(origin[0])
    origin_y = float(origin[1])

    x_axis = config.get("reference_x_axis_robot_per_mm")
    y_axis = config.get("reference_y_axis_robot_per_mm")

    # New calibration method: three taught corners.
    if x_axis is not None and y_axis is not None:

        if len(x_axis) != 2 or len(y_axis) != 2:
            raise RuntimeError(
                "Invalid workspace axis calibration."
            )

        robot_x = (
            origin_x
            + float(x_mm) * float(x_axis[0])
            + float(y_mm) * float(y_axis[0])
        )

        robot_y = (
            origin_y
            + float(x_mm) * float(x_axis[1])
            + float(y_mm) * float(y_axis[1])
        )

        return robot_x, robot_y

    # Backward compatibility with the old two-point calibration.
    rotation_deg = float(
        config.get("reference_rotation_deg", 0.0)
    )

    theta = math.radians(rotation_deg)

    robot_x = (
        origin_x
        + float(x_mm) * math.cos(theta)
        - float(y_mm) * math.sin(theta)
    )

    robot_y = (
        origin_y
        + float(x_mm) * math.sin(theta)
        + float(y_mm) * math.cos(theta)
    )

    return robot_x, robot_y


def robot_to_reference(robot_x, robot_y, config=None):
    """
    Convert robot X,Y coordinates back to Letter-sheet coordinates.

    Primarily useful for testing and calibration verification.
    """

    if config is None:
        config = load_config()

    origin = config.get("reference_origin_robot_xy_mm")
    x_axis = config.get("reference_x_axis_robot_per_mm")
    y_axis = config.get("reference_y_axis_robot_per_mm")

    if origin is None or x_axis is None or y_axis is None:
        raise RuntimeError(
            "Three-point workspace calibration is required "
            "for robot_to_reference()."
        )

    ox = float(origin[0])
    oy = float(origin[1])

    a = float(x_axis[0])
    b = float(y_axis[0])
    c = float(x_axis[1])
    d = float(y_axis[1])

    dx = float(robot_x) - ox
    dy = float(robot_y) - oy

    det = a * d - b * c

    if abs(det) < 1e-9:
        raise RuntimeError(
            "Workspace calibration matrix is singular."
        )

    x_mm = (d * dx - b * dy) / det
    y_mm = (-c * dx + a * dy) / det

    return x_mm, y_mm


def main():
    config = load_config()

    print()
    print("WORKSPACE COORDINATE TEST")
    print()

    width = float(config.get("reference_width_mm", 215.9))
    height = float(config.get("reference_height_mm", 279.4))

    test_points = [
        ("TOP-LEFT", 0.0, 0.0),
        ("TOP-RIGHT", width, 0.0),
        ("BOTTOM-LEFT", 0.0, height),
        ("CENTER", width / 2.0, height / 2.0),
    ]

    for name, x_mm, y_mm in test_points:
        rx, ry = reference_to_robot(x_mm, y_mm, config)

        print(
            f"{name:12s} "
            f"Paper=({x_mm:7.1f}, {y_mm:7.1f}) mm  "
            f"Robot=({rx:8.2f}, {ry:8.2f}) mm"
        )


if __name__ == "__main__":
    main()