from pathlib import Path
import json
import math


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.json"


def read_xy(label):
    print()
    print(label)
    print("Use UFACTORY to move the TCP to this corner of the Letter sheet.")
    print("Then read the robot X and Y coordinates.")

    while True:
        try:
            x = float(input("Robot X [mm]: ").strip())
            y = float(input("Robot Y [mm]: ").strip())
            return x, y
        except ValueError:
            print("Please enter valid numbers.")


def main():
    print()
    print("======================================")
    print("LETTER SHEET -> ROBOT CALIBRATION")
    print("======================================")
    print()
    print("The Letter sheet must remain fixed during calibration.")
    print()
    print("You will teach three corners:")
    print("  1. TOP-LEFT")
    print("  2. TOP-RIGHT")
    print("  3. BOTTOM-LEFT")
    print()
    print("Move the CENTER OF THE TCP above each corner.")
    print("Only X and Y are used for this calibration.")
    print()

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    width_mm = float(config.get("reference_width_mm", 215.9))
    height_mm = float(config.get("reference_height_mm", 279.4))

    print(f"Reference sheet size: {width_mm:.1f} mm x {height_mm:.1f} mm")

    input("\nPress ENTER to begin...")

    x_tl, y_tl = read_xy("1. TOP-LEFT corner")
    x_tr, y_tr = read_xy("2. TOP-RIGHT corner")
    x_bl, y_bl = read_xy("3. BOTTOM-LEFT corner")

    dx_x = x_tr - x_tl
    dx_y = y_tr - y_tl

    dy_x = x_bl - x_tl
    dy_y = y_bl - y_tl

    measured_width = math.hypot(dx_x, dx_y)
    measured_height = math.hypot(dy_x, dy_y)

    if measured_width < 1.0:
        raise RuntimeError("TOP-LEFT and TOP-RIGHT are too close.")

    if measured_height < 1.0:
        raise RuntimeError("TOP-LEFT and BOTTOM-LEFT are too close.")

    # Robot displacement produced by 1 mm movement on the paper X axis
    x_axis_robot = [
        dx_x / width_mm,
        dx_y / width_mm,
    ]

    # Robot displacement produced by 1 mm movement on the paper Y axis
    y_axis_robot = [
        dy_x / height_mm,
        dy_y / height_mm,
    ]

    rotation_deg = math.degrees(math.atan2(dx_y, dx_x))

    # Check approximately how perpendicular the taught axes are
    dot = (
        x_axis_robot[0] * y_axis_robot[0]
        + x_axis_robot[1] * y_axis_robot[1]
    )

    x_scale = math.hypot(*x_axis_robot)
    y_scale = math.hypot(*y_axis_robot)

    determinant = (
        x_axis_robot[0] * y_axis_robot[1]
        - x_axis_robot[1] * y_axis_robot[0]
    )

    print()
    print("======================================")
    print("CALIBRATION RESULTS")
    print("======================================")
    print(f"Measured top edge:  {measured_width:.1f} mm")
    print(f"Expected top edge:  {width_mm:.1f} mm")
    print()
    print(f"Measured left edge: {measured_height:.1f} mm")
    print(f"Expected left edge: {height_mm:.1f} mm")
    print()
    print(f"Paper X-axis angle in robot coordinates: {rotation_deg:.2f} deg")
    print(f"X scale: {x_scale:.4f}")
    print(f"Y scale: {y_scale:.4f}")
    print(f"Axis dot product: {dot:.4f}")
    print(f"Transform determinant: {determinant:.4f}")

    if abs(measured_width - width_mm) > 15.0:
        print()
        print("WARNING:")
        print("The measured sheet width differs significantly from a Letter sheet.")
        print("Check the taught TOP-LEFT and TOP-RIGHT points.")

    if abs(measured_height - height_mm) > 15.0:
        print()
        print("WARNING:")
        print("The measured sheet height differs significantly from a Letter sheet.")
        print("Check the taught TOP-LEFT and BOTTOM-LEFT points.")

    if abs(dot) > 0.15:
        print()
        print("WARNING:")
        print("The taught X and Y axes are not close to perpendicular.")
        print("Check the three taught corner positions.")

    if abs(determinant) < 0.2:
        raise RuntimeError(
            "Invalid workspace calibration. "
            "The three taught points do not define a valid workspace."
        )

    config["reference_origin_robot_xy_mm"] = [x_tl, y_tl]
    config["reference_x_axis_robot_per_mm"] = x_axis_robot
    config["reference_y_axis_robot_per_mm"] = y_axis_robot

    # Keep this for display / backward compatibility.
    config["reference_rotation_deg"] = rotation_deg

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("Workspace calibration saved successfully.")
    print(f"Saved to: {CONFIG_PATH}")
    print()
    print("The Letter sheet must remain in this position while the system is used.")


if __name__ == "__main__":
    main()