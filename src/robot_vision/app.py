"""
HTTP interface for vision and simulated target coordinates.

Author: Javier G. Fontanet
"""

from flask import Flask, jsonify

from vision.detection import detect_piece
from robot.coordinates import load_config, reference_to_robot

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        message="Vision application is ready"
    )


@app.get("/detect")
def detect():
    """Detect the object and calculate a simulated approach target."""
    try:
        config = load_config()

        if config["mode"] != "simulation":
            raise ValueError("This endpoint requires simulation mode.")

        result, _, _ = detect_piece()

        x_robot, y_robot = reference_to_robot(
            result["reference_x_mm"],
            result["reference_y_mm"],
            config
        )

        result.update({
            "mode": "simulation",
            "robot_x_mm": x_robot,
            "robot_y_mm": y_robot,
            "robot_z_mm": config["approach_z_mm"],
            "tool_orientation_deg": config["tool_orientation_deg"]
        })

        print(
            "Simulated target: X={:.2f}, Y={:.2f}, Z={:.2f} mm".format(
                result["robot_x_mm"],
                result["robot_y_mm"],
                result["robot_z_mm"]
            ),
            flush=True
        )

        return jsonify(result)

    except (FileNotFoundError, ValueError) as error:
        return jsonify(
            status="error",
            message=str(error)
        ), 422

    except Exception:
        app.logger.exception("Unexpected detection error")
        return jsonify(
            status="error",
            message="Detection failed. Check the application terminal."
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5050,
        debug=False,
        use_reloader=False
    )