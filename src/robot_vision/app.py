"""
Live vision server for the xArm vision project.

The webcam runs continuously.
The program detects the part, estimates its orientation,
calculates the taught grasp point, and provides the target
coordinates to UFACTORY through /detect.

Author: Javier G. Fontanet
"""

import threading

import cv2
import numpy as np
from flask import Flask, jsonify

from vision.detection import load_calibration, process_frame
from vision.orientation import calculate_grasp, load_grasp_config
from robot.coordinates import load_config, reference_to_robot


app = Flask(__name__)

latest_result = None
result_lock = threading.Lock()

MIN_ORIENTATION_MATCH = 0.50


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        message="Vision application is ready"
    )


@app.get("/detect")
def detect():

    with result_lock:
        result = (
            latest_result.copy()
            if latest_result is not None
            else None
        )

    if result is None:
        return jsonify(
            status="error",
            message="No valid grasp target detected"
        ), 422

    try:

        config = load_config()
        grasp_config = load_grasp_config()

        pick_z = grasp_config.get("pick_z_mm")

        if pick_z is None:
            return jsonify(
                status="error",
                message=(
                    "PICK_Z has not been calibrated. "
                    "Run configure_grasp.py first."
                )
            ), 422

        # Convert taught grasp point to robot coordinates
        x_robot, y_robot = reference_to_robot(
            result["grasp_x_mm"],
            result["grasp_y_mm"],
            config
        )

        result.update({

            "mode":
                config["mode"],

            "robot_x_mm":
                x_robot,

            "robot_y_mm":
                y_robot,

            "approach_z_mm":
                config["approach_z_mm"],

            "pick_z_mm":
                float(pick_z),

            "tool_orientation_deg":
                config["tool_orientation_deg"]
        })

        print(
            "Grasp target: "
            "X={:.2f}, Y={:.2f}, "
            "Approach Z={:.2f}, Pick Z={:.2f} mm".format(
                result["robot_x_mm"],
                result["robot_y_mm"],
                result["approach_z_mm"],
                result["pick_z_mm"]
            ),
            flush=True
        )

        return jsonify(result)

    except Exception as error:

        app.logger.exception(
            "Target coordinate error"
        )

        return jsonify(
            status="error",
            message=str(error)
        ), 500


def run_flask():

    app.run(
        host="0.0.0.0",
        port=5050,
        debug=False,
        use_reloader=False
    )


def main():

    global latest_result

    calibration = load_calibration()

    camera_index = calibration["camera_index"]

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        calibration["image_width_px"]
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        calibration["image_height_px"]
    )

    server_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    server_thread.start()

    print()
    print("======================================")
    print("         XARM LIVE VISION")
    print("======================================")
    print()
    print("Camera connected.")
    print("Vision server running.")
    print("Press Q to quit.")
    print()

    pixels_per_mm = 4.0

    matrix = np.asarray(
        calibration["pixel_to_mm_matrix"],
        dtype=np.float64
    )

    scale_matrix = np.diag([
        pixels_per_mm,
        pixels_per_mm,
        1.0
    ])

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                continue

            try:

                result, rectified, mask = process_frame(
                    frame,
                    calibration
                )

                grasp_result = calculate_grasp(
                    result,
                    mask
                )

                result.update(
                    grasp_result
                )

                match_score = result[
                    "orientation_match"
                ]

                grasp_x_px = round(
                    result["grasp_x_mm"]
                    * pixels_per_mm
                )

                grasp_y_px = round(
                    result["grasp_y_mm"]
                    * pixels_per_mm
                )

                cv2.drawMarker(
                    rectified,
                    (
                        grasp_x_px,
                        grasp_y_px
                    ),
                    (255, 0, 255),
                    cv2.MARKER_TILTED_CROSS,
                    30,
                    3
                )

                cv2.putText(
                    rectified,
                    "GRASP",
                    (
                        grasp_x_px + 10,
                        grasp_y_px
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    rectified,
                    f"Angle: {result['part_angle_deg']:.1f} deg",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    rectified,
                    f"Match: {match_score:.2f}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                if match_score >= MIN_ORIENTATION_MATCH:

                    with result_lock:
                        latest_result = result.copy()

                    cv2.putText(
                        rectified,
                        "READY",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                else:

                    with result_lock:
                        latest_result = None

                    cv2.putText(
                        rectified,
                        "ORIENTATION NOT RELIABLE",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

            except Exception:

                with result_lock:
                    latest_result = None

                rectified = cv2.warpPerspective(
                    frame,
                    scale_matrix @ matrix,
                    (
                        round(
                            calibration["reference_width_mm"]
                            * pixels_per_mm
                        ),
                        round(
                            calibration["reference_height_mm"]
                            * pixels_per_mm
                        )
                    )
                )

                cv2.putText(
                    rectified,
                    "Waiting for valid part...",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "xArm Live Vision",
                rectified
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()