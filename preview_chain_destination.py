"""Capture the current Zone 2 opening without connecting to the robot."""
import json
from datetime import datetime

import cv2
from robot_vision.robot.test_pick_motion import (
    PROJECT_ROOT, read_json, capture_frame, save_images,
)
from robot_vision.vision.preview_zone2_horseshoe import analyze


def main():
    calibration = read_json("data/calibration/zone2_camera.json")
    output = PROJECT_ROOT / "data/chain_destination_previews" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    frame = capture_frame(calibration)
    save_images(output, {"camera.png": frame})
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    try:
        preview, detection, geometry = analyze(frame, calibration)
        save_images(output, {"opening_preview.png": preview})
        record = {"status": "candidate_not_verified", "detection": detection,
                  "geometry": geometry, "robot_connected": False,
                  "free_chain_end_verified": False}
        (output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
        print("OPENING DIRECTION [deg]:", geometry["direction_image_deg"])
        print("OPENING CENTER [sheet mm]:", [float(v)/4 for v in geometry["mouth_center_px"]])
        print("Yellow line and magenta cross must mark the FREE end, not the occupied joint.")
        print("SAVED:", output)
        window = "Current chain destination - no robot movement"
        cv2.imshow(window, preview)
        while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(30) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    except Exception as error:
        (output / "results.json").write_text(json.dumps({
            "status": "rejected", "error": str(error), "robot_connected": False,
        }, indent=2) + "\n")
        print("SAVED:", output)
        raise
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
