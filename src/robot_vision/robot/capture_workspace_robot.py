"""Capture printed fiducial poses with Robot 1; never command robot motion.

Move the robot manually in UFACTORY. Keep the calibration pointer, TCP
configuration, tool orientation and height unchanged for all three markers.
The resulting linear map is usable for displacements. Its absolute origin
is provisional unless the pointer center coincides with the configured TCP.
Existing pick, transfer and camera calibration files are never overwritten.
"""

import argparse
import ipaddress
import json
import math
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk

import numpy as np
from xarm.wrapper import XArmAPI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS = ("TL", "TR", "BL")


def angle_error(first, second):
    return max(abs((a-b+180) % 360-180) for a, b in zip(first[3:6], second[3:6]))


def read_stationary_pose(arm):
    """Read three fresh poses, rejecting motion and controller errors."""
    readings = []
    for _ in range(3):
        code, diagnostics = arm.get_err_warn_code()
        if code != 0 or list(diagnostics) != [0, 0]:
            raise ValueError(f"Controller diagnostics: {code}, {diagnostics}")
        code, state = arm.get_state()
        if code != 0 or state not in (0, 2, 3, 4, 5):
            raise ValueError(f"Robot must be stationary: {code}, {state}")
        code, pose = arm.get_position(is_radian=False)
        if code != 0 or len(pose) < 6 or not all(math.isfinite(v) for v in pose[:6]):
            raise ValueError("Could not read a valid robot pose.")
        readings.append(list(pose[:6]))
        time.sleep(0.15)
    for pose in readings[1:]:
        distance = math.dist(pose[:3], readings[0][:3])
        if distance > 0.2 or angle_error(pose, readings[0]) > 0.2:
            raise ValueError("Robot moved during capture. Stop and capture again.")
    return readings[-1]


def rotation_matrix(pose):
    """SDK roll/pitch/yaw degrees: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = np.radians(pose[3:6])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]]) @ np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]]) @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])


def fit_pointer(poses):
    """Solve p_i + R_i * offset = one fixed physical point."""
    if len(poses) < 6:
        raise ValueError("Capture at least six orientations at the SAME physical point.")
    design = np.vstack([np.column_stack((rotation_matrix(p), -np.eye(3))) for p in poses])
    values = -np.concatenate([np.asarray(p[:3]) for p in poses])
    solution, _, rank, singular = np.linalg.lstsq(design, values, rcond=None)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 1e-12 else float("inf")
    if rank != 6 or condition > 50:
        raise ValueError("Insufficient tilt diversity. Add poses tilted around different axes; yaw alone is insufficient.")
    offset, point = solution[:3], solution[3:]
    errors = [float(np.linalg.norm(np.asarray(p[:3])+rotation_matrix(p) @ offset-point)) for p in poses]
    if max(errors) > 1.0:
        raise ValueError(f"Fixed-point error {max(errors):.2f} mm exceeds 1 mm. Realign or undo inaccurate poses.")
    if np.linalg.norm(offset) > 300:
        raise ValueError("Estimated pointer offset exceeds 300 mm. Check alignment.")
    return {"offset_in_reported_tcp_axes_mm":offset.tolist(), "fixed_point_robot_mm":point.tolist(),
            "max_fit_error_mm":max(errors), "condition_number":condition,
            "rotation_convention":"Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            "status":"fitted_not_independently_validated"}


def compute_mapping(records, calibration, pointer=None):
    """Fit physical sheet directions to measured robot-base directions."""
    sheet = np.asarray(calibration["reference_points_mm"], dtype=float)[[0, 1, 3]]
    poses = np.asarray([r["pose_mm_deg"] for r in records], dtype=float)
    if pointer is not None:
        offset = np.asarray(pointer["offset_in_reported_tcp_axes_mm"])
        poses = poses.copy()
        for index, item in enumerate(records):
            raw = item["pose_mm_deg"]
            poses[index, :3] = np.asarray(raw[:3])+rotation_matrix(raw) @ offset
    if pointer is None and any(angle_error(p, poses[0]) > 1.0 for p in poses[1:]):
        raise ValueError("Tool orientation differs by over 1 deg. Recapture with fixed orientation.")
    if np.ptp(poses[:, 2]) > 2.0:
        raise ValueError("Height differs by over 2 mm. Recapture at a fixed height.")
    sheet_axes = np.column_stack((sheet[1]-sheet[0], sheet[2]-sheet[0]))
    robot_axes = np.column_stack((poses[1, :2]-poses[0, :2], poses[2, :2]-poses[0, :2]))
    matrix = robot_axes @ np.linalg.inv(sheet_axes)
    scales = np.linalg.norm(matrix, axis=0)
    cosine = float(np.dot(matrix[:, 0], matrix[:, 1]) / np.prod(scales)) if np.prod(scales) else 1.0
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 10:
        raise ValueError("Invalid marker geometry. Check marker identities.")
    if np.any(abs(scales-1) > 0.1) or abs(cosine) > 0.15:
        raise ValueError("Measured marker spacing or perpendicularity is inconsistent with the sheet.")
    return {
        "sheet_to_robot_linear_matrix": matrix.tolist(),
        "provisional_sheet_origin_robot_xy_mm": (poses[0, :2]-matrix @ sheet[0]).tolist(),
        "absolute_origin_validated": False,
        "axis_scales": scales.tolist(),
        "axis_cosine": cosine,
        "tool_to_pointer_offset_corrected": pointer is not None,
        "physical_robot_execution_enabled": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", required=True)
    parser.add_argument("--zone", type=int, choices=(1, 2), default=1)
    parser.add_argument("--calibrate-pointer", action="store_true", help="Fit pointer offset at one fixed point before capturing markers.")
    parser.add_argument("--record-only", action="store_true", help="Save raw marker poses at any orientation; do not calculate an uncorrected mapping.")
    args = parser.parse_args()
    address = ipaddress.ip_address(args.ip)
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        parser.error("Use the physical robot IP address.")
    calibration_path = PROJECT_ROOT / f"data/calibration/zone{args.zone}_camera.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("reference_type") != "printed_fiducial_centers" or calibration.get("point_order") != ["TL", "TR", "BR", "BL"] or calibration.get("zone") != args.zone:
        raise ValueError("A matching printed-fiducial camera calibration is required.")
    points = np.asarray(calibration["reference_points_mm"], dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Invalid sheet coordinates.")
    output = PROJECT_ROOT / "data/workspace_calibrations" / f"robot1_zone{args.zone}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "capturing", "created_at": datetime.now().astimezone().isoformat(),
        "robot_ip": args.ip, "robot_name": "Robot 1", "zone": args.zone,
        "camera_calibration": calibration, "marker_order": list(LABELS), "markers": [],
        "alignment": "pointer_cross_center_over_printed_fiducial_center",
        "cycle_profiles_updated": False, "pointer_poses": [],
    }

    def save():
        pending = output / "robot_mapping.pending.json"
        pending.write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
        pending.replace(output / "robot_mapping.json")

    arm = None
    window = None
    try:
        arm = XArmAPI(args.ip)
        save()
        window = tk.Tk()
        window.title(f"Robot 1 - Zone {args.zone} Marker Capture")
        window.geometry("760x440")
        window.configure(background="#10223A")
        title = tk.StringVar()
        message = tk.StringVar(value="Move manually in UFACTORY. Return to this window to capture.")
        tk.Label(window, textvariable=title, font=("Arial", 25, "bold"), fg="white", bg="#10223A", wraplength=720).pack(pady=24)
        tk.Label(window, text="Cross center over BLACK SQUARE CENTER\nPointer mode: keep CROSS at one fixed physical point.\nSPACE / ENTER: capture   BACKSPACE: undo   ESC: close", font=("Arial", 15), fg="white", bg="#10223A").pack(pady=12)
        tk.Label(window, textvariable=message, font=("Arial", 13), fg="#8FE0D0", bg="#10223A", wraplength=720).pack(pady=20)
        tk.Label(window, text="READ ONLY: no movement, enable, mode or state commands", font=("Arial", 11), fg="white", bg="#10223A").pack()

        phase = {"pointer": args.calibrate_pointer}
        def refresh():
            if phase["pointer"]:
                title.set(f"Pointer calibration: {len(record['pointer_poses'])} poses")
                return
            count = len(record["markers"])
            title.set(f"Next: {LABELS[count]} - Zone {args.zone}" if count < 3 else "All three markers captured")

        def capture(event=None):
            if phase["pointer"]:
                try:
                    pose = read_stationary_pose(arm)
                    record["pointer_poses"].append(pose)
                    save()
                    total = len(record["pointer_poses"])
                    if total >= 6:
                        record["pointer_calibration"] = fit_pointer(record["pointer_poses"])
                        phase["pointer"] = False
                        save()
                        message.set("Pointer fitted. Now align at TL and SPACE. Different tool orientations accepted.")
                        print("POINTER FIT:", record["pointer_calibration"], flush=True)
                    else:
                        message.set(f"Pose {total} saved. Tilt tool around another axis, REALIGN cross to SAME TL point and same physical height; SPACE.")
                except Exception as error:
                    message.set(str(error)+" BACKSPACE: undo. SPACE: add another pose.")
                    print("Pointer check:",error,flush=True)
                refresh()
                return
            count = len(record["markers"])
            if count == 3:
                return
            try:
                pose = read_stationary_pose(arm)
                if not args.record_only and "pointer_calibration" not in record and count and angle_error(pose, record["markers"][0]["pose_mm_deg"]) > 1.0:
                    raise ValueError("Keep the TL tool orientation (within 1 deg). Point not saved.")
                if not args.record_only and "pointer_calibration" not in record and count and abs(pose[2]-record["markers"][0]["pose_mm_deg"][2]) > 2.0:
                    raise ValueError("Keep the TL height (within 2 mm). Point not saved.")
                label = LABELS[count]
                record["markers"].append({"marker": label, "pose_mm_deg": pose, "captured_at": datetime.now().astimezone().isoformat()})
                record.pop("mapping", None)
                record["status"] = "capturing"
                save()
                print(f"{label}: {pose}\nSaved: {output / 'robot_mapping.json'}", flush=True)
                if len(record["markers"]) == 3:
                    if args.record_only:
                        record["status"] = "raw_poses_saved_pointer_compensation_pending"
                        save()
                        message.set(f"RAW POSES SAVED: {output}. Pointer compensation pending. No mapping activated.")
                        refresh()
                        return
                    record["mapping"] = compute_mapping(record["markers"], calibration, record.get("pointer_calibration"))
                    record["status"] = "directions_calibrated_absolute_anchor_pending"
                    save()
                    message.set(f"SAVED: {output}\nNext: teach the actual grasp anchor. Existing profiles unchanged.")
                else:
                    message.set(f"{label} SAVED. Move to {LABELS[count+1]}; align cross at the same physical height; then SPACE here.")
            except Exception as error:
                message.set(str(error)+"  BACKSPACE: undo last point.")
                print(f"Capture/check rejected: {error}", flush=True)
            refresh()

        def undo(event=None):
            if phase["pointer"]:
                if record["pointer_poses"]:
                    record["pointer_poses"].pop()
                    save()
                message.set("Last pointer pose removed. Realign at the SAME fixed point and capture.")
                refresh()
                return
            if record["markers"]:
                removed = record["markers"].pop()["marker"]
                record.pop("mapping", None)
                record["status"] = "capturing"
                save()
                message.set(f"Removed {removed}. Align and capture again.")
            refresh()

        if args.calibrate_pointer:
            message.set("Capture first pose at TL. Then vary tilts and realign cross to EXACT same 3D point for each pose. At least 6 poses.")
        refresh()
        window.bind("<space>", capture)
        window.bind("<Return>", capture)
        window.bind("<BackSpace>", undo)
        window.bind("<Escape>", lambda event: window.destroy())
        print(f"Capture window ready. Records: {output}", flush=True)
        window.mainloop()
    finally:
        if arm is not None:
            arm.disconnect()
        print(f"Saved points remain at: {output / 'robot_mapping.json'}", flush=True)


if __name__ == "__main__":
    main()
