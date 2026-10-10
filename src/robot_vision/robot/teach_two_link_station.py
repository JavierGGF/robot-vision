"""Teach current two-link station poses. Read-only robot connection.

All movement and gripper operations remain manual in UFACTORY.
This tool saves new profiles; it never activates or executes them.
"""
import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
import numpy as np
from xarm.wrapper import XArmAPI
from robot_vision.robot.capture_workspace_robot import read_stationary_pose, angle_error

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGES = [
    ("retreat", "Open the gripper, clear the first link manually, and park outside the camera view. Capture the stationary retreat pose."),
    ("second_entry", "Manually pick the SECOND link. Align its opposite end just BEFORE the first link's horseshoe, without contact. Capture the entry pose."),
    ("second_final", "Manually approach slowly and fit the second link into the first horseshoe. Keep the gripper closed. Capture the final pose. Never force the fit."),
    ("transfer_height", "Release only when both links are supported. Manually clear the parts and reach a height with clearance for transfer. Capture this pose; only its Z is used."),
]


def repair_height(station_value, robot_ip):
    """Capture only transfer height; keep every other taught pose unchanged."""
    folder = Path(station_value)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    record = json.loads((folder / "station.json").read_text())
    transfer = json.loads((folder / "transfer_profile.json").read_text())
    pick = json.loads((folder / "pick_profile.json").read_text())
    threshold = max(transfer["retreat_pose_mm_deg"][2], transfer["placement_pose_mm_deg"][2],
                    record["poses"]["second_entry"][2],record["poses"]["second_final"][2],pick["motion"]["pick_z_mm"])
    arm = XArmAPI(robot_ip)
    try:
        window = tk.Tk()
        window.title("Correct Transfer Height - READ ONLY")
        message = tk.StringVar(value=f"Manually reach a clear transfer height ABOVE {threshold:.2f} mm. SPACE saves only this height.")
        tk.Label(window,textvariable=message,font=("Arial",16),wraplength=700).pack(padx=30,pady=40)
        def capture(event=None):
            try:
                pose = read_stationary_pose(arm)
                if pose[2] <= threshold:
                    raise ValueError(f"Height must exceed {threshold:.2f} mm.")
                record["poses"]["transfer_height"] = pose
                transfer["motion"]["transfer_z_mm"] = pose[2]
                pick["motion"]["approach_z_mm"] = pose[2]
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                for name,value in (("station.json",record),("transfer_profile.json",transfer),("pick_profile.json",pick)):
                    target=folder/name
                    backup=folder/(target.stem+"_before_height_"+stamp+".json")
                    backup.write_bytes(target.read_bytes())
                    target.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
                print("TRANSFER HEIGHT SAVED:",pose[2],flush=True)
                print("STATION:",folder,flush=True)
                window.destroy()
            except Exception as error:
                message.set(str(error))
        window.bind("<space>",capture)
        window.bind("<Return>",capture)
        window.bind("<Escape>",lambda event:window.destroy())
        window.mainloop()
    finally:
        arm.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--ip", default="192.168.1.185")
    parser.add_argument("--reteach-first", action="store_true", help="Capture a new first placement before the other station poses.")
    parser.add_argument("--repair-height", help="Existing station directory: update only transfer height.")
    args = parser.parse_args()
    if args.repair_height:
        repair_height(args.repair_height,args.ip)
        return
    stages = list(STAGES)
    if args.reteach_first:
        stages.insert(0, ("first_placement", "First link supported at its FINAL location, gripper still CLOSED. SPACE saves its placement. Do not move the link after release."))
    directory = Path(args.capture)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    placements = sorted(directory.glob("first_placement_*.json"))
    if not placements:
        raise FileNotFoundError("First placement record is required.")
    placement = json.loads(placements[-1].read_text())
    pick = json.loads((directory / "pick_profile.json").read_text())
    output = directory / ("station_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir()
    record = {"status":"teaching", "robot_ip":args.ip, "first_placement":placement,
              "capture_directory":str(directory.relative_to(PROJECT_ROOT)), "poses":{},
              "automatic_path_validated":False, "active_profiles_updated":False}

    def save():
        pending = output / "station.pending.json"
        pending.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        pending.replace(output / "station.json")

    def finish():
        poses = record["poses"]
        first = placement["pose_mm_deg"]
        final = poses["second_final"]
        entry = poses["second_entry"]
        if np.linalg.norm(np.asarray(entry[:3])-final[:3]) < 1.0:
            raise ValueError("Entry and fitted pose are identical or too close. Recapture entry BEFORE fitting.")
        height = poses["transfer_height"][2]
        if height <= max(first[2],final[2],entry[2],poses["retreat"][2],pick["motion"]["pick_z_mm"]):
            raise ValueError("Transfer height must exceed pick, placement and entry heights. Undo and recapture height.")
        transfer = copy.deepcopy(json.loads((PROJECT_ROOT / "config/robot1_transfer.json").read_text()))
        transfer["placement_pose_mm_deg"] = first
        transfer["retreat_pose_mm_deg"] = poses["retreat"]
        transfer["motion"]["transfer_z_mm"] = height
        transfer["source_records"] = {"current_station":str((output / "station.json").relative_to(PROJECT_ROOT))}
        transfer["validation"] = {"check_required_before_each_execution":True,
                                  "automatic_transfer_path_verified":False, "placement_verified":False,
                                  "part_retention_verified":False, "saved_route_check_passed":False}
        pick["motion"]["approach_z_mm"] = height
        offsets = {"offset_robot_mm":(np.asarray(final[:3])-first[:3]).tolist(),
                   "entry_offset_mm":(np.asarray(entry[:3])-final[:3]).tolist(),
                   "orientation_difference_first_to_second_deg":angle_error(first,final),
                   "orientation_difference_entry_to_final_deg":angle_error(entry,final),
                   "compatible_with_current_constant_orientation_coupling":
                       angle_error(first,final)<=1 and angle_error(entry,final)<=1,
                   "physical_contact_observed_by_operator":False,
                   "final_pose_taught_as_fitted":True,
                   "automatic_execution_validated":False}
        for filename, value in (("pick_profile.json",pick),("transfer_profile.json",transfer),("coupling.json",offsets)):
            (output / filename).write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
        record["status"] = "poses_saved_automatic_route_validation_pending"
        record["coupling"] = offsets
        save()
        print("STATION SAVED:",output,flush=True)
        print("COUPLING:",offsets,flush=True)
        return offsets

    arm = None
    try:
        arm = XArmAPI(args.ip)
        save()
        window = tk.Tk()
        window.title("Teach Two-Link Station - READ ONLY")
        window.geometry("800x440")
        heading, instruction, status = tk.StringVar(),tk.StringVar(),tk.StringVar()
        tk.Label(window,textvariable=heading,font=("Arial",24,"bold")).pack(pady=20)
        tk.Label(window,textvariable=instruction,font=("Arial",15),wraplength=750).pack(pady=20)
        tk.Label(window,text="Move manually in UFACTORY. SPACE / ENTER: save | BACKSPACE: undo | ESC: close",wraplength=750).pack(pady=10)
        tk.Label(window,textvariable=status,wraplength=750).pack(pady=20)

        def refresh():
            index = len(record["poses"])
            if index < len(stages):
                heading.set(f"{index+1}/{len(stages)}: {stages[index][0]}")
                instruction.set(stages[index][1])
            else:
                heading.set("All poses saved")
                instruction.set("Existing active profiles unchanged. Automatic route checks still required.")

        def capture(event=None):
            index = len(record["poses"])
            if index == len(stages):
                return
            try:
                key = stages[index][0]
                pose = read_stationary_pose(arm)
                if key == "second_final" and np.linalg.norm(np.asarray(pose[:3])-record["poses"]["second_entry"][:3]) < 1.0:
                    raise ValueError("Same as entry. Move to the actual fitted position before capturing final.")
                if key == "first_placement":
                    placement["pose_mm_deg"] = pose
                    placement["stage"] = "first_link_placement"
                    placement["gripper_state"] = "closed"
                record["poses"][key] = pose
                save()
                print(key,pose,flush=True)
                status.set(f"{key} SAVED: {output}")
                if len(record["poses"]) == len(stages):
                    try:
                        offsets = finish()
                    except Exception:
                        record["poses"].pop(key)
                        save()
                        raise
                    if not offsets["compatible_with_current_constant_orientation_coupling"]:
                        status.set("Poses saved. Different coupling orientations require a planner adjustment before execution.")
            except Exception as error:
                status.set(str(error))
                print("Capture rejected:",error,flush=True)
            refresh()

        def undo(event=None):
            if record["poses"]:
                key = list(record["poses"])[-1]
                record["poses"].pop(key)
                record["status"] = "teaching"
                save()
                status.set(f"Removed {key}. Capture again.")
            refresh()

        window.bind("<space>",capture)
        window.bind("<Return>",capture)
        window.bind("<BackSpace>",undo)
        window.bind("<Escape>",lambda event:window.destroy())
        refresh()
        print("First placement reused:",placement["pose_mm_deg"],flush=True)
        window.mainloop()
    finally:
        if arm is not None:
            arm.disconnect()
        print("Records:",output,flush=True)


if __name__ == "__main__":
    main()
