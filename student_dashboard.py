"""Read-only classroom display. Reads trial files; never connects to hardware."""
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 820


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def text(canvas, label, x, y, size=.65, color=(222,230,240)):
    cv2.putText(canvas, str(label), (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                size, color, 1, cv2.LINE_AA)


def picture(canvas, image, x, y, width, height):
    if image is None:
        text(canvas, "Waiting for capture...", x+30,y+80)
        return
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width/image.shape[1],height/image.shape[0])
    w,h = max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))
    resized = cv2.resize(image,(w,h))
    px,py = x+(width-w)//2,y+(height-h)//2
    canvas[py:py+h,px:px+w] = resized


def load_image(path):
    return cv2.imread(str(path)) if path.is_file() else None


def render(folder=None, record=None):
    record = record or {}
    canvas = np.full((HEIGHT,WIDTH,3),(25,20,15),np.uint8)
    accent = (100,220,70)
    text(canvas,"ROBOT VISION LAB",30,45,1.0,accent)
    text(canvas,"SEE  /  PLAN  /  PICK  /  PLACE  /  CONNECT",580,42,.65)
    status = record.get("status","Waiting for the next trial")
    text(canvas,str(status).replace("_"," ").upper(),30,85,.65)
    blocks = record.get("blocks",[])
    stage = "Ready"
    for block in blocks:
        stage = block.get("name","").replace("_"," ")
        actions = block.get("execution",[])
        if actions:
            stage += "  |  " + actions[-1].get("label","").replace("_"," ")
    text(canvas,stage[:105],30,119,.6,(110,210,245))
    for x in (25,650):
        cv2.rectangle(canvas,(x,140),(x+600,585),(48,39,30),-1)
    text(canvas,"ZONE 1 - CAMERA AND GRASP",45,171,.7,accent)
    text(canvas,"ZONE 2 - ASSEMBLY WORKSPACE",670,171,.7,accent)
    vision = record.get("vision",[])
    latest = vision[-1] if vision else {}
    index = latest.get("link_index",1)
    analysis = latest.get("analysis",{})
    frame = detection = geometry = None
    if folder:
        image_dir = folder / f"link{index}_vision"
        frame = load_image(image_dir/"camera.png")
        detection = load_image(image_dir/"detection.png")
        geometry = load_image(image_dir/"current_geometry.png")
    picture(canvas,detection if detection is not None else frame,45,190,405,330)
    picture(canvas,geometry,465,210,140,235)
    zone2 = None
    calibration = read_json(ROOT/"data/calibration/zone2_camera.json")
    if frame is not None and calibration:
        try:
            if frame.shape[:2] != (calibration["image_height_px"],calibration["image_width_px"]):
                raise ValueError("Resolution differs")
            matrix = np.diag([2.,2.,1.]) @ np.asarray(calibration["pixel_to_mm_matrix"])
            zone2 = cv2.warpPerspective(frame,matrix,(
                round(calibration["reference_width_mm"]*2),
                round(calibration["reference_height_mm"]*2)))
        except (KeyError,ValueError,cv2.error):
            zone2 = None
    picture(canvas,zone2,670,190,550,330)
    text(canvas,"Last shared-camera snapshot (not live)",670,548,.52)
    text(canvas,"Zone 2 angle is taught; not measured by vision",670,574,.48)
    text(canvas,f"PART {index} / 2" if vision else "PART -- / 2",45,548,.7)
    rotation = analysis.get("rotation_change_deg")
    text(canvas,f"Rotation vs reference: {rotation:+.1f} deg" if isinstance(rotation,(int,float))
         else "Rotation: waiting for measurement",45,574,.52)
    xy = analysis.get("target_xy_mm")
    tool = analysis.get("target_orientation_deg")
    text(canvas,"GRASP TARGET",35,622,.65,accent)
    text(canvas,f"X {xy[0]:.1f} mm    Y {xy[1]:.1f} mm" if xy else "X --    Y --",35,652,.65)
    text(canvas,f"Tool yaw: {tool[2]:.1f} deg" if tool else "Tool yaw: --",35,682,.65)
    destination = record.get("transfer_profile",{}).get("placement_pose_mm_deg",[])
    # Prefer the actual taught coupling destination once its plan exists.
    for block in blocks:
        for step in block.get("plan",[]):
            if step.get("label") == "couple_second_link":
                destination = step.get("pose",destination)
    text(canvas,"TAUGHT ASSEMBLY TARGET",665,622,.65,accent)
    if len(destination)==6:
        text(canvas,f"X {destination[0]:.1f}  Y {destination[1]:.1f}  Z {destination[2]:.1f} mm",665,652,.6)
        text(canvas,f"Tool yaw: {destination[5]:.1f} deg",665,682,.65)
    else:
        text(canvas,"Waiting for station data...",665,652,.6)
    labels = ("Startup","First link","Second pick","Coupling")
    names = ("startup_retreat","first_link_cycle","second_link_approach","second_link_coupling_and_retreat")
    for i,(label,name) in enumerate(zip(labels,names)):
        block = next((b for b in blocks if b.get("name")==name),{})
        state = block.get("status","pending")
        color = accent if state=="completed" else ((65,180,245) if state=="started" else (100,105,110))
        cv2.rectangle(canvas,(30+i*310,713),(320+i*310,757),color,-1)
        text(canvas,label+": "+state,40+i*310,742,.5,(15,20,25))
    text(canvas,"Stage colors describe program progress; grip and assembly are not sensed.",30,789,.5)
    return canvas


def main():
    import tkinter as tk
    started = time.time()
    window = tk.Tk()
    window.title("Robot Vision Lab - read-only student dashboard")
    window.configure(bg="#101419")
    label = tk.Label(window,bg="#101419")
    label.pack()
    info = {"photo":None,"folder":None,"record":{}}
    def refresh():
        try:
            root = ROOT/"data/robot1_two_link_trials"
            candidates = [p for p in root.glob("*/results.json") if p.stat().st_mtime >= started-2]
            if candidates:
                path = max(candidates,key=lambda p:p.stat().st_mtime)
                record = read_json(path)
                if record:
                    info["folder"],info["record"] = path.parent,record
            image = render(info["folder"],info["record"])
            ok,encoded = cv2.imencode(".png",image)
            if ok:
                photo = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"))
                label.configure(image=photo)
                info["photo"] = photo
        except Exception as error:
            window.title("Robot Vision Lab - display waiting: "+str(error)[:90])
        window.after(750,refresh)
    refresh()
    window.mainloop()


if __name__=="__main__":
    main()
