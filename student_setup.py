"""
Guided student menu for the robotics assembly demonstration.

Select simulation or real-robot preparation before choosing an activity.
Keep offline practice separate from existing hardware setup tools.

Selecting a mode does not change robot configuration or start motion.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"


def pause():
    """Keep the result visible until the student continues."""

    input("\nPress ENTER to continue...")


def heading(title):
    """Display a consistent screen heading."""

    print("\n" + "=" * 54)
    print(title)
    print("=" * 54)


def run_module(module, *arguments):
    """Run a project activity using the current Python environment."""

    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SOURCE_ROOT)

    if existing:
        environment["PYTHONPATH"] += os.pathsep + existing

    try:
        result = subprocess.run(
            [sys.executable, "-m", module, *arguments],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )

        if result.returncode:
            print("\nThe activity ended with an error.")
            print("Review the message above before continuing.")

    except KeyboardInterrupt:
        print("\nActivity interrupted.")

    except OSError as error:
        print(f"\nCould not open the activity: {error}")

    pause()


def show_workflow():
    """Explain the complete assembly workflow without operating hardware."""

    heading("HOW THE SYSTEM WORKS")

    steps = [
        (
            "1. Camera Setup",
            "Choose a camera and confirm that both workspaces are visible.",
        ),
        (
            "2. Robot Connections",
            "Identify Robot 1 and Robot 2 and check their connections.",
        ),
        (
            "3. Workspace Calibration",
            "Use four black marker centers to convert pixels to millimeters.",
        ),
        (
            "4. Robot Calibration",
            "Relate positions on each sheet to its robot coordinates.",
        ),
        (
            "5. Part Detection and Orientation",
            "Find the contour, center, and horseshoe opening direction.",
        ),
        (
            "6. Grasp Point",
            "Choose where to hold the part and rotate that point with it.",
        ),
        (
            "7. Gripper and Grasp Height",
            "Teach the final grasp height for the installed fingers.",
        ),
        (
            "8. Home and Placement",
            "Use recorded poses and check the route from the actual start.",
        ),
        (
            "9. Assembly Preview",
            "Place one link, insert the next link, then insert the pin.",
        ),
        (
            "10. Step-by-Step Demonstration",
            "Check each operation before running the complete sequence.",
        ),
    ]

    for title, explanation in steps:
        print(f"\n{title}")
        print(f"   {explanation}")

    print("\nA saved setup can be inspected without recalibrating.")
    print("Simulation does not establish physical grasp or insertion success.")
    pause()


def explore_orientation():
    """Open the working offline activity in either menu."""

    heading("EXPLORE PART ORIENTATION")
    print("This activity uses a saved silhouette.")
    print("It does not require a camera, Docker, or a robot.")
    print()
    print("Click the image window before using its keyboard controls.")
    print("Slider / A / D: rotate")
    print("Z: set practice zero")
    print("Mouse click: select a practice grasp point")
    print("R: reset")
    print("Q / ESC: close")
    print()
    print("Practice zero and grasp point do not change robot calibration.")

    run_module("robot_vision.vision.explore_orientation")


def inspect_saved_setup():
    """Show saved project data without treating file presence as validation."""

    heading("VIEW SAVED SETUP")
    print("Saved values are references, not confirmation of readiness.")

    profiles = [
        ("Robot connection", "config/simulation.json"),
        ("Robot 1 pick", "config/robot1_pick.json"),
        ("Robot 1 transfer", "config/robot1_transfer.json"),
        ("Zone 1 camera", "data/calibration/zone1_camera.json"),
        (
            "Zone 1 reference camera",
            "data/calibration/zone1_reference_camera.json",
        ),
    ]

    while True:
        print()
        for index, (label, relative_path) in enumerate(profiles, 1):
            exists = (PROJECT_ROOT / relative_path).is_file()
            state = "Saved" if exists else "Not found"
            print(f"{index}. {label} [{state}]")

        print("0. Back")
        choice = input("\nSelect a profile to inspect: ").strip()

        if choice == "0":
            return

        if not choice:
            continue

        if not choice.isdigit() or not 1 <= int(choice) <= len(profiles):
            print("Please select an available option.")
            continue

        label, relative_path = profiles[int(choice) - 1]
        path = PROJECT_ROOT / relative_path

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            heading(label.upper())
            print(json.dumps(data, indent=2))
        except (OSError, ValueError) as error:
            print(f"Could not read this profile: {error}")

        pause()


def simulator_status():
    """Check Docker and the named container without starting services."""

    heading("CHECK SIMULATOR SETUP")
    print("Required model: UFACTORY Lite6")
    print("Expected container: uf_software")
    print("This operation only reads Docker status.")

    docker = shutil.which("docker")

    if docker is None:
        print("\nDocker was not found.")
        print("Docker Desktop must be installed for robot simulation.")
        pause()
        return

    try:
        result = subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )

        if result.returncode:
            print("\nDocker Engine is not available.")
            print("Open Docker Desktop and wait until it is ready.")
            print(result.stderr.strip()[-1000:])
            pause()
            return

        print("\nDocker Engine:", result.stdout.strip())

        result = subprocess.run(
            [
                docker,
                "container",
                "inspect",
                "--format",
                "{{.State.Status}}",
                "uf_software",
            ],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )

        if result.returncode:
            print("The simulator container could not be inspected.")
            print(result.stderr.strip()[-1000:])
        else:
            print("Container:", result.stdout.strip())
            print("Robot model and simulator readiness are not yet verified.")

    except subprocess.TimeoutExpired:
        print("Docker did not respond in time.")
    except OSError as error:
        print(f"Could not query Docker: {error}")

    pause()


# Current physical two-link station. Selection never starts robot motion.
from datetime import datetime
import ast
import copy
import math

DEFAULT_STATION = 'data/orientation_tests/20261010_094517_994548/station_20261010_100335_354623'
DEFAULT_MARKERS = 'data/workspace_calibrations/robot1_zone1_20261010_093932_909289/robot_mapping.json'
SETTINGS_FILE = PROJECT_ROOT / 'config/student_menu.json'
STATE = {'station': DEFAULT_STATION, 'markers': DEFAULT_MARKERS,
         'capture': 'data/orientation_tests/20261010_105015_008545',
         'robot_ip': '192.168.1.185', 'camera': 0}


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8'))


def write_backup(path, value):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path.with_name(path.name + '.backup_' + stamp).write_bytes(path.read_bytes())
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def remember():
    write_backup(SETTINGS_FILE, STATE)


def shared_functions():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from robot_vision.robot.test_pick_motion import capture_frame, save_images
    return capture_frame, save_images


def choose_path(key, title, candidates, required):
    heading(title)
    choices = sorted(set(candidates))[-12:][::-1]
    for number, path in enumerate(choices, 1):
        print(f'{number}. {relative(path)}')
    print('P. Enter a project-relative path')
    print('0. Cancel')
    answer = input('Select: ').strip()
    if answer == '0':
        return
    if answer.lower() == 'p':
        selected = resolve(input('Path: ').strip())
    elif answer.isdigit() and 1 <= int(answer) <= len(choices):
        selected = choices[int(answer)-1]
    else:
        raise ValueError('Invalid selection')
    for filename in required:
        if not (selected / filename).is_file():
            raise ValueError(f'Missing {filename} in {selected}')
    STATE[key] = relative(selected)
    if key == 'station':
        record = read(selected / 'station.json')
        STATE['capture'] = record['capture_directory']
        STATE['robot_ip'] = record.get('robot_ip', STATE['robot_ip'])
    remember()
    print('Selected:', STATE[key])


def select_station():
    choose_path('station', 'SELECT TWO-LINK STATION',
                [p.parent for p in (PROJECT_ROOT / 'data').glob('orientation_tests/*/station_*/station.json')],
                ['station.json', 'pick_profile.json', 'transfer_profile.json', 'coupling.json'])


def select_capture():
    choose_path('capture', 'SELECT GRASP CAPTURE',
                [p.parent for p in (PROJECT_ROOT / 'data').glob('orientation_tests/*/camera.png')],
                ['camera.png'])


def select_markers():
    heading('SELECT ROBOT MARKER RECORD')
    choices = sorted((PROJECT_ROOT / 'data/workspace_calibrations').glob('robot1_zone1_*/robot_mapping.json'))[::-1]
    for i, path in enumerate(choices[:12], 1):
        print(f'{i}. {relative(path)}')
    answer = input('Number, or project-relative JSON path (0 cancels): ').strip()
    if answer == '0':
        return
    selected = choices[int(answer)-1] if answer.isdigit() and 1 <= int(answer) <= min(12,len(choices)) else resolve(answer)
    record = read(selected)
    if record.get('marker_order') != ['TL', 'TR', 'BL'] or len(record.get('markers', [])) != 3:
        raise ValueError('Select a complete TL/TR/BL record')
    STATE['markers'] = relative(selected)
    remember()


def station_status():
    heading('CURRENT STATION - READ ONLY')
    print('Station:', STATE['station'])
    print('Preparation capture:', STATE['capture'])
    print('Marker record:', STATE['markers'])
    folder = resolve(STATE['station'])
    record = read(folder / 'station.json')
    pick = read(folder / 'pick_profile.json')
    transfer = read(folder / 'transfer_profile.json')
    print('Execution reference:', record['capture_directory'])
    print('Robot IP:', STATE['robot_ip'])
    print('Pick height [mm]:', pick['motion']['pick_z_mm'])
    print('Transfer height [mm]:', transfer['motion']['transfer_z_mm'])
    print('Travel speed [mm/s]:', transfer['motion']['travel_speed_mm_s'])
    print('Retreat pose:', transfer['retreat_pose_mm_deg'])
    print('Destination: taught first placement and second coupling poses.')
    print('Zone 2 free-horseshoe detection, third link and Robot 2 pin insertion are not implemented.')
    print('Current runner uses a 190 mm target-height cap and a 20 mm slow final approach.')
    pause()


def settings_menu():
    while True:
        heading('STATION AND HARDWARE SETTINGS')
        print(f"Station: {STATE['station']}\nRobot IP: {STATE['robot_ip']}\nCamera index: {STATE['camera']}")
        print('1. Select station\n2. Select grasp capture\n3. Select marker record\n4. Set preparation robot IP\n5. Set calibration camera index\n6. View current station\n0. Back')
        choice = input('Select: ').strip()
        if choice == '0': return
        try:
            if choice == '1': select_station()
            elif choice == '2': select_capture()
            elif choice == '3': select_markers()
            elif choice == '4':
                import ipaddress
                value = input('Physical robot IPv4 address: ').strip()
                address = ipaddress.ip_address(value)
                if address.version != 4 or address.is_loopback or address.is_unspecified or address.is_multicast:
                    raise ValueError('Use a physical unicast IPv4 address')
                STATE['robot_ip'] = value; remember()
                print('Preparation IP saved. Execution still uses config/simulation.json; configure its connection separately.')
            elif choice == '5':
                value = int(input('Camera index: '))
                if not 0 <= value <= 20: raise ValueError('Camera index must be 0..20')
                STATE['camera'] = value; remember()
                print('Used for the next camera calibration. Existing calibrations are unchanged.')
            elif choice == '6': station_status()
        except Exception as error:
            print('Settings unchanged or incomplete:', error); pause()


def camera_preview():
    import cv2
    cap = cv2.VideoCapture(int(STATE['camera']))
    try:
        if not cap.isOpened(): raise RuntimeError('Camera unavailable; close other camera windows')
        calibration = PROJECT_ROOT / 'data/calibration/zone1_camera.json'
        if calibration.exists():
            cal = read(calibration)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal['image_width_px'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal['image_height_px'])
        print('Camera preview only. Q / ESC closes. No robot connection.')
        failures = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures >= 30: raise RuntimeError('Cannot read camera')
            else:
                failures = 0
                cv2.imshow('Camera - both workspaces', frame)
            key = cv2.waitKey(30) & 255
            if key in (27, ord('q')): break
            if ok and cv2.getWindowProperty('Camera - both workspaces', cv2.WND_PROP_VISIBLE) < 1: break
    finally:
        cap.release(); cv2.destroyAllWindows()


def capture_reference():
    capture_frame, save_images = shared_functions()
    calibration = read('data/calibration/zone1_camera.json')
    print('ONE part in Zone 1. Keep robot outside image; keep part still after capture.')
    input('Press ENTER when ready (camera only): ')
    folder = PROJECT_ROOT / 'data/orientation_tests' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    save_images(folder, {'camera.png': capture_frame(calibration)})
    write_backup(folder / 'capture_calibration.json', calibration)
    STATE['capture'] = relative(folder); remember()
    print('CAPTURE SAVED:', folder)
    print('Next: manually teach grasp, then select its point in this image.')
    pause()


def capture_pose(stage):
    shared_functions()
    from xarm.wrapper import XArmAPI
    from robot_vision.robot.capture_workspace_robot import read_stationary_pose
    folder = resolve(STATE['capture'])
    if not (folder / 'camera.png').exists(): raise ValueError('Capture a reference first')
    if stage == 'grasp':
        print('Place open jaws at correct grasp height. Do not move the part from its captured position.')
        name = 'taught_grasp_pose.json'
    else:
        print('Hold first link at FINAL Zone 2 placement, jaws CLOSED. Read only; no commanded movement.')
        name = 'first_placement_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json'
    input('Press ENTER to read and save this stationary pose: ')
    arm = XArmAPI(STATE['robot_ip'])
    try:
        pose = read_stationary_pose(arm)
        record = {'robot_ip': STATE['robot_ip'], 'pose_mm_deg': pose,
                  'created_at': datetime.now().astimezone().isoformat()}
        if stage == 'grasp': record.update(gripper_state='open', grasp_verified=False)
        else: record.update(stage='first_link_placement', gripper_state='closed')
        write_backup(folder / name, record)
        print('POSE SAVED:', pose, '\n', folder / name)
    finally: arm.disconnect()
    pause()


def link_reference():
    folder = resolve(STATE['station']); capture = resolve(STATE['capture'])
    corrected = read(capture / 'pick_profile.json')
    reference_cal = read(capture / 'reference_camera.json')
    current_cal = read('data/calibration/zone1_camera.json')
    keys = ('image_width_px', 'image_height_px', 'pixel_to_mm_matrix', 'reference_points_mm')
    if any(reference_cal[k] != current_cal[k] for k in keys):
        raise ValueError('Capture uses a different camera calibration; create a fresh reference')
    pick = read(folder / 'pick_profile.json'); station = read(folder / 'station.json')
    print('New reference:', STATE['capture'])
    print('Existing assembly destinations and speeds will be retained.')
    if input('Type SAVE to link: ').strip() != 'SAVE': return
    pick['local_reference'] = corrected['local_reference']
    for key in ('tool_orientation_deg','pick_z_mm'):
        pick['motion'][key] = corrected['motion'][key]
    station['capture_directory'] = STATE['capture']; station['automatic_path_validated'] = False
    write_backup(folder / 'pick_profile.json', pick)
    write_backup(folder / 'station.json', station)
    print('REFERENCE LINKED. Use preview before physical execution.'); pause()


def runner_empty_folder():
    # Resolve the current runner's actual reference, rather than pretending a menu setting activates it.
    path = SOURCE_ROOT / 'robot_vision/robot/robot1_two_link_cycle.py'
    tree = ast.parse(path.read_text())
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'project_path' and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and 'empty_zone1_' in arg.value:
                matches.append(resolve(arg.value))
    if len(matches) != 1: raise ValueError('Cannot resolve active empty-sheet reference from runner')
    folder = matches[0]
    if folder.parent.resolve() != resolve(STATE['station']).resolve():
        raise ValueError('Runner empty reference belongs to another station; execution integration required')
    return folder


def capture_empty():
    capture_frame, save_images = shared_functions()
    folder = runner_empty_folder()
    print('Remove all parts from Zone 1; park robot outside the image. Keep camera and sheet fixed.')
    if input('Type EMPTY to save the new empty-sheet baseline: ').strip() != 'EMPTY': return
    calibration = read('data/calibration/zone1_camera.json')
    frame = capture_frame(calibration)
    archive = folder / ('previous_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    archive.mkdir(parents=True)
    for name in ('camera.png','calibration.json'):
        if (folder / name).exists(): shutil.copy2(folder / name, archive / name)
    save_images(folder, {'camera.png': frame})
    write_backup(folder / 'calibration.json', calibration)
    print('ACTIVE EMPTY REFERENCE UPDATED:', folder)
    print('Validate an empty capture and an occupied capture before execution.'); pause()


def live_analysis():
    capture_frame, save_images = shared_functions()
    from robot_vision.robot.test_rotated_pick import calculate_rotated_target
    from robot_vision.robot.test_pick_motion import show_preview
    import cv2
    folder = resolve(STATE['station']); record = read(folder / 'station.json')
    pick = read(folder / 'pick_profile.json')
    calibration = read('data/calibration/zone1_camera.json')
    reference = read(resolve(record['capture_directory']) / 'reference_camera.json')
    print('ONE part in Zone 1; keep robot outside image. Camera preview only, no robot connection.')
    frame = capture_frame(calibration)
    output = PROJECT_ROOT / 'data/student_vision_previews' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    save_images(output, {'camera.png': frame})
    analysis, images = calculate_rotated_target(frame, calibration, reference, pick)
    save_images(output, images)
    write_backup(output / 'results.json', {'status':'preview_only', 'analysis':analysis})
    print('Angle change [deg]:', analysis['rotation_change_deg'])
    print('Grasp XY [mm]:', analysis['target_xy_mm'])
    print('Tool orientation:', analysis['target_orientation_deg'])
    print('SAVED:', output)
    show_preview(images, analysis)


def motion_settings():
    folder = resolve(STATE['station']); pick = read(folder / 'pick_profile.json'); transfer = read(folder / 'transfer_profile.json')
    print('Current pick settings:', json.dumps(pick['motion'], indent=2))
    print('Current transfer settings:', json.dumps(transfer['motion'], indent=2))
    print('Fixed in current runner: 190 mm target cap; final slow approach 20 mm; fast approach 30 mm/s.')
    def number(prompt, default, minimum, maximum):
        answer = input(f'{prompt} [{default}]: ').strip()
        value = float(answer) if answer else float(default)
        if not math.isfinite(value) or not minimum <= value <= maximum: raise ValueError(f'Value must be {minimum}..{maximum}')
        return value
    height = number('Transfer / approach height mm', transfer['motion']['transfer_z_mm'], 130, 190)
    speed = number('Travel speed mm/s', transfer['motion']['travel_speed_mm_s'], 1, 50)
    lift = number('Lift / clearance speed mm/s', pick['motion']['lift_speed_mm_s'], 1, 30)
    slow = number('Final descent / placement speed mm/s', pick['motion']['descent_speed_mm_s'], 1, 10)
    station = read(folder / 'station.json')
    minimum = max(transfer['retreat_pose_mm_deg'][2], transfer['placement_pose_mm_deg'][2],
                  pick['motion']['pick_z_mm'] + pick['motion']['lift_mm'],
                  station['poses']['second_entry'][2], station['poses']['second_final'][2])
    if height <= minimum: raise ValueError(f'Transfer height must exceed {minimum:.3f} mm')
    if input('Type SAVE to save settings with backups (no movement): ').strip() != 'SAVE': return
    pick['motion'].update(approach_z_mm=height, travel_speed_mm_s=speed, lift_speed_mm_s=lift, descent_speed_mm_s=slow)
    transfer['motion'].update(transfer_z_mm=height, travel_speed_mm_s=speed, clearance_speed_mm_s=lift, placement_speed_mm_s=slow)
    station['automatic_path_validated'] = False
    write_backup(folder / 'pick_profile.json', pick); write_backup(folder / 'transfer_profile.json', transfer)
    write_backup(folder / 'station.json', station)
    print('Settings saved; paths require fresh checking. Retreat remains unchanged.'); pause()


def calibration_tools_menu():
    actions = {
        '1': camera_preview,
        '2': lambda: run_module('robot_vision.vision.calibrate_zone','--camera',str(STATE['camera']),'--zone','1'),
        '3': lambda: run_module('robot_vision.vision.calibrate_zone','--camera',str(STATE['camera']),'--zone','2'),
        '4': lambda: run_module('robot_vision.robot.capture_workspace_robot','--ip',STATE['robot_ip'],'--zone','1','--record-only'),
        '5': select_markers,
        '6': capture_reference,
        '7': lambda: capture_pose('grasp'),
        '8': lambda: run_module('robot_vision.vision.select_grasp','--capture',STATE['capture'],'--markers',STATE['markers']),
        '9': link_reference,
        '10': lambda: capture_pose('placement'),
        '11': lambda: run_module('robot_vision.robot.teach_two_link_station','--capture',STATE['capture'],'--ip',STATE['robot_ip'],'--reteach-first'),
        '12': select_station,
        '13': capture_empty,
        '14': motion_settings,
        '15': lambda: run_module('robot_vision.robot.configure_robot'),
        '16': settings_menu,
        '17': station_status,
        '18': records_menu,
    }
    while True:
        heading('CAMERA, SHEET AND TWO-LINK CALIBRATION')
        print('1. Camera preview / check both sheets\n2. Calibrate camera - Zone 1\n3. Calibrate camera - Zone 2\n4. Capture Robot 1 sheet markers - TL/TR/BL (read only)\n5. Select captured marker record\n6. Capture ONE part reference image\n7. Save taught grasp pose / height (read only)\n8. Click grasp point in saved image\n9. Link new grasp reference to selected station\n10. Save first placement pose (read only)\n11. Teach assembly station - first / retreat / entry / final / height\n12. Select saved station\n13. Recapture active empty Zone 1 reference\n14. Adjust speeds and transfer height\n15. Configure execution robot connection\n16. Select camera / IP / station settings\n17. Inspect saved station\n18. Trial records\n0. Back')
        print('Move the robot manually in UFACTORY for teaching. Capture tools only read poses.')
        print('Record-only sheet mapping is approximate; pointer rotation is not compensated.')
        answer = input('Select: ').strip()
        if answer == '0': return
        try:
            if answer in actions: actions[answer]()
            else: print('Select an available option.')
        except (KeyboardInterrupt, EOFError): print('\nActivity cancelled.')
        except Exception as error: print('Calibration stopped:', error); pause()


def setup_submenu(title, entries):
    while True:
        heading(title)
        for index, (label, action) in enumerate(entries, 1):
            print(f'{index}. {label}')
        print('0. Back')
        answer = input('Select: ').strip()
        if answer == '0': return
        try:
            if answer.isdigit() and 1 <= int(answer) <= len(entries):
                entries[int(answer)-1][1]()
            else: print('Select an available option.')
        except (KeyboardInterrupt, EOFError): print('Activity cancelled.')
        except Exception as error: print('Activity stopped:', error); pause()


def calibration_menu():
    grasp_steps = [
        ('Capture part reference image', capture_reference),
        ('Save taught grasp pose and height', lambda:capture_pose('grasp')),
        ('Click grasp point in saved image', lambda:run_module('robot_vision.vision.select_grasp','--capture',STATE['capture'],'--markers',STATE['markers'])),
        ('Link grasp reference to selected station', link_reference),
    ]
    station_steps = [
        ('Save first placement pose', lambda:capture_pose('placement')),
        ('Teach placement, retreat, entry, final and transfer height', lambda:run_module('robot_vision.robot.teach_two_link_station','--capture',STATE['capture'],'--ip',STATE['robot_ip'],'--reteach-first')),
        ('Select saved station', select_station),
    ]
    sheet_steps = [
        ('Capture TL / TR / BL robot positions', lambda:run_module('robot_vision.robot.capture_workspace_robot','--ip',STATE['robot_ip'],'--zone','1','--record-only')),
        ('Select captured marker record', select_markers),
    ]
    setup_submenu('CALIBRATION AND SETUP', [
        ('Select camera and robot connection', settings_menu),
        ('Calibrate camera / sheet - Zone 1', lambda:run_module('robot_vision.vision.calibrate_zone','--camera',str(STATE['camera']),'--zone','1')),
        ('Calibrate camera / sheet - Zone 2', lambda:run_module('robot_vision.vision.calibrate_zone','--camera',str(STATE['camera']),'--zone','2')),
        ('Calibrate sheet to Robot 1', lambda:setup_submenu('SHEET TO ROBOT 1',sheet_steps)),
        ('Teach grasp point and height', lambda:setup_submenu('GRASP REFERENCE',grasp_steps)),
        ('Teach placement and coupling', lambda:setup_submenu('ASSEMBLY STATION',station_steps)),
        ('Capture empty Zone 1', capture_empty),
        ('Adjust speed and transfer height', motion_settings),
        ('Advanced tools and records', calibration_tools_menu),
    ])


def execute_cycle(second=False, preview=False):
    folder = resolve(STATE['station'])
    for name in ('station.json','pick_profile.json','transfer_profile.json','coupling.json'):
        read(folder / name)
    args = ['--station', STATE['station']]
    if second: args.append('--second-only')
    if not preview:
        config = read('config/simulation.json')
        if config.get('mode') == 'simulation': raise ValueError('Configure the physical robot connection first')
        runner_empty_folder()
        heading('PHYSICAL TWO-LINK EXECUTION')
        print('This activity commands Robot 1 and its gripper.')
        print('Keep the robot outside the camera view; disable manual mode; clear the movement area.')
        print('The runner enables stopped state 4 at startup if diagnostics allow it.')
        print('After VERIFIED: no OPEN/HELD prompts. Physical gripper state is NOT sensed.')
        print('Load one part at a time. A vacant Zone 1 ends the run; load the second only while stopped.')
        print('Keep first placement fixed for second-only. No automatic destination detection in Zone 2.')
        args.append('--execute')
    if not preview:
        dashboard = PROJECT_ROOT / 'student_dashboard.py'
        if dashboard.is_file():
            try:
                subprocess.Popen([sys.executable, str(dashboard)], cwd=PROJECT_ROOT)
            except OSError as error:
                print('Student display unavailable:', error)
    run_module('robot_vision.robot.robot1_two_link_cycle', *args)


def records_menu():
    folders = sorted((PROJECT_ROOT / 'data/robot1_two_link_trials').glob('*/results.json'))[::-1][:20]
    heading('RECENT PHYSICAL / PREVIEW TRIALS')
    for i, path in enumerate(folders, 1):
        record = read(path)
        print(f"{i}. {path.parent.name} | {record.get('status','unknown')} | {record.get('mode','')}")
    answer = input('Number to inspect; 0 returns: ').strip()
    if answer == '0': return
    if not answer.isdigit() or not 1 <= int(answer) <= len(folders): raise ValueError('Invalid trial')
    folder = folders[int(answer)-1].parent
    record = read(folder / 'results.json')
    print('Folder:', folder)
    print('Status:', record.get('status')); print('Error:', record.get('error','none'))
    print('Physical assembly verified:', record.get('physical_assembly_verified',False))
    print('Vision captures:', len(record.get('vision',[])))
    print('Blocks:', [(b.get('name'),b.get('status')) for b in record.get('blocks',[])])
    if (folder / 'operator_validation.json').exists(): print('Operator record:', read(folder / 'operator_validation.json'))
    if input('Open folder? [y/N]: ').strip().lower() == 'y':
        if sys.platform == 'darwin': subprocess.run(['open',str(folder)],check=False)
        elif os.name == 'nt': os.startfile(str(folder))
        else: subprocess.run(['xdg-open',str(folder)],check=False)
    pause()


def real_menu():
    actions = {'1':lambda:execute_cycle(), '2':live_analysis, '3':calibration_menu}
    while True:
        heading('REAL ROBOT')
        print('1. Start two-piece assembly\n2. View camera and detected part\n3. Calibration and setup\n0. Back')
        answer = input('Select: ').strip()
        if answer == '0': return
        try:
            if answer in actions: actions[answer]()
            else: print('Select an available option.')
        except (KeyboardInterrupt, EOFError): print('\nActivity cancelled.')
        except Exception as error: print('Activity stopped:',error); pause()


def simulation_menu():
    actions = {
        '1':simulator_status,
        '2':lambda:run_module('robot_vision.robot.start_simulator'),
        '3':explore_orientation,
        '4':lambda:run_module('robot_vision.robot.simulator_demo_cycle'),
        '5':lambda:run_module('robot_vision.robot.simulator_demo_cycle','--execute'),
        '6':lambda:run_module('robot_vision.robot.simulator_two_link_cycle','--entry-offset','10','0','0'),
        '7':lambda:run_module('robot_vision.robot.simulator_two_link_cycle','--entry-offset','10','0','0','--execute'),
        '8':lambda:run_module('robot_vision.robot.simulator_two_link_cycle','--entry-offset','10','0','0','--execute','--auto'),
        '9':lambda:run_module('robot_vision.vision.preview_link_coupling'),
        '10':lambda:execute_cycle(preview=True),
    }
    while True:
        heading('LOCAL SIMULATION AND OFFLINE PRACTICE')
        print('1. Check Docker simulator\n2. Start Lite6 simulator\n3. Explore orientation - saved silhouette\n4. Preview one-link pick/place\n5. Run one-link LOCAL simulator\n6. Preview two-link simulator plan\n7. Run two-link LOCAL simulator with pauses\n8. Run two-link LOCAL simulator automatically\n9. Offline link-coupling geometry preview\n10. Preview selected real-station plan (offline)\n0. Back')
        print('Simulator tools enforce their own configuration and start-pose checks.')
        print('Virtual gripper events; contact and insertion are not physically simulated.')
        answer = input('Select: ').strip()
        if answer == '0': return
        try:
            if answer in actions: actions[answer]()
            else: print('Select an available option.')
        except Exception as error: print('Simulation stopped:',error); pause()


def show_workflow():
    heading('HOW THE DEMO WORKS')
    print('1. Calibrate camera pixels to the printed sheet using TL/TR/BR/BL centers.')
    print('2. Pair sheet markers with Robot 1; teach a grasp reference and height.')
    print('3. Detect one part in Zone 1 and rotate its grasp offset and tool yaw.')
    print('4. Pick and place first link at the taught Zone 2 pose.')
    print('5. Capture next part, then approach and couple at taught entry/final poses.')
    print('6. Open gripper and retreat; stop before picking if Zone 1 matches the empty baseline.')
    print('The current demo assembles TWO links. It does not locate the Zone 2 joint by vision.')
    print('Automatic commands do not prove grip retention or successful physical assembly.')
    print('After relocation: recalibrate camera/sheet and reteach changed station poses; do not edit source code.')
    pause()


def main():
    if SETTINGS_FILE.exists():
        try:
            saved = read(SETTINGS_FILE)
            STATE.update({key:saved[key] for key in STATE if key in saved})
        except Exception as error: print('Menu settings ignored:',error)
    try:
        while True:
            heading('ROBOTICS ASSEMBLY DEMO')
            print('1. Simulation\n2. Real Robot\n3. How the system works\n0. Exit')
            actions = {'1':simulation_menu,'2':real_menu,'3':show_workflow}
            answer = input('Select: ').strip()
            if answer == '0': print('Demonstration menu closed.'); return
            try:
                if answer in actions: actions[answer]()
                else: print('Select an available option.')
            except (KeyboardInterrupt, EOFError): print('\nActivity cancelled.')
            except Exception as error: print('Activity stopped:',error); pause()
    except (KeyboardInterrupt, EOFError): print('\nDemonstration menu closed.')


if __name__ == '__main__':
    main()
