# October 13 function consolidation

Reviewed repository: `JavierGGF/robot-vision`, branch `assembly-demo`, commit `0b035b1`.

## What changes

- Chain extension lives in `robot_vision.robot.chain_assembly`; the existing root scripts remain compatible launchers.
- A single `build_release_plan` explicitly opens the gripper before lifting and retreating. Existing controller opening and closing methods each send two immediate SDK commands. SDK returns do not sense physical finger position.
- `robot_vision.vision.chain_targets` supplies the shared planar translation/rotation transform for live extension and saved target previews. Fixed-name taught pose files are accepted as well as uniquely matching timestamped files.
- `config/chain_assembly.json` selects current grasp, coupling and Zone 2 marker references. It does not edit their calibration or pose values.
- The menu uses the October 13 grasp reference for camera previews and automatically selects complete Zone 1 marker records matching the current camera snapshot when teaching a grasp. Zone 2 robot-marker calibration is available under Calibration and setup.
- Chain previews, elevated approaches, fitting-pose recording and release/retreat are available under Real Robot -> Vision-guided chain extension. Existing saved-station and simulation activities remain available; stale station grasp selection is rejected before physical execution.
- The combined dashboard displays chain trial images, measured opening angle and estimated destination. Foreground priority is removed shortly after startup, allowing normal access to VS Code.
- Trial records preserve detection/targets, stage results, timing, failures and local operator fit records.

## Evidence and limits

An October 13 elevated pick/approach completed. Its estimated final TCP was `[259.627554, -141.457075, 86.205742, -177.747264, -5.680819, 9.750973]`. The operator manually fitted the third link at `[255.627686, -141.457581, 86.578529, -177.747264, -5.680819, 9.750997]`. The observed local correction was approximately X -4.00 mm and Z +0.37 mm, with unchanged orientation.

This supports an operator-confirmed three-link fit, not automatic three-link insertion. The local correction remains evidence associated with that destination and grasp; it is never applied globally. Pointer-origin/height accuracy, grasp retention and contact forces remain unvalidated. Route checking does not model external obstacles.

Physical insertion retains a 10-degree local rotation limit. Elevated approaches permit up to 20 degrees, with a 100 mm opening displacement limit. These are trial restrictions, not experimentally established accuracy bounds. A current opening preview must remain consistent with the capture used for execution.

The preserved controller already contains the 2-degree marker-orientation checks and redundant gripper commands. It is not replaced by this update. No calibration, taught pose, raw experiment image or original trial record is overwritten.

The new menu extends a supported existing chain, one link per execution. A complete automatic three-link cycle from an empty Zone 2 is still pending: current first-placement teaching and destination-error validation are required. Robot 2/pin insertion, two-minute cycle performance and distributable macOS/Windows binaries remain pending.

## Installation

From the repository root, use the included `install_update.py --project . --apply`. It verifies reviewed source hashes, creates `update_backups/<timestamp>`, applies the source/configuration update and runs offline checks. Divergent local source files cause an abort before replacement. Failed offline checks restore replaced files. Neither installation nor these tests access a camera or robot.

Open the menu with `python student_setup.py`. Inspect the menu first; selecting a physical activity requires its explicit execution command and initial VERIFIED check where applicable. Saving a fitted pose reads the robot but issues no motion or gripper commands.

## Offline verification

`python tests/test_chain_update.py` checks the published-target regression, independent rotation arithmetic, unchanged poses for zero displacement, calibration mismatch rejection, height limits, release ordering and failures in the first/second gripper command. These are software checks; they do not validate physical alignment or grip success.
