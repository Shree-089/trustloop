# TrustLoop — a self-verifying robot manipulation system (MuJoCo)

**Execute → Verify → Recover → Verify again.**
A 6-DOF arm with a parallel-jaw gripper picks a red cube off a table and places it in a blue bin, in real MuJoCo physics (no teleporting, no welding the cube to the hand). The task is **never** declared successful because the controller finished its motions — an independent verifier reads the physical simulation state (poses, velocities, contacts, finger positions) and decides. Failures trigger bounded, whitelisted recovery (max 3 attempts), and every step is logged as JSON.

Runs on a CPU-only laptop. One run takes ~0.5 s headless (≈13 s of simulated time) or ≈13 s with the real-time viewer.

```
            ┌────────── EXECUTE ──────────┐        ┌──────── VERIFY (physics state) ────────┐
 HOME → APPROACH_CUBE → OPEN_GRIPPER →     │        │ grasp_closure : both pads touch cube,   │
 DESCEND_TO_CUBE → CLOSE_GRIPPER ──────────┼──────► │   gap ≈ cube width, cube centred        │
                                           │        │ grasp_lift_test: 3 cm test lift – cube  │
 LIFT_CUBE → MOVE_ABOVE_CONTAINER →        │        │   rises, no relative drift, off table   │
 DESCEND_INTO_CONTAINER ───────────────────┼──────► │ carry : still pinched, no slip          │
 OPEN_GRIPPER → RETREAT ───────────────────┼──────► │ placement: inside, below rim, on floor, │
                                           │        │   not on rim, released, settled         │
            └──────────────────────────────┘        └──────────────┬──────────────────────────┘
                     ▲                                     pass    │   fail (categorised)
                     │                              SUCCESS ◄──────┤
                     │                                             ▼
                     └── new attempt (≤3) ◄── RECOVERY: whitelisted action ──► abort → FAILURE
```

---

## 1. Setup

Requires Python 3.10+ (3.12 recommended). Run every command from the project folder (the one containing this README) — nothing assumes a fixed path.

**Linux / WSL**
```bash
cd trustloop
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd trustloop
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```
(Windows cmd.exe: `.venv\Scripts\activate.bat`.)

**macOS**: same as Linux, but the interactive viewer must be started with `mjpython src/main.py ...` instead of `python`.

Dependencies: `mujoco`, `numpy`, `pytest`. Tested with MuJoCo 3.17.0 and 3.2.7.

## 2. Validate the model
```bash
python src/main.py --validate
```
Expected: `MODEL OK` (loads the MJCF, simulates 2 s, checks the cube rests on the table).

## 3. Run the simulation
| Command | What you see |
|---|---|
| `python src/main.py` | Viewer, real time, nominal pick-and-place → SUCCESS |
| `python src/main.py --fault grasp_miss` | Perception off by 5 cm → fingers close on air → **grasp_missed** detected → retry → SUCCESS |
| `python src/main.py --fault weak_grip` | Grip force 0.1 N: fingers *do* touch the cube, but the test lift shows it stays on the table → **grasp_slipped** → raise force → SUCCESS |
| `python src/main.py --fault place_offset` | Cube released 17 cm off target → **placed_outside_container** → re-pick from where it really landed → SUCCESS |
| `python src/main.py --fault rim` | Cube released on top of a bin wall, where it balances → **resting_on_rim** → re-pick → SUCCESS |
| `python src/main.py --fault unrecoverable` | Perception wrong every time → 3 failed attempts → honest **FAILURE** |
| `python src/main.py --headless --runs 20 --random-cube --seed 1` | Batch of 20 fresh runs with random cube pose/yaw (no viewer) |

Useful flags: `--headless` (no window, max speed), `--fast` (viewer without real-time pacing), `--keep-open` (leave the viewer open at the end), `--cube X Y Z --cube-yaw DEG` (custom start), `--screenshots`, `--record` (MP4), `--advisor nemotron` (optional, see §7).

Exit code: `0` if every run ended in SUCCESS, `2` otherwise — handy for scripts.

**Resetting:** every run calls `Simulation.reset()` (HOME keyframe + cube start pose + contact settling), so simply re-running the command, or using `--runs N`, starts from a fresh physical state. Close the viewer window to stop a run early (it ends as FAILURE "Interrupted"). Don't use the viewer's own Backspace reset mid-task — the controller isn't told about it.

## 4. Run the tests
```bash
python -m pytest -q
```
63 tests, about 20 s on a CPU (the Octo-shadow test needs offscreen GL: on headless Linux run `MUJOCO_GL=egl python -m pytest -q`; otherwise it is skipped). Run one file: `python -m pytest -q tests/test_placement_verification.py`.

## 5. Logs, screenshots, video
All outputs go to `logs/` (override with `--log-dir`):

| File | Content |
|---|---|
| `logs/run_<timestamp>[_<fault>].jsonl` | One JSON object per event: every state transition, verification result (all checks + measurements), recovery decision, final outcome |
| `logs/run_<...>_summary.json` | Outcome, reason, attempts used, all recoveries and verifications, final cube pose |
| `logs/batch_latest.json` | One line per run of the last `--runs N` batch |
| `logs/screenshots/*.png` | With `--screenshots`: home / grasp / placed / final for each attempt |
| `logs/videos/*.mp4` | With `--record` (needs `ffmpeg` on PATH) |

Screenshots/video use MuJoCo's offscreen renderer. They work normally on Windows/macOS desktops; on a headless Linux box prefix the command with `MUJOCO_GL=egl` (or `osmesa`). If rendering isn't possible the run still completes and prints a notice.

Real example of a placement verification record (from `--fault place_offset`, attempt 1):
```json
{
  "success": false,
  "reason": "Cube landed outside the container",
  "failure": "placed_outside_container",
  "checks": {
    "inside_horizontal_bounds": false, "below_rim": true, "resting_on_floor": false,
    "not_on_rim": true, "released_from_gripper": true, "stable": true
  },
  "measurements": {"cube_pos_world": [0.3999, 0.0799, 0.4199], "interior_y_local": [-0.07, 0.07],
                   "cube_xy_extent_local": [[-0.0201, 0.0199], [-0.1901, -0.1501]], "contacts": ["table_top"], "...": "..."}
}
```

## 6. Architecture
```
trustloop/
├── assets/trustloop_scene.xml   MJCF: table, 6-DOF arm, 2-finger gripper, cube, bin, cameras
├── src/
│   ├── main.py                  CLI: validate / run / batch / faults / screenshots / video
│   ├── simulation.py            model loading, reset, stepping, passive viewer, offscreen render
│   ├── kinematics.py            damped-least-squares IK (mj_kinematics + mj_jacSite), top-down targets
│   ├── robot_controller.py      min-jerk Cartesian moves → IK @100 Hz → position actuators, timeouts
│   ├── gripper.py               force-limited position-controlled fingers, open/close/settle
│   ├── verification.py          INDEPENDENT verifier: grasp, carry, placement (reads physics only)
│   ├── recovery.py              failure → whitelisted actions, deterministic policy, advisor validation
│   ├── llm_advisor.py           OPTIONAL Nemotron (Nebius Token Factory) advisor – off by default
│   ├── experiments.py           reproducible experiments: baseline / verify_only / verify_recover / octo_shadow / nemotron
│   ├── octo_shadow.py           Octo shadow mode: frame + manifest export, merge of Colab predictions
│   ├── task_manager.py          state machine, attempt budget, fault injection, orchestration
│   └── task_logger.py           JSONL event log + summary JSON
├── tests/                       63 pytest tests (model, IK, grasp, placement, end-to-end, recovery, advisor, modes, experiments)
├── octo_colab/                  Octo-Small 1.5 inference for Colab (isolated env) + shadow batch mode
└── logs/
```

**Robot.** Self-contained arm (no external download): base yaw, shoulder pitch, elbow pitch, forearm roll, wrist pitch, wrist roll — 6 revolute joints with limits — plus two slide-joint fingers. Joint-space position actuators with force limits and critical damping (`dampratio=1`); link gravity is compensated in the model (`gravcomp`). Arm links don't self-collide but collide with the world.

**Control.** Each move is a straight-line min-jerk trajectory of the grasp point with the tool pointing down. Every 10 ms, warm-started DLS IK (joint-limit clamped) converts it to joint targets, which are rate-limited (≤3 rad/s) before going to the actuators. A move succeeds only when the *measured* end effector is within 4 mm and joints have stopped; otherwise it times out (1.5 s) and raises `motion_failed`. Unreachable goals are rejected up front by IK instead of waiting.

**Gripper.** Pads with high friction (μ=1.5, torsional friction, elliptic cones, `impratio=10`). Each finger squeezes with at most 12 N. The cube (4 cm, 50 g) is held purely by contact friction.

**Verification signals.**

| Check | Evidence from simulation state |
|---|---|
| grasp_closure | contact between cube and *each* pad; finger gap consistent with cube width; cube centred between pads (tool frame) |
| grasp_lift_test | after a 3 cm test lift: cube height rose ≥70 % of gripper rise; cube–gripper offset drift < 8 mm; no cube–table contact; still pinched |
| carry | after LIFT / MOVE / DESCEND: both pad contacts and drift < 12 mm |
| placement | all 8 cube corners inside the bin interior (from bin geoms, cube half-size included, any yaw); cube top below rim; cube–floor contact and bottom at floor height; no contact at wall-top height; no robot contact and gripper ≥3 cm away; lin/ang velocity < 0.01 m/s / 0.1 rad/s after a settle wait |

Container dimensions and pose are read from the model's geoms, so moving or resizing the bin in the XML needs no code change (a test does exactly this).

**Recovery (bounded).** An *attempt* = one full pick-and-place cycle; max 3 per task.

| Failure | Allowed actions (first = deterministic default) |
|---|---|
| grasp_missed | retry_grasp → (2nd time) adjust_approach, abort_task |
| grasp_partial | adjust_approach, retry_grasp, abort_task |
| grasp_slipped / dropped_in_transit | increase_grip_force, retry_grasp, abort_task |
| placed_outside_container / resting_on_rim | repick_and_place, abort_task |
| not_settled / not_resting_on_floor | wait_and_reverify (no new attempt), … |
| still_in_gripper | reopen_and_retreat (no new attempt), abort_task |
| motion_failed | return_home_and_retry, abort_task |
| cube_unreachable / cube_fell_off_table | abort_task |

Every new attempt first does a safe retreat (open gripper, back straight up, return HOME) and **re-observes** the cube where it physically is. When the budget is spent only `abort_task` is allowed, and the run ends with FAILURE.

## 7. Nemotron recovery advisor (optional, Nebius Token Factory)
**Status: implemented and unit-tested against mocked HTTP responses. Never run against the live API**, because the build environment cannot reach `api.tokenfactory.nebius.com`. Verify it on your machine with `--check` (below) before claiming it in a demo.

What it does: when the verifier reports a failure, `RecoveryPolicy` sends Nemotron a JSON object with
`failure`, `allowed_actions`, `attempts_used`, `max_attempts`, and `evidence` (the verifier's reason, its boolean checks and a **whitelisted** set of measurements: finger gap, drift, cube extents relative to the bin, velocities, contact names). It does not send poses, joint states or other simulator internals. The model must answer `{"action": ..., "rationale": ...}`. The action is checked against the allowed list for that failure. Anything else, including joint commands, unknown actions, no JSON, an HTTP error, a timeout or a rate limit, falls back to the deterministic policy, and the log records `source: fallback` and the reason. The attempt budget is enforced in `recovery.py`, so the advisor cannot extend retries. When only `abort_task` is allowed, it isn't called at all.

API handling (`src/llm_advisor.py`): OpenAI-compatible `GET /v1/models` checks that `NEMOTRON_MODEL` exists for your key at start-up, and `POST /v1/chat/completions` handles the advice (temperature 0). Errors are classified as `auth` (401/403), `model_unavailable` (404, or not in `/models`), `rate_limited` (429; retried with bounded backoff, honouring `Retry-After` up to 5 s), `timeout`, `http_error` (5xx retried) and `malformed_response`. After an auth or model error, the advisor turns itself off for the rest of the run instead of retrying on every failure. `<think>` reasoning traces and `reasoning_content` are handled. The key is read only from `NEBIUS_API_KEY`, never printed or logged, and scrubbed from error text (this is tested).

**Linux / macOS**
```bash
export NEBIUS_API_KEY='...'                                 # never commit this
python src/llm_advisor.py --list-models                     # Nemotron ids available to YOUR key
export NEMOTRON_MODEL='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B'   # pick one from the list
python src/llm_advisor.py --check                           # key + model + one real advisory call
python src/main.py --headless --advisor nemotron --fault place_offset
```
**Windows PowerShell**
```powershell
$env:NEBIUS_API_KEY = "..."
python src/llm_advisor.py --list-models
$env:NEMOTRON_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
python src/llm_advisor.py --check
```
`--check` exits 0 only if the real model's answer was accepted (`"source": "nemotron"`). Nemotron ids listed by Nebius at the time of writing include `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/Nemotron-3_5-Lightning`, `nvidia/Nemotron-3.5-Super-VL` and `nvidia/Nemotron-3-Ultra-550b-a55b`. Always trust `--list-models` over this list. Optional: `NEBIUS_BASE_URL` (default `https://api.tokenfactory.nebius.com/v1`). In the logs, `recovery` events show `source` (`deterministic` / `nemotron` / `fallback`) and `advisor_call` (status, latency, attempts, error, rationale).

## 8. Evaluation modes and experiments

### Modes (`--mode`, same controller in all)
| Mode | Verifier | Recovery | "SUCCESS" means |
|---|---|---|---|
| `baseline` | audit only (measured, never acted on) | none | controller completed every commanded motion |
| `verify_only` | gates every stage | none, 1 attempt | physically verified placement |
| `verify_recover` (default) | gates every stage | bounded, ≤3 attempts | physically verified placement |

Every run ends with an independent **final audit** of the physical state, logged as `final_audit`. The summary JSON separates `outcome` (what the system reported), `controller_reported_success`, `verified_success` (the audit) and `false_success` (reported SUCCESS but the audit failed).

```bash
python src/main.py --headless --mode baseline --fault grasp_miss       # reports SUCCESS, false_success=True
python src/main.py --headless --mode verify_only --fault grasp_miss    # FAILURE, recovery disabled
python src/main.py --headless --mode verify_recover --fault grasp_miss # recovers, verified SUCCESS
```

### Reproducible experiments (`src/experiments.py`)
All experiments use the same seeded cube poses (`--seed`, `--trials` per condition) and the same injected faults (`--conditions`). Default conditions: `none, grasp_miss, weak_grip, place_offset, rim, unrecoverable`.
```bash
# 1-3: baseline / verification without recovery / verification + bounded recovery
python src/experiments.py --experiments baseline,verify_only,verify_recover --trials 5 --seed 0
# 4: Octo shadow mode (same controller; exports frames for Octo, see §9)
MUJOCO_GL=egl python src/experiments.py --experiments octo_shadow --trials 5 --seed 0 --out logs/experiments/shadow
# 5: Nemotron-assisted recovery (needs NEBIUS_API_KEY + NEMOTRON_MODEL; otherwise SKIPPED, never faked)
python src/experiments.py --experiments verify_recover,nemotron --trials 5 --seed 0
```
(On Windows or a desktop Linux session, omit `MUJOCO_GL=egl`.) Output in `logs/experiments/<name>/`: `results.csv` (one row per run: outcome, controller vs verified success, false success, attempts, recovery, failure reasons, advisor calls/fallbacks/latency, Octo frames, sim/wall time), `summary.json` (per experiment and per experiment×condition), `summary.md` (tables), and `logs/` (full JSONL for every run).

**Results from one actual run** (`--trials 5 --seed 0`, Linux CPU, MuJoCo 3.17, 30 runs per experiment; Nemotron skipped because no key was available):

| experiment | reported success | verified success | false successes | recovered / first-attempt failures |
|---|---|---|---|---|
| baseline | 100% | 17% | 25 | 0 / 25 |
| verify_only | 17% | 17% | 0 | 0 / 25 |
| verify_recover | 83% | 83% | 0 | 20 / 25 |
| octo_shadow | 83% | 83% | 0 | 20 / 25 |

How to read it: 25 of the 30 runs per experiment have a **deliberately injected** fault, so these rates measure fault handling, not normal-operation reliability (with no fault, every mode was 5/5). The 5 failures left under `verify_recover` are the `unrecoverable` condition, which ends cleanly as FAILURE after 3 attempts. Re-run the command to reproduce; nothing in this table is hard-coded.

## 9. Octo shadow mode
**Octo does not control the robot.** Its actions follow a training dataset's convention (Bridge: WidowX end-effector deltas plus gripper), which has not been mapped or validated for this arm. Shadow mode records what Octo predicts at the same decision points the scripted controller passes through, next to the verified outcome, for offline analysis.

1. **Export (laptop):** `--octo-shadow DIR` on `main.py`, or the `octo_shadow` experiment. This renders the overview camera at 256×256 at `pre_approach`, `grasp_closed`, `pre_release` and `final` for each attempt, writes `DIR/frames/*.png` and `DIR/manifest.jsonl` (instruction, run/attempt/state, image, previous frame), and writes task outcomes to `DIR/outcomes.jsonl`. Rendering does not touch the physics.
2. **Infer (Colab, model loaded once):** zip `DIR`, upload it, and run section 9 of `octo_colab/TrustLoop_Octo_Inference.ipynb`, i.e. `octo_inference.py --manifest DIR/manifest.jsonl --use-history`. This writes `octo_predictions.jsonl` (action chunk, normalised actions, mask, latency, JIT flag, errors).
3. **Merge (laptop):** copy `octo_predictions.jsonl` into `DIR`, then run `python src/octo_shadow.py merge DIR`. This writes `octo_shadow_merged.jsonl` (frame + Octo prediction + task outcome) and `octo_shadow_summary.json` (coverage, errors, warm latency).

Setup and caveats for the Octo environment are in `octo_colab/README_OCTO.md`.

## 10. Limitations (honest list)
- **Perception is an oracle**: the cube pose is read from the simulator (with injectable errors). No camera-based detection yet.
- **Custom arm, not a Franka Panda**: chosen so the project has no external asset dependency. It is a 6-DOF arm (not 7); top-down grasps only.
- **Top-down grasps only**: a cube lying tilted or against a wall may not be graspable; the cube is assumed upright.
- **Joint-space PD tracking**, no torque/impedance control and no collision-aware planning; straight-line moves at a fixed transport height are safe for this scene but not in general clutter.
- The deterministic recovery strategies are simple (retry, lower approach, more force, re-pick). Verification thresholds are hand-tuned for this cube/bin.
- Fault injection changes inputs to the controller (perception offset, grip force limit, place target) — the resulting failures themselves are produced by the physics, not scripted.

**Implemented and tested here:** §2–§6, the evaluation modes, the experiment runner, Octo shadow export/merge, advisor validation and fallback with mocked HTTP. **Implemented, needs your run:** live Nemotron (`--check`), real Octo-Small 1.5 weights in Colab. **Not done:** Octo controlling the arm (no validated action mapping), camera-based perception, Franka Panda model swap, collision-aware motion planning.

## 11. Success-criteria checklist (what was actually tested)
All of these were run on Linux, CPU only, MuJoCo 3.17.0. The test suite has 63 tests (`python -m pytest -q`, about 20 s). The Octo shadow test needs offscreen GL, so on headless Linux use `MUJOCO_GL=egl python -m pytest -q`; elsewhere it is skipped.

| # | Criterion | Evidence |
|---|---|---|
| 1 | Model loads without errors | `test_model_loads_with_required_elements`, `--validate` |
| 2 | Simulation starts, viewer shows robot & objects | Viewer path run under a virtual display (Xvfb) to SUCCESS; offscreen screenshots inspected. *Not tested on a physical monitor/Windows.* |
| 3 | Robot moves toward the cube | `test_arm_moves_toward_cube`, `test_ik_accuracy` |
| 4 | Gripper opens and closes | `test_gripper_opens_and_closes` |
| 5 | Grasps and lifts the cube (physics) | `test_gripper_physically_lifts_cube` (cube rises 15 cm) |
| 6 | Grasp verification distinguishes success/failure | `test_successful_grasp_passes_both_checks`, `test_closing_on_empty_space_is_detected`, `test_weak_grip_slip_is_detected_by_lift_test`, `test_issuing_close_command_alone_is_not_success`, rotated-cube cases |
| 7 | Places the cube in the container | `test_nominal_task_succeeds_with_verified_placement` |
| 8 | Placement verified independently | `test_cube_outside_container_is_not_success` (required test), `..._just_next_to_container_...`, `..._on_rim_...`, `test_falling_cube_...`, `test_cube_still_in_gripper_...`, `test_verifier_uses_model_geometry_not_hardcoded_target` |
| 9 | Failures trigger bounded recovery | `test_injected_failure_is_detected_and_recovered` ×4 faults, `test_unrecoverable_failure_reports_failure_after_max_attempts` |
| 10 | Clear SUCCESS / FAILURE result | same tests; console `OUTCOME` line + summary JSON + exit code |
| 11 | Machine-readable log | `test_machine_readable_log` (parses JSONL, checks every state appears) |
| 12 | Repeatable from fresh state | `test_repeated_runs_from_fresh_state`; manual batches: 50/50 + 50/50 nominal random poses (seeds 99, 123), 8/8 per recoverable fault, 0/8 (correctly FAILURE) for `unrecoverable` |
| — | Advisor output validation | `test_advisor_invalid_action_is_rejected`, `..._valid_action_is_used`, `..._error_falls_back`, `test_budget_exhausted_only_allows_abort` |
| — | Advisor HTTP handling (mocked) | `tests/test_advisor.py`: request format, evidence whitelist, invalid/unsupported/joint-command replies, 401/403, model not listed, 429 retry + persistent 429, timeout, malformed JSON, missing config, key never leaks, budget cannot be extended |
| — | Modes / false success | `tests/test_modes_and_experiments.py`: baseline false success, verify_only stops after 1 attempt, bogus/aborting advisor, unrecoverable terminates cleanly, experiment CSV/summary, Nemotron skipped (not faked) without a key, Octo shadow export + merge |
| — | **Not tested here** | live Nebius/Nemotron calls; real Octo weights; GPU JAX on Colab; the viewer on Windows/macOS |

## 12. Demo plan (about 3 minutes)
1. `python src/main.py --mode baseline --fault grasp_miss`: the arm "finishes" and prints SUCCESS, but the summary shows `verified_success=False, false_success=True`. *Trusting the controller lies.*
2. `python src/main.py --fault grasp_miss`: the verifier catches the empty grasp, recovery retries, and placement is verified. Show the JSONL log lines `verification` → `recovery` → `outcome`.
3. `python src/main.py --fault unrecoverable`: three bounded attempts, then a clean, honest FAILURE.
4. Show `logs/experiments/<run>/summary.md`: false successes 25 → 0, with recovery restoring 20 of 25 injected failures.
5. Nemotron, only if `python src/llm_advisor.py --check` passed on the demo machine: `python src/main.py --advisor nemotron --fault place_offset`, showing `source: nemotron` and the rationale in the log. Otherwise say it is implemented with a validated fallback, not demonstrated live.
6. Octo: show the Colab self-check and a merged shadow record. Say clearly that Octo runs in shadow mode and does not drive the arm.
