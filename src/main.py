"""TrustLoop entry point.

Examples (run from the project root):
    python src/main.py                         # viewer, nominal task
    python src/main.py --fault grasp_miss      # demo grasp failure + recovery
    python src/main.py --headless --runs 5 --random-cube --seed 1
    python src/main.py --validate              # load model & step, then exit
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation import DEFAULT_SCENE, PROJECT_ROOT, Simulation  # noqa: E402
from task_logger import TaskLogger  # noqa: E402
from task_manager import DEFAULT_INSTRUCTION, FAULTS, MODES, TrustLoopTask  # noqa: E402


def validate_model(scene: Path) -> int:
    import mujoco
    sim = Simulation(scene)
    sim.reset()
    m = sim.model
    for _ in range(1000):
        sim.step()
    names = {"bodies": m.nbody, "joints": m.njnt, "actuators": m.nu, "geoms": m.ngeom}
    ok = np.all(np.isfinite(sim.data.qpos)) and abs(sim.cube_pos()[2] - 0.42) < 0.005
    print(f"MuJoCo {mujoco.__version__} | model: {scene.name} | {names}")
    print(f"cube resting at {np.round(sim.cube_pos(), 4).tolist()} after 2 s of physics")
    print("MODEL OK" if ok else "MODEL PROBLEM")
    return 0 if ok else 1


def random_cube(rng: np.random.Generator) -> tuple[list[float], float]:
    return ([float(rng.uniform(0.32, 0.48)), float(rng.uniform(-0.30, -0.12)), 0.42],
            float(rng.uniform(-np.pi / 4, np.pi / 4)))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="TrustLoop: self-verifying pick-and-place in MuJoCo")
    p.add_argument("--scene", default=str(DEFAULT_SCENE))
    p.add_argument("--headless", action="store_true", help="no viewer, runs as fast as possible")
    p.add_argument("--fast", action="store_true", help="viewer without real-time pacing")
    p.add_argument("--fault", choices=FAULTS, default="none", help="inject a controlled failure")
    p.add_argument("--mode", choices=MODES, default="verify_recover",
                   help="baseline (unverified scripted), verify_only (no recovery), verify_recover (default)")
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--octo-shadow", metavar="DIR",
                   help="export 256x256 observation frames + manifest for Octo shadow inference")
    p.add_argument("--runs", type=int, default=1, help="repeat the task from a fresh reset N times")
    p.add_argument("--random-cube", action="store_true", help="randomise cube start pose per run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cube", type=float, nargs=3, metavar=("X", "Y", "Z"), help="cube start position")
    p.add_argument("--cube-yaw", type=float, default=0.0, help="cube start yaw (degrees)")
    p.add_argument("--log-dir", default=str(PROJECT_ROOT / "logs"))
    p.add_argument("--screenshots", action="store_true", help="save PNGs to logs/screenshots")
    p.add_argument("--record", action="store_true",
                   help="save an MP4 of each run to logs/videos (needs ffmpeg)")
    p.add_argument("--advisor", choices=["none", "nemotron"], default="none",
                   help="optional LLM recovery advisor (needs NEBIUS_API_KEY, NEMOTRON_MODEL)")
    p.add_argument("--keep-open", action="store_true", help="keep viewer open after the task")
    p.add_argument("--validate", action="store_true", help="validate the MJCF model and exit")
    args = p.parse_args(argv)

    if args.validate:
        return validate_model(Path(args.scene))

    advisor = None
    if args.advisor == "nemotron":
        from llm_advisor import AdvisorError, NemotronAdvisor
        try:
            advisor = NemotronAdvisor()      # verifies the model id against /v1/models
            print(f"[advisor] Nemotron enabled: {advisor.model}")
        except AdvisorError as exc:
            print(f"[advisor] Nemotron disabled ({exc.category}: {exc}); "
                  "using deterministic recovery policy.")
    recorder = None
    if args.octo_shadow:
        from octo_shadow import OctoShadowRecorder
        recorder = OctoShadowRecorder(args.octo_shadow)

    rng = np.random.default_rng(args.seed)
    sim = Simulation(args.scene, viewer=not args.headless, realtime=not args.fast,
                     screenshot_dir=Path(args.log_dir) / "screenshots" if args.screenshots else None)
    results = []
    try:
        for i in range(args.runs):
            if args.random_cube:
                pos, yaw = random_cube(rng)
            elif args.cube:
                pos, yaw = list(args.cube), np.radians(args.cube_yaw)
            else:
                pos, yaw = None, np.radians(args.cube_yaw)
            sim.reset(cube_pos=pos, cube_yaw=yaw)          # fresh simulation state every run
            sim.open_viewer()
            run_id = time.strftime("run_%Y%m%d_%H%M%S") + (f"_{args.fault}" if args.fault != "none" else "")
            if args.runs > 1:
                run_id += f"_r{i + 1:02d}"
            logger = TaskLogger(args.log_dir, run_id=run_id)
            print(f"\n=== TrustLoop run {i + 1}/{args.runs} | fault={args.fault} | "
                  f"cube={np.round(sim.cube_pos(), 3).tolist()} ===")
            task = TrustLoopTask(sim, logger, fault=args.fault, advisor=advisor,
                                 screenshots=args.screenshots, mode=args.mode,
                                 frame_hook=recorder.hook if recorder else None,
                                 instruction=args.instruction)
            if args.record:
                sim.start_recording(Path(args.log_dir) / "videos" / f"{run_id}.mp4")
            summary = task.run()
            if recorder is not None:
                recorder.record_outcome(summary)
            if args.record:
                video = sim.stop_recording()
                if video:
                    summary["video"] = str(video)
                    print(f"    video: {video}")
            logger.close()
            results.append(summary)
            print(f"==> {summary['outcome']} after {summary['attempts_used']} attempt(s): "
                  f"{summary['reason']}\n    mode={summary['mode']} controller_reported_success="
                  f"{summary['controller_reported_success']} verified_success={summary['verified_success']}"
                  f" false_success={summary['false_success']}"
                  f"\n    log: {summary['log_file']}\n    summary: {summary['summary_file']}")
            if not sim.viewer_alive():
                break
        if args.runs > 1:
            n_ok = sum(r["outcome"] == "SUCCESS" for r in results)
            print(f"\nBatch: {n_ok}/{len(results)} SUCCESS")
            batch = Path(args.log_dir) / "batch_latest.json"
            batch.write_text(json.dumps([{k: r[k] for k in ("run_id", "outcome", "reason", "attempts_used")}
                                         for r in results], indent=2))
        if args.keep_open and sim.viewer is not None:
            print("Viewer left open – close the window to exit.")
            while sim.viewer.is_running():
                sim.step(10)
    finally:
        if recorder is not None:
            recorder.close()
            print(f"Octo shadow: {recorder.frames_written} frame(s) -> {recorder.manifest}"
                  + (f" (disabled: {recorder.disabled_reason})" if recorder.disabled_reason else ""))
        sim.close()
    return 0 if results and all(r["outcome"] == "SUCCESS" for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
