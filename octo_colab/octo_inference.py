"""Standalone Octo-Small 1.5 inference for TrustLoop (run on Colab or any Octo env).

This file is deliberately independent of the TrustLoop simulator: it takes an
RGB image (a MuJoCo screenshot) plus a language instruction and returns the
action chunk that the *pretrained* Octo-Small 1.5 model predicts. It does NOT
send anything to a robot and does NOT map actions onto the TrustLoop arm.

Everything follows Octo's own documented interface (github.com/octo-models/octo,
README + examples/01_inference_pretrained.ipynb + octo/model/octo_model.py):

  model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
  observation = {"image_primary": uint8[B, T, 256, 256, 3],
                 "timestep_pad_mask": bool[B, T]}
  task = model.create_tasks(texts=["..."])
  actions = model.sample_actions(observation, task,
                unnormalization_statistics=model.dataset_statistics[KEY]["action"],
                rng=jax.random.PRNGKey(seed))         # -> [B, action_horizon, action_dim]

Nothing about shapes or normalisation is hard-coded here: image size and the
maximum history window are read from the checkpoint's `example_batch`, the
action horizon/dim are read from the model output, and unnormalisation uses the
checkpoint's own `dataset_statistics` (including its "mask", which leaves the
gripper dimension un-normalised for EEF_POS datasets such as Bridge).

Usage (inside the Octo Python 3.10 environment):
  python octo_inference.py --list-stats
  python octo_inference.py --image shot.png --instruction "put the red cube in the blue bin"
  python octo_inference.py --image t1.png --prev-image t0.png --instruction "..." --out result.json
  python octo_inference.py --image shot.png --instruction "..." --self-check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")      # as in Octo's example notebook
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402

DEFAULT_CHECKPOINT = "hf://rail-berkeley/octo-small-1.5"
DEFAULT_STATS_KEY = "bridge_dataset"   # what Octo's own pretrained-inference example uses


# --------------------------------------------------------------------------- results
@dataclass
class OctoPrediction:
    """One inference call. Arrays have the batch dimension removed."""

    instruction: str
    stats_key: Optional[str]
    seed: int
    window_size: int
    timestep_pad_mask: list
    actions_normalized: np.ndarray            # [action_horizon, model_action_dim] (or [S, H, D])
    actions: Optional[np.ndarray]             # unnormalised with stats_key, [H, D] (or [S, H, D])
    normalization_mask: Optional[list]        # True = dim was unnormalised with mean/std
    stats: Optional[dict] = None
    inference_seconds: float = 0.0
    jit_compiled_this_call: bool = False
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        def arr(a):
            return None if a is None else np.asarray(a).round(6).tolist()

        return {
            "instruction": self.instruction,
            "stats_key": self.stats_key,
            "seed": self.seed,
            "window_size": self.window_size,
            "timestep_pad_mask": self.timestep_pad_mask,
            "actions_shape": None if self.actions is None else list(np.shape(self.actions)),
            "actions_normalized_shape": list(np.shape(self.actions_normalized)),
            "actions": arr(self.actions),
            "actions_normalized": arr(self.actions_normalized),
            "normalization_mask": self.normalization_mask,
            "stats": self.stats,
            "inference_seconds": round(self.inference_seconds, 4),
            "jit_compiled_this_call": self.jit_compiled_this_call,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- policy
class OctoPolicy:
    """Thin wrapper around a pretrained OctoModel. Inference only, no fine-tuning."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, verbose: bool = True):
        import jax
        from octo.model.octo_model import OctoModel

        self._jax = jax
        self.checkpoint = checkpoint
        t0 = time.perf_counter()
        self.model = OctoModel.load_pretrained(checkpoint)
        self.load_seconds = time.perf_counter() - t0
        self._compiled_shapes: set = set()

        obs_example = self.model.example_batch["observation"]
        if "image_primary" not in obs_example:
            raise RuntimeError(
                f"Checkpoint has no 'image_primary' observation; keys = {sorted(obs_example)}"
            )
        prim = np.shape(obs_example["image_primary"])        # (B, T, H, W, C)
        self.image_size = (int(prim[-3]), int(prim[-2]))
        self.max_window = int(prim[1])
        self.observation_keys = sorted(obs_example.keys())
        self.task_keys = sorted(self.model.example_batch["task"].keys())
        if verbose:
            print(self.describe())

    # ----------------------------------------------------------------- introspection
    def describe(self) -> str:
        devs = ", ".join(f"{d.platform}:{getattr(d, 'device_kind', '?')}" for d in self._jax.devices())
        head = self.model.config["model"]["heads"]["action"]["kwargs"]
        return (
            f"Loaded {self.checkpoint} in {self.load_seconds:.1f}s | JAX devices: {devs}\n"
            f"  example_batch observation keys: {self.observation_keys}\n"
            f"  image_primary size: {self.image_size}, max history window: {self.max_window}\n"
            f"  task keys: {self.task_keys}\n"
            f"  action head config: action_horizon={head.get('action_horizon')}, "
            f"action_dim={head.get('action_dim')}\n"
            f"  dataset_statistics keys: {len(self.model.dataset_statistics)} datasets"
        )

    def dataset_keys(self) -> list[str]:
        return sorted(self.model.dataset_statistics.keys())

    def action_statistics(self, stats_key: str) -> dict:
        """The checkpoint's own action statistics for one training dataset."""
        all_stats = self.model.dataset_statistics
        if stats_key not in all_stats:
            raise KeyError(
                f"'{stats_key}' not in this checkpoint's dataset_statistics. "
                f"Available: {self.dataset_keys()}"
            )
        return all_stats[stats_key]["action"]

    # ----------------------------------------------------------------- preprocessing
    def preprocess_image(self, image: Any) -> np.ndarray:
        """RGB image (path / PIL / HxWx3 or HxWx4 uint8 array) -> uint8[H, W, 3] at the model's size.

        Resizing uses tf.image.resize(method="lanczos3", antialias=True) followed by
        round+clip to uint8, matching dlimp.transforms.resize_image, which Octo's
        training data pipeline uses. Pixel normalisation is NOT done here: Octo's
        image encoder normalises raw uint8 images internally.
        """
        import tensorflow as tf

        if isinstance(image, (str, Path)):
            from PIL import Image

            image = np.asarray(Image.open(image).convert("RGB"))
        elif not isinstance(image, np.ndarray):          # PIL.Image
            image = np.asarray(image.convert("RGB"))
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"expected HxWx3 RGB image, got shape {image.shape}")
        image = image[..., :3]
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                raise ValueError("float image in [0,1] given; pass uint8 0-255 RGB")
            image = np.clip(image, 0, 255).astype(np.uint8)
        if tuple(image.shape[:2]) == self.image_size:
            return image
        out = tf.image.resize(image, self.image_size, method="lanczos3", antialias=True)
        out = tf.cast(tf.clip_by_value(tf.round(out), 0, 255), tf.uint8)
        return out.numpy()

    def build_observation(self, image, prev_image=None) -> dict:
        """Observation dict in Octo's documented format.

        * window 1: {"image_primary": [1,1,H,W,3], "timestep_pad_mask": [[True]]}
        * window 2 (Octo was pretrained with history 2): previous frame first,
          current frame last, pad mask [[True, True]].
        Only the primary (third-person) camera is supplied. Octo treats a missing
        key such as image_wrist as padding (README FAQ, `pad_mask_dict`).
        """
        frames = [self.preprocess_image(image)]
        if prev_image is not None:
            if self.max_window < 2:
                raise ValueError("checkpoint was trained with window 1; prev_image not supported")
            frames.insert(0, self.preprocess_image(prev_image))
        stacked = np.stack(frames)[None]                  # [1, T, H, W, 3]
        return {
            "image_primary": stacked,
            "timestep_pad_mask": np.ones((1, stacked.shape[1]), dtype=bool),
        }

    # ----------------------------------------------------------------- inference
    def predict(
        self,
        image,
        instruction: str,
        prev_image=None,
        stats_key: Optional[str] = DEFAULT_STATS_KEY,
        seed: int = 0,
        num_samples: int = 1,
        argmax: bool = False,
    ) -> OctoPrediction:
        """Run real Octo inference. Returns both normalised and unnormalised chunks.

        num_samples > 1 draws several samples from the diffusion head (sample_shape),
        which is useful for seeing how certain the model is.
        """
        jax = self._jax
        obs = self.build_observation(image, prev_image)
        task = self.model.create_tasks(texts=[instruction])
        rng = jax.random.PRNGKey(seed)
        sample_shape = () if num_samples == 1 else (num_samples,)
        shape_sig = (obs["image_primary"].shape, num_samples, argmax, stats_key)
        compiled = shape_sig not in self._compiled_shapes

        t0 = time.perf_counter()
        # 1) model-space (normalised) actions: no unnormalisation statistics
        normalized = self.model.sample_actions(
            obs, task, rng=rng, argmax=argmax, sample_shape=sample_shape
        )
        normalized = np.asarray(jax.device_get(normalized))
        # 2) embodiment actions: same rng, so the same diffusion sample, unnormalised by Octo
        actions, mask_list, stats_summary = None, None, None
        if stats_key is not None:
            stats = self.action_statistics(stats_key)
            actions = self.model.sample_actions(
                obs, task, unnormalization_statistics=stats, rng=rng,
                argmax=argmax, sample_shape=sample_shape,
            )
            actions = np.asarray(jax.device_get(actions))
            mask = np.asarray(stats.get("mask", np.ones_like(stats["mean"], dtype=bool)))
            mask_list = mask.astype(bool).tolist()
            stats_summary = {
                k: (np.asarray(v).round(6) if np.issubdtype(np.asarray(v).dtype, np.floating)
                    else np.asarray(v)).tolist()
                for k, v in stats.items()
            }
        elapsed = time.perf_counter() - t0
        self._compiled_shapes.add(shape_sig)

        # remove batch dim: (*sample_shape, B=1, H, D) -> (*sample_shape, H, D)
        normalized = normalized[..., 0, :, :]
        if actions is not None:
            actions = actions[..., 0, :, :]

        notes = [
            "actions are Octo's prediction in the action convention of the training "
            f"dataset '{stats_key}', NOT TrustLoop joint or end-effector commands",
        ]
        if mask_list is not None and not all(mask_list):
            idx = [i for i, m in enumerate(mask_list) if not m]
            notes.append(f"dims {idx} are not unnormalised (mask=False) and are returned as predicted")
        return OctoPrediction(
            instruction=instruction,
            stats_key=stats_key,
            seed=seed,
            window_size=int(obs["image_primary"].shape[1]),
            timestep_pad_mask=obs["timestep_pad_mask"].tolist(),
            actions_normalized=normalized,
            actions=actions,
            normalization_mask=mask_list,
            stats=stats_summary,
            inference_seconds=elapsed,
            jit_compiled_this_call=compiled,
            notes=notes,
        )

    # ----------------------------------------------------------------- self-check
    def self_check(self, image, instruction: str, stats_key: str = DEFAULT_STATS_KEY) -> dict:
        """Checks that can be verified without a robot. Returns {check: bool, ...}."""
        p1 = self.predict(image, instruction, stats_key=stats_key, seed=0)
        p2 = self.predict(image, instruction, stats_key=stats_key, seed=0)
        p3 = self.predict(image, instruction, stats_key=stats_key, seed=1)
        stats = self.action_statistics(stats_key)
        mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
        mask = np.asarray(stats.get("mask", np.ones_like(mean, dtype=bool)))
        d = len(mean)
        # recompute Octo's NORMAL unnormalisation by hand from the normalised output
        manual = np.where(mask, p1.actions_normalized[..., :d] * std + mean, p1.actions_normalized[..., :d])
        head = self.model.config["model"]["heads"]["action"]["kwargs"]
        t0 = time.perf_counter()
        self.predict(image, instruction, stats_key=stats_key, seed=2)
        warm = time.perf_counter() - t0
        checks = {
            "output_is_2d_chunk": p1.actions.ndim == 2,
            "action_horizon_matches_config": p1.actions.shape[0] == head.get("action_horizon"),
            "action_dim_matches_stats": p1.actions.shape[1] == d,
            "all_finite": bool(np.all(np.isfinite(p1.actions))),
            "deterministic_for_same_seed": bool(np.allclose(p1.actions, p2.actions)),
            "differs_for_other_seed": bool(not np.allclose(p1.actions_normalized, p3.actions_normalized)),
            "unnormalisation_matches_mean_std_mask": bool(np.allclose(manual, p1.actions, atol=1e-4)),
            "window2_runs": self.predict(image, instruction, prev_image=image,
                                         stats_key=stats_key).actions.shape == p1.actions.shape
            if self.max_window >= 2 else True,
        }
        return {
            "checks": checks,
            "all_passed": all(checks.values()),
            "actions_shape": list(p1.actions.shape),
            "warm_inference_seconds": round(warm, 4),
            "first_call_seconds_incl_jit": round(p1.inference_seconds, 2),
        }


# --------------------------------------------------------------------------- batch (shadow mode)
def run_manifest(policy: "OctoPolicy", manifest: str, out_path: str | None, stats_key: Optional[str],
                 seed: int = 0, use_history: bool = False, limit: int | None = None) -> dict:
    """Run Octo on every frame listed in a TrustLoop shadow manifest (src/octo_shadow.py).

    The model is loaded once (by the caller) and reused for all frames. Image paths
    are resolved relative to the manifest's folder. One output row per frame; a
    failing frame is recorded with an "error" field instead of stopping the batch.
    Rows already present in out_path are skipped, so an interrupted run can resume.
    """
    mpath = Path(manifest)
    base = mpath.parent
    out = Path(out_path) if out_path else base / "octo_predictions.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["frame_id"] for l in out.read_text().splitlines() if l.strip()}
    rows = [json.loads(l) for l in mpath.read_text().splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    n_ok = n_err = n_skip = 0
    with out.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            if row["frame_id"] in done:
                n_skip += 1
                continue
            rec = {"frame_id": row["frame_id"], "run_id": row.get("run_id"),
                   "checkpoint": policy.checkpoint, "error": None}
            try:
                prev = base / row["prev_image"] if (use_history and row.get("prev_image")) else None
                pred = policy.predict(base / row["image"], row["instruction"], prev_image=prev,
                                      stats_key=stats_key, seed=seed)
                d = pred.to_dict()
                d.pop("stats", None)          # same for every frame; keep rows small
                rec.update(d)
                n_ok += 1
            except Exception as exc:          # record and continue
                rec["error"] = f"{type(exc).__name__}: {exc}"
                n_err += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i + 1}/{len(rows)}] {row['frame_id']}: "
                  f"{'ERROR ' + rec['error'] if rec['error'] else 'ok %.3fs' % rec['inference_seconds']}")
    return {"frames": len(rows), "predicted": n_ok, "errors": n_err, "skipped_existing": n_skip,
            "output": str(out)}


# --------------------------------------------------------------------------- CLI
def _print_chunk(pred: OctoPrediction) -> None:
    a = pred.actions if pred.actions is not None else pred.actions_normalized
    label = f"unnormalised with '{pred.stats_key}'" if pred.actions is not None else "normalised"
    print(f"\nInstruction: {pred.instruction!r}  window={pred.window_size}  seed={pred.seed}")
    print(f"Action chunk {list(a.shape)} ({label}); inference {pred.inference_seconds:.3f}s"
          f"{' (incl. JIT compile)' if pred.jit_compiled_this_call else ''}")
    if a.ndim == 2:
        for t, row in enumerate(a):
            print(f"  step {t}: " + "  ".join(f"{v:+.5f}" for v in row))
    if pred.normalization_mask is not None:
        print(f"Normalisation mask: {pred.normalization_mask}")
    for n in pred.notes:
        print(f"NOTE: {n}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pretrained Octo-Small 1.5 inference on an RGB image")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--image", help="current RGB frame (png/jpg)")
    ap.add_argument("--prev-image", help="optional previous frame -> history window of 2")
    ap.add_argument("--instruction", default="put the red cube in the blue bin")
    ap.add_argument("--stats-key", default=DEFAULT_STATS_KEY,
                    help="dataset_statistics key used to unnormalise; 'none' = normalised only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=1, help="number of diffusion samples")
    ap.add_argument("--argmax", action="store_true", help="pass argmax=True to sample_actions")
    ap.add_argument("--out", help="write the prediction as JSON")
    ap.add_argument("--list-stats", action="store_true", help="print dataset_statistics keys and exit")
    ap.add_argument("--self-check", action="store_true", help="run shape/normalisation/determinism checks")
    ap.add_argument("--manifest", help="TrustLoop shadow manifest.jsonl: batch-predict all frames")
    ap.add_argument("--pred-out", help="batch output (default <manifest dir>/octo_predictions.jsonl)")
    ap.add_argument("--use-history", action="store_true",
                    help="batch: pass each frame's previous frame (window 2)")
    ap.add_argument("--limit", type=int, help="batch: only the first N frames")
    args = ap.parse_args(argv)

    policy = OctoPolicy(args.checkpoint)
    if args.list_stats:
        for k in policy.dataset_keys():
            s = policy.model.dataset_statistics[k]["action"]
            print(f"  {k:<55} action_dim={len(s['mean'])} mask={np.asarray(s.get('mask', [])).astype(int).tolist()}")
        return 0
    stats_key = None if args.stats_key.lower() == "none" else args.stats_key
    if args.manifest:
        res = run_manifest(policy, args.manifest, args.pred_out, stats_key, args.seed,
                           args.use_history, args.limit)
        print(json.dumps(res, indent=2))
        return 0 if res["errors"] == 0 else 1
    if not args.image:
        ap.error("--image is required (unless --list-stats or --manifest)")

    if args.self_check:
        report = policy.self_check(args.image, args.instruction, stats_key or DEFAULT_STATS_KEY)
        print(json.dumps(report, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2))
        return 0 if report["all_passed"] else 1

    pred = policy.predict(args.image, args.instruction, prev_image=args.prev_image,
                          stats_key=stats_key, seed=args.seed, num_samples=args.samples,
                          argmax=args.argmax)
    _print_chunk(pred)
    if args.out:
        out = pred.to_dict()
        out.update(checkpoint=args.checkpoint, image=args.image, prev_image=args.prev_image)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
