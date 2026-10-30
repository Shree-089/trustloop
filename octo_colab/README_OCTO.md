# TrustLoop × Octo-Small 1.5: stage 1, standalone inference

This stage only runs the **pretrained** Octo-Small 1.5 model (no fine-tuning) on a TrustLoop RGB screenshot and records the action chunk it predicts. It does **not** touch the robot controller, verifier, recovery manager or scripted baseline. Nothing in `src/` was changed.

## Minimal architecture (target, built in stages)

```
 Laptop (CPU, existing TrustLoop env)                 Colab (T4, isolated Octo env)
 ─────────────────────────────────────                ────────────────────────────────────────
 MuJoCo sim ──► RGB frame (PNG) ───── upload ───────► octo_inference.py
                                                         preprocess: resize → uint8 256×256
                                                         obs  = {image_primary [1,T,256,256,3],
                                                                 timestep_pad_mask [1,T]}
                                                         task = create_tasks(texts=[instruction])
                                                         sample_actions(..., dataset stats) ──►
 (stage 3) adapter: Octo action ──► TrustLoop  ◄── JSON  action chunk [4, 7] in the training
           safety limits ──► RobotController             dataset's convention + metadata
           ──► Verifier ──► RecoveryPolicy  (unchanged; scripted baseline kept as fallback)
```

* **Stage 1 (this folder):** Colab-only inference on screenshots → JSON. ✅ code written
* **Stage 2 (next):** calibrate what Octo's outputs mean against TrustLoop (frames, scale, gripper), offline.
* **Stage 3:** a guarded adapter that turns an Octo chunk into a bounded TrustLoop end-effector target, behind the existing verifier and recovery. The scripted baseline stays the default.

Why Colab plus a separate venv: Octo pins `tensorflow==2.15.0`, `jax==0.4.20`, `flax==0.7.5` and `numpy==1.24.3`. TF 2.15 doesn't install on Colab's current Python, and installing these pins into Colab's (or your laptop's) main environment would break other things. `setup_octo_colab.sh` builds a Python 3.10 venv at `/content/octo_env/venv`, and Octo runs there as a subprocess.

## Files

| File | Purpose |
|---|---|
| `octo_inference.py` | Standalone module + CLI: `OctoPolicy.load → predict(image, instruction, prev_image=None, stats_key=...)` |
| `setup_octo_colab.sh` | Builds the isolated Octo environment in Colab (Octo pinned to commit `241fb35`) |
| `octo_constraints.txt` | 3 extra pins Octo is missing today; must sit next to the setup script |
| `TrustLoop_Octo_Inference.ipynb` | Colab notebook: GPU check → setup → list stats → upload screenshots → self-check → inference → plots → download |
| `offline_plumbing_test.py` | Tests the module against the real Octo library with random weights (no Hugging Face needed) |

## Shadow mode batch (stage 2a, implemented)
`octo_inference.py --manifest <dir>/manifest.jsonl [--use-history] [--pred-out FILE] [--limit N]` runs Octo on every frame exported by TrustLoop's shadow recorder (`src/octo_shadow.py`). The model loads once, each frame produces one JSONL row (with an `error` field instead of crashing), and an interrupted run resumes. Notebook section 9 does this in Colab. Merge back on the laptop with `python src/octo_shadow.py merge <dir>`. Octo's predictions are logged next to the verified task outcome. They are **not** executed.

## Put it in your project
Unzip so the folder sits at `~/Downloads/trustloop/octo_colab/`. It's self-contained: it adds no dependencies to your laptop's TrustLoop environment and changes nothing in `src/`, `tests/` or `assets/`.

## Offline plumbing test (optional, any Linux machine with the Octo env)
```bash
PY=/content/octo_env/venv/bin/python        # or your own Octo env
$PY offline_plumbing_test.py some_screenshot.png /content/octo_env/octo
```
It should end with `ALL PLUMBING CHECKS PASSED`.

## Run it in Colab

1. Open `TrustLoop_Octo_Inference.ipynb` in Colab (File → Upload notebook). Runtime → Change runtime type → **T4 GPU**.
2. Run the cells top to bottom. When asked, upload `octo_inference.py`, `setup_octo_colab.sh` and `octo_constraints.txt`, then your screenshots.
   * To make screenshots on the laptop: `python src/main.py --headless --screenshots` → `logs/screenshots/*.png`.
3. Cell 5 (`--self-check`) must print `"all_passed": true`.
4. Cell 8 downloads `octo_results.zip` (one JSON per image + `self_check.json`).

To use the terminal instead of the notebook (in a Colab cell, prefix each with `!`):
```bash
bash setup_octo_colab.sh
PY=/content/octo_env/venv/bin/python
$PY octo_inference.py --list-stats
$PY octo_inference.py --image home.png --instruction "put the red cube in the blue bin" --self-check
$PY octo_inference.py --image home.png --instruction "put the red cube in the blue bin" --out home_octo.json
$PY octo_inference.py --image t1.png --prev-image t0.png --instruction "..." --samples 5 --out t1_octo.json
```

Python API (inside the Octo env):
```python
from octo_inference import OctoPolicy
policy = OctoPolicy()                           # hf://rail-berkeley/octo-small-1.5
pred = policy.predict("home.png", "put the red cube in the blue bin", stats_key="bridge_dataset", seed=0)
pred.actions            # np.ndarray [4, 7]  unnormalised with bridge_dataset stats
pred.actions_normalized # np.ndarray [4, 7]  raw model space
pred.to_dict()          # JSON-ready, includes mask, stats and timing
```

## What the code uses from Octo, and where it comes from

| Item | Value / behaviour | Source |
|---|---|---|
| Load | `OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")` | README, example notebook |
| Observation | `image_primary` uint8 `[B, T, H, W, 3]` + `timestep_pad_mask` bool `[B, T]` | example notebook, `sample_actions` docstring |
| Image size | read at runtime from `example_batch` (pretraining config: primary 256×256) | `octo_pretrain_config.py` |
| History | window 1, or 2 with `--prev-image` (Octo was pretrained with window 2) | README FAQ |
| Wrist camera | omitted → treated as padding | README FAQ (`pad_mask_dict`), `tokenizers.py` |
| Task | `model.create_tasks(texts=[instruction])` | README |
| Output | `[*sample_shape, B, action_horizon, action_dim]`; horizon 4 for the pretrained model | `sample_actions` docstring, README FAQ |
| Unnormalisation | `model.dataset_statistics[key]["action"]` (mean, std, mask). `mask=False` dims are left as predicted | `octo_model.py` (`NormalizationType.NORMAL` is the default) |
| Resize | `tf.image.resize(lanczos3, antialias)` → round → uint8, as in Octo's data pipeline (dlimp) | `obs_transforms.py` |

**Not invented:** the action dimension comes from `len(stats["mean"])`, the horizon from the model output, and normalisation from the checkpoint's own statistics. The self-check recomputes `normalized*std+mean` (with mask) by hand and compares it with Octo's result.

## What the numbers mean, and what they don't

With `--stats-key bridge_dataset`, the 7 dims follow Octo's **EEF_POS** encoding for Bridge (from `oxe_dataset_configs.py`): `[Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]`. Octo relabels the 6 motion dims as the difference between consecutive end-effector states of the **WidowX** robot (`relabel_actions`). The gripper is binarised in the training data (≈1 open, ≈0 closed, `binarize_gripper_actions`) and is not unnormalised.

They are **not** TrustLoop commands. Unknowns that need testing before stage 3:
* the coordinate frame (WidowX base frame vs your world frame), the axis signs, and the rotation convention;
* the time scale: Bridge's control rate vs your 100 Hz controller, i.e. how far one Δ should move your arm;
* whether a 0.5 gripper threshold separates open from close for your scenes;
* the domain gap: your MuJoCo camera, arm and colours look nothing like Bridge. Zero-shot outputs may be noisy or meaningless. That is expected and is exactly why the verifier and recovery stay in charge.

Other `dataset_statistics` keys (see `--list-stats`) give other robots' conventions. Their action dims and masks can differ, and the code adapts to them.

Aspect ratio: 1280×720 screenshots are squashed to 256×256, like Octo's training resize. A square 256×256 MuJoCo render would avoid the distortion; that's a stage-2 option.

## Testing status

| Item | Status |
|---|---|
| Octo API, shapes and normalisation checked against Octo source (commit `241fb35`) | ✅ done |
| `setup_octo_colab.sh` from scratch in a clean folder (`CPU_ONLY=1`, Linux): Python 3.10 venv, Octo @ `241fb35`, pinned stack imports (jax 0.4.20, tf 2.15.0, flax 0.7.5, numpy 1.24.3, FlaxT5) | ✅ tested. Without `octo_constraints.txt` the import **fails** today (protobuf / scipy / transformers drift) |
| Module + CLI against the real Octo library with **random weights**, Octo-Small (ViT-S) architecture (`offline_plumbing_test.py`): window 1 → `[4,7]`, pad mask `[[True]]`; window 2 → pad mask `[[True,True]]`; 3 samples → `[3,4,7]`; gripper dim not unnormalised; output dim follows the chosen stats; normalised-only mode; unknown stats key rejected; `--self-check` passes | ✅ 10/10 passed, in two separately built envs |
| CPU timing with random weights (no meaning for quality) | first call 17–20 s (JIT compile), then ≈0.5–0.6 s per call |
| Downloading the real `octo-small-1.5` checkpoint and T5 tokenizer from Hugging Face | ❗ **needs testing in Colab** (blocked in the build environment) |
| `--self-check` with real weights | ❗ **needs testing in Colab** |
| CUDA JAX (`jax[cuda11_pip]==0.4.20`) on the Colab T4 | ❗ **needs testing in Colab**; fallback: `CPU_ONLY=1 bash setup_octo_colab.sh` |
| Inference speed on T4 | ❗ measure (the self-check reports warm and first-call times) |
| Meaning of the actions for TrustLoop | ❗ stage 2 |
