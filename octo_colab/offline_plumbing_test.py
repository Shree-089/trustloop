"""OFFLINE plumbing test for octo_inference.py (no Hugging Face access needed).

What it tests: that octo_inference.py drives the *real* Octo library correctly —
observation/task construction, history window, jit'd sample_actions, multi-sample
draws, unnormalisation with mean/std/mask, JSON output, and the self-check.

What it does NOT test: the pretrained weights or the quality of predictions.
OctoModel.load_pretrained is replaced by OctoModel.from_config with Octo's own
Octo-Small (ViT-S) pretraining config and RANDOM weights. Because the T5 language
encoder also lives on Hugging Face, the language tokenizer is fed pre-computed
embeddings, which is a code path Octo's LanguageTokenizer supports natively.
The dataset statistics are synthetic but structured like Octo's.

Run inside the Octo env:  python offline_plumbing_test.py <image.png> [octo_repo_dir]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_fake_model(octo_repo: str):
    import importlib.util
    import jax
    import numpy as np
    from octo.model.components.tokenizers import LanguageTokenizer
    from octo.model.octo_model import OctoModel
    from octo.utils.spec import ModuleSpec

    spec = importlib.util.spec_from_file_location(
        "octo_pretrain_config", f"{octo_repo}/scripts/configs/octo_pretrain_config.py")
    cfg_mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, f"{octo_repo}/scripts/configs")
    spec.loader.exec_module(cfg_mod)
    config = cfg_mod.get_config("vit_s").to_dict()
    config["model"]["task_tokenizers"]["language"] = ModuleSpec.create(LanguageTokenizer, encoder=None)

    class FakeTextProcessor:
        def encode(self, strings):
            out = np.zeros((len(strings), 16, 768), np.float32)
            for i, s in enumerate(strings):
                out[i] = np.random.default_rng(abs(hash(s)) % 2**32).normal(size=(16, 768))
            return out

    W = 2
    example_batch = {
        "observation": {
            "image_primary": np.zeros((1, W, 256, 256, 3), np.uint8),
            "image_wrist": np.zeros((1, W, 128, 128, 3), np.uint8),
            "timestep_pad_mask": np.ones((1, W), bool),
            "pad_mask_dict": {"image_primary": np.ones((1, W), bool),
                              "image_wrist": np.ones((1, W), bool)},
        },
        "task": {
            "image_primary": np.zeros((1, 256, 256, 3), np.uint8),
            "image_wrist": np.zeros((1, 128, 128, 3), np.uint8),
            "language_instruction": np.zeros((1, 16, 768), np.float32),
            "pad_mask_dict": {"image_primary": np.ones((1,), bool),
                              "image_wrist": np.ones((1,), bool),
                              "language_instruction": np.ones((1,), bool)},
        },
    }
    rng = np.random.default_rng(0)
    stats = {
        "bridge_dataset": {"action": {
            "mean": rng.normal(0, 0.01, 7).astype(np.float32),
            "std": rng.uniform(0.01, 0.1, 7).astype(np.float32),
            "max": np.ones(7, np.float32), "min": -np.ones(7, np.float32),
            "p99": np.ones(7, np.float32), "p01": -np.ones(7, np.float32),
            "mask": np.array([True] * 6 + [False]),
        }},
        # Octo pretraining pads every dataset to max_action_dim=7, so real stats have <=7 dims.
        "fake_nav_dataset": {"action": {
            "mean": np.full(2, 0.1, np.float32), "std": np.full(2, 0.5, np.float32),
            "mask": np.array([True, True]),
        }},
    }
    return OctoModel.from_config(config, example_batch, FakeTextProcessor(),
                                 rng=jax.random.PRNGKey(0), dataset_statistics=stats)


def main():
    image = sys.argv[1]
    octo_repo = sys.argv[2] if len(sys.argv) > 2 else str(HERE.parent / "octo")
    if len(sys.argv) > 3 and sys.argv[3] == "--child":
        # child process: patch load_pretrained, then run the real CLI
        from octo.model.octo_model import OctoModel
        model = build_fake_model(octo_repo)
        OctoModel.load_pretrained = classmethod(lambda cls, path, step=None: model)
        sys.path.insert(0, str(HERE))
        import octo_inference
        return octo_inference.main(sys.argv[4:])

    tmp = Path(tempfile.mkdtemp())

    def cli(*args):
        r = subprocess.run([sys.executable, __file__, image, octo_repo, "--child", *args],
                           capture_output=True, text=True)
        tail = (r.stdout + r.stderr)[-2500:]
        return r.returncode, r.stdout, tail

    results = {}
    code, out, tail = cli("--list-stats")
    results["list_stats"] = code == 0 and "bridge_dataset" in out and "mask=[1, 1, 1, 1, 1, 1, 0]" in out
    print(tail)

    code, out, tail = cli("--image", image, "--instruction", "put the red cube in the blue bin",
                          "--out", str(tmp / "w1.json"))
    print(tail)
    w1 = json.loads((tmp / "w1.json").read_text()) if code == 0 else {}
    results["window1_shape_4x7"] = w1.get("actions_shape") == [4, 7]
    results["window1_pad_mask"] = w1.get("timestep_pad_mask") == [[True]]
    results["gripper_dim_not_unnormalised"] = (
        code == 0 and abs(w1["actions"][0][6] - w1["actions_normalized"][0][6]) < 1e-5)

    code, out, tail = cli("--image", image, "--prev-image", image, "--instruction", "pick up the cube",
                          "--samples", "3", "--out", str(tmp / "w2.json"))
    w2 = json.loads((tmp / "w2.json").read_text()) if code == 0 else {}
    results["window2_multisample_shape_3x4x7"] = w2.get("actions_shape") == [3, 4, 7]
    results["window2_pad_mask"] = w2.get("timestep_pad_mask") == [[True, True]]

    code, out, tail = cli("--image", image, "--stats-key", "fake_nav_dataset", "--out", str(tmp / "j.json"))
    j = json.loads((tmp / "j.json").read_text()) if code == 0 else {}
    results["other_stats_key_dim_from_stats_2"] = j.get("actions_shape") == [4, 2]

    code, out, tail = cli("--image", image, "--stats-key", "none", "--out", str(tmp / "n.json"))
    n = json.loads((tmp / "n.json").read_text()) if code == 0 else {}
    results["normalized_only"] = n.get("actions") is None and n.get("actions_normalized_shape") == [4, 7]

    code, out, tail = cli("--image", image, "--stats-key", "not_a_dataset")
    results["bad_stats_key_rejected"] = code != 0 and "Available" in tail

    code, out, tail = cli("--image", image, "--self-check")
    print(tail)
    results["self_check_passes"] = code == 0

    print(json.dumps(results, indent=2))
    print("ALL PLUMBING CHECKS PASSED" if all(results.values()) else "SOME CHECKS FAILED")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
