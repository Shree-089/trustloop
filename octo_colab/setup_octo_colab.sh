#!/usr/bin/env bash
# Builds an isolated Python 3.10 environment for Octo inside Google Colab.
#
# Why a separate environment: Octo pins tensorflow==2.15.0, jax==0.4.20, flax==0.7.5,
# numpy==1.24.3 (requirements.txt). TensorFlow 2.15 has no wheels for the Python
# version current Colab runtimes ship, and downgrading Colab's own packages breaks
# the runtime. So Octo gets its own venv and is run as a subprocess:
#     /content/octo_env/venv/bin/python octo_inference.py ...
#
# Needs octo_constraints.txt next to this script (extra pins, see that file).
#
# Usage (Colab cell):  !bash setup_octo_colab.sh
# Optional:            !CPU_ONLY=1 bash setup_octo_colab.sh   (skip CUDA jax)
set -euo pipefail

OCTO_COMMIT="241fb3514b7c40957a86d869fecb7c7fc353f540"   # octo-models/octo main, 2024-07-30
ENV_DIR="${OCTO_ENV_DIR:-/content/octo_env}"
VENV="$ENV_DIR/venv"
PY="$VENV/bin/python"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTRAINTS="$HERE/octo_constraints.txt"
[ -f "$CONSTRAINTS" ] || { echo "missing $CONSTRAINTS (upload it next to this script)"; exit 1; }
mkdir -p "$ENV_DIR"
echo "== [1/5] uv + Python 3.10"
python3 -m pip install -q uv
uv python install 3.10
[ -x "$PY" ] || uv venv -q --python 3.10 "$VENV"
export VIRTUAL_ENV="$VENV"

echo "== [2/5] Octo source @ $OCTO_COMMIT"
if [ ! -d "$ENV_DIR/octo/.git" ]; then
  git clone -q https://github.com/octo-models/octo.git "$ENV_DIR/octo"
fi
git -C "$ENV_DIR/octo" fetch -q --depth 1 origin "$OCTO_COMMIT" 2>/dev/null || true
git -C "$ENV_DIR/octo" checkout -q "$OCTO_COMMIT"

echo "== [3/5] Octo + pinned requirements (a few minutes)"
uv pip install -q -e "$ENV_DIR/octo" -c "$CONSTRAINTS"
uv pip install -q -r "$ENV_DIR/octo/requirements.txt" -c "$CONSTRAINTS"

echo "== [4/5] JAX 0.4.20"
if [ "${CPU_ONLY:-0}" = "1" ] || ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "   no GPU requested/found -> CPU jax"
  uv pip install -q "jax==0.4.20" "jaxlib==0.4.20" -c "$CONSTRAINTS"
else
  # The command documented in Octo's README for CUDA GPUs.
  if ! uv pip install -q "jax[cuda11_pip]==0.4.20" -c "$CONSTRAINTS" \
        --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; then
    echo "   WARNING: CUDA jax install failed -> falling back to CPU jax (inference still works, slower)"
    uv pip install -q "jax==0.4.20" "jaxlib==0.4.20" -c "$CONSTRAINTS"
  fi
fi

echo "== [5/5] smoke test"
"$PY" - <<'EOF'
import jax, tensorflow as tf, flax, numpy as np
from octo.model.octo_model import OctoModel  # noqa: F401
from transformers import FlaxT5EncoderModel  # noqa: F401  (Octo's language encoder)
print("python ok | jax", jax.__version__, "| devices", jax.devices(),
      "| tf", tf.__version__, "| flax", flax.__version__, "| numpy", np.__version__)
if jax.default_backend() != "gpu":
    print("NOTE: JAX is running on CPU. Expected on a CPU runtime; on a T4 runtime the CUDA install did not take.")
EOF
echo "Octo environment ready: $PY"
