# syntax=docker/dockerfile:1

# Keep the CUDA/PyTorch/SSH contract supplied by RunPod's official pod image.
# The runner uploads the Emergence source separately, so this image contains
# only the stable Python runtime layer.
FROM --platform=linux/amd64 runpod/pytorch@sha256:517c88bc0264c688b3bf83f9d684cc0236b160f708a78de2804f26da91280eae

WORKDIR /opt/emergence-runtime
COPY requirements.lock .

# The Ubuntu 24.04 base is externally managed.  These are application runtime
# dependencies only; torch/CUDA remain owned by the pinned RunPod base image.
RUN python -m pip install \
      --break-system-packages \
      --no-cache-dir \
      --only-binary=:all: \
      --requirement requirements.lock \
    && python - <<'PY'
import importlib.metadata
import numba
import numpy
import torch
import yaml

assert numpy.__version__ == "2.3.5", numpy.__version__
assert numba.__version__ == "0.63.1", numba.__version__
assert importlib.metadata.version("PyYAML") == "6.0.3"
assert torch.__version__.startswith("2.9.1"), torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print("EMERGENCE_CUDA_RUNTIME_OK")
PY

# Deliberately inherit the base image's /start.sh CMD.  It establishes SSH and
# the web terminal that the pod runner needs; the simulation is launched later
# over SSH after CUDA admission and source transfer.
