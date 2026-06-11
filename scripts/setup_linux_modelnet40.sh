#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python - <<'PY'
import re
import subprocess
import sys
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_HOME:", CUDA_HOME)
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch is required before running this script.")
if CUDA_HOME is None:
    raise SystemExit(
        "CUDA toolkit was not found. Install nvcc and set CUDA_HOME before compiling."
    )

try:
    output = subprocess.check_output(
        [f"{CUDA_HOME}/bin/nvcc", "--version"],
        text=True,
        stderr=subprocess.STDOUT,
    )
except (OSError, subprocess.CalledProcessError) as exc:
    raise SystemExit(f"failed to run nvcc under CUDA_HOME={CUDA_HOME}: {exc}")

match = re.search(r"release\s+(\d+\.\d+)", output)
if match is None:
    raise SystemExit(f"could not parse CUDA version from nvcc output:\n{output}")

nvcc_cuda = match.group(1)
torch_cuda = torch.version.cuda
print("nvcc CUDA:", nvcc_cuda)
if torch_cuda is None or nvcc_cuda != torch_cuda:
    raise SystemExit(
        "\nCUDA version mismatch:\n"
        f"  PyTorch was compiled with CUDA {torch_cuda}\n"
        f"  CUDA_HOME/nvcc provides CUDA {nvcc_cuda}\n\n"
        "Use a CUDA toolkit with the same major.minor version as torch.version.cuda, "
        "then set CUDA_HOME and PATH accordingly. The NVIDIA driver version shown by "
        "nvidia-smi does not need to match exactly."
    )
PY

python -m pip install -r requirements-modelnet40.txt

pushd openpoints/cpp/pointnet2_batch >/dev/null
python -m pip install -v .
popd >/dev/null

python tools/check_dataset.py
echo "PointNeXt/OpenPoints ModelNet40 environment is ready."
