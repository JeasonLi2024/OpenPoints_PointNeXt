#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch is required before running this script.")
PY

python -m pip install -r requirements-modelnet40.txt

pushd openpoints/cpp/pointnet2_batch >/dev/null
python -m pip install -v .
popd >/dev/null

python tools/check_dataset.py
echo "PointNeXt/OpenPoints ModelNet40 environment is ready."
