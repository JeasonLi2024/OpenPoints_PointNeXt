#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python examples/classification/main.py \
  --cfg cfgs/modelnet40_normals/pointnext-s_c64_xyz.yaml \
  "$@"
