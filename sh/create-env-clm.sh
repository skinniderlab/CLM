#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="clm"

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Conda env '${ENV_NAME}' already exists. Using it."
else
  echo "[INFO] Creating env '${ENV_NAME}'..."
  conda create -y -n "${ENV_NAME}" python=3.10 pip \
    -c conda-forge --override-channels
fi

conda activate "${ENV_NAME}"
echo "[INFO] Conda env '${ENV_NAME}' activated."

conda env update -n "${ENV_NAME}" --file environment.yml
pip install -e . --no-deps

echo "[INFO] Done. Activate with: conda activate ${ENV_NAME}"