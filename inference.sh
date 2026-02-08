#!/usr/bin/env bash
set -euo pipefail

# Simple local inference runner (Mac/Linux).
#
# Usage:
#   ./inference.sh v1.0
#   ./inference.sh v1.5
#   ./inference.sh v1.5 ./configs/inference/test.yaml
#
# Notes:
# - This runs the pipeline from `app/` by setting PYTHONPATH to include `./app`
# - Edit the YAML config to point at your video/audio paths.

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <version> [inference_config]"
  echo "  <version>: v1.0 | v1.5"
  echo "  [inference_config] (optional): path to YAML (default: ./app/configs/inference/test.yaml or ./configs/inference/test.yaml)"
  exit 1
fi

version="$1"

# Prefer configs moved under app/, but allow old location.
default_config_app="./app/configs/inference/test.yaml"
default_config_root="./configs/inference/test.yaml"
if [[ $# -eq 2 ]]; then
  config_path="$2"
elif [[ -f "$default_config_app" ]]; then
  config_path="$default_config_app"
else
  config_path="$default_config_root"
fi

result_dir="./results"

if [[ "$version" == "v1.0" ]]; then
  model_dir="./models/musetalk"
  unet_model_path="$model_dir/pytorch_model.bin"
  unet_config="$model_dir/musetalk.json"
  version_arg="v1"
elif [[ "$version" == "v1.5" ]]; then
  model_dir="./models/musetalkV15"
  unet_model_path="$model_dir/unet.pth"
  unet_config="$model_dir/musetalk.json"
  version_arg="v15"
else
  echo "Invalid version '$version' (use v1.0 or v1.5)."
  exit 1
fi

export PYTHONPATH="$(pwd)/app:${PYTHONPATH:-}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" app/scripts/inference.py \
  --inference_config "$config_path" \
  --result_dir "$result_dir" \
  --unet_model_path "$unet_model_path" \
  --unet_config "$unet_config" \
  --version "$version_arg"
