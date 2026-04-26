#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-data/grpo_crisis_prompts}"
OUTPUT_DIR="${OUTPUT_DIR:-models/grpo_qwen3_14b}"
HUB_MODEL_ID="${HUB_MODEL_ID:-balarajr/triage-qwen3-14b-grpo}"
LOG_FILE="${LOG_FILE:-training_qwen3_14b_grpo.log}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required for model download and Hub push." >&2
  exit 1
fi

python scripts/train_grpo_qwen3_14b.py \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --hub-model-id "$HUB_MODEL_ID" \
  --resume \
  --merge-16bit \
  "$@" 2>&1 | tee "$LOG_FILE"

if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
  echo "Training complete. Stopping instance..."
  sudo shutdown -h now
fi
