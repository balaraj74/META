#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# run_live_benchmark.sh — Run the TRIAGE benchmark with REAL LLM
# ═══════════════════════════════════════════════════════════════════
#
# This script:
#   1. Checks Ollama is running
#   2. Checks the base model is available
#   3. Runs demo_comparison.py in LIVE mode
#   4. Exports results to data/grpo/live_comparison.json
#
# Usage:
#   chmod +x scripts/run_live_benchmark.sh
#   ./scripts/run_live_benchmark.sh
#
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════"
echo "  TRIAGE — Live LLM Benchmark"
echo "═══════════════════════════════════════════════════════"

# 1. Check Ollama
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running."
    echo "   Start it with: ollama serve"
    exit 1
fi
echo "✅ Ollama is running"

# 2. Check models
MODELS=$(curl -sf http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(m['name'])
")
echo "   Available models: $MODELS"

# 3. Check base model
BASE_MODEL="${OLLAMA_BASE_MODEL:-qwen3.5:0.8b}"
if ! echo "$MODELS" | grep -q "$BASE_MODEL"; then
    echo "⚠ Base model '$BASE_MODEL' not found. Pulling..."
    ollama pull "$BASE_MODEL"
fi
echo "✅ Base model: $BASE_MODEL"

# 4. Check for trained model
TRAINED_MODEL="${OLLAMA_TRAINED_MODEL:-triage-grpo}"
if echo "$MODELS" | grep -q "$TRAINED_MODEL"; then
    echo "✅ Trained model: $TRAINED_MODEL"
    TRAINED_FLAG="--trained-model $TRAINED_MODEL"
else
    echo "⚠ Trained model '$TRAINED_MODEL' not found."
    echo "   Running with same model + enhanced prompts to simulate GRPO effect."
    echo "   For true comparison, train GRPO first and create the model."
    TRAINED_FLAG="--trained-model $BASE_MODEL"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Starting live benchmark..."
echo "═══════════════════════════════════════════════════════"
echo ""

# 5. Run the benchmark
python3 scripts/demo_comparison.py \
    --live \
    --base-model "$BASE_MODEL" \
    $TRAINED_FLAG \
    --export data/grpo/live_comparison.json

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Done! Results saved to data/grpo/live_comparison.json"
echo "═══════════════════════════════════════════════════════"
