#!/usr/bin/env bash
# Regression guard: run e2e_bench --json and fail if sustained decode speed
# drops below THRESHOLD_TOK_S (default: 60 tok/s for SmolLM2-135M on any CI).
#
# Usage:
#   ./scripts/check_perf_regression.sh
#   THRESHOLD_TOK_S=40 ./scripts/check_perf_regression.sh
#   MODEL_PATH=path/to/model.gguf THRESHOLD_TOK_S=20 ./scripts/check_perf_regression.sh
#
# Outputs JSON benchmark result on stdout.
# Exits 0 if performance is above threshold, 1 if below or if model is missing.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"
MODEL_NAME="${MODEL_NAME:-smollm2-135m-instruct-q8_0.gguf}"
MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/$MODEL_NAME}"
THRESHOLD_TOK_S="${THRESHOLD_TOK_S:-60}"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Benchmark model not found at: $MODEL_PATH" >&2
  echo "Run ./scripts/download_baseline_model.sh first." >&2
  exit 1
fi

echo "Running e2e_bench (this takes ~30s)..." >&2

JSON=$(MODEL_PATH="$MODEL_PATH" cargo run -p flarellm-server \
  --example e2e_bench --release --quiet -- --json 2>/dev/null)

echo "$JSON"

# Extract sustained decode speed using Python (available on all CI runners)
ACTUAL_TOK_S=$(python3 -c "
import sys, json
d = json.loads(sys.stdin.read())
print(d['sustained_512_tok_s'])
" <<< "$JSON")

echo "" >&2
echo "Sustained decode: ${ACTUAL_TOK_S} tok/s  (threshold: ${THRESHOLD_TOK_S} tok/s)" >&2

# Compare floats: fail if actual < threshold
PASSED=$(python3 -c "
import sys
print('yes' if float('$ACTUAL_TOK_S') >= float('$THRESHOLD_TOK_S') else 'no')
")

if [ "$PASSED" = "yes" ]; then
  echo "PASS: performance is above threshold." >&2
  exit 0
else
  echo "FAIL: ${ACTUAL_TOK_S} tok/s is below the ${THRESHOLD_TOK_S} tok/s threshold." >&2
  echo "Check recent commits for regressions." >&2
  exit 1
fi
