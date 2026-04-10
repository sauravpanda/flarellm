#!/usr/bin/env bash
# Run the Flare e2e benchmark against every GGUF model found in models/.
#
# Usage:
#   ./scripts/bench_multi.sh              # human-readable table
#   ./scripts/bench_multi.sh --json       # one JSON object per line (machine-readable)
#   ./scripts/bench_multi.sh --log        # append results to BENCHMARK_HISTORY.md
#   MODEL_DIR=path/to/models ./scripts/bench_multi.sh
#
# Exit code: 0 if at least one model was benchmarked, 1 otherwise.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"
BENCH_BIN="cargo run -p flarellm-server --example e2e_bench --release --quiet"
JSON_MODE=false
LOG_MODE=false

for arg in "$@"; do
  case "$arg" in
    --json) JSON_MODE=true ;;
    --log)  LOG_MODE=true  ;;
  esac
done

# Build once so the per-model runs are fast
echo "Building e2e_bench (release)..." >&2
cargo build -p flarellm-server --example e2e_bench --release --quiet 2>&1 | grep -v "^$" >&2 || true
BENCH_BIN_PATH="$(cargo metadata --format-version 1 --no-deps 2>/dev/null | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d['target_directory'])" \
  2>/dev/null || echo "target")/release/examples/e2e_bench"

if [ ! -x "$BENCH_BIN_PATH" ]; then
  # Fallback: use cargo run each time (slower first run, cached after)
  BENCH_BIN_PATH=""
fi

# Discover models
mapfile -t MODELS < <(find "$MODEL_DIR" -maxdepth 2 -name "*.gguf" 2>/dev/null | sort)

if [ ${#MODELS[@]} -eq 0 ]; then
  echo "No .gguf models found in $MODEL_DIR/" >&2
  echo "Run ./scripts/download_baseline_model.sh to download the baseline model." >&2
  exit 1
fi

echo "Found ${#MODELS[@]} model(s) in $MODEL_DIR/" >&2
echo "" >&2

# Header for human-readable mode
if ! $JSON_MODE; then
  printf "%-45s  %8s  %8s  %8s\n" "Model" "Load(s)" "Prefill" "Decode" >&2
  printf "%-45s  %8s  %8s  %8s\n" "-----" "-------" "-------" "------" >&2
fi

BENCH_ARGS=""
$LOG_MODE  && BENCH_ARGS="$BENCH_ARGS --log"
$JSON_MODE && BENCH_ARGS="$BENCH_ARGS --json"

EXIT_CODE=0
for MODEL_PATH in "${MODELS[@]}"; do
  MODEL_NAME="$(basename "$MODEL_PATH")"
  echo "Benchmarking: $MODEL_NAME" >&2

  if [ -n "$BENCH_BIN_PATH" ]; then
    OUTPUT=$(MODEL_PATH="$MODEL_PATH" "$BENCH_BIN_PATH" $BENCH_ARGS 2>/dev/null) || {
      echo "  FAILED — skipping $MODEL_NAME" >&2
      EXIT_CODE=1
      continue
    }
  else
    OUTPUT=$(MODEL_PATH="$MODEL_PATH" cargo run -p flarellm-server \
      --example e2e_bench --release --quiet -- $BENCH_ARGS 2>/dev/null) || {
      echo "  FAILED — skipping $MODEL_NAME" >&2
      EXIT_CODE=1
      continue
    }
  fi

  if $JSON_MODE; then
    echo "$OUTPUT"
  else
    # Extract key numbers from human-readable output for summary line
    LOAD=$(echo "$OUTPUT"  | grep "^Load:"     | awk '{print $2}')
    DECODE=$(echo "$OUTPUT" | grep "Sustained" | grep -oE '[0-9]+\.[0-9]+')
    PREFILL=$(echo "$OUTPUT" | grep "gen= *16" | awk '{print $3}')
    printf "%-45s  %8s  %8s  %8s\n" "$MODEL_NAME" "$LOAD" "${PREFILL:-N/A}" "${DECODE:-N/A}"
    echo ""
    echo "$OUTPUT"
    echo ""
  fi
done

exit $EXIT_CODE
