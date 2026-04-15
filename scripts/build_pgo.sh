#!/bin/bash
# Profile-Guided Optimization build for e2e_bench.
#
# Runs a three-stage build:
#   1. Instrumented build (records branch/hot-path counters)
#   2. Training run on a representative workload (1B Q8_0 decode)
#   3. Final build compiled with the collected profile
#
# Usage:
#   ./scripts/build_pgo.sh                     # defaults to 1B Q8_0
#   PGO_MODEL=models/smollm2-135m-instruct-q8_0.gguf ./scripts/build_pgo.sh
#
# Requires `llvm-tools-preview` (install via `rustup component add llvm-tools-preview`).
#
# On success, the final optimized binary is at:
#   target/release/examples/e2e_bench
# and all non-bench crates in the workspace are also PGO-built.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PGO_DATA_DIR="${PGO_DATA_DIR:-$ROOT_DIR/target/pgo-data}"
PGO_MODEL="${PGO_MODEL:-models/llama-3.2-1b-instruct-q8_0.gguf}"

SYSROOT="$(rustc --print sysroot)"
HOST_TRIPLE="$(rustc -vV | awk '/^host:/ {print $2}')"
LLVM_PROFDATA="$SYSROOT/lib/rustlib/$HOST_TRIPLE/bin/llvm-profdata"

if [ ! -x "$LLVM_PROFDATA" ]; then
    echo "error: llvm-profdata not found at $LLVM_PROFDATA" >&2
    echo "run: rustup component add llvm-tools-preview" >&2
    exit 1
fi

if [ ! -f "$PGO_MODEL" ]; then
    echo "error: training model not found at $PGO_MODEL" >&2
    echo "set PGO_MODEL to a valid .gguf path, or download a 1B model first" >&2
    exit 1
fi

echo "=== stage 1: clean previous PGO data ==="
rm -rf "$PGO_DATA_DIR"
mkdir -p "$PGO_DATA_DIR"

echo "=== stage 2: instrumented build ==="
# -Cprofile-generate implies codegen-units=1 semantics for instrumentation.
# We also keep target-cpu=native so the instrumented binary still dispatches
# NEON/Accelerate paths the final build will profile.
RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR -Ctarget-cpu=native" \
    cargo build --release --example e2e_bench

echo "=== stage 3: training run on $PGO_MODEL ==="
LLVM_PROFILE_FILE="$PGO_DATA_DIR/e2e_bench-%m.profraw" \
    ./target/release/examples/e2e_bench \
    --model "$PGO_MODEL"

echo "=== stage 4: merge raw profiles ==="
"$LLVM_PROFDATA" merge \
    -o "$PGO_DATA_DIR/merged.profdata" \
    "$PGO_DATA_DIR"

echo "=== stage 5: optimized build using merged profile ==="
RUSTFLAGS="-Cprofile-use=$PGO_DATA_DIR/merged.profdata -Ctarget-cpu=native" \
    cargo build --release --example e2e_bench

echo
echo "PGO build complete."
echo "  profile:  $PGO_DATA_DIR/merged.profdata"
echo "  binary:   target/release/examples/e2e_bench"
echo
echo "Run the optimized bench with:"
echo "  ./target/release/examples/e2e_bench --model $PGO_MODEL"
