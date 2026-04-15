#!/usr/bin/env python3
"""Compare e2e_bench JSON output against benchmarks/baseline.json.

Usage:
    check_perf_regression.py <current.json> <baseline.json>

The current file is the raw output of `cargo run --example e2e_bench -- --json`
(a single JSON object). The baseline file groups expected tok/s numbers by a
model id, e.g. `smollm2-135m-q8_0`.

Supported baseline metrics:
    decode_16      — decode tok/s for the generation run with gen_count == 16
    sustained_512  — sustained decode tok/s over 512 tokens
    prefill_peak   — highest prefill tok/s across all generation runs

For now this script is informational: it prints a human-readable comparison
and always exits 0. Flip HARD_FAIL to True once we have a stable baseline on
the CI runner.
"""

from __future__ import annotations

import json
import sys
from typing import Any


HARD_FAIL = False  # flip to True when the baseline is trustworthy on CI
REGRESSION_THRESHOLD = 0.95  # >5% slower than baseline is a regression


def extract_metrics(current: dict[str, Any]) -> dict[str, float]:
    """Project the raw e2e_bench JSON onto the baseline metric keys."""
    metrics: dict[str, float] = {}

    gen = current.get("generation") or []
    if isinstance(gen, list):
        for entry in gen:
            if not isinstance(entry, dict):
                continue
            if entry.get("gen_count") == 16:
                decode = entry.get("decode_tok_s")
                if isinstance(decode, (int, float)):
                    metrics["decode_16"] = float(decode)
                break

        prefill_values = [
            float(e["prefill_tok_s"])
            for e in gen
            if isinstance(e, dict) and isinstance(e.get("prefill_tok_s"), (int, float))
        ]
        if prefill_values:
            metrics["prefill_peak"] = max(prefill_values)

    sustained = current.get("sustained_512_tok_s")
    if isinstance(sustained, (int, float)):
        metrics["sustained_512"] = float(sustained)

    return metrics


def model_id_from_current(current: dict[str, Any]) -> str:
    """Pick a model id key that matches the baseline file.

    The e2e_bench `model` field is a free-form string like
    `SmolLM2-135M Q8_0` — normalize it to `smollm2-135m-q8_0`.
    """
    raw = str(current.get("model") or "").strip().lower()
    if not raw:
        return ""
    return (
        raw.replace(" ", "-")
        .replace("_", "-")
        .replace("--", "-")
    )


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <current.json> <baseline.json>", file=sys.stderr)
        return 2

    current_file, baseline_file = sys.argv[1], sys.argv[2]

    try:
        with open(current_file) as f:
            current = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"failed to read current bench JSON {current_file}: {e}", file=sys.stderr)
        return 2

    try:
        with open(baseline_file) as f:
            baseline = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"failed to read baseline JSON {baseline_file}: {e}", file=sys.stderr)
        return 2

    metrics = extract_metrics(current)
    normalized_id = model_id_from_current(current)

    # Pick the baseline entry that most closely matches the current model id.
    # Fall back to the first non-underscore-prefixed key if we can't tell.
    model_keys = [k for k in baseline.keys() if not k.startswith("_")]
    if not model_keys:
        print("baseline.json has no model entries — nothing to compare.")
        return 0

    chosen = None
    for key in model_keys:
        if key.lower() == normalized_id or key.lower() in normalized_id:
            chosen = key
            break
    if chosen is None:
        chosen = model_keys[0]

    expected = baseline.get(chosen, {})
    if not isinstance(expected, dict):
        print(f"baseline entry for {chosen} is not an object — skipping.")
        return 0

    print(f"Model: {chosen}")
    print(f"Current metrics: {metrics}")
    print("")
    print("| metric | current | baseline | ratio | status |")
    print("|---|---:|---:|---:|---|")

    regressions: list[str] = []
    for metric, baseline_val in expected.items():
        if not isinstance(baseline_val, (int, float)):
            continue
        current_val = metrics.get(metric)
        if current_val is None:
            print(f"| {metric} | — | {baseline_val} | — | MISSING |")
            continue

        ratio = current_val / baseline_val if baseline_val else float("inf")
        status = "ok"
        if current_val < baseline_val * REGRESSION_THRESHOLD:
            status = "REGRESSION"
            regressions.append(
                f"{chosen}/{metric}: {current_val:.1f} < "
                f"{baseline_val * REGRESSION_THRESHOLD:.1f} "
                f"(baseline {baseline_val})"
            )
        print(
            f"| {metric} | {current_val:.1f} | {baseline_val} | "
            f"{ratio:.2f}x | {status} |"
        )

    print("")
    if regressions:
        print("Performance regressions detected:")
        for r in regressions:
            print(f"  - {r}")
        if HARD_FAIL:
            return 1
        print("")
        print("(informational only — not failing CI until baseline is stable)")
    else:
        print("No performance regressions detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
