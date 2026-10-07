# Qwen3 and NanoJev routing comparison

This opt-in experiment measures whether the published NanoJev game checkpoint
transfers to Flare's support-routing use case. The production decision API and
its `qwen3-choice-v1` prompt stay unchanged. No Rust/WASM head is added here.

## What is being compared

- **Qwen3:** [original Qwen3-0.6B weights](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), existing Flare decision prompt, four
  single-token label logits. Token IDs and rendering are checked against all
  32 preserved independent llama.cpp reference prompts before inference.
- **NanoJev:** [C-Tianyu/NanoJev unified-games-v1](https://huggingface.co/C-Tianyu/NanoJev/tree/047b927b30882a1138fc504821b82ac145a4b81a), strict full checkpoint,
  unchanged upstream `DecisionModel`, candidate token preparation and answer
  semantics. Candidate strings are mapped to `criteria[choice] = choice`;
  question and state are identical to Qwen's. No label targets enter inference.
- Both run locally on **PyTorch CPU float32**, four intra-op threads and one
  inter-op thread, SDPA, one question per request and a 512-token path limit.
  NanoJev encodes four candidate paths, without prefix sharing. Qwen performs
  one backbone pass and projects its final state onto only the four LM-head
  rows. This is a comparison of these complete implementations, not an isolated
  experiment on classifier heads or a browser benchmark.

Upstream `DecisionPredictor` requires CUDA in its constructor. `run.py` provides
CPU initialization for the unmodified model and tokenizer. `audit_adapter.py`
checks real checkpoint outputs against the unchanged upstream `predict` method
with CPU-initialized fields. No CUDA/BF16 equivalence is claimed. The checkpoint
was trained on games; this is an out-of-domain experiment. It is an independent
NanoJev project, not TypeSafe Jev.

The original-weight Qwen model is not the deployed Q8 model. `bridge.py` compares
it with the old Q8 llama.cpp oracle; the separate native profile measures actual
Flare Q8 decisions. Historical browser results remain in the parent directory.

## Data and protocol

Read [PROTOCOL.md](PROTOCOL.md) for the predeclared comparison and integration
gate. The data and protocol were committed before inference. `dev.json` has
16 tickets, and `test.json` has 64 new held-out tickets; both have balanced class
counts. Four fixed permutations put every class at each option position once.
The test contains 48 clear requests, 12 out-of-scope requests, and four ambiguous
requests. Under the existing policy the latter two groups target `other`.

These are original synthetic English examples with author-assigned labels,
without independent human adjudication or real customer traffic. No training,
prompt selection, calibration fitting or threshold selection occurred. Once
published, these tickets must be treated as exposed evaluation examples.
Repeated orders are paired observations, not extra independent cases. Order
and Qwen's letter assignment change together; we do not isolate those effects.

## Reproduce

Use Python >=3.12 on macOS/Linux. The recorded run uses native Python 3.13.7.
Core dependency versions match the published NanoJev requirements (which used
Python 3.14.4 on A100). The first older-Transformers NanoJev loader attempt failed
before inference; all final Qwen results were rerun on the compatible runtime.

```sh
python3 -m venv /tmp/decision-env
/tmp/decision-env/bin/pip install -r evaluations/decision/comparison/requirements.txt
# Explicit public model download, about 4 GB plus dependencies. No credentials.
/tmp/decision-env/bin/python evaluations/decision/comparison/download.py /tmp/decision-assets
mkdir -p /tmp/decision-results
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
for model in qwen3 nanojev; do
  for split in dev test; do
    /tmp/decision-env/bin/python evaluations/decision/comparison/run.py \
      --model "$model" --assets "/tmp/decision-assets/$model" \
      --dataset "evaluations/decision/comparison/$split.json" \
      --output "/tmp/decision-results/$model-$split.json"
  done
done
for split in dev test; do
  python3 evaluations/decision/comparison/report.py \
    "evaluations/decision/comparison/$split.json" \
    "/tmp/decision-results/qwen3-$split.json" \
    "/tmp/decision-results/nanojev-$split.json" \
    "/tmp/decision-results/$split-summary.json"
done
/tmp/decision-env/bin/python evaluations/decision/comparison/audit_adapter.py \
  /tmp/decision-assets/nanojev /tmp/decision-results/adapter-audit.json
/tmp/decision-env/bin/python evaluations/decision/comparison/run.py \
  --model qwen3 --assets /tmp/decision-assets/qwen3 \
  --dataset evaluations/decision/cases.json \
  --output /tmp/decision-results/qwen3-legacy.json
python3 evaluations/decision/comparison/bridge.py \
  /tmp/decision-results/qwen3-legacy.json /tmp/decision-results/bridge-summary.json
```

Only `download.py` accesses the network. Inference verifies all local assets
against [assets.json](assets.json), disables Hub access and rejects truncation.
Upstream source is fetched at a commit and verified before import, not vendored
or silently updated. No trained weights are committed or downloaded by CI.
Upstream source is MIT; Qwen retains Apache-2.0. The download includes licenses.

Per-attempt JSON records preserve ordered probabilities, logits, exact token
paths, timing, failures and provenance. Interrupted runs retain completed
attempts, but reporting rejects incomplete or duplicate attempt sets. Ordinary
failures count as incorrect with Brier 2; ECE uses completed requests only.
Five equal-width ECE bins are descriptive, not evidence of calibration. Load
includes local initialization and Qwen prompt verification, excludes download
and asset hashing. Filesystem caches may be warm. RSS is a process high-water
mark including initialization, not steady-state model memory. First and warm
request times are separate; shared-host timings are not stable performance SLAs.

### Profile the actual native Q8 implementation

Supply the Q8 file pinned in the [original evaluation](../README.md).
The example's `--profile` option uses the existing monotonic phase profiler;
normal calls retain their existing behavior. Four fixed development tickets
are each measured twice, followed by an unprofiled control on the first ticket.

```sh
cargo build --release --locked -p flarellm-loader --example decision_eval
python3 evaluations/decision/comparison/prefill_profile.py requests /tmp/profile-requests.json
# profile-control-request.json is the first request from that generated file.
python3 evaluations/decision/comparison/measure_native.py \
  target/release/examples/decision_eval /path/to/Qwen3-0.6B-Q8_0.gguf \
  /tmp/decision-assets/qwen3/tokenizer.json /tmp/profile-requests.json \
  /tmp/decision-results/flare-profile.json --profile
python3 evaluations/decision/comparison/measure_native.py \
  target/release/examples/decision_eval /path/to/Qwen3-0.6B-Q8_0.gguf \
  /tmp/decision-assets/qwen3/tokenizer.json \
  evaluations/decision/comparison/profile-control-request.json \
  /tmp/decision-results/flare-unprofiled.json
python3 evaluations/decision/comparison/prefill_profile.py report \
  /tmp/profile-requests.json /tmp/decision-results/flare-profile.json \
  /tmp/decision-results/flare-unprofiled.json /tmp/decision-results/profile-summary.json
```

### Lightweight checks

```sh
python3 -m unittest discover -s evaluations/decision/comparison -p 'test_*.py'
cargo test --locked -p flarellm-core --test decision
cargo clippy --locked -p flarellm-loader --example decision_eval -- -D warnings
cargo fmt --all -- --check
```

CI validates frozen hashes, split disjointness, choice permutations, candidate
identity, malformed/missing outputs, failure accounting and known metric values
without installing model libraries. Full model runs remain explicit local work.

## Recorded result: 2026-10-07

**Do not port this NanoJev checkpoint for support routing.** It fails the
predeclared integration gate: it chooses `billing` for every development and
test request, in every order. The existing Qwen prompt also needs quality work.
The original choice order makes Qwen choose `other` for all 64 test tickets.
Neither behavior supports unattended routing.

Host: Apple M1 Max, 64 GiB RAM, macOS 15.7.3, native arm64 Python 3.13.7.
The model processes ran serially on a shared desktop, with no evaluation builds
running during final collection. Host activity was not otherwise isolated.
Full runtime/build metadata and exact package versions are embedded in each run.

### Held-out quality (64 unique tickets)

| Metric | Qwen3 original weights | NanoJev game checkpoint |
|---|---:|---:|
| Original order accuracy | 16/64 (25.0%) | 16/64 (25.0%) |
| Reversed order accuracy | 16/64 (25.0%) | 16/64 (25.0%) |
| Third order accuracy | 25/64 (39.1%) | 16/64 (25.0%) |
| Fourth order accuracy | 17/64 (26.6%) | 16/64 (25.0%) |
| Cases changing winner across orders | 14/64 (21.9%) | 0/64 |
| Original-order multiclass Brier | 1.3415 | 0.7771 |
| Original-order ECE, five bins | 0.6892 | 0.1613 |
| Failed / overflow requests, all orders | 0 / 0 | 0 / 0 |

A constant label achieves 25% on this balanced test. Uniform probabilities have
Brier 0.75; NanoJev's 0.7771 does not beat that trivial probabilistic baseline.
Its stable ordering and smaller ECE do not establish usefulness or calibration:
it misses all technical, account and `other` tickets. Qwen gets zero of the
48 clear tickets right in original order, while selecting the prespecified
`other` target for all 16 ambiguous/out-of-scope tickets. That is class collapse,
not demonstrated ambiguity detection. Four orders are 256 paired attempts per
model, not 256 independent tickets. Per-order confusion matrices and paired
correct/incorrect counts are in [test-summary.json](results/test-summary.json).

The separate [development results](results/dev-summary.json) are Qwen
4/16, 4/16, 4/16, 5/16 across the orders, and NanoJev 4/16 throughout.
No changes were selected from either split's performance.

### Same-runtime timing and memory

| Held-out run | Qwen3 | NanoJev |
|---|---:|---:|
| Local model initialization | 2.28 s | 11.12 s |
| First request | 360 ms | 320 ms |
| Warm request median | 226 ms | 382 ms |
| Warm request p95 | 302 ms | 498 ms |
| Peak process RSS including initialization | 4.63 GiB | 4.87 GiB |

These are observed CPU float32 Python implementation timings. They do not
predict Rust/WASM latency, Q8 latency, memory after quantization, or CUDA
performance. Initialization paths differ: NanoJev constructs then loads the
model; Qwen uses `from_pretrained` and additionally verifies historical prompts.
Warm medians cover 255 completed requests after the first.

### Adapter and precision checks

The [adapter audit](results/adapter-audit.json) matches the unmodified upstream
prediction method exactly on the first development ticket in original and
reversed order (maximum probability difference 0). This verifies the CPU
adapter, not equivalence to the publisher's CUDA/BF16 execution.

The [precision/runtime bridge](results/bridge-summary.json) agrees with the
preserved llama.cpp Q8 winner on **30/32** historical attempts. The two changes
are `b1` and `b4` in original order, which switch from correct `billing` under Q8
to incorrect `other` under original-weight FP32. Maximum candidate-logit error
is 1.276 and probability error 0.0901. Prompts and IDs match exactly. Weight/KV
precision and backend differences are confounded; do not use this as a Flare
numerical regression or assume these new FP32 accuracies are Q8 measurements.
The old Q8 native/browser baseline remains unchanged in the parent directory.

Exact token arrays are formatted inline after collection to reduce diff noise;
JSON values and recorded source hashes are preserved. Those source hashes refer
to collection-time code, before the output-formatting change in `common.py`.

### Native Flare Q8 prefill profile

[Eight measured calls](results/profile-summary.json), covering the first
development ticket per class twice, have a **28.82 s warm median**. Mean prefill
is 28.98 s. The four packed-weight projection phases consume **98.94%** of it:

| Phase, summed across 28 layers | Mean per request |
|---|---:|
| FFN gate/up projections | 11.47 s |
| Q/K/V projections | 7.62 s |
| FFN down projection | 5.63 s |
| Attention output projection | 3.95 s |
| Attention computation | 0.179 s |
| LM head | 0.024 s |
| Outside prefill (preparation/reset/result) | 0.007 s |

The code confirms that `Qwen3CpuBackend::batched_dequant_matmul` sends each
prompt token through `matvec_q8_0_scalar`, retaining f32 activations for numerical
correctness. This localizes the next performance task to packed Q8 × f32
projection kernels. It does not justify restoring activation requantization
that previously harmed Qwen reference parity, nor changing the decision head
to fix latency. No kernel changes are made in this experiment.

The profiled process peaks at 1.90 GiB RSS, with local load 0.482 s. A fresh
unprofiled control peaks at 1.76 GiB, loads in 0.443 s and takes 30.98 s for its
first request. Its complete decision result, including logits, scores and token
IDs, exactly matches the first profiled call. Those two individual timings are
not a controlled estimate of profiler overhead. These are native CPU data;
this PR does not remeasure the browser or certify a browser speed improvement.

## Recommended follow-up

1. Improve Qwen decision prompt/scoring on development data. Measure whether
   offered letters receive meaningful probability mass, and compare a small
   prespecified set of prompts or scoring methods. Use new held-out data if
   these now-exposed test cases influence selection. Keep `other` policy explicit.
2. Optimize the measured native Qwen Q8 prefill bottleneck while retaining f32
   activation accuracy and the independent logit reference. Verify native and
   WASM speed and numerical parity before enabling a new kernel.
3. Reconsider a learned decision head only with domain-relevant supervision and
   representative validation. This result does not establish that NanoJev is
   ineffective on its published game tasks or all possible decision tasks.
