# Qwen3 bounded-decision routing baseline

This directory is the reusable **held-out test** for `qwen3-choice-v1`, not a
training set. `cases.json` contains 16 original synthetic tickets (repository
MIT OR Apache-2.0 license), authored before measuring this baseline. There is no
customer data. It has four labels, twelve clear cases, two ambiguous cases and
two none-of-the-above cases. The label definitions are in the fixed question.
Ambiguous cases have a prespecified `other` target: this is a routing policy, not
an objectively unique interpretation. No calibration or prompt tuning was done
on this set. Future tuning needs separate development data and a new held-out
set; do not silently reuse these exposed cases as unseen data.

Each case is evaluated once in original order and once reversed, changing both
option order and letter assignment. These two effects are **confounded**, not
individually measured. Repeated orders are paired measurements, not independent
examples. A later NanoJev comparison should preserve the inputs, labels, failures
and score metrics, while documenting that a game-trained head may not transfer
to support tickets. A learned head with incompatible label semantics cannot be
compared as though its outputs were these four classes without an explicit mapping.

## Independent correctness oracle

`reference.json` was produced by **llama.cpp**, not Flare, using the identical
Q8_0 file and prompt IDs. Hugging Face `tokenizers==0.22.2` and Jinja2 `3.1.6`
render the official pinned template with `enable_thinking=false`. Every prompt
and every appended A–H label is verified independently. The reduced tokenizer
retains all merges needed by these exact strings; other texts can tokenize
differently with it and it is not for application use.

Pins (also see [Qwen3 provenance](../../flare-loader/tests/fixtures/qwen3/README.md)):

- Model: `Qwen/Qwen3-0.6B-GGUF@23749fefcc72300e3a2ad315e1317431b06b590a`,
  `Qwen3-0.6B-Q8_0.gguf`, SHA-256
  `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.
- Tokenizer/config: `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`,
  tokenizer SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`,
  tokenizer_config SHA-256 `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`.
  Public Qwen assets, including the derivative reduced tokenizer, use Apache-2.0
  ([license copy](../../flare-core/tests/fixtures/qwen3/LICENSE-APACHE)).
  No trained weights are committed.
- llama.cpp `f3f1a8f2760f28325a5ec20c05b171e5b7c83a29`, CPU, four threads,
  context/batch/ubatch 512, flash attention disabled, f16 KV. The local independent
  client was compiled x86_64 under Rosetta on the Apple host; it supplies numerical
  expectations, not comparative performance claims. Flare uses f32 KV/activations
  and packed Q8 projections, without sampling or decode steps.
- Prompt version `qwen3-choice-v1`; labels A–H, IDs 32–39. All evaluation prompts
  are 134–146 original-tokenizer tokens. Full prompts and IDs are committed.

The real-model candidate-logit bound is the existing Qwen3 smoke bound
`0.8 + 0.02 * abs(reference)`. Different activation quantization and f16/f32 KV
make bitwise parity inappropriate. An additional **0.15 absolute score bound**
checks the downstream four-way softmax; it is a coarse cross-kernel tolerance,
not a claim of 15% quality uncertainty or calibration. The untrained fixture uses
`0.04 + 0.002 * abs(reference)` logits and 0.02 absolute scores. Measured maximum
errors are reported separately; tolerances are not performance targets. The
reference exporter and bounds were fixed before observing Flare evaluation
outputs. Winning-choice agreement is reported rather than forcing equality for
arbitrarily close logits.

`fixture-reference.json` is independently generated from the existing Qwen3 tiny
model generator with `--decision`: a full tokenizer-sized embedding vocabulary,
two untrained layers, 128 hidden dimensions and a 512-token context. The generated
GGUF is about 21 MB, not committed or downloaded. Its hash is checked in the
portable packed-consumer CI job. This exercises actual decisions, not mocked
logits. The original smaller Qwen3/GQA and Llama fixtures remain unchanged.

## Reproduce

Use Python `tokenizers==0.22.2` and `Jinja2==3.1.6` for regeneration only. Build the
reference client against the pinned CPU llama.cpp library. Keep the pinned
`tokenizer_config.json` beside `tokenizer.json`. Follow the existing
[reference build instructions](../../flare-loader/tests/fixtures/rope/README.md):

```sh
c++ -std=c++17 evaluations/decision/reference.cpp \
  -I/path/to/llama.cpp/include -I/path/to/llama.cpp/ggml/include \
  -L/path/to/llama.cpp/build/bin -lllama \
  -Wl,-rpath,/path/to/llama.cpp/build/bin -o /tmp/decision-reference
python3 evaluations/decision/export_reference.py /path/to/llama.cpp \
  /tmp/decision-reference /path/to/Qwen3-0.6B-Q8_0.gguf /path/to/tokenizer.json
python3 flare-loader/tests/fixtures/qwen3/generate.py --decision /tmp/decision-fixture.gguf
python3 evaluations/decision/export_fixture_reference.py /path/to/llama.cpp \
  /tmp/decision-reference /tmp/decision-fixture.gguf

cargo test -p flarellm-core --test decision
cargo run --release -p flarellm-loader --example decision_eval -- \
  /path/to/Qwen3-0.6B-Q8_0.gguf /path/to/tokenizer.json \
  evaluations/decision/reference.json /tmp/native.json
python3 evaluations/decision/report.py evaluations/decision/reference.json \
  /tmp/native.json /tmp/native-metrics.json

wasm-pack build flare-web --target web
npm ci --prefix flare-web
npm run build:sdk --prefix flare-web
npm run test:sdk --prefix flare-web
node .github/scripts/check_wasm_package.mjs /tmp/decision-consumer
python3 flare-web/tests/browser/make_ci_fixture.py /tmp/decision-consumer
# Portable CI: no model download, full real SDK lifecycle on generated fixture.
node flare-web/tests/browser/run.mjs /tmp/decision-consumer
# Optional trained-model run. The existing real-0 full-vocabulary reference is
# regenerated by the Qwen3 reference exporter documented in QWEN3.md.
DECISION_EVAL=1 REFERENCE_ARCHITECTURE=qwen3 \
REFERENCE_MODEL_GGUF=/path/to/Qwen3-0.6B-Q8_0.gguf \
REFERENCE_LOGITS_JSON=/path/to/Qwen3-0.6B-Q8_0.0.reference.json \
ORIGINAL_TOKENIZER_JSON=/path/to/tokenizer.json \
BROWSER_ARTIFACTS=/tmp/decision-browser \
node flare-web/tests/browser/run.mjs /tmp/decision-consumer
python3 evaluations/decision/report.py evaluations/decision/reference.json \
  /tmp/decision-browser/result.json /tmp/browser-metrics.json
```

No downloads happen during ordinary CI or evaluation scripts. Supply the pinned
public assets explicitly for the opt-in full-model run. Linux users must build
the reference client/library for the same native architecture; the local Rosetta
compiler flag is not part of portable reproduction.

## Metrics and interpretation

Accuracy includes **every attempted case**; failures count as incorrect.
Multiclass Brier is the mean sum of squared class errors (range 0–2); failed
requests receive worst-case 2. ECE uses five fixed equal-width bins and completed
predictions only, so always read it with coverage/failure counts. Empty bins
contribute nothing. These estimates are unstable at n=16, especially with four
classes; no calibration is fitted or claimed. Overflow rate is reported for the
held-out requests; separate intentional overflow tests must reject.

Load timing is a fresh engine and model parse from local disk/localhost, with
OS filesystem caches potentially warm, **not an internet cold-download timing**.
First decision and subsequent warm decisions are separate. SDK worker decision
time includes tokenization/validation plus one prefill; end-to-end time includes
worker transit. There are no decode tokens. Native RSS and observed WASM linear
memory capacity are different measures and cannot be compared as peak process
memory. Recorded results and hardware notes follow below.

## Recorded native result (2026-10-04)

[native-results.json](native-results.json) retains every attempt and candidate
logit/score. Host: Apple M1 Max, 64 GiB RAM, macOS 15.7.3 (24G419), native aarch64
Rust 1.92.0 release build. Reference implementation/asset hashes are above.

| Metric | Original choices | Reversed choices |
|---|---:|---:|
| Accuracy | 6/16 (37.5%) | 4/16 (25.0%) |
| Multiclass Brier | 1.1145 | 1.4532 |
| ECE, 5 fixed bins | 0.5139 | 0.7344 |
| Request failures / overflows | 0 / 0 | 0 / 0 |

The original ordering gets only **2/12 clear cases** right; it selects `other` on
most tickets. All four ambiguous/none-of-the-above cases target `other`, so success
on those cases must not be read as reliable ambiguity detection. Reversing options
changes 2/16 selected queues (12.5%). The prompt and data were not changed to
improve these disappointing results.

All 32 winning labels match the independent oracle. Maximum candidate-logit
absolute error is 0.17928; maximum normalized-score absolute error is 0.01925.
These are numerical parity observations, not quality guarantees.

Fresh engine load from local disk: **0.429 s**. First decision: **28.950 s**.
Remaining 31 decisions: median **28.301 s**, range **27.194–39.163 s**. Timing
includes validation/tokenization and one prefill, no generation. This shared
host was also running local builds/tests during part of the native run; timings
are observations, not an isolated benchmark or stable speed target. The Qwen3
correctness-first scalar Q8 prefill path is unchanged by this PR.

Observed native RSS at 12m35s was **1,589,056 KiB (1.52 GiB)**. This is a process
snapshot, not peak RSS. The platform's `/usr/bin/time -l` peak-memory collection
was unavailable in the sandbox; its elapsed time was 930.80 s for all 32 requests
plus load. Browser memory capacity and results are recorded separately.


## Recorded browser result (2026-10-04)

[browser-results.json](browser-results.json) retains all 32 evaluation attempts.
Same M1 Max host; installed npm tarball, Playwright 1.58.2 Chromium
**145.0.7632.6 arm64 headless shell**, WASM CPU with GPU disabled. The browser was
an isolated automated instance, not the user's personal Chrome. The local WASM
build used release Cargo with configured SIMD128 and wasm-bindgen 0.2.117, without
wasm-opt. Full environment and measurement caveats are in
[environment.json](environment.json).

Browser accuracy, failures and order changes match native: **6/16 (37.5%)**, then
**4/16 (25.0%)** reversed, zero failures/overflows, and 2/16 changed predictions.
Brier is **1.1145 / 1.4532**; five-bin ECE is **0.5139 / 0.7344**. All 32 winning
labels match the independent reference. Maximum candidate-logit error is
**0.17928**, maximum normalized-score error **0.01926**.

Fresh worker/model load from localhost: **1.303 s**. First decision: **32.118 s**.
Warm decision median: **32.545 s**, range **31.630–34.993 s**. These timings include
full original-tokenizer validation and a 134–146-token prefill, no autoregressive
decode. The 32 held-out attempts are distinct from additional lifecycle requests.
Some timing overlapped a separate small SwiftShader regression run on this shared
host; do not treat small latency differences as controlled speed comparisons.

Observed worker WASM linear-memory capacity after the first decision:
**2,194,145,280 bytes (2.04 GiB)**, excluding JS/browser allocations. A separate
renderer RSS snapshot was **3,085,712 KiB (2.94 GiB)**. Neither is peak total
browser-process memory; download buffers, tokenizer and other browser processes
need additional memory.

The actual package passed cancellation during a live request and automatic
reload, pre-aborted signals, reset/repeat, BUSY rejection, dispose, invalid-input
and overflow recovery, and decision → chat → decision state isolation. The
existing full-vocabulary real-Qwen chat checks also passed. Portable generated
fixture maximum errors were 0.01041 logits and 0.00208 scores. Local SwiftShader
regressions passed; they exercise existing Llama/GPU behavior and Qwen3 CPU
fallback, **not Qwen3 GPU inference**. Hosted CI covered Rust checks/tests/Clippy,
docs, Docker, WASM/package/types, Chromium CPU and missing-adapter rejection.

The [captured routing demo](routing-demo.png) loaded and displayed all four
scores in the real browser. It **incorrectly selected `other` (65.96%)** for
“I was charged twice for my subscription.” (`billing`: 33.82%). That demo example
is separate from the 16-case evaluation and is not added to its denominator.
We retain this failure rather than changing the default text or prompt to make
the screenshot look successful. Numerical correctness and lifecycle safety do
not establish classification quality.

The baseline and reusable evaluation are ready for a later NanoJev comparison.
This checkpoint/prompt combination is **not ready for unattended routing**, and
these conditional scores provide no Jev-equivalent calibration claim.
