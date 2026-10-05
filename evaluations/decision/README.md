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
reference client against the pinned CPU llama.cpp library, as in the existing
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
memory. Results and hardware notes are recorded below once runs finish.
