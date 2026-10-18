# Qwen3 Q8 decision-scoring experiment

This experiment compares four fixed prompts on the published development split,
then evaluates a frozen choice on new synthetic tickets. It measures full-vocabulary
probability mass on offered answer labels as well as their conditional scores.
Production `Flare.decide()` retains `qwen3-choice-v1` until a separate adoption
change is justified and validated for the SDK.

## Design

Read [PROTOCOL.md](PROTOCOL.md) for the predeclared selection rule and preliminary
quality gates. The protocol, prompts and 32 test cases were committed before
inference (`e00698a`). The test has eight examples per class; 24 specific-queue
requests, six out-of-scope requests and two ambiguous requests. Four balanced
choice permutations are paired observations, not 128 independent examples.
All text is original synthetic English with author-assigned labels and no
independent human adjudication. Semantic overlap between synthetic examples and
demonstrations limits generalization claims.

Candidates, evaluated on the same 16 development cases in four orders:

| Candidate | Input/scoring difference |
|---|---|
| `baseline` | Exact existing system/user prompt, labels A–D (IDs 32–35). |
| `concise` | A shorter user-only routing instruction, same answer letters. |
| `fewshot` | Concise prompt plus five independent routing demonstrations; example letters follow the offered order. |
| `answer_prefix` | Concise prompt with `Answer:` appended to the assistant prefix; scores single tokens for ` A`–` D` (IDs 362, 425, 356, 422). |

All candidates use the pinned original Qwen3 template with thinking disabled.
Every answer continuation must append exactly one token without changing the
prompt prefix. The original baseline is independently checked against all old
reference prompts and token IDs before running a sweep. No input truncation,
generation, calibration fitting or adaptive prompt search is used.

Selection maximizes worst-order development accuracy, then pooled paired
accuracy, then minimizes pooled Brier. Exact ties follow the candidate order
above. `selection.json` records the decision and hashes before held-out inference.
Only the winner and baseline run on the new test; the test never selects a prompt.
The development evidence and selected `answer_prefix` were committed at `1507e16`
before the held-out run.

## Results and decision

**Do not adopt the selected prompt.** It fails every preliminary gate. The
experiment does not justify a production prompt change, a NanoJev port, or an
unattended routing feature.

Development accuracy on 16 cases per order:

| Candidate | Original | Reversed | Swap pairs | Rotate |
|---|---:|---:|---:|---:|
| baseline | 25.00% | 25.00% | 18.75% | 31.25% |
| concise | 25.00% | 25.00% | 25.00% | 25.00% |
| fewshot | 25.00% | 25.00% | 25.00% | 25.00% |
| answer_prefix (selected) | 25.00% | 31.25% | 31.25% | 25.00% |

`answer_prefix` ties the two concise variants on worst-order accuracy and wins
on pooled development accuracy. This is a weak winner: it predicts `other` in
61/64 development attempts. All 256 development attempts completed.

Fresh held-out accuracy on 32 cases per order:

| Candidate | Original | Reversed | Swap pairs | Rotate | Paired total |
|---|---:|---:|---:|---:|---:|
| baseline | 31.25% | 25.00% | 25.00% | 25.00% | 34/128 (26.56%) |
| answer_prefix | 25.00% | 28.13% | 28.13% | 25.00% | 34/128 (26.56%) |

The selected prompt scores 0/8 on each specific queue and 8/8 on `other` in
original order. Across all orders it predicts `other` 125/128 times. Its apparent
order stability improves (2/32 cases change, versus 11/32 for baseline), but
mostly because it repeats the same wrong queue. All 256 test attempts completed.
These repeated orders do not increase the independent sample size beyond 32.

| Held-out diagnostic | Baseline | Selected |
|---|---:|---:|
| Mean full-vocabulary mass on offered labels | 99.970% | 98.805% |
| Top next token outside offered labels | 0/128 | 0/128 |
| Mean winning conditional score | 86.512% | 95.764% |
| Paired multiclass Brier (lower is better) | 1.2738 | 1.3717 |
| Paired five-bin ECE (lower is better) | 0.5995 | 0.6920 |

Low probability mass on letters does not explain these errors. The selected
prompt is confidently wrong; its Brier and ECE both worsen. Label mass is not a
correctness signal, and this experiment does not establish a usable confidence
threshold. Per-order confusion matrices and paired outcomes are preserved in
[report.json](results/report.json); raw logits and top tokens are in
[test-results.json](results/test-results.json).

The predeclared native Flare subset agrees with llama.cpp on **8/8 winners**.
Maximum candidate-logit error is **0.38021**, conditional-score error **0.03015**,
and full-vocabulary label-mass error **0.00451 or less**. Every candidate logit
and score satisfies the predeclared tolerance; see [flare-audit.json](results/flare-audit.json).
This supports numerical agreement for the sampled inputs, including their wrong
routes. It does not establish full-set Flare quality or browser parity.

**Next:** build a task-specific supervised routing baseline with independently
labeled data, including out-of-scope and ambiguous requests. Freeze a new test
split before fitting the classifier or selecting abstention thresholds. Compare
per-class accuracy, order dependence, coverage, and errors among accepted routes
against this baseline. These exposed synthetic tickets can support regression
checks but should not serve as the next untouched quality test. Q8 kernel work
remains a separate performance task; it cannot resolve the measured routing errors.

## What the measurements mean

`result.scores` is a stable softmax over the offered label logits. `labelMass`
is the **sum of those labels' probabilities over the entire model vocabulary**.
A high conditional score can coexist with very little vocabulary probability.
The top five next-token IDs, decoded text and full-vocabulary probabilities are
retained so such behavior can be inspected. Label mass is a diagnostic, not an
abstention threshold or calibrated probability of a correct route.

The full sweep uses the actual pinned Q8 model in llama.cpp. Flare runs a
predeclared numerical subset: first test case of each class, original/reversed
order, eight requests. The diagnostic `decision_scores` executable accepts
explicit prompt/label IDs, validates their bounds and finite logits, then uses
Flare's native Qwen3 CPU engine. It is an example, not an application API. It
checks the frozen prompt's behavior without changing production defaults.

The numerical audit uses the existing real-model candidate-logit tolerance
`0.8 + 0.02 * abs(reference)` and 0.15 absolute conditional-score tolerance.
Winner agreement, maximum errors and label-mass differences are reported
separately. Passing this smoke does not establish universal backend parity,
browser support, calibration or deployment quality.

Failures stay in the denominator as incorrect with Brier 2. ECE uses completed
predictions only. Missing or duplicate attempts invalidate a report. Incremental
client diagnostics are retained locally if inference stops; only complete runs
are reportable. Timings cover the reference client's prefill and scoring work,
not tokenization/model load, and are not controlled performance comparisons.

## Reproduce

Assets are the public Apache-2.0 Qwen model/tokenizer pinned in the
[parent evaluation](../README.md):

- Q8 GGUF: `Qwen/Qwen3-0.6B-GGUF@23749fefcc72300e3a2ad315e1317431b06b590a`,
  `Qwen3-0.6B-Q8_0.gguf`, SHA-256
  `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.
- Original tokenizer/config:
  `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
- llama.cpp: `f3f1a8f2760f28325a5ec20c05b171e5b7c83a29`, CPU, four threads,
  512 context/batch/ubatch, flash attention disabled, default f16 KV.

The recorded macOS reference build also enables the default Accelerate BLAS
backend. Flare uses its scalar Q8 × f32 CPU path; the engines differ in numerical
implementation. Neither run is a controlled speed benchmark.

No trained weights are committed or downloaded by normal CI. Supply assets
explicitly; the runner verifies their hashes. Install `requirements.txt` in an
isolated environment. The recorded build targets native arm64 on an M1 Max;
select the appropriate architecture for another host.

```sh
# Use the pinned llama.cpp checkout at /tmp/llama-reference.
cmake -S /tmp/llama-reference -B /tmp/llama-reference/build \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.4-a+dotprod \
  -DGGML_METAL=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_COMMON=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build /tmp/llama-reference/build --target llama -j 4
c++ -std=c++17 -arch arm64 evaluations/decision/scoring/reference.cpp \
  -I/tmp/llama-reference/include -I/tmp/llama-reference/ggml/include \
  -L/tmp/llama-reference/build/bin -lllama \
  -Wl,-rpath,/tmp/llama-reference/build/bin -o /tmp/scoring-reference
mkdir -p evaluations/decision/scoring/results
python3 evaluations/decision/scoring/experiment.py \
  --model /path/to/Qwen3-0.6B-Q8_0.gguf --tokenizer /path/to/tokenizer.json \
  --llama /tmp/llama-reference --client /tmp/scoring-reference \
  --dataset evaluations/decision/comparison/dev.json \
  --output evaluations/decision/scoring/results/dev-results.json
# Freeze and retain selection.json before the next command.
python3 evaluations/decision/scoring/experiment.py \
  --model /path/to/Qwen3-0.6B-Q8_0.gguf --tokenizer /path/to/tokenizer.json \
  --llama /tmp/llama-reference --client /tmp/scoring-reference \
  --dataset evaluations/decision/scoring/test.json \
  --selection evaluations/decision/scoring/results/selection.json \
  --output evaluations/decision/scoring/results/test-results.json
python3 evaluations/decision/scoring/report.py \
  evaluations/decision/scoring/test.json \
  evaluations/decision/scoring/results/test-results.json \
  evaluations/decision/scoring/results/selection.json \
  evaluations/decision/scoring/results/report.json
python3 evaluations/decision/scoring/verify_flare.py requests \
  evaluations/decision/scoring/results/test-results.json \
  evaluations/decision/scoring/results/flare-requests.json
cargo build --release --locked -p flarellm-loader --example decision_scores
target/release/examples/decision_scores /path/to/Qwen3-0.6B-Q8_0.gguf \
  evaluations/decision/scoring/results/flare-requests.json \
  evaluations/decision/scoring/results/flare-results.json
python3 evaluations/decision/scoring/verify_flare.py audit \
  evaluations/decision/scoring/results/flare-requests.json \
  evaluations/decision/scoring/results/flare-audit.json \
  --actual evaluations/decision/scoring/results/flare-results.json
```

Linux reproduction should omit `CMAKE_OSX_ARCHITECTURES` and `-arch arm64`, and
use CPU flags supported by its host. Actual environment and binary hashes are
saved in run artifacts. For lightweight validation:

```sh
python3 -m unittest discover -s evaluations/decision/scoring -p 'test_*.py'
cargo clippy --locked -p flarellm-loader --example decision_scores -- -D warnings
cargo fmt --all -- --check
```
