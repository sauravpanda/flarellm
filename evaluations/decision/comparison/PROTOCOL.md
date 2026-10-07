# Frozen comparison protocol

Authored 2026-10-07 before any inference on these tickets. This is an exploratory
transfer evaluation, not a deployment qualification or a test of TypeSafe Jev.

- Preserve the original 16-case `routing-synthetic-v1` baseline as historical evidence.
- New split: 16 development tickets (four per class) and 64 held-out test tickets
  (16 per class). Test `other`: 12 out-of-scope and four ambiguous tickets.
  All labels are author-assigned under the existing queue policy. No independent
  human adjudication, customer data, translation, or paraphrase augmentation.
- Freeze these files by SHA-256 before inference:
  - `dev.json`: `af00a4e161064b1b39ec6fddbc778feb1d9e32719f765b589e22cc29ffd4ed8d`
  - `test.json`: `c3b797790eaf8c1efcae4a4609ba2cef3a0ddfea2e99c575808e586f3293867f`
- Use the same state, question and choice strings for both models. Qwen uses the
  unchanged `qwen3-choice-v1` system/chat prompt and single-token label logits.
  NanoJev maps each choice to `criteria[choice] = choice` with upstream segment
  encoding/EOS and the trained attention head. This repeats the label in its
  candidate path, as required by that interface; no extra class descriptions.
- Use four predeclared permutations, each placing each class in each position
  exactly once. Option ordering and Qwen label assignment remain confounded.
  Process orders consecutively for each ticket; reset each request, no KV reuse.
- Compare Qwen3 original weights and NanoJev on PyTorch CPU, float32, SDPA,
  four intra-op threads, one inter-op thread, one question per call, temperature 1.
  NanoJev batches four candidate paths without prefix sharing. Qwen scores only
  the four needed output rows after one backbone pass. No generated text.
- NanoJev's published `DecisionPredictor` requires CUDA. The CPU adapter loads
  the same strict state dict and calls its unchanged `DecisionModel` and
  `prepare_examples`. It is not the published CUDA/BF16 runtime. Verify the
  adapter's probabilities against upstream `answer_from_probabilities`.
- Qwen original-weight float32 is not Flare's Q8 runtime. Bridge it to the old
  32 llama.cpp/Q8 reference decisions and record disagreements and score error;
  never equate numerical tolerance with quality. Separately profile actual
  Flare Q8 calls on development inputs with the existing phase profiler.
- Keep load, first decision and warm latency separate. Record process peak RSS,
  package versions, host, thread count, hashes and per-attempt outputs. Run model
  processes serially; compare CPU PyTorch timings only within that runtime.
  Native Flare and historical browser timings do not establish a model speedup.
- Report accuracy per order (64 independent synthetic cases, not 256), class
  confusion, kind breakdown, order changes, multiclass Brier, five-bin ECE,
  failures, context overflows, latency and memory. Failure counts as incorrect
  with worst-case Brier 2; ECE is completed-only. Missing/duplicate attempts or
  invalid probabilities invalidate a report. Never discard failed requests.
- No training, prompt selection, temperature fitting or threshold fitting in
  this PR. Development data remains available for a later separate experiment.
  Neither result is evidence of calibrated probabilities or broad task ability.
- Integration gate: NanoJev should outperform the existing scoring approach
  across all four test orders and avoid a per-class collapse before considering
  a Rust/WASM port. Even then, a representative human-reviewed evaluation and
  browser latency/memory feasibility are required. Otherwise retain the head
  as research and prioritize the measured Flare bottleneck or prompt work on dev.

Runtime setup amendment, before any NanoJev inference: its published bundle uses
Transformers 5.17.0 tokenizer/config serialization. The initial Transformers
4.57.1 loader failed before making a NanoJev request. Align both models to the
published torch 2.14.0 / transformers 5.17.0 / safetensors 0.8.0 / numpy 2.5.3
versions, on available native Python 3.13.7 (upstream used 3.14.4). Rerun all Qwen
measurements on this same environment. Dataset, prompts, metrics and gate stay
as frozen above. No extra tokenizer aliases or RoPE config translations.
