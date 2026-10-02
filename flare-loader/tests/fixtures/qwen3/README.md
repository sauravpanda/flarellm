# Independent Qwen3 reference

`gqa.gguf` is a deterministic untrained model generated under this repository's
license, without ML dependencies or downloaded weights. Its two layers have
128 hidden dimensions, four 64-wide Q heads, two KV heads, 192 FFN dimensions,
128 vocabulary entries, nonuniform learned Q/K RMSNorm weights, and a tied
embedding/output matrix. Q8_0 projections retain **split-half** row order.
This deliberately tests `num_heads * head_dim != hidden_dim`.

`reference.json` contains all 128 logits for prefill `[2,4,7]` followed by 15
successive greedy decode steps from llama.cpp, not Flare. Every logit uses the
bound `0.04 + 0.002 * abs(reference)` with exact greedy tokens. This allows
Q8 kernels and f16-versus-f32 KV rounding; it is not bitwise parity. Removing
Q/K normalization fails most first-step reference logits. The Llama layout
regression is kept separate and unchanged.

## Pinned implementation and source semantics

- llama.cpp converter audit and executable reference:
  `f3f1a8f2760f28325a5ec20c05b171e5b7c83a29`.
  `conversion/qwen.py` SHA-256:
  `b783f53598e1a65cb0ef7fbe0d0f52da472e2784f63acf13a5aea00f623cfaa6`.
  Unlike LlamaModel, Qwen3Model does not permute Q/K rows; its graph uses
  per-head Q/K RMSNorm and split-half RoPE. The loader therefore leaves its
  rows untouched in normal, chunked, f32 and raw-attachment routes.
- Transformers v4.51.3, commit `5f4ecf2d9f867a1255131d2461d75793c0cf1db2`;
  `src/transformers/models/qwen3/modeling_qwen3.py` SHA-256
  `704c914530530a1acb0b443add1f520404e3ac2c28c0ab7e16f80f86cfe8ccb2`.
  Audited independent head dimensions, head-shared norms, GQA, no attention
  biases, SiLU, RMSNorm epsilon, RoPE and tied output semantics.

## Real model

Official `Qwen/Qwen3-0.6B-GGUF`, revision
`23749fefcc72300e3a2ad315e1317431b06b590a`, file `Qwen3-0.6B-Q8_0.gguf`,
639,446,688 bytes, SHA-256
`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`.
The official artifact does not record its original converter commit. The
converter revision above is the pinned **audit/reproduction implementation**,
not an invented provenance claim about that published file. File identity is
pinned by the official repository revision and SHA-256.

Original model/config/tokenizer revision:
`Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`.
Config SHA-256: `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
Tokenizer provenance and license are in `flare-core/tests/fixtures/qwen3`.
The Qwen model and its official quantization are Apache-2.0 licensed.

`real-0.json` through `real-2.json` record independent greedy decisions and
selected logits (every 384th vocabulary entry plus each step's reference top
ten). The exporter also saves all 151,936 logits beside the local model for
browser checks. Settings: temperature 0, no repetition penalty/filtering,
256-token llama.cpp context, four CPU threads, no Metal/flash attention,
default f16 KV. Flare uses f32 KV and exact scalar SiLU for Qwen3. Its Q8
matmuls retain f32 activations instead of adding another quantization step.
The real-model bound is `0.8 + 0.02 * abs(reference)`, with exact tokens. This
is a coarse numerical smoke check across different kernels, not a precision
claim or general quality evaluation. It does not change prior model tolerances.

## Reproduce

Build `reference.cpp` against the pinned llama.cpp CPU library as described in
`../rope/README.md` (substitute this client's path). The client stops on the
reference vocabulary's EOG tokens instead of a hardcoded Llama ID.

```sh
python3 flare-loader/tests/fixtures/qwen3/generate.py
python3 flare-loader/tests/fixtures/qwen3/export_reference.py \
  /tmp/llama-reference /tmp/qwen3-reference \
  /path/to/Qwen3-0.6B-Q8_0.gguf /path/to/tokenizer.json
cargo test -p flarellm-loader --test qwen3
QWEN3_GGUF=/path/to/Qwen3-0.6B-Q8_0.gguf \
  cargo test --release -p flarellm-loader --test qwen3 -- --include-ignored
```

Keep the pinned `tokenizer_config.json` beside tokenizer.json. The exporter
verifies model/tokenizer hashes and llama.cpp commit. No trained model file
is committed. Browser reproduction uses the existing package checker and runner;
see `flare-web/tests/browser/README.md` and `QWEN3.md` at the repository root.
