# Independent Llama GGUF rotary reference

`gqa.gguf` is a deterministic, untrained model created by `generate.py` under
this repository's license. It has two layers, 128 hidden units, two query heads,
one KV head, 64 units per head, 192 FFN units and 128 vocabulary entries.
F32 embeddings/norms/output and Q8_0 projections exercise non-square matrices,
GQA and whole packed rows. The data is in GGUF adjacent-pair rotary order.
It has no answer-quality significance and contains no downloaded model weights.

`reference.json` contains all 128 logits and greedy tokens for a three-token
prefill followed by 15 decode steps. They were produced by **llama.cpp commit
f3f1a8f2760f28325a5ec20c05b171e5b7c83a29**, not Flare. `reference.cpp` passes exact
IDs directly to `llama_decode`, with CPU execution, 256-token context, four
threads, flash attention disabled and default f16 KV. The recorded run used
x86_64 macOS/Rosetta. Model SHA-256 is checked by the exporter and browser worker.

The independent-engine tolerance is `0.04 + 0.002 * abs(reference)` for every
logit, with exact greedy tokens. This accommodates the engines' different
quantized matrix kernels, accumulation and native activation implementations;
it is not bitwise equivalence. The observed maxima were 0.0355 for native ARM
f32 and 0.0341 for WASM raw Q8. The former six-token candidate prompt encountered
a close greedy decision and diverged after two tokens despite similar logits;
the committed three-token prompt has 16 matching decisions. The fixture is
therefore a bounded regression, not a claim of universal generation parity.
Removing normalization fails its first-prefill logit check and row tests.
Existing GPU-vs-CPU tolerances in the separate lifecycle fixture stay unchanged.

## Reproduce

Check out the pinned commit above as `/tmp/llama-reference`, then build its CPU
library and this small reference client (no server or tokenizer needed):

```sh
cmake -S /tmp/llama-reference -B /tmp/llama-reference/build \
  -DGGML_METAL=OFF -DGGML_OPENMP=OFF -DLLAMA_BUILD_COMMON=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF
cmake --build /tmp/llama-reference/build --target llama -j 4
c++ -std=c++17 flare-loader/tests/fixtures/rope/reference.cpp \
  -I/tmp/llama-reference/include -I/tmp/llama-reference/ggml/include \
  -L/tmp/llama-reference/build/bin -lllama \
  -Wl,-rpath,/tmp/llama-reference/build/bin -o /tmp/rope-reference
python3 flare-loader/tests/fixtures/rope/generate.py
python3 flare-loader/tests/fixtures/rope/export_reference.py \
  /tmp/llama-reference /tmp/rope-reference
cargo test -p flarellm-loader --test gguf_rope
```

The compiler and CMake architectures must agree. The recorded local CMake was
x86_64, so the client was compiled with `-arch x86_64`. Ordinary CI consumes the
committed binary and reference without CMake, Python dependencies or downloads.
The exporter checks the llama.cpp checkout's commit. Build the client against
that checkout; do not substitute an unpinned installed library.

## Real-model evidence

`smollm2-reference.json` records the model revision/checksum from #526, rendered
prompt, 46 exact original-tokenizer IDs, ten greedy decisions including EOS, and
selected logits: every 384th vocabulary entry plus each step's reference top ten.
The full original tokenizer is pinned in `flare-core/tests/fixtures/tokenizer`.
The exporter checks its hash and verifies IDs with `tokenizers==0.22.2`.

To reproduce, append the pinned model and original tokenizer paths to the export
command. This also writes a full-vocabulary `.reference.json` beside the local
model for the optional browser run. No trained weights are committed. The
selected reference's provenance is the same public Apache-2.0 SmolLM2 model
identified in #526; its output is the synthetic Aster/Mira Vale test prompt.

The real-model tolerance is `0.8 + 0.02 * abs(reference)`, with exact tokens.
This is a coarse numerical smoke check, not a high-precision logits claim.
WASM CPU matched all ten decisions with maximum full-vocabulary absolute error
0.7181. Native ARM generation remains affected by its pre-existing NEON SiLU
approximation; the loader repair does not fix or certify that path.

## Loader invariant

Low-level `GgufFile` readers retain GGUF file order. The model assembly boundary
normalizes only Llama GGUF Q/K weights and optional biases to split-half order.
It consumes the source maps once. `weights::load_raw_layer_weights` performs the
same conversion for later attachment; the lower-level method on `GgufFile`
remains a source-format reader. Never normalize an assembled model a second time.
SafeTensors and other GGUF architecture strings retain their existing order.
Tests cover both heads independently, f32/raw equality, skipped f32 copies,
chunked assembly, attachment, biases, malformed shapes/blocks and SafeTensors.
Byte-preservation tests include F16/BF16/Q8_0/Q4_0/Q4_K/Q6_K; they do not add
execution support for quantization formats or architectures.
