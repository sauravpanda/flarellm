# Qwen3-0.6B

Flare supports the official **Qwen/Qwen3-0.6B Q8_0 GGUF with its original
JSON tokenizer** on native CPU and browser WASM CPU. Browser chat defaults to
non-thinking mode. Use the packaged demo's **Use Qwen3-0.6B Q8_0** button, or:

```js
import { Flare } from '@sauravpanda/flare';

const flare = await Flare.init({
  backend: 'cpu',
  tokenizerUrl: 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json',
  cache: false,
});
await flare.loadModel('https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf');
const answer = await flare.chat({
  message: 'What is the capital of France?',
  maxTokens: 32,
  temperature: 0, topP: 1, topK: 0, repeatPenalty: 1,
  onToken: text => console.log(text),
});
console.log(answer.text);
flare.dispose();
```

Build/install the package from this branch to use this support; this change
publishes no npm release. See [SDK setup](flare-web/README.md) for module workers,
import maps/bundlers, CORS and serving `dist/` and `pkg/`. The sample uses greedy
sampling to reproduce the recorded reference, not as a universal sampling
recommendation. No model weights are included in the package or repository.

## Scope and memory

- Dense Qwen3: 28 layers, hidden size 1,024, FFN 3,072, 16 Q heads and 8 KV
  heads, **head_dim 128** (Q projection width 2,048), vocabulary 151,936.
- Head-shared Q/K RMSNorm precedes full split-half RoPE. Epsilon is 1e-6;
  RoPE base is 1,000,000 with no scaling. No attention biases. SiLU is computed
  without the existing native ARM approximation. Embedding/output storage is
  shared with copy-on-write mutation semantics.
- The browser caps total context at **512 tokens**, including prompt and output.
  Reserve output space and use short histories. The source config's 40,960
  positions and model card's 32,768 context are not browser memory guarantees.
  Native callers should also override `max_seq_len` before constructing a model;
  validation uses 256. A 512-position f32 KV cache alone is 112 MiB.
- The model download is **639,446,688 bytes** (639 MB / 610 MiB), plus the
  tokenizer. Expect several GB of free memory: parsing, JS download bytes,
  dequantized embeddings and WASM allocations exceed the file size. Measured
  WASM linear-memory capacity reached 2,193,555,456 bytes (2.04 GiB); that excludes
  JS buffers and browser overhead and is not a peak process-memory measurement.
  Cache Storage is optional and needs additional disk space. Mobile devices
  and low-memory tabs have not been validated.
- WebGPU Qwen3 execution is **unsupported**. `init_gpu()` returns false with a
  Q/K-normalization diagnostic. An explicit SDK `backend: 'webgpu'` request
  rejects loading and tells the caller to select CPU. Native backend selection
  falls back to the Qwen3 CPU implementation. No Qwen3 GPU speed or physical GPU
  correctness is claimed.
- The supported tokenizer implements the exact official NFC and Split→ByteLevel
  sequence, including array-form merges and added thinking tokens. Other
  explicit pipelines are rejected. Encoding inserts no BOS. The official GGUF
  disables automatic BOS, and EOS `<|im_end|>` (151645) or `<|endoftext|>` (151643) is consumed
  by the SDK, not emitted.
- Chat supports text system/user/assistant messages with `enable_thinking=false`.
  It appends the official empty thinking block and strips historical reasoning
  as the official text-only template does. It does not execute arbitrary Jinja,
  implement tools, or provide a thinking-mode API/reasoning parser. Raw prompt
  generation is available, but thinking-mode behavior is not certified.
- Q8_0/F32 GGUF tensors are supported for this architecture. Qwen3 SafeTensors
  auto-inference is rejected because config semantics cannot be inferred from
  shapes. Q4, MoE, Qwen3.5/DeltaNet, multimodal models and custom classifier heads remain unsupported. An
  experimental CPU [bounded-decision API](DECISIONS.md) is available separately.

## Reference results

Pinned llama.cpp CPU is the independent logit/token oracle. The exact same
Q8_0 file, original-tokenizer IDs and greedy sampling are used by Flare.

| Prompt | Reference and Flare output | Decisions including EOS | Browser max logit error |
|---|---|---:|---:|
| What is the capital of France? | The capital of France is **Paris**. | 10 | 0.6910 |
| Reply with exactly the word hello. | Hello! | 3 | 0.6173 |
| What is 2 + 2? | 2 + 2 = 4. | 9 | 0.5846 |

All 22 decisions match in native ARM CPU and the installed npm package in
Playwright Chromium **145.0.7632.6**, macOS. The instruction prompt demonstrates
reference parity but fails the requested exact wording in both engines. These
three prompts are not a broad instruction-following or factuality evaluation.

Browser checks compare all 151,936 logits at each step, with the documented
`0.8 + 0.02 * abs(reference)` bound. Native committed real-model checks compare
selected logits. The small redistributable two-layer fixture compares every
logit over 16 decisions with a tighter bound, separate from this coarse
real-model smoke test. Normal, chunked, f32 and raw-attachment routes are tested
on the fixture. Actual Q8_0 normal/chunked loads and two SDK chat requests with
reset pass for all three prompts. Existing Llama layout/tokenizer/GPU tests
retain their previous tolerances.

Observed localhost SDK load was 1.26–1.41 seconds, TTFT 4.49–4.81 seconds, and
decode 3.11–3.18 tokens/second. These are single-host observations from short
prompts, excluding internet download time. CPU preserves f32 activations while
reading packed Q8 weights; performance optimization can follow correctness.
No stable performance threshold is claimed. Local Chrome through browser-harness
was not used; the browser actually executed was the isolated automated Chromium
runner. See the validation record for software GPU fallback checks.

## Provenance and reproduction

The model and tokenizer are published by Qwen under **Apache-2.0**. See the
[official model](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
and [official quantization](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/23749fefcc72300e3a2ad315e1317431b06b590a).
Their revisions/checksums, converter audit, Transformers source pin, reference
client and regeneration commands are in
[the numerical fixture README](flare-loader/tests/fixtures/qwen3/README.md) and
[the tokenizer fixture README](flare-core/tests/fixtures/qwen3/README.md).
The upstream GGUF does not identify its original converter commit; that gap is
explicitly recorded rather than attributed to the audit converter.

After building WASM/SDK, install the actual package and run the existing harness:

```sh
node .github/scripts/check_wasm_package.mjs /tmp/flare-consumer
python3 flare-web/tests/browser/make_ci_fixture.py /tmp/flare-consumer
REFERENCE_ARCHITECTURE=qwen3 \
REFERENCE_MODEL_GGUF=/path/to/Qwen3-0.6B-Q8_0.gguf \
REFERENCE_LOGITS_JSON=/path/to/Qwen3-0.6B-Q8_0.0.reference.json \
ORIGINAL_TOKENIZER_JSON=/path/to/tokenizer.json \
BROWSER_ARTIFACTS=/tmp/qwen-browser \
node flare-web/tests/browser/run.mjs /tmp/flare-consumer
```

Repeat with `.1.reference.json` and `.2.reference.json`; `--gpu` exercises
SwiftShader regressions plus Qwen3's deliberate CPU fallback. Ordinary CI runs
only the committed small fixtures, with no model download. The actual model
checks are opt-in. The **CPU non-thinking typed-decision baseline** is documented in
[DECISIONS.md](DECISIONS.md); NanoJev evaluation remains separate.
