# GGUF Q/K reference validation — October 2, 2026

## Reproduction and fix

Issue #526 was closed as completed at 18:41:57 UTC with no closing commit in its
timeline. Main `f031c0a` includes tokenizer PR #538, whose description explicitly
excludes Q/K layout. Closure did not establish correctness. Using the issue's
checksum-pinned SmolLM2-360M Q8_0 file, the original tokenizer from #538 and 46
identical prompt IDs, the previous packaged WASM produced
`The director of the 20000000000`. The normalized loader produces
`The director of Aster is Mira Vale.` All ten reference decisions, including
EOS, match pinned llama.cpp `f3f1a8f2760f28325a5ec20c05b171e5b7c83a29` CPU.
Hugging Face tokenizers 0.22.2 independently verifies all prompt IDs.

Low-level GGUF readers retain file order. Model assembly and the new normalized
raw-attachment helper restore Llama Q/K weights and biases to split-half order,
using separate Q and KV head counts. Tests cover row mapping, f32/raw equivalence,
quantized byte preservation, malformed dimensions/blocks, skipped-f32 and chunked
loads, separate raw attachment, other architectures and SafeTensors. Disabling
the normalization makes three regression tests fail, including the first
reference-prefill logit assertion. No issue was reopened or rewritten.

## Local evidence

- 645 workspace tests pass, 47 ignored. Workspace check, clippy with warnings
  denied, rustfmt, WASM build, strict SDK types, packaged ESM/worker imports and
  all 16 SDK tests pass.
- Actual packaged Chromium 145.0.7632.6 on macOS passes CPU lifecycle checks and
  all four load routes against the new independent GQA reference: three prompt
  IDs, 16 steps, all 128 logits. Original full-tokenizer parity also passes.
- The real-model Chromium CPU run compares all 49,152 logits over ten steps,
  including EOS. Maximum absolute error is 0.718062 under the documented coarse
  `0.8 + 0.02 * abs(reference)` bound. This establishes token parity and one
  factual answer, not precise logits or broad answer quality.
- Local Chromium SwiftShader passes both default and forced-f32 independent
  reference modes, plus the existing CPU/GPU numerical and fault-recovery suite.
  This adapter has no shader-f16. Default and forced-f32 use f32 KV; CPU prefill
  precedes async GPU decode. No uncaptured GPU errors occur.
- The system-adapter Chromium run also selected SwiftShader, not physical Metal.
  It passed the real 360M model on async GPU decode, matching all ten decisions
  with maximum absolute logit error 0.636692. Physical f16 coverage is not claimed.
- The previous lifecycle model's expected tokens remain `[36,28,20,12,4]`.
  Its writer now interleaves Q/K row blocks before storing GGUF, preserving the
  exact original split-half model after loading; only its GGUF checksum changes.

The real Chrome browser-harness connection failed its existing debugging
handshake. It was not used for inference. Isolated Playwright Chromium is the
browser actually tested. Initial GPU runs hit a report serialization error
(`GPUAdapterInfo` is not structured-cloneable); plain metadata fields fix it.

## Scope and remaining limits

Reference generation, model/tokenizer revisions and SHA-256, numerical bounds,
regeneration tools and reference-client source are committed under
`flare-loader/tests/fixtures/rope`. The small fixture is independently generated
and redistributable. The real-model reference commits selected logits and all
tokens; the local full-vocabulary report remains available separately. Models
are never downloaded by ordinary CI. The existing browser harness runs the new
fixture in ordinary CPU CI and the existing software GPU workflow.

Native ARM's existing approximate NEON SiLU path still produces an incorrect
real-model answer. A temporary diagnostic using scalar SiLU for that path
restored all ten reference decisions; the diagnostic was reverted. Native x86 macOS cross-compilation also exposes a pre-existing
`matvec` cfg overlap; neither implementation was changed. Native Metal resident
limitations remain as recorded below. Qwen3, Q4_0 execution, other browser engines,
GPU prefill and broad model quality remain outside this change. No package was
published and no PR was merged.

---

# Issue #530 tokenizer parity — October 2, 2026

On latest main `8302852`, the new independent Rust test failed for literal
`a\n\nb`: Flare returned `[81,1116,82]` instead of `[81,198,198,82]`.
The fixed implementation respects Digits-before-ByteLevel boundaries. `a  b`
now returns `[81,216,278]` rather than `[81,256,82]`; trailing `x\n\n`
remains `[104,1116]`.

Local validation passed on macOS ARM64, Node 25.6.1, Playwright 1.58.2 /
Chromium headless shell **145.0.7632.6**, with GPU disabled. The actual installed
npm tarball's `FlareTokenizer` matched **81 cases / 498 IDs** for both the full
original SmolLM2 JSON and the committed reduced fixture. Expectations came from
Hugging Face tokenizers **0.22.2**, `add_special_tokens=False`. Provenance,
checksums, license and regeneration are in
[the fixture README](../../../flare-core/tests/fixtures/tokenizer/README.md).

Both browser runs also checked the old behavior through an explicit null
pre-tokenizer negative control, and rejected an unsupported explicit pipeline.
Existing CPU inference, streaming, cancellation, cache and context-boundary
stages passed. The established runner saves the usual JSON, screenshot, log
and trace artifacts. Local Chrome via browser-harness remained unavailable
because its debugging handshake could not attach; the isolated automated
Chromium runner was actually tested. Firefox, WebKit and new GPU runs were not
performed for this tokenizer-only change.

Other checks: **638 workspace tests passed, 47 ignored**, all-target workspace
check, clippy with warnings denied, rustfmt, WASM release build (`wasm-pack
0.13.1 --no-opt`, wasm-bindgen 0.2.117), strict SDK/packed-consumer TypeScript,
all 16 SDK tests, packed ESM/worker syntax and WASM initialization. No native
GPU tests were newly executed; the ignored count remains explicit.

Scope: the exact ordered Digits/ByteLevel pipeline is supported; other explicit
pipelines now return a load error. Missing/null preserves legacy behavior.
BOS/EOS insertion and the embedded GGUF tokenizer remain unchanged. Independent
generation parity, #520 real-answer acceptance, #526 Q/K layout and #529 Q4_0
loading remain separate. No package was published.

---

# PR #537 follow-up: native CI driver compatibility

The native job now uses **Ubuntu 22.04, Mesa 23.2.1-1ubuntu3.1~22.04.4,
LLVM 15.0.7**, where **all 44 ignored GPU tests pass**. The test runner still
requires an adapter, executes each test separately and fails on assertions,
crashes or timeouts. The browser WebGPU job remains on Ubuntu 24.04 / SwiftShader.

[Driver comparison and all-test artifacts](https://github.com/sauravpanda/flarellm/actions/runs/36982135714)
confirm that the unchanged Q3_K/Q6_K shaders execute on the configured stack.
On Ubuntu 24.04 with Mesa 25.2.8 / LLVM 20.1.2, the four original Q3_K/Q6_K tests
exit with SIGSEGV. A GDB run stopped in generated shader code with corrupted
stack frames. Disabling LLVM optimization and inlining byte reads did not resolve
the failure; that source experiment was reverted. This narrows the compatibility
problem to the software-driver stack but does not identify the exact upstream
compiler defect. Ubuntu 24.04 native Mesa coverage remains unsupported.

Two additional ignored tests compare each packed kernel against CPU dequantization
and dot products using nonuniform bytes, three rows, two batches, and 1/2/65 blocks
per row. They cover u32/half-word-aligned block starts, terminal padding and the
second 64-lane loop iteration. These pass on local Metal and hosted Mesa 23.2.1;
the existing numerical assertions and tolerances are retained.

Workspace tests: 634 passed / 47 ignored. Clippy and rustfmt pass. The historical
Metal SiLU mismatch and the broader #521 coverage gaps below remain separate.
Revalidate this suite when upgrading the native CI image or software driver.

---

# Issue #521 automated coverage — October 2, 2026

## Browser CI evidence

The installed-package job passed on hosted Linux x86_64, Node 20.20.2 and
Playwright Chromium 145.0.7632.6. The model and tokenizer were generated from
committed scripts and checked against `fixture.json`; no private assets were used.
The ordinary job disables GPU and verifies that `requestAdapter()` returns null.
All eight lifecycle/numerical stages passed, including streamed Unicode from
actual generated tokens, cancellation/reset/reload, model-cache reuse with HTTP
503 downloads, and the CPU context boundary. The deliberate required-adapter run
exited 1 with the expected missing-adapter error; a corrupted GGUF also exited 1
with a checksum error before browser startup.

The software WebGPU job passed all six dedicated worker modes: default and
forced-f32 numerical comparisons, insufficient adapter storage, insufficient
device storage, invalid pipeline, and oversized dispatch. Adapter metadata says
Google SwiftShader with 32,768-byte workgroup storage; Flare requested 16,416.
The device did not enable shader-f16 or subgroups. Both are explicitly reported
as skipped; this run certifies neither path. Five comparison tokens match
`[36,28,20,12,4]`; maximum absolute logit error was about 0.001900 under the unchanged
`0.002 + 0.002 * abs(cpu)` bound. Prefill is CPU in both comparison paths.

Artifacts include JSON reports, browser/server/network logs, screenshot and
Playwright trace. Example observed CPU measurements across two hosted runs were
16.1–17.6 ms load, 2.9–6.1 ms TTFT and roughly 2,222–5,714 tokens/s for this tiny
fixture. This variance and sub-millisecond decode intervals make these unsuitable
as performance gates; no threshold or stable baseline is claimed.

Evidence runs:

- [Ordinary CI including CPU browser and negative adapter check](https://github.com/sauravpanda/flarellm/actions/runs/36980374157)
- [Software GPU workflow and downloadable artifacts](https://github.com/sauravpanda/flarellm/actions/runs/36980374150)

## Native capability and test results

The initial Mesa llvmpipe run rejected Flare's unconditional 1 GiB storage binding
request because the adapter supports 128 MiB. Device creation now caps buffer
limits at the adapter's advertised values, with the existing 1 GiB ceiling and
attention workgroup minimum retained. A unit test covers 128 MiB, 1 GiB and 2 GiB
adapters. This enables small kernels; it does not certify large-model allocation
or sharding on lower-limit devices.

After negotiation, the parallel Mesa run passed all six integration tests but its
library test process exited with SIGSEGV. A serial run also crashed, specifically
at `test_dequant_matvec_q3k_matches_cpu`. The workflow now discovers every ignored
test and runs each in its own process; crashes remain failures and the remaining
tests still execute. Each has a 120-second timeout and its own saved log. This
isolates a software-driver failure without weakening correctness assertions.

The final isolated Mesa 25.2.8 / LLVM 20.1.2 run exercised all 42 ignored tests:
**38 passed, 4 failed**. The single/multi-row Q3_K and Q6_K matvec tests each exited
with SIGSEGV (`-11`); all remaining tests ran and passed. Each failure has a log
and a JSON entry. The workflow intentionally remains red. A deliberate fake test
executable that crashes followed by one that passes separately verified that the
runner continues after a signal and still exits 1 overall.

Local native Metal: 41 passed and the existing SiLU comparison failed
(`0.7310586` vs `0.732906` at `1e-3`). No tolerance or shader was changed. Full
workspace: 634 passed / 45 ignored; clippy, rustfmt, WASM release build,
packaged SDK types/ESM/worker checks, 16 SDK tests and workflow actionlint pass.

The authorized local Chrome harness could not complete its debugging handshake;
local Chrome was not tested for this PR. Hosted Chromium was actually executed.
Firefox, WebKit, physical GPU CI, independent reference generation/tokenizer
fixtures, GPU prefill/more quantization/context coverage, fully offline startup
and stable performance baselines remain outstanding. This is **Refs #521**,
not closure of the issue. The synthetic tokenizer and regression outputs are
not an independent trusted model reference; see `fixture.json` provenance.

---

# Issue #527 validation — October 2, 2026

## Established cause and fix

On main `ea1c5c8`, the previously packed SDK in Chrome 154/macOS requested a
16,384-byte device limit from an adapter supporting 32,768 bytes. Chrome reported
`attention_scores_f16` needing 16,416 bytes, then rejected its command buffers.
Initialization returned true and successful readback exposed zero-filled logits.
For `Hello`, CPU returned `[198,57,5248,18948,346,2316,11904,260]`; GPU returned
`[198,0,0,0,0,0,0,0]`. Only the first token used CPU prefill logits.

The fix requests 16,416 bytes, or 16,672 when subgroup attention is enabled.
These values follow [WebGPU's compute pipeline validation rule](https://www.w3.org/TR/webgpu/):
each workgroup variable is rounded separately to 16 bytes. The baseline, f16,
and prefill attention shaders use 4096 f32 scores plus two scalar f32 variables;
the subgroup shader adds 64 f32 reduction slots. A source-based Naga test guards
these requirements. An insufficient adapter returns `GpuError` before device
creation; the existing `init_gpu()` boolean API returns false.

The checked async path covers lazy pipeline creation, encoding and submission
with validation/internal/out-of-memory scopes, and propagates readback and
non-finite-logit failures. It never samples output from an invalid command.
A failed model step drops GPU state, clears conversation KV and selects CPU;
`next_token_async()` rejects, clears stale logits and ends the stream with
`error`. A new generation must prefill its prompt. Existing infallible Rust
async methods remain compatibility wrappers; callers should use the additive
`try_forward_async` / `try_forward_single_token_gpu_async` APIs for recovery.
Successful JavaScript return values and SDK method signatures are unchanged.

## Chrome and installed-package results

Built WASM with wasm-pack 0.13.1 (`--no-opt`) and wasm-bindgen 0.2.117, built strict
TypeScript, and installed the actual npm tarball into a separate consumer.
Chrome used the retained SmolLM2-135M Q8_0 and original tokenizer whose hashes
are recorded below. All seven worker modes in the README passed:

- Eight real-model steps have finite, nonzero logits, with no uncaptured errors.
  CPU and GPU both returned `[198,57,5248,18948,346,2316,11904,260]`, decoded as
  `\nI'm glad you're enjoying the`. The device requested/granted 16,416 bytes.
  Real-model logits are **not numerically identical**; matching this short
  token sequence does not establish broad numerical parity or answer quality.
- The controlled fixture passes five steps (one CPU prefill plus four GPU
  decodes), including f16 and f32 KV paths. CPU/GPU tokens are
  `[36,28,20,12,4]`. With f16 KV, the maximum absolute error across all steps
  is approximately `0.001699`; every logit satisfies
  `abs(gpu-cpu) <= 0.002 + 0.002 * abs(cpu)`. The f32-KV run reaches
  approximately `0.001901` maximum absolute error.
- A simulated 16 KiB adapter fails initialization without a device request.
  A forced 16 KiB device, an invalid attention entry point and an oversized
  dispatch each reject after the first token and recover with a fresh CPU
  generation. Errors are caught by the scopes rather than console-only errors.
- The full packed SDK GPU lifecycle passes cancellation, reload, reset,
  concurrent-request rejection, download abort, disposal, OPFS, Unicode and
  responsive UI checks. Long-prompt cancellation settled in about 21 ms.
- The new fixture and forced-device regressions fail against the previous SDK:
  it emits zero logits / extra tokens instead of rejecting the invalid GPU step.

## Local checks and limits

- Workspace tests: 633 passed, 45 ignored. Workspace clippy with warnings denied,
  rustfmt, strict TypeScript, 16 SDK tests, WASM build and packed consumer checks
  pass.
- Actual native Metal run: 36 existing GPU unit tests, two new error-recovery
  tests, and three existing matvec integration tests pass. The existing
  `test_gpu_silu_mul_vec` fails: GPU `0.7310586` versus CPU `0.732906` exceeds
  its `1e-3` tolerance. The same failure occurs on unchanged main.
- Native resident inference is not certified: wgpu 24/Naga rejects `enable f16`
  in the current shaders. During diagnosis, a temporary portable-shader run
  also returned non-finite resident logits; native buffer reuse needs separate
  investigation. Those native shader-selection changes are not part of this PR.
  The new checked path reports these failures and clears state.
- This Chrome/wgpu build exercised f16 and f32 KV, but not subgroup attention.
  The subgroup storage requirement is covered by the source-derived limit test.
- GGUF layout #526, Q4_0 load #529, tokenizer parity #530, broad browser/GPU CI
  #521 and release gates #532 remain separate. #520 still needs its real-answer
  acceptance established. No package was published.

---

# Historical validation for PR #535

Tested on macOS with Chrome 154 using the installed npm tarball, through the
local browser-harness. Neither the model nor the tokenizer is committed.

Assets:

- Existing SmolLM2-135M-Instruct Q8_0 GGUF, 144,811,360 bytes;
  SHA-256 `5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83`.
- Original HuggingFaceTB/SmolLM2-135M-Instruct tokenizer JSON;
  SHA-256 `9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.

## Passed

CPU and WebGPU lifecycle runs both completed initialization/load error recovery,
model loading with byte progress, chat-history formatting, real streamed output,
cancellation after the first decoded token, reload and another request, reset,
concurrent-request rejection, cancellation during a long prompt, slow-download
abort, another load/generation, and disposal. Cancellation during the long-prompt
request settled in approximately 21 ms. A main-thread interval continued ticking
throughout inference. CPU/GPU inference uses the async WASM entry points.

The packaged demo loaded the model, streamed 64 tokens, and disposed successfully.

The final CPU consumer also round-tripped `[1,2,3]` through OPFS in a module worker,
read worker device/storage capabilities, exercised the progressive loader's
worker fetch path (HTTP 404, rather than a missing-window error), and decoded
three byte tokens to `€` through the installed generated binding and TextDecoder.

Local checks passed: TypeScript strict compilation, 16 SDK/worker tests,
workspace Rust tests (630 passed, 43 ignored), an additional chat-history Rust
test, workspace check, clippy with warnings denied, and rustfmt. Package exports,
worker syntax and WASM initialization passed with Node 20.19.2 and 25.6.1.
The local WASM build used wasm-pack 0.13.1 with `--no-opt` because its optional
wasm-opt step was unavailable in the sandbox.

## Output quality remains unresolved

For raw prompt `Hello`, greedy CPU produced token IDs
`[198,57,5248,18948,346,2316,11904,260]`, decoded as
`\nI'm glad you're enjoying the`. The matching GPU request produced
`[198,0,0,0,0,0,0,0]` (newline followed by `<|endoftext|>`).

For system `Answer briefly.` and user `Hello!`, CPU produced
`I'm a math student\nI'm`; GPU produced `I` followed by repeated
`<|endoftext|>`. These are recorded outputs, not reference-quality assertions.
No llama.cpp parity or broad model compatibility is claimed.

The embedded GPT-2 GGUF vocabulary also returned visible byte-level markers in
an initial diagnostic run. The SDK now requires an original tokenizer URL for
that tokenizer family instead of silently returning encoded marker strings.
Original JSON tokenizer boundary parity remains tracked in #530. Engine/model
issues #526, #527 and #529 remain outside this change. Therefore #520's full
real-answer acceptance criterion remains open. Broader CI and release validation
are tracked in #521 and #532.
