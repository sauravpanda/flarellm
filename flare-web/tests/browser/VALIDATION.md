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
