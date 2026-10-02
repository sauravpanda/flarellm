# Browser SDK validation

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
