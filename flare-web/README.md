# Flare browser SDK

Build with `npm install && npm run build`. The build generates WASM bindings first,
then compiles and typechecks the TypeScript SDK against those bindings.

```js
import { Flare } from '@sauravpanda/flare';

const flare = await Flare.init({
  backend: 'cpu', // or 'webgpu' to request async GPU decode
  cache: true,   // opt in to browser Cache Storage
  onProgress: (loaded, total) => console.log(loaded, total),
  // tokenizerUrl: '/models/tokenizer.json', // optional original HF tokenizer
});
await flare.loadModel('/models/model.gguf');
const controller = new AbortController();
const result = await flare.chat({
  message: 'Hello!',
  system: 'Answer briefly.',
  maxTokens: 64,
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  repeatPenalty: 1.1,
  signal: controller.signal,
  onToken: text => { document.querySelector('#output').textContent += text; },
});
console.log(result.text, result.tokenIds, result.stopReason);
flare.dispose();
```

`generate({prompt, ...options})` accepts a rendered prompt. `chat({messages,
...options})` accepts role/content history. `chat({message, system, ...options})`
formats one user turn. Both use the engine's detected ChatML, Qwen3 non-thinking, Llama3, Phi3, Gemma,
Alpaca or raw template. They do not execute arbitrary Jinja or keep implicit
conversation history. Each generation resets the KV cache. `reset()` clears it explicitly.
Sampling also accepts `minP` and `seed`. `onToken` receives text and the token ID;
a final incomplete UTF-8 suffix is replaced by U+FFFD with token ID `-1`.
Empty text chunks are possible while a multi-byte character is buffered.

## Lifecycle and cancellation

Downloads report incremental byte progress, but the complete GGUF is buffered
before parsing. Cache Storage retains model bytes; tokenizer JSON is fetched
on each load. This is not a complete offline asset cache.

Only one operation can run at a time; a concurrent call rejects with
`FlareError.code === 'BUSY'`. All worker responses carry a request ID. Progress
callbacks are local to the instance. Download, parsing, prefill and decode all
run in a module worker. CPU prefill is synchronous WASM, even through the async
binding. `cancel()` or an AbortSignal **terminates that worker immediately**,
rejects the active operation with `ABORTED`, and releases its WASM/GPU resources.
The next generation creates a worker and reloads the last model URL (from cache
when enabled). Cancellation discards partial output from the returned result;
already delivered callback chunks remain with the application.

`loadModel(url, {signal})` supports cancellation. Use `Flare.init()` followed by
`loadModel` when you need to cancel loading; `init({modelUrl})` is a convenience
for uncancellable initialization. URLs resolve relative to the consumer page.
`dispose()` is idempotent, terminates outstanding work with `DISPOSED`, and
permanently closes the instance. Worker crashes reject with `WORKER`, decoding
message failures with `PROTOCOL`, and worker operation failures with `INIT`,
`LOAD`, `GENERATE`, or `RESET`. Failed operations discard the worker, so retries
can initialize cleanly. Callbacks that throw reject their operation too.

## Packaging and browser requirements

The root ESM export provides `Flare` and preserves the generated WASM exports
(including the default WASM initializer). `@sauravpanda/flare/wasm` explicitly
exposes the low-level API. `@sauravpanda/flare/worker` resolves the compiled module
worker; the wrapper creates it via `new URL('./worker.js', import.meta.url)`.
Use a bundler that supports module worker URLs, or an import map pointing to the
installed `dist/index.js`. Deploy `dist/` and `pkg/` with their relative paths
intact. Serve WASM as `application/wasm`. CSP must permit module workers and
WASM; remote model/tokenizer URLs must allow CORS. WebGPU and Cache Storage
require a secure context (localhost is supported).

CPU is the default. Selecting WebGPU fails loading when device initialization
fails. Capability detection is not an inference correctness guarantee. The
embedded GGUF tokenizer has limited model coverage. GPT-2 byte-level BPE GGUF
models (including SmolLM2) require an original `tokenizerUrl`; loading fails with
a helpful error if it is missing. Original SmolLM2 JSON supports the ordered
`Digits(individual_digits=true)` → `ByteLevel(add_prefix_space=false,
use_regex=true)` pipeline, validated against Hugging Face tokenizers 0.22.2.
Qwen3 supports its exact NFC and Split→ByteLevel pipeline; see the
[Qwen3 quick start and limits](../QWEN3.md). Other explicit pre-tokenizer pipelines fail at load; missing/null pipelines
retain legacy whole-chunk BPE. Encoding does not insert BOS/EOS automatically.
See [tokenizer support and fixture provenance](../flare-core/tests/fixtures/tokenizer/README.md).
The embedded GGUF tokenizer is unchanged. Known engine/model issues (#526 Q/K layout, #527 GPU attention,
#529 Q4 loading) can affect real answers. This SDK does not repair those issues.

Serve the package directory to try `demo/`. `demo/advanced.html` retains the
previous low-level experimental demo. Run `npm run test:sdk` after building.
For the installed tarball browser test, see `tests/browser/README.md`. Broader CI
and release checks are tracked in #521 and #532.

## Experimental bounded decisions

Use `flare.decide({ state, question, choices })` to score 2–8 text choices with
one Qwen3 CPU prefill. Returned scores are relative to the offered labels, not
calibrated confidence. See [the contract and evaluation](../DECISIONS.md) and
[the packaged routing demo](demo/routing.html).
