# Installed package browser check

Build WASM and the SDK first. Install the actual tarball into a retained consumer:

```sh
node .github/scripts/check_wasm_package.mjs /tmp/flare-consumer
cp /absolute/path/tokenizer.json /tmp/flare-consumer/tokenizer.json
python3 flare-web/tests/browser/serve.py /tmp/flare-consumer /absolute/path/model.gguf
```

Open `http://127.0.0.1:8520` in an existing Chrome instance and call `await run()`
from its console (or through an authorized CDP harness). Results appear on the
page and in `window.validation`. Call `run('webgpu')` separately to exercise
async GPU decode. A running UI tick counter checks main-thread responsiveness.
The model stays local. No automated model download or browser launch is involved.

Coverage: failed initialization and load, download progress, actual inference
chunks/result equality, cancellation after a decoded token, a follow-up request,
reset, cancellation while a long prompt prefills, BUSY rejection, slow download
abort, reload, disposal, and Unicode bytes from the installed WASM binding.
The fixture defaults to `/tokenizer.json`; pass a second argument
`null` to `run('cpu', null)` for a supported embedded SentencePiece vocabulary.
Output quality is deliberately reported separately from lifecycle assertions;
these checks do not establish parity with a reference engine.

## GPU storage and decode regressions (#527)

After installing the packed consumer above, generate the small deterministic
model fixture (no downloads):

```sh
python3 flare-web/tests/browser/make_gpu_fixture.py /tmp/flare-consumer/fixture.gguf
```

In the consumer page's console, run these modes in separate module workers:

```js
for (const mode of ['normal', 'fixture', 'fixture-f32', 'adapter16', 'device16', 'pipeline', 'dispatch']) {
  const result = await new Promise((resolve, reject) => {
    const worker = new Worker('./gpu-regression.mjs', { type: 'module' });
    worker.onmessage = ({ data }) => { worker.terminate(); resolve(data); };
    worker.onerror = error => { worker.terminate(); reject(error); };
    worker.postMessage({ mode });
  });
  console.log(result);
  if (!result.passed) throw new Error(result.failure);
}
```

The `normal` mode requires the local real model and tokenizer. It checks eight
steps, including GPU decode after CPU prefill, and records tokens and logit
statistics. `fixture` and `fixture-f32` compare five CPU/GPU steps with absolute
plus relative tolerance `0.002 + 0.002 * abs(cpu_logit)`. The generated two-layer
Q8_0 fixture has deterministic nonzero weights, 128-wide square tensors (to
isolate GGUF layout #526), two 64-element heads, and a 128-token vocabulary.
The f32 mode omits the optional shader-f16 device feature to exercise f32 KV.

Fault modes override browser APIs only inside their dedicated worker:

- `adapter16`: report a 16 KiB adapter; initialization must return false before
  requesting a device.
- `device16`: force a 16 KiB device despite a capable adapter. This recreates
  the original invalid attention pipeline.
- `pipeline`: replace the attention entry point with a missing name.
- `dispatch`: exceed the maximum workgroups per dimension.

Execution failures must reject after the CPU-prefill token, emit no failed GPU
token, clear stale logits and the stream, switch to CPU, and generate the same
CPU sequence after a fresh prefill. No validation errors may escape their scopes.
Native error recovery tests require a real adapter:

```sh
cargo test -p flarellm-gpu --test decode_errors -- --ignored --nocapture
```
