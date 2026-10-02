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

## Automated installed-package coverage (#521)

The ordinary `CI / WASM build` job now drives Chromium from the locked Playwright
version in `flare-web/package-lock.json`. It installs the npm tarball with the
same consumer checker above. No external model downloads or private paths are
needed:

```sh
wasm-pack build flare-web --target web --no-opt
npm ci --prefix flare-web
npm run build:sdk --prefix flare-web
node .github/scripts/check_wasm_package.mjs /tmp/flare-consumer
python3 flare-web/tests/browser/make_ci_fixture.py /tmp/flare-consumer
(cd flare-web && npx playwright install --with-deps chromium)
node flare-web/tests/browser/run.mjs /tmp/flare-consumer
```

For an already running local Chrome, serve with `serve.py ... --ci`, then call
`await runCI()` in the consumer page, or `await runCI({gpu: true})`. Each GPU
fault uses a disposable worker. The CI driver launches a fresh browser/context;
it never attaches to user tabs. Set `BROWSER_PORT` or `BROWSER_ARTIFACTS` to change
its localhost port or artifact directory.

### Fixture and assertions

`fixture.json` pins SHA-256 for the generated 370 KiB GGUF and synthetic tokenizer,
prompt IDs, greedy sampling settings, regression token IDs and logit tolerances.
Generation verifies checksums; the driver verifies them again before launching.
The GGUF uses the existing deterministic two-layer generator: F32 embeddings,
norms and output plus Q8_0 layer matrices. It is redistributable under the repo
license. No trained weights are downloaded. The tokenizer deliberately maps the
first three generated tokens to the three bytes of `€`; actual WASM worker
inference must stream two empty chunks then the complete character.

Assertions cover worker init/load errors, download progress, actual CPU generation,
stream/result equality, token-boundary Unicode, cancellation, automatic reload,
reset, BUSY rejection, download abort, disposal, and 27 prompt + 5 generated tokens
at the 32-token context boundary. The CPU CI job disables GPU and verifies no
adapter exists. Cache coverage constructs a fresh worker with the model download
returning HTTP 503. **WASM and tokenizer still load online**; fully offline app
startup is not covered. Tiny-model cancellation does not certify cancellation
during a long-running prefill; the original manual real-model check remains.

These are synthetic regression expectations from Flare, **not an independent
trusted generation reference**. `referenceImplementation` is explicitly null.
They do not establish real-answer quality or original model tokenizer parity.

### GPU jobs and capability requirements

`GPU correctness (software Vulkan)` runs weekly, on workflow dispatch, and on PRs changing that workflow:

- Native: Ubuntu 24.04 + Mesa software Vulkan; executes all ignored `flarellm-gpu`
  library/integration tests with `--no-fail-fast -- --ignored --nocapture --test-threads=1`.
  Tests run serially to avoid concurrent software-driver device initialization. `FLARE_REQUIRE_GPU=1` converts
  the integration helper's optional adapter skip into an explicit failure.
  Existing unit tests and decode error tests already require an adapter.
  Device creation caps buffer limits at the adapter's advertised values (up to
  1 GiB); the previous unconditional 1 GiB binding request prevented llvmpipe's
  128 MiB adapter from running even tiny kernels. Large-model sharding on these
  lower-limit adapters is not certified by the tiny fixtures.
- Browser: pinned Chromium + SwiftShader, requiring WebGPU initialization. It runs
  f16/default and forced f32 KV comparisons and adapter/device storage, invalid
  pipeline and oversized-dispatch fault regressions against the generated model.
  A missing/insufficient adapter fails, not skips. Default mode uses f16 only
  when the adapter offers it; features are recorded, not assumed.

There were zero repository self-hosted runners when this was implemented. These
jobs use hosted runners and software adapters for correctness only. Physical GPU
coverage needs an available managed runner and remains outstanding. The browser
compares all logits over CPU prefill + four GPU decode steps with
`abs(gpu-cpu) <= 0.002 + 0.002 * abs(cpu)`. Prefill runs on CPU in both paths;
this is **not GPU prefill parity**. Q4/mixed-quantized layer formats and GPU context
boundaries remain outside this fixture. No tolerance is widened to hide failures.
Known native SiLU tolerance and resident shader issues described in VALIDATION.md
may keep the diagnostic workflow red; its failures are retained, not suppressed.

### Reports and performance

Artifacts include `result.json`, console/network/server logs, a screenshot and a
Playwright trace; native jobs preserve adapter/platform/compiler metadata and
full test output. The result explicitly reports Chromium, Firefox/WebKit not run,
CPU-job GPU skips and required-GPU failures. Rust tests print their ignored counts.
Load milliseconds, TTFT (first token event, which may be an incomplete UTF-8
character), and decode tokens/second exclude setup. These tiny-model measurements
are informational: hosted runner variance and software GPUs are unsuitable for
physical GPU performance gates. Gather repeated baselines on a stable physical
runner before adding thresholds. No speed threshold is enforced.

Remaining #521 work: independent versioned reference fixtures, original tokenizer
parity, GPU prefill/more quantization/context coverage, fully offline startup,
other browser/platform configurations, physical adapters and stable performance
baselines. This change deliberately references rather than closes #521.

The ordinary job also runs `run.mjs ... --gpu --disable-adapter` and requires an
explicit missing-WebGPU error and exit 1. This deliberate failure has its own
artifact subdirectory; setup failures cannot satisfy that assertion. For a
checksum negative test, alter `fixture.gguf` in a disposable consumer and run the
driver: it must exit 1 with `Fixture checksum: fixture.gguf` before browser startup.
