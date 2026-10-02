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
Q8_0 fixture has deterministic nonzero weights, 128-wide square tensors with Q/K exported in GGUF adjacent-pair order, two 64-element heads, and a 128-token vocabulary.
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
They do not establish real-answer quality. Original tokenizer parity is checked
separately by the independent fixtures described below.

### GPU jobs and capability requirements

`GPU correctness (software Vulkan)` runs weekly, on workflow dispatch, and on PRs changing that workflow:

- Native: Ubuntu 22.04 + Mesa software Vulkan; executes all ignored `flarellm-gpu`
  library/integration tests through `.github/scripts/run_gpu_tests.py`.
  It discovers ignored tests from Cargo's actual test executables and runs each
  in a separate process, serially, with a 120-second per-test timeout. Crashes,
  timeouts and assertions remain failures, but cannot hide the remaining tests.
  Per-test logs and `results.json` record every pass/failure/not-run state. `FLARE_REQUIRE_GPU=1` converts
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
remain unresolved on the affected native platforms. The configured Ubuntu 22.04
Mesa 23.2.1 / LLVM 15 stack passes all 44 ignored tests. Ubuntu 24.04's Mesa 25.2.8 /
LLVM 20 stack crashes in Q3_K/Q6_K kernels even with serial execution and LLVM
optimization disabled; it is not the supported native CI stack. See VALIDATION.md
for the comparison. Revalidate all tests before upgrading the runner/driver.

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

Remaining #521 work: independent generation reference fixtures, additional tokenizer
configurations, GPU prefill/more quantization/context coverage, fully offline startup,
other browser/platform configurations, physical adapters and stable performance
baselines. This change deliberately references rather than closes #521.

The ordinary job also runs `run.mjs ... --gpu --disable-adapter` and requires an
explicit missing-WebGPU error and exit 1. This deliberate failure has its own
artifact subdirectory; setup failures cannot satisfy that assertion. For a
checksum negative test, alter `fixture.gguf` in a disposable consumer and run the
driver: it must exit 1 with `Fixture checksum: fixture.gguf` before browser startup.

## Independent SmolLM2 tokenizer parity (#530)

Every ordinary packed-consumer run also executes `tokenizer-parity.mjs` through
the installed `FlareTokenizer` WASM binding. It compares 81 cases / 498 IDs with
Hugging Face tokenizers 0.22.2 expectations, verifies the reduced JSON checksum,
checks unsupported-pipeline errors, and reproduces the old incorrect IDs by
explicitly selecting legacy (null pre-tokenizer) behavior. Failures propagate
through the existing report/trace/exit-status path. GPU runs use the same check;
no model inference or GPU is needed for tokenizer parity.

The ~18 KiB committed subset preserves original IDs and all relevant merges,
including forbidden cross-boundary merges for the negative controls. It is
verified independently against the full original during fixture generation.
See [provenance, regeneration and supported semantics](../../../flare-core/tests/fixtures/tokenizer/README.md).
CI requires no downloads or Python tokenizer installation. The older synthetic
inference fixture's `referenceImplementation: null` remains accurate.

To additionally test the full, checksum-pinned original JSON with the same runner:

```sh
ORIGINAL_TOKENIZER_JSON=/absolute/path/tokenizer.json \
  node flare-web/tests/browser/run.mjs /tmp/flare-consumer
```

This adds a separate full-original stage to the report. The runner copies the
verified file into the consumer. In a manually served consumer with that file,
call `await runCI({originalTokenizer: true})`.

## Independent GGUF Q/K reference (#526)

The same packed-consumer runner now checks a checksum-pinned GQA model against
16 steps of external llama.cpp logits and tokens. It exercises bulk raw loading,
chunked raw loading, progressive f32 loading and separate raw attachment. GPU
runs cover default and forced-f32 KV; prefill still runs on CPU. Source, pinned
revision, regeneration, tolerances and limitations are documented in
`flare-loader/tests/fixtures/rope/README.md`.

The earlier lifecycle fixture retains its exact token expectations. Its generator
now interleaves the original split-half Q/K rows when writing GGUF, so the new
loader restores exactly the original model weights. Only the GGUF checksum
changes; no generated token expectations were replaced.

An optional real-model comparison uses local assets and the same runner:

```sh
ORIGINAL_TOKENIZER_JSON=/path/to/tokenizer.json \
REFERENCE_MODEL_GGUF=/path/to/smollm2-360m-instruct-q8_0.gguf \
REFERENCE_LOGITS_JSON=/path/to/full.reference.json \
node flare-web/tests/browser/run.mjs /tmp/flare-consumer
```

The committed `smollm2-reference.json` can also serve as `REFERENCE_LOGITS_JSON`
for selected-logit checks. Add `--gpu` for hosted Linux SwiftShader, or
`--gpu-system` to use the system adapter in isolated Chromium. Adapter fields,
features and each actual backend are included in the report. These options
launch a dedicated test browser and do not touch existing user tabs. Real-model
checks require the original tokenizer checksum and verify the exact prompt IDs.
