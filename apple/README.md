# FlareKit native Apple preview

FlareKit downloads, verifies, caches, loads, and runs a model locally. It contains
no agent loop, tools, personal memory, or task orchestration; those belong to the
application using it. The Swift package wraps the existing Rust CPU engine with
an owned, serial C bridge. It does not run a server or embed a browser.

## Supported first slice

- iOS 18+ deployment target, ARM64 device and simulator slices; ARM64 macOS 14+
  slice for development and automated tests. Intel Mac/simulator slices are not
  included by this build script.
- Dense Qwen3 up to 0.6B, Q8_0/F32 GGUF plus the original `tokenizer.json`.
  The public model manifest pins revisions, SHA-256 hashes, and exact lengths.
- Greedy raw-prompt generation or the existing Qwen3 non-thinking text chat
  template. Context is explicitly bounded to 32–512 total tokens; reserve output
  space. Overflow is an error, never silent truncation.
- Streaming UTF-8 text, EOS/length completion, cancellation between native steps,
  and reusable loaded weights. Each new request resets KV state; conversation
  history is supplied by the caller. Sampling, tool schemas, and prefix reuse are
  outside this preview.
- CPU only. Qwen3 Metal is not implemented. A successful simulator test is not a
  physical-iPhone memory, battery, speed, or thermal qualification.

## Build and integrate

Use an Apple Silicon Mac with Xcode and Rust. Build the local binary dependency
before opening `apple/Package.swift` or adding `apple/` as a local package:

```sh
rustup target add aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin
bash apple/scripts/build.sh
swift test --package-path apple
```

The script creates the ignored `apple/Frameworks/FlareNative.xcframework`, with
C headers/module map and three platform slices. The bundle is built from source;
this PR does not publish a signed binary release or remote Swift package.
Link the `FlareKit` product in the host application. A later distribution release
can package this artifact with a checksum and a versioned binary-target URL.

```swift
import FlareKit

let store = try ModelStore() // Application Support/Flare/Models, excluded from backup
let assets = try FlareModels.qwen3_06B_Q8() // No network operation yet
let cached = try await store.prepare(model: assets.model, tokenizer: assets.tokenizer) {
    received, total in
    // Progress runs off the main actor. Forward UI updates explicitly.
}
let model = try await FlareModel.load(cached, context: 256)
let request = Task {
    try await model.chat(
        messages: [.init(role: .user, content: "What is the capital of France?")],
        maxTokens: 32,
        onToken: { text in /* forward the text to your UI */ }
    )
}
let result = try await request.value
print(result.text, result.firstTokenMs, result.elapsedMs)
// For a Stop button: request.cancel(); await completion before close/reuse.
try await model.close()
```

Callbacks execute synchronously on the model's dedicated queue. Keep them short;
they provide backpressure and must not synchronously wait for another call on the
same model. The wrapper does not create an unbounded AsyncStream. Concurrent
requests or close-during-generation return `busy`. `Task.cancel()` stops at the
next boundary before/after the full prompt prefill or between decoded tokens;
it cannot interrupt prefill or a kernel already executing. Loading can be cancelled before/after the
native load, not halfway through weight allocation. Interrupted generation resets
KV state before another request runs. Explicit `close()` releases native resources;
the ownership wrapper also releases them when its last reference disappears.

## Cache behavior

- The supplied model and tokenizer total **650,869,342 bytes**. Downloads use
  URLSession's temporary disk files; neither hashing nor the Swift wrapper loads
  the entire GGUF into a `Data` buffer. Rust reads it through a buffered file.
- Files are keyed by SHA-256. Both exact byte length and a streaming SHA-256 are
  checked before atomic publication in the cache directory. HTTP failures,
  redirects to HTTP, hash mismatches, and cancelled downloads cannot publish an
  incomplete model. Production downloads require HTTPS.
- Every cache hit re-verifies the file and makes no network request. This adds
  disk/hash work but detects corruption rather than trusting a filename.
- Application Support preserves files for offline reuse. The model directory is
  excluded from backups. Use one `ModelStore` per app-private directory; shared
  app-group/multi-process cache coordination is not implemented.
- The default cache limit is 2 GB. `maximumBytes` configures it; in-flight assets
  reserve their full size. Existing assets are not silently evicted. Use
  `cachedBytes()`, `cachedURL(for:)`, and `remove(_:)` to manage storage.
  This is a final-cache budget, not a promise about temporary disk or RAM usage.
- Identical concurrent downloads return `busy`; different assets may download
  concurrently within the reservation budget. A completed model file remains
  reusable if the separate tokenizer download fails.
- This version uses foreground downloads. After interruption, a retry restarts
  the missing asset. Background URLSession restoration and HTTP range resume are
  follow-ups. A process crash can leave a hidden `.partial` staging file; the next store
  initialization removes these abandoned files from its exclusively owned root.

The pinned Qwen weights/tokenizer are Apache-2.0. Applications distributing these
assets must preserve upstream license notices. Model files are not shipped with
this package; only a small deterministic repository-owned test fixture is included.

## Validation and performance

Normal tests use the same untrained Qwen3 fixture and independent llama.cpp token
oracle as `flare-loader/tests/fixtures/qwen3`. The copy in the test bundle allows
Xcode to install it in the simulator. `expected.json` retains the first 16 oracle
tokens; `tokenizer.json` maps `ace` to prompt IDs `[2,4,7]`. `real-expected.json`
contains the non-EOS tokens from the existing `real-0.json` independent reference.

```sh
cargo test --locked -p flarellm-native
cargo test --locked -p flarellm-core neon_dot_fallback
swift test --package-path apple
cd apple
xcodebuild test -scheme FlareKit \
  -destination 'platform=iOS Simulator,name=iPhone 16 Pro' CODE_SIGNING_ALLOWED=NO
```

Tests cover token agreement, streaming UTF-8, context rejection, EOS, full-context
prefill, cancellation/reset, concurrent-request rejection, unload, corrupt cache
repair, HTTP/hash/length failures, storage reservations, and cache reuse. Apple
CI builds the device/simulator/macOS slices and runs Swift tests on macOS and an
installed iPhone simulator. Ordinary CI never downloads trained weights.

Optional real-model smoke, explicitly opting into the 651 MB download:

```sh
FLARE_REAL_CACHE=/tmp/flare-models FLARE_REAL_RESULT=/tmp/flare-result.json \
  swift test --package-path apple --filter RealModelTests
```

This tests the actual HTTPS download, a verified cache hit, native load, Qwen3
chat formatting, and exact greedy tokens against the preserved reference. The
result records prepare/cache/load/first-token/request timings and model hashes.
Generation timings start after tokenization, at native request initialization;
they include prefill, time between native steps, and previous callbacks. They
exclude cache verification, queue wait, and prompt encoding. First-token time
measures the first model decision, including EOS, rather than first visible text. Native f32 activations and packed Q8
projection matrices match the existing correctness path. Embeddings/output
storage still uses substantial f32 memory; this preview does not promise a low
memory footprint just because the download is quantized.

### Recorded validation

On 2026-10-18, all seven native bridge tests and the per-lane ARM fallback test
passed. The Swift suite passed eight offline tests on the iPhone 16 Pro simulator
(iOS 18.6), with the opt-in real-model test skipped. All nine Swift tests passed
on an M1 Max Mac, including the pinned model's nine non-EOS reference tokens and
correct stop termination. Device and simulator Rust slices compiled successfully.

The [Mac smoke record](results/macos-qwen3-smoke.json) preserves the exact output,
IDs, hashes and environment. A verified disk hit took 313 ms; native load 306 ms;
first token 3.76 s; the nine-token answer plus EOS completed in 6.69 s. Assets were
already cached for this recorded run after the preceding actual HTTPS download.
No other Flare tests ran concurrently, but this remains a single shared-host
observation and provides no iPhone speed or performance-improvement claim. Peak iPhone memory,
battery and thermal measurements remain outstanding.

### Repeatable runtime benchmark

Build the XCFramework from the revision under test, then run the opt-in benchmark:

```sh
bash apple/scripts/build.sh
FLARE_BENCHMARK=1 FLARE_BENCH_CACHE=/tmp/flare-models \
  FLARE_BENCH_REVISION="$(git rev-parse HEAD)" FLARE_BENCH_LABEL=neon \
  FLARE_BENCH_RESULT=/tmp/flare-benchmark.json \
  swift test --package-path apple --filter BenchmarkTests
```

It may download the pinned 651 MB assets. There are three fixed prompts, one
warmup per prompt, and three measured requests per prompt (`FLARE_BENCH_REPEATS`
can set 1–20). Each request resets KV state and reuses loaded weights. The short
prompt must match the independent token oracle; all measured runs must match
their warmup. Compare token IDs and finish reasons across revisions as well.
Keep other inference and builds idle during measurement. Record the host model,
toolchain and power conditions alongside the JSON; `sourceRevision` and
`buildLabel` are caller-supplied labels, not automatic binary verification.

The JSON includes native load time, first-decision latency, request time, decode
steps/second, and thermal state before/after each measured request. Decode steps
include terminal EOS, so they differ from visible output token counts. Memory is
Darwin's process-lifetime high-water resident size in bytes, including XCTest,
download/cache work and loading. It is not per-request memory or a measurement
of memory released on close. Use Instruments for allocation attribution.

To use a physical iPhone, open the package in Xcode, select a signed device test
destination, and add `FLARE_BENCHMARK=1` to the test scheme's environment. Omit
`FLARE_BENCH_CACHE` to use the device's Application Support cache. Leave
`FLARE_BENCH_RESULT` unset to print JSON to the test log, or supply a writable
path inside that test app's container. Record the device, power state, and thermal
conditions, and repeat on both revisions. Mac and simulator numbers do not
establish an iPhone speedup, battery benefit, or sustained thermal behavior.

For a model-free comparison of the Qwen projection shapes:

```sh
cargo run --locked --release -p flarellm-core --example q8_f32_bench
```

This alternates scalar and platform-dispatch timing order, checks bitwise output
agreement, and reports median times. ARM64 dispatch uses NEON with float
activations and the scalar accumulation order; other targets use scalar. Neither
path introduces activation quantization. CI checks the kernel's float results
and the independent Qwen logits/token reference on ARM64 in release mode.

### Recorded NEON comparison

The [2026-10-20 Mac comparison](results/macos-qwen3-neon.json) records scalar main
`cb6a9d7` versus NEON `8af57d0` on an M1 Max, with Rust release libraries and Swift
debug tests. These are medians of three measured requests after one warmup per
prompt, using the same cached model, context 512, and maximum 16 output tokens.

| Prompt tokens | Scalar first token | NEON first token | Speedup | Scalar decode steps/s | NEON decode steps/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 19 | 3.69 s | 1.07 s | 3.46× | 3.12 | 5.50 |
| 64 | 14.17 s | 3.55 s | 3.99× | 3.07 | 5.48 |
| 127 | 24.49 s | 7.04 s | 3.48× | 3.11 | 5.42 |

All nine measured token sequences and finish reasons match across revisions.
The short answer completes in 2.70 s versus 6.57 s. The projection microbenchmark
reports 3.51–3.52× speedups across Qwen's Q and FFN shapes with exact float results.
Independent fixture logits, trained-model reference tokens/sampled logits across
three prompts, and native bridge tests also pass.

No other Flare builds/tests ran concurrently with these recorded measurements.
This is one sequential comparison on a shared Mac; other activity, frequency and
power conditions were not controlled. All measured thermal states were nominal.
Process high-water RSS reached 1.85 GB for scalar and 1.89 GB for NEON, including
XCTest/cache/load work. This change does not establish a memory reduction or
physical-iPhone performance, battery, or thermal results.

### Next performance work

1. Measure a physical iPhone: peak resident memory during load/inference,
   first-token latency at several prompt lengths, decode throughput, cancellation
   delay, and sustained thermal/battery behavior. Simulator speed is not phone speed.
2. Profile the NEON Q8 × f32 path on-device to choose further batching/threading
   improvements. Avoid extra activation quantization without a separate accuracy audit.
3. Reduce expanded embedding/loading memory and qualify larger context budgets.
4. Implement and validate Qwen3 Metal kernels before offering a GPU selector.

The generic ARM64 build also exposed an unconditional SDOT instruction. It now
uses a lane-correct NEON fallback when `dotprod` is absent from the compilation
target; targets that enable it retain SDOT. A per-lane arithmetic regression test
covers both paths. This is a portability correction, not a measured speedup.
