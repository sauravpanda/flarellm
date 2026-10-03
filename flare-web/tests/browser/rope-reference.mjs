// Independent llama.cpp numerical reference, executed in a disposable worker.
import { Flare } from './node_modules/@sauravpanda/flare/dist/index.js';
import init, { FlareEngine, FlareProgressiveLoader, FlareTokenizer } from './node_modules/@sauravpanda/flare/pkg/flare_web.js';
const assert = (v, message) => { if (!v) throw new Error(message); };
self.onmessage = async ({ data: { gpu = false, real = false, forceF32 = false, qwen = false } }) => {
  const result = { passed: false, gpu, real, forceF32, qwen, paths: [], devices: [], errors: [] };
  let engine;
  try {
    if (gpu) {
      assert(typeof GPUAdapter !== 'undefined', 'Required WebGPU API unavailable');
      const request = GPUAdapter.prototype.requestDevice;
      GPUAdapter.prototype.requestDevice = async function (descriptor) {
        if (forceF32) descriptor.requiredFeatures = descriptor.requiredFeatures.filter(f => f !== 'shader-f16');
        const device = await request.call(this, descriptor);
        result.devices.push({ features: [...device.features], adapter: { vendor: this.info?.vendor, architecture: this.info?.architecture, device: this.info?.device, description: this.info?.description } });
        device.addEventListener('uncapturederror', ({ error }) => result.errors.push(error.message));
        return device;
      };
    }
    const wasm = await init();
    const reference = await (await fetch(real ? '/real-reference.json' : (qwen ? '/qwen3-reference/reference.json' : '/rope-reference/reference.json'))).json();
    const url = real ? '/real.gguf' : (qwen ? '/qwen3-reference/gqa.gguf' : '/rope-reference/gqa.gguf');
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2,'0')).join('');
    assert(hash === reference.modelSha256, 'Reference model checksum');
    if (real) {
      const tokenizer = FlareTokenizer.from_json(await (await fetch('/original-tokenizer.json')).text());
      assert(JSON.stringify(Array.from(tokenizer.encode(reference.prompt))) === JSON.stringify(reference.promptIds), 'Original tokenizer prompt IDs');
      tokenizer.free();
    }
    const paths = real ? (qwen ? ['bulk', 'chunked'] : ['bulk']) : ['bulk', 'chunked', 'f32', 'attached'];
    for (const path of paths) {
      const loadStart = performance.now();
      if (path === 'bulk') engine = FlareEngine.load(bytes);
      else if (path === 'chunked') {
        const loader = FlareEngine.begin_load(bytes);
        for (let i = 0; i < loader.total_tensors; i++) {
          const start = Number(loader.tensor_byte_offset(i)), length = Number(loader.tensor_byte_length(i));
          loader.load_tensor(i, bytes.subarray(start, start + length));
        }
        engine = loader.finalize();
      } else {
        const loader = new FlareProgressiveLoader(url);
        engine = await loader.load(() => {});
        loader.free();
        if (path === 'attached') assert(engine.load_raw_weights(bytes), 'Separate raw attachment failed');
      }
      // GPU requires raw weights; the f32-only route is compared on CPU.
      const onGpu = gpu && path !== 'f32';
      if (onGpu) {
        const enabled = await engine.init_gpu();
        assert(qwen ? !enabled : enabled, qwen ? 'Qwen3 must explicitly refuse incomplete GPU math' : 'Required WebGPU adapter unavailable');
      }
      if (qwen) assert(engine.max_seq_len <= 512, 'Qwen3 browser context cap');
      const start = performance.now();
      const loadMs = performance.now() - loadStart;
      engine.reset();
      await engine.begin_stream_with_params_async(new Uint32Array(reference.promptIds), reference.steps.length, 0, 1, 0, 1, 0);
      const steps = [];
      const prefillMs = performance.now() - start;
      const decodeStart = performance.now();
      for (let s = 0; s < reference.steps.length; s++) {
        const expected = reference.steps[s];
        const token = await engine.next_token_async();
        // The public stream consumes EOS without emitting it; still compare its logits.
        assert(token === expected.token || (token === undefined && expected.token === reference.eosTokenId), `${path} step ${s} token ${token} != ${expected.token}`);
        const logits = engine.last_logits;
        assert(logits.length === (reference.vocabSize ?? expected.logits.length), 'Reference vocabulary size');
        let maxAbsoluteError = 0;
        for (let i = 0; i < expected.logits.length; i++) {
          const index = expected.logitIndices?.[i] ?? i;
          const delta = Math.abs(logits[index] - expected.logits[i]);
          const bound = reference.tolerance.absolute + reference.tolerance.relative * Math.abs(expected.logits[i]);
          assert(Number.isFinite(logits[index]) && delta <= bound, `${path} step ${s} logit ${i}: ${logits[index]} vs ${expected.logits[i]} (bound ${bound})`);
          maxAbsoluteError = Math.max(maxAbsoluteError, delta);
        }
        steps.push({ token: expected.token, maxAbsoluteError });
      }
      result.paths.push({ path, backend: onGpu && !qwen ? 'async WebGPU decode; CPU prefill' : 'CPU/WASM', gpuRejected: onGpu && qwen, loadMs, prefillMs, decodeMs: performance.now() - decodeStart, wasmMemoryBytes: wasm.memory.buffer.byteLength, steps });
      engine.free(); engine = undefined;
    }
    if (qwen) {
      const requestedGpu = await Flare.init({ backend: 'webgpu', cache: false });
      let rejected = false;
      try { await requestedGpu.loadModel('/qwen3-reference/gqa.gguf'); }
      catch (error) { rejected = error.code === 'LOAD' && error.message.includes('Qwen3 WebGPU execution is unsupported'); }
      finally { requestedGpu.dispose(); }
      assert(rejected, 'Explicit SDK Qwen3 GPU request must reject with a CPU diagnostic');
      result.sdkGpuRejection = 'passed';
    }
    if (real && qwen) {
      // Exercise the public worker-backed SDK with the original tokenizer and
      // official chat template, including EOS consumption and a second request.
      const flare = await Flare.init({ backend: 'cpu', tokenizerUrl: '/original-tokenizer.json', cache: false });
      try {
        const loadStart = performance.now();
        await flare.loadModel('/real.gguf');
        result.sdk = { loadMs: performance.now() - loadStart, requests: [] };
        for (let request = 0; request < 2; request++) {
          const start = performance.now();
          const times = [];
          let streamed = '';
          const generated = await flare.chat({ messages: reference.messages,
            maxTokens: 32, temperature: 0, topP: 1, topK: 0, repeatPenalty: 1,
            onToken: (text, id) => { streamed += text; if (id >= 0) times.push(performance.now()); } });
          const expected = reference.steps.map(s => s.token).filter(t => t !== reference.eosTokenId);
          assert(JSON.stringify(generated.tokenIds) === JSON.stringify(expected), 'SDK original-tokenizer chat tokens');
          assert(generated.stopReason === 'eos' && generated.text === streamed, 'SDK EOS and streamed assistant text');
          assert(!generated.text.includes('<think>') && !generated.text.includes('<|im_end|>'), 'Assistant output contains template tokens');
          result.sdk.requests.push({ ...generated, ttftMs: times[0] - start,
            decodeTokensPerSecond: 1000 * (times.length - 1) / (times.at(-1) - times[0]) });
          await flare.reset();
        }
      } finally { flare.dispose(); }
    }
    assert(result.errors.length === 0, 'GPU errors escaped scopes');
    result.passed = true;
  } catch (error) { result.failure = String(error); }
  finally { engine?.free(); postMessage(result); }
};
