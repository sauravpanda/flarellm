// Independent llama.cpp numerical reference, executed in a disposable worker.
import init, { FlareEngine, FlareProgressiveLoader, FlareTokenizer } from './node_modules/@sauravpanda/flare/pkg/flare_web.js';
const assert = (v, message) => { if (!v) throw new Error(message); };
self.onmessage = async ({ data: { gpu = false, real = false, forceF32 = false } }) => {
  const result = { passed: false, gpu, real, forceF32, paths: [], devices: [], errors: [] };
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
    await init();
    const reference = await (await fetch(real ? '/real-reference.json' : '/rope-reference/reference.json')).json();
    const url = real ? '/real.gguf' : '/rope-reference/gqa.gguf';
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2,'0')).join('');
    assert(hash === reference.modelSha256, 'Reference model checksum');
    if (real) {
      const tokenizer = FlareTokenizer.from_json(await (await fetch('/original-tokenizer.json')).text());
      assert(JSON.stringify(Array.from(tokenizer.encode(reference.prompt))) === JSON.stringify(reference.promptIds), 'Original tokenizer prompt IDs');
      tokenizer.free();
    }
    const paths = real ? ['bulk'] : ['bulk', 'chunked', 'f32', 'attached'];
    for (const path of paths) {
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
      if (onGpu) assert(await engine.init_gpu(), 'Required WebGPU adapter unavailable');
      engine.reset();
      await engine.begin_stream_with_params_async(new Uint32Array(reference.promptIds), reference.steps.length, 0, 1, 0, 1, 0);
      const steps = [];
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
      result.paths.push({ path, backend: onGpu ? 'async WebGPU decode; CPU prefill' : 'CPU/WASM', steps });
      engine.free(); engine = undefined;
    }
    assert(result.errors.length === 0, 'GPU errors escaped scopes');
    result.passed = true;
  } catch (error) { result.failure = String(error); }
  finally { engine?.free(); postMessage(result); }
};
