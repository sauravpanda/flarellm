// Run only in a dedicated module worker. Fault injection stays inside this worker.
import init, { FlareEngine, FlareTokenizer } from './node_modules/@sauravpanda/flare/pkg/flare_web.js';
const assert = (value, message) => { if (!value) throw new Error(message); };
self.onmessage = async ({ data: { mode = 'normal', synthetic = false } }) => {
  const result = { mode, errors: [], devices: [] };
  let engine, tokenizer;
  try {
    const requestAdapter = GPU.prototype.requestAdapter;
    GPU.prototype.requestAdapter = async function (...args) {
      const adapter = await requestAdapter.apply(this, args);
      if (adapter && mode === 'adapter16') {
        const limits = adapter.limits;
        Object.defineProperty(adapter, 'limits', { value: new Proxy(limits, {
          get(target, key) { return key === 'maxComputeWorkgroupStorageSize' ? 16384 : Reflect.get(target, key, target); },
        }) });
      }
      return adapter;
    };
    const requestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (descriptor) {
      if (mode.endsWith('f32')) descriptor.requiredFeatures = descriptor.requiredFeatures.filter(f => f !== 'shader-f16');
      if (mode === 'device16') descriptor.requiredLimits.maxComputeWorkgroupStorageSize = 16384;
      const device = await requestDevice.call(this, descriptor);
      result.devices.push({ adapterStorage: this.limits.maxComputeWorkgroupStorageSize,
        requestedStorage: descriptor.requiredLimits.maxComputeWorkgroupStorageSize,
        features: [...device.features], deviceStorage: device.limits.maxComputeWorkgroupStorageSize });
      device.addEventListener('uncapturederror', ({ error }) => result.errors.push(error.message));
      return device;
    };
    const pipeline = GPUDevice.prototype.createComputePipeline;
    GPUDevice.prototype.createComputePipeline = function (descriptor) {
      if (mode === 'pipeline' && descriptor.label?.startsWith('attention_scores')) {
        descriptor.compute.entryPoint = 'injected_missing_entry_point';
      }
      return pipeline.call(this, descriptor);
    };
    const dispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups;
    let injected = false;
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      if (mode === 'dispatch' && !injected) { args[0] = 65536; injected = true; }
      return dispatch.apply(this, args);
    };
    await init();
    if (mode === 'fixture' || mode === 'fixture-f32') {
      engine = FlareEngine.load(new Uint8Array(await (await fetch('/fixture.gguf')).arrayBuffer()));
      const steps = async () => {
        engine.reset();
        await engine.begin_stream_with_params_async(new Uint32Array([2, 4, 7]), 5, 0, 1, 0, 1, 0);
        const output = [];
        while (!engine.stream_done) {
          const token = await engine.next_token_async();
          assert(token !== undefined, 'Fixture must generate all steps');
          output.push({ token, logits: engine.last_logits });
        }
        return output;
      };
      const cpu = await steps();
      assert(await engine.init_gpu(), 'Fixture GPU initialization failed');
      const gpu = await steps();
      assert(cpu.length === 5 && gpu.length === 5, 'Compare multiple decode steps');
      result.steps = cpu.map((reference, index) => {
        const actual = gpu[index];
        assert(reference.token === actual.token, 'Fixture context diverged');
        assert(actual.logits.some(v => Math.abs(v) > 0.01), 'Fixture output must be nonzero');
        let maxAbsoluteError = 0;
        for (let i = 0; i < reference.logits.length; i++) {
          const a = actual.logits[i], b = reference.logits[i];
          const delta = Math.abs(a - b);
          assert(Number.isFinite(a) && Number.isFinite(b), 'Fixture logits must be finite');
          assert(delta <= 0.002 + 0.002 * Math.abs(b), `Fixture step ${index}, logit ${i}: ${a} vs ${b}`);
          maxAbsoluteError = Math.max(maxAbsoluteError, delta);
        }
        return { token: actual.token, maxAbsoluteError };
      });
      assert(result.errors.length === 0, 'Fixture GPU errors escaped scopes');
      result.passed = true;
      return;
    }
    engine = FlareEngine.load(new Uint8Array(await (await fetch('/model.gguf')).arrayBuffer()));
    tokenizer = FlareTokenizer.from_json(await (await fetch('/tokenizer.json')).text());
    const generate = async () => {
      engine.reset();
      await engine.begin_stream_with_params_async(tokenizer.encode(synthetic ? 'Hel' : 'Hello'), synthetic ? 5 : 8, 0, 1, 0, 1, 0);
      const run = { ids: [], logits: [] };
      try {
        while (!engine.stream_done) {
          const token = await engine.next_token_async();
          if (token === undefined) break;
          run.ids.push(token);
          const logits = engine.last_logits;
          run.logits.push({ finite: logits.every(Number.isFinite), nonzero: logits.filter(x => x !== 0).length,
            min: Math.min(...logits), max: Math.max(...logits) });
        }
      } catch (error) { run.error = String(error); }
      run.text = tokenizer.decode(new Uint32Array(run.ids));
      return run;
    };
    result.cpu = await generate();
    assert(!result.cpu.error && result.cpu.ids.length === (synthetic ? 5 : 8), 'CPU baseline failed');
    result.initialized = await engine.init_gpu();
    if (mode === 'adapter16') {
      assert(!result.initialized && result.devices.length === 0, 'Insufficient adapter must fail before requesting a device');
    } else {
      assert(result.initialized, 'GPU initialization failed');
      result.gpu = await generate();
      if (mode === 'normal') {
        assert(!result.gpu.error && result.gpu.ids.length === (synthetic ? 5 : 8), 'Multiple GPU decode steps must succeed');
        assert(result.gpu.logits.every(x => x.finite && x.nonzero > 0), 'Meaningful model logits must be finite and nonzero');
      } else {
        assert(result.gpu.ids.length === 1 && result.gpu.error, 'Failure must reject after the CPU prefill token, without emitting a GPU token');
        assert(engine.stream_done && engine.stream_stop_reason === 'error', 'Failed stream must stop');
        assert(engine.last_logits.length === 0, 'Failed stream must clear stale logits');
        assert(JSON.parse(engine.backend_info()).backend === 'cpu', 'Failed GPU context must be discarded');
        result.recovery = await generate();
        assert(JSON.stringify(result.recovery.ids) === JSON.stringify(result.cpu.ids), 'CPU recovery must restart with a fresh context');
      }
    }
    assert(result.errors.length === 0, 'GPU errors escaped their scopes');
    result.passed = true;
  } catch (error) { result.failure = String(error); }
  finally { engine?.free(); tokenizer?.free(); postMessage(result); }
};
