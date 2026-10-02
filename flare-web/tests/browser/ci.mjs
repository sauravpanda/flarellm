// The same installed consumer is driven by CI and the local Chrome harness.
import { tokenizerParity } from './tokenizer-parity.mjs';
import { Flare } from '@sauravpanda/flare';
import init, { FlareEngine, FlareTokenizer } from '@sauravpanda/flare/wasm';
const assert = (value, message) => { if (!value) throw new Error(message); };
const equal = (a, b, message) => assert(JSON.stringify(a) === JSON.stringify(b), `${message}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
const rejects = async (promise, code) => {
  try { await promise; } catch (error) { assert(error.code === code, `Expected ${code}: ${error}`); return; }
  throw new Error(`Expected rejection ${code}`);
};
const workerRun = mode => new Promise((resolve, reject) => {
  const worker = new Worker('./gpu-regression.mjs', { type: 'module' });
  const timer = setTimeout(() => { worker.terminate(); reject(new Error(`${mode} timed out`)); }, 60000);
  const finish = () => { clearTimeout(timer); worker.terminate(); };
  worker.onmessage = ({ data }) => { finish(); data.passed ? resolve(data) : reject(new Error(JSON.stringify(data))); };
  worker.onerror = event => { finish(); reject(new Error(event.message)); };
  worker.postMessage({ mode, synthetic: true });
});
const ropeRun = (gpu, forceF32 = false) => new Promise((resolve, reject) => {
  const worker = new Worker('./rope-reference.mjs', { type: 'module' });
  const timer = setTimeout(() => { worker.terminate(); reject(new Error('Reference timed out')); }, 120000);
  const finish = () => { clearTimeout(timer); worker.terminate(); };
  worker.onmessage = ({ data }) => { finish(); data.passed ? resolve(data) : reject(new Error(JSON.stringify(data))); };
  worker.onerror = e => { finish(); reject(new Error(e.message)); };
  worker.postMessage({ gpu, forceF32 });
});
window.runCI = async ({ gpu = false, originalTokenizer = false } = {}) => {
  const report = { passed: false, platform: navigator.platform, userAgent: navigator.userAgent,
    stages: [], metrics: { backend: 'cpu', informational: true }, coverage: { chromium: 'running', firefox: 'not run', webkit: 'not run',
      gpu: gpu ? 'required' : 'skipped: CPU job', independentReference: 'tokenizer pending; generation not implemented' } };
  const record = (name, detail = {}) => {
    report.stages.push({ name, status: 'passed', ...detail });
    document.querySelector('#result').textContent = JSON.stringify(report, null, 2);
  };
  let flare;
  try {
    const fixture = await (await fetch('/fixture.json')).json();
    report.fixture = fixture;
    await init();
    record('independent Llama GGUF logits and load paths', await ropeRun(gpu));
    if (gpu) record('independent Llama GGUF logits with f32 KV', await ropeRun(true, true));
    const parity = await tokenizerParity();
    if (originalTokenizer) record('full original-tokenizer IDs', await tokenizerParity('/original-tokenizer.json'));
    record('independent original-tokenizer IDs', parity);
    report.coverage.independentReference = 'SmolLM2 tokenizer and pinned llama.cpp GQA logits passed';
    const tokenizer = FlareTokenizer.from_json(await (await fetch('/tokenizer.json')).text());
    try { equal(Array.from(tokenizer.encode(fixture.prompt)), fixture.promptIds, 'Pinned synthetic tokenizer IDs'); }
    finally { tokenizer.free(); }
    record('synthetic tokenizer IDs');
    await rejects(Flare.init({ wasmUrl: '/missing.wasm' }), 'INIT');
    const config = { backend: 'cpu', tokenizerUrl: '/tokenizer.json', cache: true };
    flare = await Flare.init(config);
    report.device = flare.deviceInfo;
    await rejects(flare.loadModel('/missing.gguf'), 'LOAD');
    let progress = 0;
    flare.dispose();
    flare = await Flare.init({ ...config, onProgress: loaded => { progress = loaded; } });
    const loadStart = performance.now();
    await flare.loadModel('/model.gguf');
    report.metrics.loadMs = performance.now() - loadStart;
    assert(progress > 0, 'No model download progress');
    record('worker initialization, failed load recovery, model download');
    const options = { prompt: fixture.prompt, ...fixture.sampling };
    let text = '', chunks = [], times = [];
    const start = performance.now();
    const generated = await flare.generate({ ...options, onToken: (chunk, id) => {
      text += chunk; chunks.push(chunk); if (id >= 0) times.push(performance.now());
    } });
    equal(generated.tokenIds, fixture.tokenIds, 'Pinned regression tokens');
    equal(generated.text, text, 'Streaming/result equality');
    assert(text.startsWith('€') && chunks[0] === '' && chunks[1] === '' && chunks[2] === '€', 'UTF-8 bytes did not stream across token boundaries');
    assert(times.length === 5, 'Missing token events');
    report.metrics.ttftMs = times[0] - start;
    report.metrics.decodeIntervalMs = times.at(-1) - times[0];
    report.metrics.decodeTokensPerSecond = report.metrics.decodeIntervalMs > 0
      ? 1000 * (times.length - 1) / report.metrics.decodeIntervalMs : null;
    record('actual CPU inference and Unicode streaming', { generated, chunks });
    let count = 0;
    await rejects(flare.generate({ ...options, onToken: () => { if (++count === 1) flare.cancel(); } }), 'ABORTED');
    equal((await flare.generate(options)).tokenIds, fixture.tokenIds, 'Generation after cancellation');
    await flare.reset();
    equal((await flare.generate(options)).tokenIds, fixture.tokenIds, 'Generation after reset');
    const pending = flare.generate(options);
    await rejects(flare.generate(options), 'BUSY');
    flare.cancel();
    await rejects(pending, 'ABORTED');
    record('cancel, automatic reload, reset, concurrent request rejection');
    const controller = new AbortController();
    const loading = flare.loadModel('/slow-model.gguf', { signal: controller.signal });
    setTimeout(() => controller.abort(), 50);
    await rejects(loading, 'ABORTED');
    await flare.loadModel('/model.gguf');
    record('download abort and reload');
    // Disable the model origin, then construct a fresh worker. WASM/tokenizer
    // remain online: this verifies model cache reuse, not a fully offline app.
    await fetch('/__model_offline', { method: 'POST' });
    try {
      assert((await fetch('/model.gguf')).status === 503, 'Model origin is still available');
      flare.dispose();
      flare = await Flare.init(config);
      await flare.loadModel('/model.gguf');
      equal((await flare.generate(options)).tokenIds, fixture.tokenIds, 'Cached model generation');
    } finally { await fetch('/__model_online', { method: 'POST' }); }
    record('fresh worker model-cache reuse with model origin unavailable');
    flare.dispose();
    await rejects(flare.generate(options), 'DISPOSED');
    record('disposal');
    const engine = FlareEngine.load(new Uint8Array(await (await fetch('/fixture.gguf')).arrayBuffer()));
    try {
      // The public SDK CPU default must also work when no adapter exists.
      assert(JSON.parse(engine.backend_info()).backend === 'cpu', 'Default backend is not CPU');
      await engine.begin_stream_with_params_async(new Uint32Array(Array(27).fill(2)), 5, 0, 1, 0, 1, 0);
      let steps = 0;
      while (!engine.stream_done) {
        assert(await engine.next_token_async() !== undefined, 'Context boundary ended early');
        assert(engine.last_logits.length === 128 && engine.last_logits.every(Number.isFinite) && engine.last_logits.some(v => Math.abs(v) > 0.01), 'Boundary logits invalid');
        steps++;
      }
      assert(steps === 5, 'Context boundary must decode all five steps');
    } finally { engine.free(); }
    record('CPU context boundary (27 prompt + 5 output = 32)');
    if (gpu) {
      report.coverage.gpu = 'failed';
      assert(navigator.gpu, 'Required WebGPU API unavailable');
      const adapter = await navigator.gpu.requestAdapter();
      assert(adapter, 'Required WebGPU adapter unavailable');
      report.adapter = { info: adapter.info?.toJSON?.() ?? {
        vendor: adapter.info?.vendor, architecture: adapter.info?.architecture,
        device: adapter.info?.device, description: adapter.info?.description },
        features: [...adapter.features], storage: adapter.limits.maxComputeWorkgroupStorageSize };
      for (const mode of ['fixture', 'fixture-f32', 'adapter16', 'device16', 'pipeline', 'dispatch']) {
        const result = await workerRun(mode);
        if (mode === 'fixture') {
          report.coverage.shaderF16 = result.devices.some(device => device.features.includes('shader-f16'))
            ? 'passed' : 'skipped: adapter/device did not enable shader-f16';
          report.coverage.subgroups = result.devices.some(device => device.features.includes('subgroups'))
            ? 'enabled' : 'skipped: device did not enable subgroups';
        }
        record(`GPU ${mode}`, { result });
      }
      report.coverage.gpu = 'passed';
    }
    report.coverage.chromium = 'passed';
    report.passed = true;
  } catch (error) {
    report.error = { message: error.message, stack: error.stack };
    report.coverage.chromium = 'failed';
  } finally {
    flare?.dispose();
    window.ciValidation = report;
    document.querySelector('#result').textContent = JSON.stringify(report, null, 2);
  }
  return report;
};
