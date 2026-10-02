import { Flare } from '@sauravpanda/flare';
import init, { FlareTokenizer } from '@sauravpanda/flare/wasm';
const result = document.querySelector('#result');
let ticks = 0;
setInterval(() => { document.querySelector('#heartbeat').textContent = `UI ticks: ${++ticks}`; }, 50);
const assert = (condition, message) => { if (!condition) throw new Error(message); };
const rejects = async (promise, code) => {
  try { await promise; } catch (e) { assert(e.code === code, `Expected ${code}, got ${e.code}: ${e.message}`); return; }
  throw new Error(`Expected ${code}`);
};
window.run = async (backend = 'cpu', tokenizerUrl = '/tokenizer.json') => {
  const log = [];
  const record = (stage, value) => { log.push({ stage, value }); result.textContent = JSON.stringify(log, null, 2); };
  const before = ticks;
  const flare = await Flare.init({ backend, tokenizerUrl, cache: true, onProgress: (loaded, total) => { window.progress = { loaded, total }; } });
  window.flare = flare;
  try {
    await rejects(Flare.init({ wasmUrl: '/missing.wasm' }), 'INIT');
    record('initialization failure', 'passed');
    await rejects(flare.loadModel('/missing.gguf'), 'LOAD');
    record('load failure', 'passed');
    await flare.loadModel('/model.gguf');
    record('load', { device: flare.deviceInfo, progress: window.progress });
    const chat = await flare.chat({ messages: [{ role: 'system', content: 'Answer briefly.' }, { role: 'user', content: 'Hello!' }], maxTokens: 8 });
    record('real chat (quality not asserted)', chat);
    let streamed = '';
    const first = await flare.generate({ prompt: 'Hello', maxTokens: 8, onToken: text => { streamed += text; } });
    assert(first.tokenIds.length > 0 && first.text.length > 0, 'No real model output');
    assert(first.text === streamed, 'Stream/result mismatch');
    record('real output (quality not asserted)', first);
    let tokenCount = 0;
    await rejects(flare.generate({ prompt: 'Hello', maxTokens: 128, onToken: () => { if (++tokenCount === 1) flare.cancel(); } }), 'ABORTED');
    record('decode cancel', { tokenCount });
    const next = await flare.generate({ prompt: 'Hello', maxTokens: 3 });
    assert(next.tokenIds.length > 0, 'Cannot generate after cancel');
    record('reload and another request', next);
    await flare.reset();
    // A long prompt keeps synchronous WASM prefill busy while the main thread cancels.
    const pending = flare.generate({ prompt: 'Hello '.repeat(128), maxTokens: 3 });
    await rejects(flare.generate({ prompt: 'concurrent' }), 'BUSY');
    const start = performance.now();
    setTimeout(() => flare.cancel(), 20);
    await rejects(pending, 'ABORTED');
    assert(performance.now() - start < 2000, 'Cancellation waited for prefill');
    record('prefill cancellation and concurrency', { ms: performance.now() - start });
    const controller = new AbortController();
    const download = flare.loadModel('/slow-model.gguf', { signal: controller.signal });
    setTimeout(() => controller.abort(), 200);
    await rejects(download, 'ABORTED');
    record('download cancellation', 'passed');
    await flare.loadModel('/model.gguf');
    const final = await flare.generate({ prompt: 'Hello', maxTokens: 2 });
    record('after download cancellation', final);
    flare.dispose();
    await rejects(flare.generate({ prompt: 'disposed' }), 'DISPOSED');
    record('dispose', 'passed');
    const probe = await new Promise((resolve, reject) => {
      const worker = new Worker('./probe-worker.mjs', { type: 'module' });
      const timer = setTimeout(() => { worker.terminate(); reject(new Error('Worker cache probe timed out')); }, 10000);
      worker.onmessage = ({ data }) => { clearTimeout(timer); worker.terminate(); resolve(data); };
      worker.onerror = event => { clearTimeout(timer); worker.terminate(); reject(new Error(event.message)); };
    });
    assert(!probe.error && String(probe.bytes) === '1,2,3', `Worker OPFS failed: ${probe.error}`);
    assert(probe.device.userAgent && probe.fetchError.includes('404'), 'Worker capability/fetch helpers failed');
    record('worker OPFS, capabilities, and progressive fetch', probe);
    await init();
    const tokenizer = FlareTokenizer.from_json(JSON.stringify({ model: { vocab: { 'â': 0, 'Ĥ': 1, '¬': 2 }, merges: [] } }));
    try {
      const decoder = new TextDecoder();
      const parts = [0, 1, 2].map(id => decoder.decode(tokenizer.decode_bytes(new Uint32Array([id])), { stream: true }));
      assert(parts.join('') + decoder.decode() === '€', 'UTF-8 boundary corrupted');
      record('packaged Unicode binding', parts);
    } finally { tokenizer.free(); }
    assert(ticks > before + 5, 'Main thread did not stay responsive');
    record('responsive UI', { ticks: ticks - before });
    window.validation = { passed: true, backend, log };
  } catch (error) {
    record('failure', { code: error.code, message: error.message, stack: error.stack });
    window.validation = { passed: false, backend, log };
  } finally { flare.dispose(); }
  return window.validation;
};
