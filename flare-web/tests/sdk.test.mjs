import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { Flare } from '../dist/index.js';
const workers = [];
globalThis.location = new URL('https://example.test/app/');
class FakeWorker {
  static failInit = false;
  static hold = '';
  static initDelay = false;
  constructor(url, options) {
    assert.ok(url.pathname.endsWith('/dist/worker.js'));
    assert.equal(options.type, 'module');
    workers.push(this); this.messages = [];
  }
  postMessage(message) {
    this.messages.push(message);
    if (message.type === FakeWorker.hold) return;
    queueMicrotask(() => this.emit(message.id, FakeWorker.failInit && message.type === 'init'
      ? { type: 'error', code: 'INIT', message: 'bad wasm' }
      : { type: 'result', value: message.type === 'init' ? { webgpu: true } : { text: 'hi', tokenIds: [1], stopReason: 'max_tokens' } }));
  }
  emit(id, data) { this.onmessage({ data: { id, ...data } }); }
  terminate() { this.terminated = true; }
}
globalThis.Worker = FakeWorker;
beforeEach(() => { workers.length = 0; FakeWorker.hold = ''; FakeWorker.failInit = false; });
test('initialization failure terminates worker and returns a structured error', async () => {
  FakeWorker.failInit = true;
  await assert.rejects(Flare.init(), { code: 'INIT' });
  assert.equal(workers[0].terminated, true);
});
test('public config URLs resolve relative to the consumer and capabilities are copied', async () => {
  const f = await Flare.init({ modelUrl: 'model.gguf' });
  assert.equal(workers[0].messages[1].args.modelUrl, 'https://example.test/app/model.gguf');
  assert.equal(f.webgpuAvailable, true);
  f.deviceInfo.webgpu = false;
  assert.equal(f.webgpuAvailable, true); f.dispose();
});
for (const stage of ['load', 'generate']) test(`cancel during ${stage}, ignore stale replies, reload and generate again`, async () => {
  const f = await Flare.init();
  if (stage === 'generate') await f.loadModel('/model.gguf');
  FakeWorker.hold = stage;
  const controller = new AbortController();
  const pending = stage === 'load' ? f.loadModel('/model.gguf', { signal: controller.signal }) : f.generate({ prompt: 'hello', signal: controller.signal });
  await new Promise(resolve => setImmediate(resolve));
  await assert.rejects(f.reset(), { code: 'BUSY' });
  controller.abort();
  await assert.rejects(pending, { code: 'ABORTED' });
  const old = workers[0]; assert.equal(old.terminated, true);
  FakeWorker.hold = '';
  const result = await f.generate({ prompt: 'again' });
  old.emit(999, { type: 'error', code: 'STALE' });
  assert.equal(result.text, 'hi');
  assert.deepEqual(workers[1].messages.map(m => m.type), ['init', 'load', 'generate']);
  await f.reset(); f.dispose(); f.dispose();
  await assert.rejects(f.generate({ prompt: 'no' }), { code: 'DISPOSED' });
});
test('request correlation, streaming callback, worker failure and recovery', async () => {
  const f = await Flare.init({ modelUrl: '/model.gguf' });
  FakeWorker.hold = 'generate';
  const chunks = [];
  const promise = f.generate({ prompt: 'hello', onToken: text => chunks.push(text) });
  await new Promise(resolve => setImmediate(resolve));
  const w = workers[0], id = w.messages.at(-1).id;
  w.emit(id - 1, { type: 'result', value: 'stale' });
  w.emit(id, { type: 'token', text: '世界', tokenId: 5 });
  assert.deepEqual(chunks, ['世界']);
  w.onerror({ preventDefault() {}, message: 'crash' });
  await assert.rejects(promise, { code: 'WORKER' });
  FakeWorker.hold = '';
  await f.generate({ prompt: 'again' }); f.dispose();
});
test('dispose rejects an active request; pre-aborted signals do not dispatch', async () => {
  const f = await Flare.init({ modelUrl: '/model.gguf' });
  const count = workers[0].messages.length;
  await assert.rejects(f.generate({ prompt: 'x', signal: AbortSignal.abort() }), { code: 'ABORTED' });
  assert.equal(workers[0].messages.length, count);
  FakeWorker.hold = 'generate';
  const p = f.generate({ prompt: 'x' });
  await new Promise(resolve => setImmediate(resolve)); f.dispose();
  await assert.rejects(p, { code: 'DISPOSED' });
});
test('consumer callback failure rejects instead of stranding a request', async () => {
  const f = await Flare.init({ modelUrl: '/model.gguf' });
  FakeWorker.hold = 'generate';
  const p = f.generate({ prompt: 'x', onToken() { throw new Error('callback'); } });
  await new Promise(resolve => setImmediate(resolve));
  workers[0].emit(workers[0].messages.at(-1).id, { type: 'token', text: 'x' });
  await assert.rejects(p, /callback/); assert.equal(workers[0].terminated, true); f.dispose();
});
test('abort during worker reinitialization terminates it before reload', async () => {
  const f = await Flare.init({ modelUrl: '/model.gguf' });
  f.cancel(); FakeWorker.hold = 'init';
  const controller = new AbortController();
  const p = f.generate({ prompt: 'x', signal: controller.signal });
  controller.abort(); await assert.rejects(p, { code: 'ABORTED' });
  assert.equal(workers[1].terminated, true);
  assert.deepEqual(workers[1].messages.map(m => m.type), ['init']); f.dispose();
});
test('message deserialization failure rejects active work', async () => {
  const f = await Flare.init({ modelUrl: '/model.gguf' });
  FakeWorker.hold = 'generate';
  const p = f.generate({ prompt: 'x' });
  await new Promise(resolve => setImmediate(resolve));
  workers[0].onmessageerror(); await assert.rejects(p, { code: 'PROTOCOL' });
  assert.equal(workers[0].terminated, true); f.dispose();
});
test('immediate abort between awaited lifecycle steps preserves ABORTED and dispatches no generation', async () => {
  const f = await Flare.init({ modelUrl: '/model.gguf' });
  const controller = new AbortController();
  const p = f.generate({ prompt: 'x', signal: controller.signal });
  controller.abort(); await assert.rejects(p, { code: 'ABORTED' });
  assert.deepEqual(workers[0].messages.map(m => m.type), ['init', 'load']);
  assert.equal((await f.generate({ prompt: 'retry' })).text, 'hi'); f.dispose();
});
