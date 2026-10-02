import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
// Run the compiled worker protocol against a controlled engine. Real bindings
// and model inference are covered separately by the installed browser consumer.
const source = (await readFile(new URL('../dist/worker.js', import.meta.url), 'utf8')).replace(/^import .*;$/m, '');
function worker(overrides = {}) {
  const messages = [], calls = [];
  let freed = 0, reset = 0, cursor = 0;
  const engine = {
    free() { freed++; }, reset() { reset++; cursor = 0; },
    metadata_json: '{}', max_seq_len: 100, vocab_size: 10,
    init_gpu: async () => true, apply_chat_messages: json => JSON.parse(json).map(m => m.content).join(' '),
    apply_chat_template: (user, system) => `${system} ${user}`,
    encode_text: text => { calls.push(['encode', text]); return new Uint32Array([3]); },
    set_rng_seed: seed => calls.push(['seed', seed]),
    begin_stream_with_params_async: async (...args) => calls.push(['prefill', ...args]),
    next_token_async: async () => cursor++,
    get stream_done() { return cursor === 3; },
    decode_token_chunk: id => ['a', 'b', 'c'][id], flush_decode: () => '', stream_stop_reason: 'length',
    ...overrides,
  };
  const self = { postMessage: msg => messages.push(msg) };
  const context = {
    self, TextDecoder, Uint8Array, Uint32Array, Response, JSON,
    init: async () => {}, device_info: () => '{"webgpu":true}',
    FlareEngine: { load: () => engine },
    FlareTokenizer: { from_json: () => ({ free() {}, encode: () => new Uint32Array([2]), decode_bytes: ids => new Uint8Array([[0xe2], [0x82], [0xac]][ids[0]]) }) },
    fetch: async () => new Response(new Uint8Array([1, 2])),
    caches: { open: async () => ({ match: async () => undefined, put: async () => {} }) },
  };
  vm.runInNewContext(source, context);
  let id = 0;
  return { messages, calls, engine, context, freed: () => freed, reset: () => reset,
    send: (type, args = {}) => self.onmessage({ data: { id: ++id, type, args } }) };
}
test('worker streams async engine output and forwards all sampling parameters', async () => {
  const w = worker(); await w.send('init'); await w.send('load', { modelUrl: '/model', cache: true });
  await w.send('generate', { prompt: 'hi', maxTokens: 3, temperature: .7, topP: .8, topK: 4, repeatPenalty: 1.2, minP: .1, seed: 7 });
  assert.deepEqual(w.messages.filter(m => m.type === 'token').map(m => m.text), ['a','b','c']);
  assert.deepEqual(w.calls.find(c => c[0] === 'prefill').slice(2), [3,.7,.8,4,1.2,.1]);
  assert.equal(w.messages.at(-1).value.text, 'abc');
  assert.equal(w.reset(), 1);
});
test('original tokenizer bytes stream a complete Unicode character', async () => {
  const w = worker(); await w.send('init'); await w.send('load', { modelUrl: '/model', tokenizerUrl: '/tok' });
  await w.send('generate', { messages: [{ role: 'user', content: 'hello' }], maxTokens: 3 });
  assert.deepEqual(w.messages.filter(m => m.type === 'token').map(m => m.text), ['', '', '€']);
  assert.equal(w.messages.at(-1).value.text, '€');
});
test('worker rejects overlapping requests while async prefill holds an engine borrow', async () => {
  let release;
  const w = worker({ begin_stream_with_params_async: () => new Promise(resolve => { release = resolve; }) });
  await w.send('init'); await w.send('load', { modelUrl: '/model' });
  const generation = w.send('generate', { prompt: 'x', maxTokens: 3 });
  await w.send('reset');
  assert.equal(w.messages.at(-1).code, 'BUSY');
  assert.equal(w.reset(), 1); release(); await generation;
});
test('load failures free resources and require another successful load', async () => {
  const w = worker({ init_gpu: async () => false });
  await w.send('init'); await w.send('load', { modelUrl: '/model', backend: 'webgpu' });
  assert.equal(w.messages.at(-1).code, 'LOAD'); assert.equal(w.freed(), 1);
  await w.send('generate', { prompt: 'x' }); assert.equal(w.messages.at(-1).code, 'GENERATE');
});
test('BPE GGUF requires an explicit original tokenizer rather than returning encoded markers', async () => {
  const w = worker({ metadata_json: '{"tokenizer.ggml.model":"gpt2"}' });
  await w.send('init'); await w.send('load', { modelUrl: '/model' });
  assert.match(w.messages.at(-1).message, /tokenizerUrl/);
});
test('invalid sampling and context limits reject before prefill', async () => {
  const w = worker(); await w.send('init'); await w.send('load', { modelUrl: '/model' });
  for (const options of [{ maxTokens: -1 }, { maxTokens: 100 }, { temperature: NaN }, { topP: 0 }, { topK: -1 }, { minP: 2 }, { repeatPenalty: 0 }, { seed: -1 }]) {
    await w.send('generate', { prompt: 'x', maxTokens: 3, ...options });
    assert.equal(w.messages.at(-1).code, 'GENERATE');
  }
  assert.equal(w.calls.some(c => c[0] === 'prefill'), false);
});
