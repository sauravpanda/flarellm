import init, { FlareEngine, FlareTokenizer, device_info } from '../pkg/flare_web.js';
import type { WorkerRequest, WorkerResponse } from './protocol.js';

let engine: FlareEngine | undefined;
let tokenizer: FlareTokenizer | undefined;
let busy = false;
let initialized = false;
const post = (message: WorkerResponse) => self.postMessage(message);
self.onmessage = async ({ data: { id, type, args } }: MessageEvent<WorkerRequest>) => {
  if (busy) { post({ id, type: 'error', code: 'BUSY', message: 'Worker is busy' }); return; }
  busy = true;
  try {
    let value: unknown;
    if (type === 'init') {
      await init(args.wasmUrl ? { module_or_path: args.wasmUrl } : undefined);
      initialized = true;
      value = JSON.parse(device_info());
    } else {
      if (!initialized) throw new Error('Worker is not initialized');
      if (type === 'load') {
        engine?.free(); engine = undefined;
        tokenizer?.free(); tokenizer = undefined;
        const cache = args.cache ? await caches.open('flare-models-v1') : undefined;
        let response = await cache?.match(args.modelUrl);
        if (!response) {
          response = await fetch(args.modelUrl);
          if (!response.ok) throw new Error(`Model HTTP ${response.status}`);
        }
        const total = Number(response.headers.get('content-length') || 0);
        const reader = response.body!.getReader();
        const chunks: Uint8Array[] = [];
        let loaded = 0;
        try {
          while (true) {
            const part = await reader.read(); if (part.done) break;
            chunks.push(part.value); loaded += part.value.length;
            post({ id, type: 'progress', loaded, total });
          }
        } finally { reader.releaseLock(); }
        const bytes = new Uint8Array(loaded);
        let offset = 0;
        for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
        chunks.length = 0;
        engine = FlareEngine.load(bytes);
        if (args.tokenizerUrl) {
          const response = await fetch(args.tokenizerUrl);
          if (!response.ok) throw new Error(`Tokenizer HTTP ${response.status}`);
          tokenizer = FlareTokenizer.from_json(await response.text());
        }
        if (!tokenizer && JSON.parse(engine.metadata_json)['tokenizer.ggml.model'] === 'gpt2') {
          throw new Error('This GGUF uses byte-level BPE; provide its original tokenizerUrl');
        }
        if (args.backend !== undefined && args.backend !== 'cpu' && args.backend !== 'webgpu') throw new Error('Unknown backend');
        if (args.backend === 'webgpu' && !await engine.init_gpu()) {
          const architecture = JSON.parse(engine.metadata_json)['general.architecture'];
          throw new Error(architecture === 'qwen3'
            ? 'Qwen3 WebGPU execution is unsupported; select backend: cpu for Q/K normalization'
            : 'WebGPU initialization failed');
        }
        // Cache only successfully parsed models. Cache failures surface to callers.
        if (cache) await cache.put(args.modelUrl, new Response(bytes));
      } else if (type === 'reset') { engine?.reset(); }
      else if (type === 'generate') {
        if (!engine) throw new Error('No model loaded');
        const maxTokens = args.maxTokens ?? 128;
        const temperature = args.temperature ?? 0;
        const topP = args.topP ?? 1;
        const topK = args.topK ?? 0;
        const minP = args.minP ?? 0;
        const repeatPenalty = args.repeatPenalty ?? 1;
        if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > engine.max_seq_len ||
            !Number.isFinite(temperature) || temperature < 0 ||
            !Number.isFinite(topP) || topP <= 0 || topP > 1 ||
            !Number.isInteger(topK) || topK < 0 || topK > engine.vocab_size ||
            !Number.isFinite(minP) || minP < 0 || minP > 1 ||
            !Number.isFinite(repeatPenalty) || repeatPenalty <= 0) throw new Error('Invalid sampling options');
        engine.reset();
        if (args.seed !== undefined && (!Number.isInteger(args.seed) || args.seed < 0 || args.seed > 0xffffffff)) throw new Error('Invalid seed');
        engine.set_rng_seed(args.seed ?? 0x12345678);
        const prompt = 'messages' in args ? engine.apply_chat_messages(JSON.stringify(args.messages))
          : 'message' in args ? engine.apply_chat_template(args.message ?? '', args.system ?? '') : args.prompt;
        if (typeof prompt !== 'string') throw new Error('Provide prompt, message, or messages');
        let tokens = tokenizer ? tokenizer.encode(prompt) : engine.encode_text(prompt);
        if (engine.add_bos_token && engine.bos_token_id !== undefined && tokens[0] !== engine.bos_token_id) {
          tokens = new Uint32Array([engine.bos_token_id, ...tokens]);
        }
        if (!tokens.length || tokens.length + maxTokens > engine.max_seq_len) throw new Error('Prompt and output exceed context or prompt is empty');
        await engine.begin_stream_with_params_async(tokens, maxTokens, temperature, topP, topK, repeatPenalty, minP);
        const decoder = new TextDecoder();
        const tokenIds: number[] = [];
        let text = '';
        while (!engine.stream_done) {
          const tokenId = await engine.next_token_async();
          if (tokenId === undefined) break;
          tokenIds.push(tokenId);
          const chunk = tokenizer ? decoder.decode(tokenizer.decode_bytes(new Uint32Array([tokenId])), { stream: true }) : engine.decode_token_chunk(tokenId);
          text += chunk; post({ id, type: 'token', text: chunk, tokenId });
        }
        const tail = tokenizer ? decoder.decode() : engine.flush_decode();
        if (tail) { text += tail; post({ id, type: 'token', text: tail, tokenId: -1 }); }
        value = { text, tokenIds, stopReason: engine.stream_stop_reason };
      } else throw new Error(`Unknown operation: ${type}`);
    }
    post({ id, type: 'result', value });
  } catch (error) {
    // A WASM trap may leave a borrowed/invalid object. Cleanup must not hide the error.
    if (type === 'load') {
      try { engine?.free(); tokenizer?.free(); } catch { /* Worker termination remains available. */ }
      engine = undefined; tokenizer = undefined;
    }
    post({ id, type: 'error', code: type.toUpperCase(), message: error instanceof Error ? error.message : String(error) });
  } finally { busy = false; }
};
