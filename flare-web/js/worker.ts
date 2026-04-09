/**
 * Web Worker bootstrap for running Flare inference off the main thread.
 *
 * Architecture:
 * - Main thread sends messages: { type: 'init' | 'generate', ... }
 * - Worker runs WASM inference and posts back tokens as they're generated
 * - All GPU operations happen in the worker (WebGPU is available in workers)
 *
 * Usage from main thread:
 * ```typescript
 * const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
 * worker.postMessage({ type: 'init' });
 * worker.postMessage({ type: 'generate', prompt: 'Hello', maxTokens: 128 });
 * worker.onmessage = (e) => {
 *   if (e.data.type === 'token') console.log(e.data.text);
 *   if (e.data.type === 'done') console.log('Generation complete');
 * };
 * ```
 */

// Message types from main thread to worker
interface InitMessage {
  type: 'init';
  modelUrl?: string;
  wasmUrl?: string;
}

interface GenerateMessage {
  type: 'generate';
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
}

interface AbortMessage {
  type: 'abort';
}

type IncomingMessage = InitMessage | GenerateMessage | AbortMessage;

// Message types from worker to main thread
interface ReadyMessage {
  type: 'ready';
  webgpu: boolean;
}

interface TokenMessage {
  type: 'token';
  text: string;
  tokenId: number;
}

interface DoneMessage {
  type: 'done';
  totalTokens: number;
  tokensPerSecond: number;
}

interface ErrorMessage {
  type: 'error';
  message: string;
}

interface ProgressMessage {
  type: 'progress';
  loaded: number;
  total: number;
}

type OutgoingMessage = ReadyMessage | TokenMessage | DoneMessage | ErrorMessage | ProgressMessage;

// Worker state
let initialized = false;
let aborted = false;

function postResult(msg: OutgoingMessage) {
  (self as unknown as { postMessage(msg: OutgoingMessage): void }).postMessage(msg);
}

async function handleInit(_msg: InitMessage) {
  try {
    // Import and initialize WASM module
    // In a real build, this path comes from wasm-pack output
    const flare = await import('../pkg/flare_web.js');
    await flare.init();

    const webgpu = flare.webgpu_available();
    initialized = true;

    postResult({ type: 'ready', webgpu });
  } catch (err) {
    postResult({
      type: 'error',
      message: `Init failed: ${err}`,
    });
  }
}

async function handleGenerate(msg: GenerateMessage) {
  if (!initialized) {
    postResult({ type: 'error', message: 'Worker not initialized. Send init message first.' });
    return;
  }

  aborted = false;
  const startTime = performance.now();
  let tokenCount = 0;

  try {
    // TODO: Wire up actual WASM inference here
    // For now, simulate token generation to validate the worker protocol
    const maxTokens = msg.maxTokens ?? 128;

    for (let i = 0; i < maxTokens && !aborted; i++) {
      // In real implementation: call flare WASM generate step
      tokenCount++;

      postResult({
        type: 'token',
        text: ' ',
        tokenId: i,
      });

      // Yield to allow abort messages to be processed
      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    const elapsed = (performance.now() - startTime) / 1000;
    postResult({
      type: 'done',
      totalTokens: tokenCount,
      tokensPerSecond: tokenCount / elapsed,
    });
  } catch (err) {
    postResult({
      type: 'error',
      message: `Generation failed: ${err}`,
    });
  }
}

// Message handler
self.onmessage = async (event: MessageEvent<IncomingMessage>) => {
  const msg = event.data;

  switch (msg.type) {
    case 'init':
      await handleInit(msg);
      break;
    case 'generate':
      await handleGenerate(msg);
      break;
    case 'abort':
      aborted = true;
      break;
  }
};
