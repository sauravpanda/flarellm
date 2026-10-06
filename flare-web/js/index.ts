/** Browser SDK. Low-level generated bindings remain available at /wasm. */
import type { WorkerRequest, WorkerResponse } from './protocol.js';

export * from '../pkg/flare_web.js';
export { default } from '../pkg/flare_web.js';
export interface FlareConfig {
  modelUrl?: string;
  wasmUrl?: string;
  /** Optional original Hugging Face tokenizer.json; otherwise uses embedded GGUF vocabulary. */
  tokenizerUrl?: string;
  backend?: 'cpu' | 'webgpu';
  cache?: boolean;
  onProgress?: (loaded: number, total: number) => void;
}
export interface GenerateOptions {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  minP?: number;
  repeatPenalty?: number;
  seed?: number;
  signal?: AbortSignal;
  onToken?: (text: string, tokenId: number) => void;
}
export interface ChatMessage { role: 'system' | 'user' | 'assistant'; content: string }
export interface ChatOptions extends Omit<GenerateOptions, 'prompt'> {
  messages?: ChatMessage[];
  message?: string;
  /** Kept for source compatibility; onToken controls streaming delivery. */
  stream?: boolean;
  system?: string;
}
export interface GenerationResult { text: string; tokenIds: number[]; stopReason: string }
/** Experimental text-only Qwen3 CPU decision; 2..8 unique trimmed choices. */
export interface DecisionOptions<C extends string = string> {
  state: string;
  question: string;
  choices: readonly C[];
  signal?: AbortSignal;
}
export interface DecisionResult<C extends string = string> {
  choice: C;
  /** Original choice index; exact logit ties select the first. */
  index: number;
  /** Original order. Relative candidate scores, NOT calibrated confidence. */
  scores: { choice: C; label: string; tokenId: number; logit: number; score: number }[];
  promptVersion: 'qwen3-choice-v1';
  prompt: string;
  promptIds: number[];
  /** Worker wall time for validation, tokenization and one CPU prefill. */
  decisionMs: number;
}
export class FlareError extends Error {
  constructor(public code: string, message: string) { super(message); this.name = 'FlareError'; }
}
interface Pending {
  id: number;
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
  onToken?: GenerateOptions['onToken'];
}
export class Flare {
  private worker?: Worker;
  private pending?: Pending;
  private sequence = 0;
  private busy = false;
  private interruption?: { error: unknown };
  private disposed = false;
  private loaded = false;
  private config: FlareConfig;
  private info: Record<string, unknown> = {};
  private constructor(config: FlareConfig) {
    this.config = { ...config };
    for (const key of ['modelUrl', 'wasmUrl', 'tokenizerUrl'] as const) {
      if (this.config[key]) this.config[key] = new URL(this.config[key], globalThis.location.href).href;
    }
  }
  static async init(config: FlareConfig = {}): Promise<Flare> {
    const flare = new Flare(config);
    try {
      await flare.exclusive(async () => { await flare.connect(); if (config.modelUrl) await flare.load(); });
      return flare;
    } catch (error) { flare.dispose(); throw error; }
  }
  get webgpuAvailable(): boolean { return this.info.webgpu === true; }
  get deviceInfo(): Record<string, unknown> { return { ...this.info }; }
  private async exclusive<T>(run: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    if (this.disposed) throw new FlareError('DISPOSED', 'Engine is disposed');
    if (this.busy) throw new FlareError('BUSY', 'Only one operation may run at a time');
    if (signal?.aborted) throw new FlareError('ABORTED', 'Operation cancelled');
    this.busy = true;
    this.interruption = undefined;
    const abort = () => this.cancel();
    signal?.addEventListener('abort', abort, { once: true });
    try {
      const result = await run();
      this.checkInterrupted();
      return result;
    }
    finally { signal?.removeEventListener('abort', abort); this.busy = false; }
  }
  private checkInterrupted(): void {
    if (this.interruption) throw this.interruption.error;
  }
  private async connect(): Promise<void> {
    this.checkInterrupted();
    if (this.worker) return;
    const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    this.worker = worker;
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (this.worker !== worker || data.id !== this.pending?.id) return;
      const pending = this.pending!;
      try {
        if (data.type === 'progress') this.config.onProgress?.(data.loaded, data.total);
        else if (data.type === 'token') pending.onToken?.(data.text, data.tokenId);
        else {
          this.pending = undefined;
          if (data.type === 'error') {
            const error = new FlareError(data.code, data.message);
            this.destroy(error);
            pending.reject(error);
          }
          else pending.resolve(data.value);
        }
      } catch (error) { this.destroy(error); }
    };
    worker.onerror = event => { event.preventDefault(); if (this.worker === worker) this.destroy(new FlareError('WORKER', event.message)); };
    worker.onmessageerror = () => { if (this.worker === worker) this.destroy(new FlareError('PROTOCOL', 'Cannot deserialize worker response')); };
    this.info = await this.request<Record<string, unknown>>('init', { wasmUrl: this.config.wasmUrl });
  }
  private request<T = void>(type: WorkerRequest['type'], args: WorkerRequest['args'] = {}, onToken?: GenerateOptions['onToken']): Promise<T> {
    if (this.interruption) return Promise.reject(this.interruption.error);
    return new Promise((resolve, reject) => {
      const id = ++this.sequence;
      this.pending = { id, resolve: value => resolve(value as T), reject, onToken };
      try { this.worker!.postMessage({ id, type, args }); }
      catch (error) { this.destroy(error); }
    });
  }
  private async load(): Promise<void> {
    if (!this.config.modelUrl) throw new FlareError('NO_MODEL', 'Call loadModel first');
    await this.request('load', { modelUrl: this.config.modelUrl, cache: this.config.cache, backend: this.config.backend, tokenizerUrl: this.config.tokenizerUrl });
    this.checkInterrupted();
    this.loaded = true;
  }
  async loadModel(modelUrl: string, options: { signal?: AbortSignal } = {}): Promise<void> {
    return this.exclusive(async () => {
      this.config.modelUrl = new URL(modelUrl, globalThis.location.href).href;
      this.loaded = false;
      await this.connect(); await this.load();
    }, options.signal);
  }
  async generate(options: GenerateOptions): Promise<GenerationResult> { return this.generateRequest(options); }
  async chat(options: ChatOptions): Promise<GenerationResult> { return this.generateRequest(options); }
  private async generateRequest(options: GenerateOptions | ChatOptions): Promise<GenerationResult> {
    return this.exclusive(async () => {
      const { signal, onToken, ...args } = options;
      await this.connect();
      if (!this.loaded) await this.load();
      return this.request<GenerationResult>('generate', args, onToken);
    }, options.signal);
  }
  async decide<const C extends string>(options: DecisionOptions<C>): Promise<DecisionResult<C>> {
    return this.exclusive(async () => {
      const { signal, ...args } = options;
      await this.connect();
      if (!this.loaded) await this.load();
      return this.request<DecisionResult<C>>('decide', args);
    }, options.signal);
  }
  async reset(): Promise<void> {
    return this.exclusive(async () => { if (this.worker) await this.request('reset'); });
  }
  /** Immediately terminate computation, including synchronous WASM prefill.
   * The next generation reloads the configured model in a fresh worker. */
  cancel(): void { this.destroy(new FlareError('ABORTED', 'Operation cancelled; model will reload on next request')); }
  dispose(): void { this.disposed = true; this.destroy(new FlareError('DISPOSED', 'Engine is disposed')); }
  private destroy(error: unknown): void {
    if (this.busy) this.interruption ??= { error };
    this.worker?.terminate(); this.worker = undefined; this.loaded = false;
    const pending = this.pending; this.pending = undefined; pending?.reject(error);
  }
}
