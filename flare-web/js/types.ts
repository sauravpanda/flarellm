/**
 * TypeScript type definitions for the Flare WASM module.
 * These map to the wasm-bindgen exports from flare-web.
 */

/** Check if WebGPU is available in this browser. */
export declare function webgpu_available(): boolean;

/** Get JSON string with device capabilities. */
export declare function device_info(): string;

/** Initialize the Flare engine (async). */
export declare function init(): Promise<string>;

/** Progress callback: (loaded_bytes, total_bytes). total_bytes is 0 when unknown. */
export type ProgressCallback = (loaded: number, total: number) => void;

/**
 * Progressive loader: fetches a GGUF model from a URL with streaming download
 * progress, then parses and returns a FlareEngine.
 */
export declare class FlareProgressiveLoader {
  constructor(url: string);
  /** Fetch, stream, and parse the model. Calls onProgress as chunks arrive. */
  load(onProgress: ProgressCallback): Promise<import('../pkg/flare_web').FlareEngine>;
}

/**
 * BPE tokenizer: encode text to token IDs and decode token IDs back to text.
 * Load from a HuggingFace tokenizer.json string.
 */
export declare class FlareTokenizer {
  /** Load from a tokenizer.json string. */
  static from_json(json: string): FlareTokenizer;
  /** Encode text to token IDs. */
  encode(text: string): Uint32Array;
  /** Decode token IDs to text. */
  decode(tokens: Uint32Array): string;
  /** Decode a single token ID to text (for streaming). */
  decode_one(tokenId: number): string;
  /** BOS token ID (may be undefined). */
  readonly bos_token_id: number | undefined;
  /** EOS token ID (may be undefined). */
  readonly eos_token_id: number | undefined;
  /** Vocabulary size. */
  readonly vocab_size: number;
}
