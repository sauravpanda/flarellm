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
 * The Flare inference engine.
 * Load a GGUF model, then use the streaming API (begin_stream / next_token)
 * or the batch API (generate_tokens) to run inference.
 */
export declare class FlareEngine {
  /** Load a GGUF model from raw bytes. */
  static load(ggufBytes: Uint8Array): FlareEngine;
  /** Reset the KV cache (start a new conversation). */
  reset(): void;
  /** Vocabulary size. */
  readonly vocab_size: number;
  /** Number of transformer layers. */
  readonly num_layers: number;
  /** Hidden dimension. */
  readonly hidden_dim: number;
  /**
   * Auto-detected chat template name: "ChatML", "Llama3", "Alpaca", or "Raw".
   * Detected from the GGUF tokenizer.chat_template metadata, falling back to
   * architecture-based detection.
   */
  readonly chat_template_name: string;
  /**
   * EOS token ID read from the GGUF model metadata, if present.
   * The generator stops automatically when this token is produced.
   */
  readonly eos_token_id: number | undefined;
  /**
   * Format a user message and optional system prompt using the model's chat
   * template.  Pass the result to FlareTokenizer.encode() before generating.
   * Pass an empty string for systemMessage to omit the system turn.
   */
  apply_chat_template(userMessage: string, systemMessage: string): string;

  // --- Token-by-token streaming API ---

  /**
   * Prepare for token-by-token streaming.  Runs the prefill pass on
   * promptTokens and initialises internal streaming state.  Call engine.reset()
   * first to start a fresh conversation, then call next_token() in a
   * requestAnimationFrame loop.
   */
  begin_stream(promptTokens: Uint32Array, maxTokens: number): void;
  /**
   * Generate the next token and return its ID, or undefined when the stream is
   * complete (EOS reached, maxTokens exhausted, or stop_stream() was called).
   * Call inside requestAnimationFrame so the browser can update the DOM between
   * tokens.
   */
  next_token(): number | undefined;
  /** Signal the current stream to stop after the next next_token() call. */
  stop_stream(): void;
  /** Whether the current stream has finished. */
  readonly stream_done: boolean;

  // --- Batch generation API (returns all tokens at once) ---

  /** Generate tokens (greedy, temperature=0). Stops at EOS. Returns token ID array. */
  generate_tokens(promptTokens: Uint32Array, maxTokens: number): Uint32Array;
  /** Generate tokens with sampling parameters. Stops at EOS. */
  generate_with_params(
    promptTokens: Uint32Array,
    maxTokens: number,
    temperature: number,
    topP: number
  ): Uint32Array;
}

/**
 * Progressive loader: fetches a GGUF model from a URL with streaming download
 * progress, then parses and returns a FlareEngine.
 */
export declare class FlareProgressiveLoader {
  constructor(url: string);
  /** Fetch, stream, and parse the model. Calls onProgress as chunks arrive. */
  load(onProgress: ProgressCallback): Promise<FlareEngine>;
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
