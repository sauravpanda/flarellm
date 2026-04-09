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
