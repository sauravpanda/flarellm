import type { ChatOptions, FlareConfig, GenerateOptions, DecisionOptions } from './index.js';

export type GenerateArgs = Omit<GenerateOptions, 'signal' | 'onToken' | 'prompt'> &
  Partial<Pick<GenerateOptions, 'prompt'>> & Pick<ChatOptions, 'message' | 'messages' | 'system'>;
export type WorkerRequest = { id: number } & (
  | { type: 'init'; args: Pick<FlareConfig, 'wasmUrl'> }
  | { type: 'load'; args: Pick<FlareConfig, 'cache' | 'backend' | 'tokenizerUrl'> & { modelUrl: string } }
  | { type: 'generate'; args: GenerateArgs }
  | { type: 'decide'; args: Omit<DecisionOptions, 'signal'> }
  | { type: 'reset'; args: Record<string, never> }
);
export type WorkerResponse = { id: number } & (
  | { type: 'progress'; loaded: number; total: number }
  | { type: 'token'; text: string; tokenId: number }
  | { type: 'result'; value: unknown }
  | { type: 'error'; code: string; message: string }
);
