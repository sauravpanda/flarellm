import init, { Flare, FlareError, FlareEngine, type ChatMessage } from '@sauravpanda/flare';
import { FlareTokenizer } from '@sauravpanda/flare/wasm';
import '@sauravpanda/flare/worker';

const messages: ChatMessage[] = [{ role: 'user', content: 'Hello' }];
async function consume() {
  const engine = await Flare.init({ tokenizerUrl: '/tokenizer.json' });
  await engine.loadModel('/model.gguf', { signal: new AbortController().signal });
  const { text, tokenIds } = await engine.chat({ messages, onToken: (text, id) => console.log(text, id) });
  const result: string = text;
  const ids: number[] = tokenIds;
  engine.cancel(); engine.dispose();
  return { result, ids, init, FlareError, FlareEngine, FlareTokenizer };
}
void consume;
