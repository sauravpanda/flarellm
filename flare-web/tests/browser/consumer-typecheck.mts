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

async function decisionTypes(flare: Flare) {
const decision = await flare.decide({state:'Charged twice',question:'Queue?',choices:['billing','other'] as const});
const selected: 'billing' | 'other' = decision.choice;
const scoreChoice: 'billing' | 'other' = decision.scores[0].choice;
void selected; void scoreChoice;
// @ts-expect-error structured state is outside the text-only contract
await flare.decide({state:{ticket:'x'},question:'Queue?',choices:['a','b']});

}
void decisionTypes;
