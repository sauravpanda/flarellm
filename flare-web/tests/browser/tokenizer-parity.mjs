// Called by the established packed-consumer browser suite after WASM init.
import { FlareTokenizer } from '@sauravpanda/flare/wasm';
const equal = (a, b, label) => {
  if (JSON.stringify(a) !== JSON.stringify(b)) throw new Error(`${label}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
};
export async function tokenizerParity(tokenizerUrl = '/tokenizer-parity/smollm2-reduced.json', qwen = false) {
  const reference = await (await fetch(qwen ? '/qwen3-tokenizer/reference.json' : '/tokenizer-parity/reference.json')).json();
  const json = await (await fetch(tokenizerUrl)).text();
  const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(json)))].map(b => b.toString(16).padStart(2, '0')).join('');
  equal(hash, tokenizerUrl.includes('-reduced.json') ? reference.fixtureSha256 : reference.originalSha256, 'Tokenizer checksum');
  const tokenizer = FlareTokenizer.from_json(json);
  let tokenIdsCompared = 0;
  try {
    for (const { text, ids } of reference.cases) {
      equal([...tokenizer.encode(text)], ids, `Independent tokenizer IDs for ${JSON.stringify(text)}`);
      tokenIdsCompared += ids.length;
    }
    equal(tokenizer.eos_token_id, qwen ? 151645 : 0, 'Existing EOS metadata');
  } finally { tokenizer.free(); }
  // Actual WASM negative control: no pre_tokenizer reproduces the old algorithm.
  const legacyJson = JSON.parse(json);
  legacyJson.pre_tokenizer = null;
  const legacy = FlareTokenizer.from_json(JSON.stringify(legacyJson));
  try {
    if (!qwen) equal([...legacy.encode('a\n\nb')], [81, 1116, 82], 'Legacy blank-line negative control');
    if (!qwen) equal([...legacy.encode('a  b')], [81, 256, 82], 'Legacy repeated-space negative control');
  } finally { legacy.free(); }
  legacyJson.pre_tokenizer = { type: 'Whitespace' };
  let rejected = false;
  try { FlareTokenizer.from_json(JSON.stringify(legacyJson)).free(); }
  catch (error) { rejected = String(error).includes('unsupported pre_tokenizer'); }
  if (!rejected) throw new Error('Unsupported pre-tokenizer was not rejected');
  return { cases: reference.cases.length, tokenIdsCompared, sha256: hash,
    referenceImplementation: reference.referenceImplementation, referenceVersion: reference.referenceVersion,
    addSpecialTokens: reference.addSpecialTokens, model: reference.model, revision: reference.revision,
    negativeControls: 'passed', unsupportedPipeline: 'rejected' };
}
