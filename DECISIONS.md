# Experimental bounded decisions

`Flare.decide()` scores 2–8 offered text choices using **one Qwen3 CPU prompt
prefill**. It does not generate or parse an answer, implement arbitrary JSON
constraints (#524), load a custom classifier head, or execute routing actions.
Build the SDK from this branch; no npm release is published by this change.

```ts
import { Flare } from '@sauravpanda/flare';
const flare = await Flare.init({
  backend: 'cpu',
  modelUrl: 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf',
  tokenizerUrl: 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json',
});
const result = await flare.decide({
  state: 'I was charged twice for my subscription.',
  question: 'Which team should handle this?',
  choices: ['billing', 'technical', 'account'] as const,
});
console.log(result.choice); // typed as 'billing' | 'technical' | 'account'
console.table(result.scores); // original order: choice, label, tokenId, logit, score
flare.dispose();
```

See [SDK setup](flare-web/README.md) for module-worker serving, imports and CORS.
The packaged [routing demo](flare-web/demo/routing.html) displays a suggestion and
scores; it does not send tickets or collect ticket text. Model/tokenizer assets
are downloaded; inference remains local. The API makes no remote inference calls.

## Contract and limits

- `state` and `question` are nonblank **strings**, initially text only. Limits
  are 16,384 and 2,048 UTF-8 bytes respectively. No objects or serialized state
  schema is accepted. Choices are 2–8 distinct, nonempty, trimmed strings of at
  most 256 UTF-8 bytes each. Uniqueness is exact string equality; no automatic
  trimming, case folding, or deduplication occurs. Option strings are display
  text and stable return values; separate option IDs are not implemented.
- Template control markers (`<|`, `<think>`, `</think>`) are rejected in inputs.
  JSON quoting preserves field and option boundaries. This is not a security
  boundary against semantic prompt injection or a guarantee of correct routing.
- The **complete** rendered system/user/assistant prompt must fit the loaded
  model's context. Browser Qwen3 is capped at 512 tokens. No hidden truncation or
  output generation occurs, so a 512-token prompt fits without a decode reserve.
  Native callers configure `max_seq_len` before creating the model.
- Internal labels are `A`–`H`, mapping to pinned token IDs 32–39, in the original
  choice order. Each call verifies that appending each label to the **actual
  answer boundary** adds exactly one expected token and leaves all prompt IDs
  unchanged; standalone label tokenization alone is insufficient. The supported
  non-thinking template ends in `</think>\n\n`; no answer space is inserted.
- Qwen3 architecture, CPU backend, Qwen3 NFC → Split → ByteLevel tokenizer
  pipeline, template control IDs, in-range prompt IDs, exact distinct label IDs
  and label decoding are checked. Incompatible combinations reject. The API does
  not authenticate downloaded model/tokenizer bytes: use the pinned public assets
  and checksums in [QWEN3.md](QWEN3.md) and the reference manifests. Compatible
  reduced tokenizers/untrained models are used only in CI, not quality claims.
- Initial validated checkpoint is official **Qwen3-0.6B Q8_0 + original JSON
  tokenizer**, native CPU and browser WASM CPU. Other checkpoint quality is not
  certified. Explicit browser WebGPU loading rejects Qwen3; existing native
  backend selection falls back to CPU. No Qwen3 GPU testing is claimed.
- The result includes `choice`, original `index`, ordered `scores`,
  `promptVersion`, complete `prompt`, and `promptIds`. SDK `decisionMs` includes
  worker-side validation, tokenization and prefill, excluding load and message
  transit. Raw prompts in results stay with the caller; do not log them unless
  appropriate for your application.
- Non-finite model logits reject the entire decision. Finite logits are
  normalized with max-subtracted f64 softmax. An exact logit tie selects the first
  offered choice; there is no random sampling, temperature, or repeat penalty.
- One SDK operation may run at a time (`BUSY`). `signal`, `cancel()`, `reset()`,
  disposal, worker request IDs and reload behavior match chat. Cancellation
  terminates the worker, including synchronous prefill; the next request reloads
  the configured model. Decision input/inference errors propagate with `DECIDE`
  and cause the existing worker error/reload recovery. Core/WASM decision entry
  points clear state before and after, including failures; chat starts fresh.
  Existing generation and low-level exports remain available.

## Scores are conditional, not confidence

For offered label logits `z`, `score[i] = exp(z[i] - max(z)) / sum(exp(z - max(z)))`.
These scores sum to one **over offered labels only**. They are neither calibrated
probabilities of correctness nor full-vocabulary probabilities. The model might
prefer an entirely different token outside this set. A score near one does not
mean the choice is correct or that any offered choice is suitable.

Adding or reordering choices changes the prompt, label assignment and denominator.
There is no automatic abstention. In the routing evaluation, `other` is an ordinary
fourth option described as “none fits or multiple queues are equally needed.” Its
score is not an uncertainty threshold. The baseline is unsuitable for unattended
routing; see the measured errors and order sensitivity in the evaluation report.

## Native API and reproduction

Native applications can call `flare_core::decision::decide(&mut model,
&tokenizer, &DecisionRequest { state, question, choices })`. It uses the same
prompt, validation, single prefill and normalization as WASM. The binding exposes
only `FlareEngine.decide(tokenizer, request_json)` returning the typed result as
JSON; it does not introduce a general unchecked prefill-logit interface.

The prompt identifier is `qwen3-choice-v1`. Exact rendered prompts, original
Hugging Face token IDs, llama.cpp candidate logits, dataset provenance, model and
reference revisions and reproduction commands live in
[evaluations/decision](evaluations/decision/README.md). Changing prompt wording
requires a new prompt version and new held-out data for any tuning comparison.

The [NanoJev comparison](evaluations/decision/comparison/README.md) evaluates the
published game checkpoint on new routing tickets. It selects `billing` for all
64 test tickets in all four orders (25% accuracy), so it does not justify adding
a NanoJev backend. The existing Qwen prompt also needs quality work. This API
remains experimental; neither model is validated for unattended routing.
