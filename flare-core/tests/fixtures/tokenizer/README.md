# Independent tokenizer parity fixture

`reference.json` contains exact IDs generated with **Hugging Face tokenizers
0.22.2**, using `Tokenizer.from_file(original).encode(text,
add_special_tokens=False).ids`. Expectations are generated from the full original
file, never from Flare or from a reconstruction of the BPE algorithm.

Source: [HuggingFaceTB/SmolLM2-360M-Instruct tokenizer.json](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/tokenizer.json),
revision `a10cc1512eabd3dde888204e902eca88bddb4951`, SHA-256
`9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c`.
The [source model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/README.md)
licenses it under Apache-2.0; see `LICENSE-APACHE`. The reduced tokenizer is a
modified subset of HuggingFaceTB's tokenizer, preserving original IDs and merge
order. No model weights or downstream private evaluation data are included.

## Regeneration

Install `tokenizers==0.22.2` in a Python environment, download the pinned source
above, then run:

```sh
python3 flare-core/tests/fixtures/tokenizer/generate.py /path/to/tokenizer.json
```

The generator verifies the source checksum, creates 81 deterministic cases
(498 IDs), and retains all byte tokens, special tokens, and merges whose result
occurs anywhere in a raw case. Retaining raw-input merges, including those across
forbidden boundaries, makes the legacy negative controls meaningful. Original
vocabulary IDs remain sparse. The generator also loads the reduced JSON into
Hugging Face and checks every case against the full original. This ~18 KiB
fixture is corpus-specific; it is not a replacement tokenizer for inference.
CI uses committed fixtures without Python packages, network access, or models.

Cases cover empty input, leading/trailing/repeated spaces, blank lines, tabs,
CRLF, mixed and Unicode whitespace, ASCII and Unicode Number categories Nd/Nl/No,
contractions (including uppercase), punctuation, combining characters, multiple
scripts, emoji, control bytes, special tokens and representative rendered ChatML
prompts. Rust and the actual packed `FlareTokenizer` browser binding consume the
same independently generated expectations. The browser checks the fixture hash.

## Supported semantics and compatibility

`BpeTokenizer::from_json` supports this explicit sequence, in this order:

1. `Digits` with `individual_digits: true`: isolate each Unicode Number scalar.
2. `ByteLevel` with `add_prefix_space: false, use_regex: true`: split with the
   case-sensitive GPT-2 expression before byte mapping and BPE. Its whitespace
   negative lookahead distinguishes interior and trailing whitespace. Each
   digit-delimited segment is processed separately. `trim_offsets` does not
   affect IDs; Flare does not expose offsets.

Missing/null `pre_tokenizer` preserves historical whole-chunk byte BPE (including
small synthetic fixtures). **Other explicit pre-tokenizers now fail at load**
with `unsupported pre_tokenizer`; they previously were silently ignored. This
includes reversed sequences, grouped digits, prefix-space mode, regex disabled,
standalone ByteLevel, and unknown pipelines. Do not remove an unsupported
pipeline to obtain reference parity: the legacy mode is compatibility only.

Special tokens are recognized before pre-tokenization, and the existing BOS/EOS
metadata and explicit-token behavior remain unchanged. No automatic BOS/EOS
insertion or post-processing is performed, matching `add_special_tokens=False`
for this original JSON. This work does not add general normalizer,
post-processor, added-token flag, decoder or BPE-model-option support. Parity is
certified for the pinned original configuration and corpus, not arbitrary JSON.
Unicode classification follows the Rust regex Unicode tables; future Unicode
versions and additional configurations require independent reference validation.
The embedded GGUF tokenizer is a separate, unchanged path. This does not resolve
GGUF tensor layout #526, Q4_0 loading #529, or real-answer acceptance #520.
