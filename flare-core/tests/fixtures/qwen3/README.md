# Qwen3 original-tokenizer and template oracle

The modified reduced vocabulary in `qwen3-reduced.json` retains original Qwen
IDs and merge order. It is derived from Qwen/Qwen3-0.6B under Apache-2.0;
see `LICENSE-APACHE`. No trained weights are included.

`reference.json` is generated from the **full original tokenizer** with
Hugging Face tokenizers 0.22.2, not Flare. Four conversations are independently
rendered by Jinja2 3.1.6 from the official `tokenizer_config.json`, with
`enable_thinking=false` and `add_generation_prompt=true`. Fixtures cover NFC,
case-insensitive contractions, Unicode numbers/letters, whitespace/newlines,
array-form merges, added tokens (including non-special `<think>`/`</think>`),
empty input, assistant history and exact prompt IDs. The reduced tokenizer is
then checked against the full tokenizer with the same independent library.

Source revision: `c1899de289a04d12100db370d81485cdf75e47ca`.

- tokenizer.json SHA-256: `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`
- tokenizer_config.json SHA-256: `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`

Download both files from that revision into one directory, then:

```sh
python3 generate.py /path/to/tokenizer.json
```

`generate.py` verifies both hashes and the tokenizers version. Ordinary Rust
and packed-consumer Chromium CI use the committed subset without downloads.
The tokenizer accepts the exact official Split→ByteLevel pipeline; modified
patterns, ordering and flags remain unsupported. This is a text-message
non-thinking template implementation, not a general Jinja interpreter or a
tool-call/reasoning parser. No implicit BOS is inserted; EOS is `<|im_end|>`.
