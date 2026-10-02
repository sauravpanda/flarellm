"""Regenerate with Python tokenizers==0.22.2 and the pinned original tokenizer.

Usage: python3 generate.py /path/to/original/tokenizer.json
No downloads and no Flare dependency. Expected IDs always come from the FULL
original tokenizer; the reduced vocabulary is independently checked afterward.
"""
import hashlib
import json
from pathlib import Path
import sys
import tokenizers

assert tokenizers.__version__ == '0.22.2'
SHA = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
source = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(source).hexdigest() == SHA
original = tokenizers.Tokenizer.from_file(sys.argv[1])
texts = [
    '', 'a\n\nb', 'a  b', 'x\n\n', ' a', '  a', 'a ', 'a  ', '   ',
    '\n', '\n\n', '\n\nhello\n\nworld\n\n', 'a\tb', 'a\t\tb', 'a\r\nb',
    'a\r\n\r\nb', 'a \t b', 'a\t b', 'a\u00a0\u00a0b', 'a\u2003\u2003b',
    'a\u0085\u0085b', 'a\u2028\u2029b', 'a\x0b\x0cb',
    '1234567890', ' 123 45', 'a\n\n123', 'a12b34', '١٢٣ １２３ १२३',
    'a²³¼½ⅧⅨb', 'a\n\n²', "don't can't won't I'm we're they've she'll he'd",
    "DON'T I'M WE'RE", "'s 't 're 've 'm 'll 'd", "a''b...!?--", 'a—b…c',
    'café naïve élève', 'e\u0301 café', '中文 日本語 한국어', 'Привет мир',
    'مرحبا بالعالم', '🙂👩\u200d💻🚀', 'a\u200bb\u200dc', '\x00a\x01b',
    '<|endoftext|>', '<|im_start|><|im_end|>',
    'a<|im_start|>b<|im_end|>c', '  <|im_start|>  a\n\n<|im_end|>\n',
    '<|im_start|>system\nAnswer briefly.<|im_end|>\n<|im_start|>user\nWhat is 12 + 34?<|im_end|>\n<|im_start|>assistant\n',
    '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!\n\nCafé ١٢٣?<|im_end|>\n<|im_start|>assistant\nHi.<|im_end|>\n<|im_start|>user\nDon\'t repeat.\tExplain ².<|im_end|>\n<|im_start|>assistant\n',
]
# Deterministic boundary matrix: whitespace before letters, punctuation, numbers
# and end-of-input; tests Digits-before-ByteLevel ordering and UTF-8 offsets.
for whitespace in [' ', '  ', '\n\n', '\r\n', '\t\t', '\t ', '\u00a0\u00a0', '\u2003 ']:
    for suffix in ['b', '!', '12', '²Ⅷ', '']:
        texts.append('a' + whitespace + suffix)
# Independently render the pinned official Jinja template with thinking disabled.
import jinja2
config_source = Path(sys.argv[1]).with_name('tokenizer_config.json').read_bytes()
assert hashlib.sha256(config_source).hexdigest() == 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'
template = jinja2.Environment().from_string(json.loads(config_source)['chat_template'])
conversations = [
    [{'role':'user','content':'What is the capital of France?'}],
    [{'role':'system','content':'Answer briefly.'},{'role':'user','content':'Reply with exactly the word hello.'}],
    [{'role':'user','content':'Hi'},{'role':'assistant','content':'<think>\nSecret\n</think>\n\nHello'},{'role':'user','content':'What is 2 + 2?'}],
    [{'role':'user','content':'Hi'},{'role':'assistant','content':'Hello'}],
]
chats = []
for messages in conversations:
    text = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    chats.append(dict(messages=messages, text=text, ids=original.encode(text, add_special_tokens=False).ids))
    texts.append(text)
texts.extend(['<think>', '</think>', 'a<think>\n\n</think>\n\nb', "DON'T I'M WE'RE", '\n \nabc'])
texts = list(dict.fromkeys(texts))
cases = [{'text': t, 'ids': original.encode(t, add_special_tokens=False).ids} for t in texts]
# Keep every merge whose result can occur in ANY raw input, even across forbidden
# boundaries. This preserves the old-behavior negative control as well as parity.
doc = json.loads(source)
bs = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
cs = bs.copy()
extra = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + extra); extra += 1
byte_map = dict(zip(bs, map(chr, cs)))
encoded = [''.join(byte_map[b] for b in t.encode()) for t in texts]
merges = [m for m in doc['model']['merges'] if any(''.join(m) in t for t in encoded)]
keep = set(byte_map.values()) | {a['content'] for a in doc['added_tokens']}
for m in merges:
    a,b = m; keep.update([a,b,a+b])
doc['model']['vocab'] = {t:i for t,i in doc['model']['vocab'].items() if t in keep}
doc['model']['merges'] = merges
for token in doc['added_tokens']:
    doc['model']['vocab'][token['content']] = token['id']
small = json.dumps(doc, ensure_ascii=False, indent=2) + '\n'
reduced = tokenizers.Tokenizer.from_str(small)
for case in cases:
    assert reduced.encode(case['text'], add_special_tokens=False).ids == case['ids'], repr(case['text'])
out = Path(__file__).parent
(out/'qwen3-reduced.json').write_text(small)
manifest = {
    'model': 'Qwen/Qwen3-0.6B',
    'revision': 'c1899de289a04d12100db370d81485cdf75e47ca',
    'originalSha256': SHA, 'referenceImplementation': 'huggingface/tokenizers',
    'referenceVersion': tokenizers.__version__, 'addSpecialTokens': False,
    'fixtureSha256': hashlib.sha256(small.encode()).hexdigest(),
    'cases': cases, 'chats': chats, 'jinjaVersion': jinja2.__version__,
}
(out/'reference.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
print(f'{len(cases)} cases, {sum(len(c["ids"]) for c in cases)} IDs, {len(small.encode())} tokenizer bytes')
print(cases[1:4])
