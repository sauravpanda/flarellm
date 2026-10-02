"""Redistributable deterministic Llama GGUF with GQA, generated without ML packages."""
import hashlib
import random
import struct
from pathlib import Path

root = Path(__file__).parent
pack = struct.pack

def string(s):
    b = s.encode()
    return pack('<Q', len(b)) + b

metadata = {
    'general.architecture': 'llama', 'llama.context_length': 256,
    'llama.embedding_length': 128, 'llama.block_count': 2,
    'llama.feed_forward_length': 192, 'llama.attention.head_count': 2,
    'llama.attention.head_count_kv': 1, 'llama.rope.dimension_count': 64,
    'llama.attention.layer_norm_rms_epsilon': 1e-5,
    'llama.vocab_size': 128, 'tokenizer.ggml.model': 'none',
}
rng = random.Random(526)
tensors = []

def tensor(name, cols, rows=None, quant=False):
    dims = [cols] if rows is None else [cols, rows]
    if rows is None:
        data = pack('<' + 'f' * cols, *[1 + rng.uniform(-.1, .1) for _ in range(cols)])
    elif quant:
        data = b''.join(pack('<e32b', 0.006, *[rng.randrange(-12, 13) for _ in range(32)])
                        for _ in range(rows * cols // 32))
    else:
        data = pack('<' + 'f' * (cols * rows), *[rng.uniform(-.15, .15) for _ in range(cols * rows)])
    # These are directly GGUF adjacent-pair rows, independently consumed by llama.cpp.
    tensors.append((name, dims, 8 if quant else 0, data))

tensor('token_embd.weight', 128, 128)
for layer in range(2):
    for kind in ['attn_norm', 'ffn_norm']:
        tensor(f'blk.{layer}.{kind}.weight', 128)
    for kind, cols, rows in [('attn_q',128,128), ('attn_k',128,64), ('attn_v',128,64),
                             ('attn_output',128,128), ('ffn_gate',128,192),
                             ('ffn_up',128,192), ('ffn_down',192,128)]:
        tensor(f'blk.{layer}.{kind}.weight', cols, rows, quant=True)
tensor('output_norm.weight', 128)
tensor('output.weight', 128, 128)
header = b'GGUF' + pack('<IQQ', 3, len(tensors), len(metadata))
for key, value in metadata.items():
    kind = 8 if isinstance(value, str) else 6 if isinstance(value, float) else 4
    header += string(key) + pack('<I', kind)
    header += string(value) if kind == 8 else pack('<f' if kind == 6 else '<I', value)
body = b''
for name, dims, kind, data in tensors:
    body += bytes(-len(body) % 32)
    header += string(name) + pack('<I',len(dims)) + pack('<'+'Q'*len(dims),*dims) + pack('<IQ',kind,len(body))
    body += data
model = header + bytes(-len(header) % 32) + body
(root / 'gqa.gguf').write_bytes(model)
print(hashlib.sha256(model).hexdigest())
