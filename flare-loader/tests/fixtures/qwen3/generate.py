"""Redistributable deterministic Qwen3 GGUF with GQA, generated without ML packages."""
import hashlib
import random
import struct
import sys
from pathlib import Path

root = Path(__file__).parent
pack = struct.pack
decision = len(sys.argv) > 1 and sys.argv[1] == '--decision'
vocab = 151936 if decision else 128

def string(s):
    b = s.encode()
    return pack('<Q', len(b)) + b

metadata = {
    'general.architecture': 'qwen3', 'qwen3.context_length': 512 if decision else 256,
    'qwen3.embedding_length': 128, 'qwen3.block_count': 2,
    'qwen3.feed_forward_length': 192, 'qwen3.attention.head_count': 4,
    'qwen3.attention.head_count_kv': 2, 'qwen3.rope.dimension_count': 64,
    'qwen3.attention.key_length': 64, 'qwen3.attention.value_length': 64,
    'qwen3.rope.freq_base': 1000000.0,
    'qwen3.attention.layer_norm_rms_epsilon': 1e-6,
    'qwen3.vocab_size': vocab, 'tokenizer.ggml.model': 'none',
}
rng = random.Random(540)
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
    # Qwen3 GGUF retains HF split-half row order, independently consumed by llama.cpp.
    tensors.append((name, dims, 8 if quant else 0, data))

tensor('token_embd.weight', 128, vocab, quant=decision)
for layer in range(2):
    for kind in ['attn_norm', 'ffn_norm']:
        tensor(f'blk.{layer}.{kind}.weight', 128)
    for kind in ['attn_q_norm', 'attn_k_norm']:
        tensor(f'blk.{layer}.{kind}.weight', 64)
    for kind, cols, rows in [('attn_q',128,256), ('attn_k',128,128), ('attn_v',128,128),
                             ('attn_output',256,128), ('ffn_gate',128,192),
                             ('ffn_up',128,192), ('ffn_down',192,128)]:
        tensor(f'blk.{layer}.{kind}.weight', cols, rows, quant=True)
tensor('output_norm.weight', 128)
# No output.weight: tied token embedding.
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
(Path(sys.argv[2]) if decision else root / 'gqa.gguf').write_bytes(model)
print(hashlib.sha256(model).hexdigest())
