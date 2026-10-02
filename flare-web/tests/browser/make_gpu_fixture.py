"""Write a deterministic, nonzero two-layer Q8_0 fixture (no downloads).

Square 128-wide tensors keep GGUF layout (#526) out of this regression.
Two 64-element heads meet WebGPU's 256-byte f32 binding alignment.
"""
import struct
import sys
from pathlib import Path

pack = struct.pack

def string(value):
    data = value.encode()
    return pack('<Q', len(data)) + data

metadata = {
    'general.architecture': 'llama',
    'llama.context_length': 32,
    'llama.embedding_length': 128,
    'llama.block_count': 2,
    'llama.feed_forward_length': 128,
    'llama.attention.head_count': 2,
    'llama.attention.head_count_kv': 2,
    'llama.rope.dimension_count': 64,
}
tensors = []

def tensor(name, norm=False, quant=False):
    seed = sum(name.encode())
    dims = [128] if norm else [128, 128]
    if norm:
        data = pack('<128f', *([1.0] * 128))
    elif quant:
        data = b''.join(pack('<e32b', 0.004, *[
            ((i * 13 + i // 128 * 7 + seed) % 15) - 7
            for i in range(block * 32, block * 32 + 32)
        ]) for block in range(128 * 4))
    else:
        data = pack('<16384f', *[
            (((i * 13 + i // 128 * 7 + seed) % 37) - 18) * 0.01
            for i in range(128 * 128)
        ])
    tensors.append((name, dims, 8 if quant else 0, data))

tensor('token_embd.weight')
for layer in range(2):
    for name in ['attn_norm', 'ffn_norm']:
        tensor(f'blk.{layer}.{name}.weight', norm=True)
    for name in ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']:
        tensor(f'blk.{layer}.{name}.weight', quant=True)
tensor('output_norm.weight', norm=True)
tensor('output.weight')
header = b'GGUF' + pack('<IQQ', 3, len(tensors), len(metadata))
for key, value in metadata.items():
    header += string(key)
    header += pack('<I', 8 if isinstance(value, str) else 4)
    header += string(value) if isinstance(value, str) else pack('<I', value)
body = b''
for name, dims, kind, data in tensors:
    body += bytes((-len(body)) % 32)
    header += string(name) + pack('<I', len(dims)) + pack('<' + 'Q' * len(dims), *dims) + pack('<IQ', kind, len(body))
    body += data
header += bytes((-len(header)) % 32)
Path(sys.argv[1]).write_bytes(header + body)
