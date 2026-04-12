// Dequantize Q4_1 blocks on GPU.
//
// Layout (20 bytes / 5 u32 per block, 32 weights per block):
//   bytes 0-1  : d  (f16 LE scale)
//   bytes 2-3  : m  (f16 LE additive bias)
//   bytes 4-19 : qs[16] (4-bit nibbles, low nibble = even weight, high = odd)
//
// Reconstruction: output[i] = d * q + m, q in [0, 15].
//
// Each thread handles one block.

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    num_blocks: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn dequant_q4_1(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let block_idx = gid.x;
    if block_idx >= params.num_blocks {
        return;
    }

    // 20 bytes per block = 5 u32
    let u = block_idx * 5u;

    // d and m are f16 LE packed as two f16 in a u32
    let dm = unpack2x16float(raw[u]);
    let d = dm.x;
    let m = dm.y;

    let out_base = block_idx * 32u;

    // qs[16] are in raw[u+1 .. u+5], 4 bytes (8 nibble-pairs) per u32
    for (var qi = 0u; qi < 4u; qi = qi + 1u) {
        let q_u32 = raw[u + 1u + qi];
        for (var bi = 0u; bi < 4u; bi = bi + 1u) {
            let byte_val = (q_u32 >> (bi * 8u)) & 0xFFu;
            let lo = byte_val & 0x0Fu;
            let hi = (byte_val >> 4u) & 0x0Fu;
            let pos = qi * 8u + bi * 2u;
            output[out_base + pos] = d * f32(lo) + m;
            output[out_base + pos + 1u] = d * f32(hi) + m;
        }
    }
}
