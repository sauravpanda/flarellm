// Dequantize Q4_K blocks on GPU.
//
// Layout (144 bytes / 36 u32 per block, 256 weights per block):
//   bytes  0-1  : d    (f16 LE overall delta)
//   bytes  2-3  : dmin (f16 LE overall min delta)
//   bytes  4-15 : scales[12] — 8 sub-block (scale, min) pairs encoded in 12 bytes
//   bytes 16-143: qs[128]    — 4-bit nibbles; low nibbles → weights 0..127,
//                              high nibbles → weights 128..255
//
// Scale decoding (matches llama.cpp Q4_K):
//   For i in 0..4:
//     sc[i]     = scales_raw[i]     & 0x3F
//     mn[i]     = scales_raw[i+4]   & 0x3F
//     sc[i+4]   = (scales_raw[i] >> 6) | ((scales_raw[i+8] & 0x0F) << 2)
//     mn[i+4]   = (scales_raw[i+4] >> 6) | ((scales_raw[i+8] >>  4) << 2)
//
// Reconstruction: output[j]     = d * sc[j/32]   * low_nibble  - dmin * mn[j/32]
//                 output[j+128] = d * sc[j/32+4] * high_nibble - dmin * mn[j/32+4]
//
// Each thread handles one block.

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    num_blocks: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn dequant_q4k(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let block_idx = gid.x;
    if block_idx >= params.num_blocks {
        return;
    }

    // 144 bytes per block = 36 u32
    let u = block_idx * 36u;

    // d and dmin as two f16 packed in the first u32
    let dm = unpack2x16float(raw[u]);
    let d = dm.x;
    let dmin = dm.y;

    // Decode 8 scale/min pairs from scales_raw[0..12] = raw[u+1 .. u+4]
    // scales_raw[0..4]  packed in raw[u+1]
    // scales_raw[4..8]  packed in raw[u+2]
    // scales_raw[8..12] packed in raw[u+3]
    var sc: array<u32, 8>;
    var mn: array<u32, 8>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let sr_i  = (raw[u + 1u] >> (i * 8u)) & 0xFFu;
        let sr_i4 = (raw[u + 2u] >> (i * 8u)) & 0xFFu;
        let sr_i8 = (raw[u + 3u] >> (i * 8u)) & 0xFFu;
        sc[i]      = sr_i & 0x3Fu;
        mn[i]      = sr_i4 & 0x3Fu;
        sc[i + 4u] = (sr_i >> 6u) | ((sr_i8 & 0x0Fu) << 2u);
        mn[i + 4u] = (sr_i4 >> 6u) | ((sr_i8 >> 4u) << 2u);
    }

    // qs[128] at bytes 16-143 = raw[u+4 .. u+36]  (32 u32 = 128 bytes)
    let out_base = block_idx * 256u;
    for (var j = 0u; j < 128u; j = j + 1u) {
        let q_u32    = raw[u + 4u + j / 4u];
        let byte_val = (q_u32 >> ((j % 4u) * 8u)) & 0xFFu;
        let lo_nibble = byte_val & 0x0Fu;
        let hi_nibble = (byte_val >> 4u) & 0x0Fu;

        let sub = j / 32u;
        output[out_base + j]       = d * f32(sc[sub])      * f32(lo_nibble) - dmin * f32(mn[sub]);
        output[out_base + j + 128u] = d * f32(sc[sub + 4u]) * f32(hi_nibble) - dmin * f32(mn[sub + 4u]);
    }
}
