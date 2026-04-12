// Dequantize Q5_K blocks on GPU.
//
// Layout (176 bytes / 44 u32 per block, 256 weights per block):
//   bytes  0-1  : d    (f16 LE)
//   bytes  2-3  : dmin (f16 LE)
//   bytes  4-15 : scales[12] — 8 (scale, min) pairs, same 6-bit encoding as Q4_K
//   bytes 16-47 : qh[32]    — one high bit per weight; qh[j/8] bit (j%8) = 5th bit
//                             of weight j; qh[(j+128)/8] bit (j%8) = 5th bit of weight j+128
//   bytes 48-175: ql[128]   — low 4 bits per weight; ql[j] & 0xF = lo nibble (weight j),
//                             ql[j] >> 4 = hi nibble (weight j+128)
//
// Weight reconstruction:
//   q_lo  = (ql[j] & 0x0F) | ((qh[j/8] >> (j%8)) & 1) << 4)    — 5-bit value
//   q_hi  = ((ql[j] >> 4) & 0x0F) | (((qh[(j+128)/8] >> (j%8)) & 1) << 4)
//   output[j]     = d * sc[j/32]   * q_lo - dmin * mn[j/32]
//   output[j+128] = d * sc[j/32+4] * q_hi - dmin * mn[j/32+4]
//
// Scale decoding: identical to Q4_K (scales_raw[0..12] in bytes 4-15).
// Each thread handles one block.

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    num_blocks: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn dequant_q5k(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let block_idx = gid.x;
    if block_idx >= params.num_blocks {
        return;
    }

    // 176 bytes per block = 44 u32; u is the base u32 index for this block.
    let u = block_idx * 44u;

    // d and dmin packed as two f16 in the first u32.
    let dm = unpack2x16float(raw[u]);
    let d    = dm.x;
    let dmin = dm.y;

    // Decode 8 scale/min pairs from scales_raw[0..12] = raw[u+1 .. u+4].
    //   scales_raw[0..4]  packed in raw[u+1]
    //   scales_raw[4..8]  packed in raw[u+2]
    //   scales_raw[8..12] packed in raw[u+3]
    // Same 6-bit encoding as Q4_K.
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

    // qh at raw[u+4 .. u+12]  (8 u32 = 32 bytes, bytes 16-47 of block)
    // ql at raw[u+12 .. u+44] (32 u32 = 128 bytes, bytes 48-175 of block)
    let out_base = block_idx * 256u;
    for (var j = 0u; j < 128u; j = j + 1u) {
        // ql[j]: u32 index u+12+j/4, byte (j%4)
        let ql_u32  = raw[u + 12u + j / 4u];
        let ql_byte = (ql_u32 >> ((j % 4u) * 8u)) & 0xFFu;
        let lo_nibble = ql_byte & 0x0Fu;
        let hi_nibble = (ql_byte >> 4u) & 0x0Fu;

        // qh bit for weight j (lo):  qh byte at index j/8, bit (j%8).
        //   qh byte j/8 is in u32[4 + j/32], at byte-within-u32 = (j/8) % 4.
        let qh_lo_u32  = raw[u + 4u + j / 32u];
        let qh_lo_byte = (qh_lo_u32 >> (((j / 8u) % 4u) * 8u)) & 0xFFu;
        let qh_lo_bit  = (qh_lo_byte >> (j % 8u)) & 1u;

        // qh bit for weight j+128 (hi): qh byte at index j/8+16, bit (j%8).
        //   qh byte (j/8+16) is in u32[8 + j/32], same byte-within-u32 = (j/8) % 4.
        let qh_hi_u32  = raw[u + 8u + j / 32u];
        let qh_hi_byte = (qh_hi_u32 >> (((j / 8u) % 4u) * 8u)) & 0xFFu;
        let qh_hi_bit  = (qh_hi_byte >> (j % 8u)) & 1u;

        // Assemble 5-bit quantized values.
        let q_lo = lo_nibble | (qh_lo_bit << 4u);
        let q_hi = hi_nibble | (qh_hi_bit << 4u);
        let sub  = j / 32u;

        output[out_base + j]        = d * f32(sc[sub])      * f32(q_lo) - dmin * f32(mn[sub]);
        output[out_base + j + 128u] = d * f32(sc[sub + 4u]) * f32(q_hi) - dmin * f32(mn[sub + 4u]);
    }
}
