// Dequantize Q6_K blocks on GPU.
//
// Layout (210 bytes per block, 256 weights per block):
//   bytes   0-127: ql[128] — lower 4 bits of the 6-bit weights (two interleaved groups)
//   bytes 128-191: qh[64]  — upper 2 bits of the 6-bit weights
//   bytes 192-207: scales[16] — signed i8 scale per sub-block
//   bytes 208-209: d (f16 LE)
//
// 210 bytes is not u32-aligned.  The Rust caller pads the raw byte buffer to the
// nearest 4-byte multiple before uploading.  Weights are accessed via byte-offset
// helpers that extract individual bytes from the u32 storage array.
//
// Reconstruction (mirrors llama.cpp dequant_row_q6_K):
//   For half in 0..2 (each covering 128 output elements):
//     ql_off = half*64,  qh_off = half*32,  sc_off = half*8,  y_off = half*128
//     For l in 0..32:
//       q1 = (ql[ql_off+l]    & 0xF)  | ((qh[qh_off+l] & 3)       << 4) − 32
//       q2 = (ql[ql_off+l+32] & 0xF)  | (((qh[qh_off+l] >> 2) & 3) << 4) − 32
//       q3 =  (ql[ql_off+l]    >> 4)  | (((qh[qh_off+l] >> 4) & 3) << 4) − 32
//       q4 =  (ql[ql_off+l+32] >> 4)  | (((qh[qh_off+l] >> 6) & 3) << 4) − 32
//       output[y_off+l]    = d * sc[sc_off + l/16]     * q1
//       output[y_off+l+32] = d * sc[sc_off + l/16 + 2] * q2
//       output[y_off+l+64] = d * sc[sc_off + l/16 + 4] * q3
//       output[y_off+l+96] = d * sc[sc_off + l/16 + 6] * q4
//
// Each thread handles one block.

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    num_blocks: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

/// Reinterpret an 8-bit unsigned value as a signed 8-bit integer (i32).
fn sign_extend_byte(b: u32) -> i32 {
    return bitcast<i32>(b << 24u) >> 24u;
}

@compute @workgroup_size(64)
fn dequant_q6k(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let block_idx = gid.x;
    if block_idx >= params.num_blocks {
        return;
    }

    // 210 bytes per block; bb is the byte offset of this block's start.
    let bb = block_idx * 210u;

    // d at bytes 208-209 of this block (LE f16).
    let d_packed = read_byte(bb + 208u) | (read_byte(bb + 209u) << 8u);
    let d = unpack2x16float(d_packed).x;

    let out_base = block_idx * 256u;

    for (var half = 0u; half < 2u; half = half + 1u) {
        let ql_off = half * 64u;
        let qh_off = half * 32u;
        let sc_off = half * 8u;
        let y_off  = half * 128u;

        for (var l = 0u; l < 32u; l = l + 1u) {
            // ql bytes: lower 4 bits of the 6-bit weight.
            let ql_a = read_byte(bb + ql_off + l);
            let ql_b = read_byte(bb + ql_off + l + 32u);
            // qh byte: upper 2 bits for four weights packed in 8 bits.
            let qh_b = read_byte(bb + 128u + qh_off + l);

            // Assemble 6-bit unsigned values, then offset-binary subtract 32.
            let q1 = i32((ql_a & 0x0Fu) | ((qh_b & 3u) << 4u)) - 32;
            let q2 = i32((ql_b & 0x0Fu) | (((qh_b >> 2u) & 3u) << 4u)) - 32;
            let q3 = i32((ql_a >> 4u)   | (((qh_b >> 4u) & 3u) << 4u)) - 32;
            let q4 = i32((ql_b >> 4u)   | (((qh_b >> 6u) & 3u) << 4u)) - 32;

            // Signed i8 scales (two scale values per 32-element group, 4 groups per half).
            let is = l / 16u;
            let sc1 = sign_extend_byte(read_byte(bb + 192u + sc_off + is));
            let sc2 = sign_extend_byte(read_byte(bb + 192u + sc_off + is + 2u));
            let sc3 = sign_extend_byte(read_byte(bb + 192u + sc_off + is + 4u));
            let sc4 = sign_extend_byte(read_byte(bb + 192u + sc_off + is + 6u));

            output[out_base + y_off + l]        = d * f32(sc1) * f32(q1);
            output[out_base + y_off + l + 32u]  = d * f32(sc2) * f32(q2);
            output[out_base + y_off + l + 64u]  = d * f32(sc3) * f32(q3);
            output[out_base + y_off + l + 96u]  = d * f32(sc4) * f32(q4);
        }
    }
}
