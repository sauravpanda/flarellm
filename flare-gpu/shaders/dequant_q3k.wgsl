// Dequantize Q3_K blocks on GPU.
//
// Layout (110 bytes per block, 256 weights per block):
//   bytes   0-31:  hmask[32] — high bit (bit 2) for each weight, 8 per byte
//   bytes  32-95:  qs[64]    — low 2 bits of each weight, 4 per byte
//   bytes  96-107: scales[12] — 8 sub-block scales (6-bit) via kmask transform
//   bytes 108-109: d (f16 LE)
//
// 110 bytes is not u32-aligned. The Rust caller pads the raw byte buffer to the
// nearest 4-byte multiple before uploading. Weights are accessed via byte-offset
// helpers that extract individual bytes from the u32 storage array.
//
// Scale decoding (kmask transform, b = scales_raw = block bytes 96..108):
//   scales[0] = (b[0] & 0x0F) | (((b[8]  >> 4) & 3) << 4) − 32
//   scales[1] = (b[1] & 0x0F) | (((b[9]  >> 4) & 3) << 4) − 32
//   scales[2] = (b[2] & 0x0F) | (((b[10] >> 4) & 3) << 4) − 32
//   scales[3] = (b[3] & 0x0F) | (((b[11] >> 4) & 3) << 4) − 32
//   scales[4] = (b[4] & 0x0F) | (((b[8]  >> 6) & 3) << 4) − 32
//   scales[5] = (b[5] & 0x0F) | (((b[9]  >> 6) & 3) << 4) − 32
//   scales[6] = (b[6] & 0x0F) | (((b[10] >> 6) & 3) << 4) − 32
//   scales[7] = (b[7] & 0x0F) | (((b[11] >> 6) & 3) << 4) − 32
//
// Weight reconstruction (mirrors dequant_q3k_block in flare-loader):
//   For oi in 0..2 (each outer iteration covers 128 output elements):
//     For si in 0..4 (each sub-block covers 32 elements):
//       m = 1 << (oi*4 + si)    — hmask bit selector
//       shift = si * 2           — qs bit position
//       scale_idx = oi*4 + si
//       For l in 0..32:
//         qs_byte = qs[oi*32 + l] = block byte 32 + oi*32 + l
//         low2  = (qs_byte >> shift) & 3
//         sub   = if (hmask[l] & m) != 0 { 0 } else { 4 }
//         q     = i32(low2) - sub     — range [−4, 3]
//         output[oi*128 + si*32 + l] = d * scales[scale_idx] * q
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

@compute @workgroup_size(64)
fn dequant_q3k(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let block_idx = gid.x;
    if block_idx >= params.num_blocks {
        return;
    }

    // 110 bytes per block; bb is the byte offset of this block's start.
    let bb = block_idx * 110u;

    // d at bytes 108-109 of this block (LE f16).
    let d_packed = read_byte(bb + 108u) | (read_byte(bb + 109u) << 8u);
    let d = unpack2x16float(d_packed).x;

    // Decode 8 scales from scales_raw bytes 96..108 using kmask transform.
    // b[k] = block byte at bb + 96 + k.
    var scales: array<i32, 8>;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let bk    = read_byte(bb + 96u + k);
        let bk4   = read_byte(bb + 96u + k + 4u);
        let bk8   = read_byte(bb + 96u + k + 8u);
        scales[k]      = i32((bk  & 0x0Fu) | (((bk8 >> 4u) & 3u) << 4u)) - 32;
        scales[k + 4u] = i32((bk4 & 0x0Fu) | (((bk8 >> 6u) & 3u) << 4u)) - 32;
    }

    let out_base = block_idx * 256u;

    for (var oi = 0u; oi < 2u; oi = oi + 1u) {
        for (var si = 0u; si < 4u; si = si + 1u) {
            let shift      = si * 2u;
            let scale_idx  = oi * 4u + si;
            let m          = 1u << scale_idx;   // hmask bit selector (fits in u8)
            let d_scale    = d * f32(scales[scale_idx]);

            for (var l = 0u; l < 32u; l = l + 1u) {
                // qs byte for weight oi*128 + si*32 + l:
                //   block byte at 32 + oi*32 + l
                let qs_byte = read_byte(bb + 32u + oi * 32u + l);
                let low2 = (qs_byte >> shift) & 3u;

                // hmask byte for weight index l within the group:
                //   block byte at l (same l for all si within same oi)
                let hmask_byte = read_byte(bb + l);
                let sub = select(4, 0, (hmask_byte & m) != 0u);
                let q = i32(low2) - sub;

                output[out_base + oi * 128u + si * 32u + l] = d_scale * f32(q);
            }
        }
    }
}
