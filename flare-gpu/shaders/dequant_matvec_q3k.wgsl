// Fused Q3_K dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  dequant(raw[row, j]) * vec[b * in_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// Q3_K block layout (110 bytes per block, 256 weights per block):
//   bytes   0-31:  hmask[32]  — high bit (bit 2) for each weight, 8 per byte
//   bytes  32-95:  qs[64]     — low 2 bits, 4 weights per byte
//   bytes  96-107: scales[12] — 8 sub-block scales (6-bit) via kmask transform
//   bytes 108-109: d (f16 LE)
//
// 110 bytes is not u32-aligned.  The Rust caller pads the raw byte buffer to the
// nearest 4-byte multiple before upload.  Weights are accessed via byte-offset
// helpers that extract individual bytes from the u32 storage array.
//
// Scale decoding (kmask transform, b = block bytes 96..108):
//   scales[0..3] : (b[k] & 0x0F) | (((b[8+k]  >> 4) & 3) << 4) − 32  for k in 0..3
//   scales[4..7] : (b[k] & 0x0F) | (((b[8+k-4] >> 6) & 3) << 4) − 32  for k in 4..7
//   i.e. scales[k+4] = (b[k+4] & 0x0F) | (((b[8+k] >> 6) & 3) << 4) − 32
//
// Weight reconstruction:
//   8 sub-blocks, each 32 weights.  Indexed as oi=0..1 (outer), si=0..3 (inner).
//   scale_idx = oi*4 + si
//   m = 1 << scale_idx  — hmask bit selector (bits cycle through bit 0..7 of each byte)
//   shift = si*2        — bit position in qs byte for this sub-block
//   For l in 0..32:
//     qs_byte   = block byte at 32 + oi*32 + l
//     low2      = (qs_byte >> shift) & 3
//     hmask_bit = (block byte at l) & m
//     sub       = if hmask_bit != 0 { 0 } else { 4 }
//     q         = i32(low2) - sub   — range [−4, 3]
//     w         = d * scales[scale_idx] * q
//
// One workgroup per (output row, batch item). 64 threads stride over blocks;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — packed Q3_K weight data [num_rows * num_blocks_per_row * 110 bytes, padded]
//   binding 1: vec    — f32 input matrix        [batch_size * num_blocks_per_row * 256]
//   binding 2: output — f32 result              [batch_size * num_rows]
//   binding 3: params — uniform

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows:           u32,
    num_blocks_per_row: u32,
    batch_size:         u32,
}
@group(0) @binding(3) var<uniform> params: Params;

/// Workgroup-shared partial sums for the tree reduction.
var<workgroup> partials: array<f32, 64>;

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn dequant_matvec_q3k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let row   = wid.x;
    let batch = wid.y;
    if row >= params.num_rows || batch >= params.batch_size {
        return;
    }

    let tid = lid.x;
    let in_cols = params.num_blocks_per_row * 256u;
    var acc: f32 = 0.0;

    // Each thread strides across blocks in this row.
    var b: u32 = tid;
    loop {
        if b >= params.num_blocks_per_row {
            break;
        }

        // Absolute byte offset of this block in the raw buffer (110 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 110u;
        // Corresponding start position in the input vector for this batch item.
        let vec_base = batch * in_cols + b * 256u;

        // d at bytes 108-109 of this block (LE f16).
        let d_packed = read_byte(bb + 108u) | (read_byte(bb + 109u) << 8u);
        let d = unpack2x16float(d_packed).x;

        // Decode 8 scales from block bytes 96..108 using kmask transform.
        var scales: array<i32, 8>;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let bk  = read_byte(bb + 96u + k);
            let bk4 = read_byte(bb + 96u + k + 4u);
            let bk8 = read_byte(bb + 96u + k + 8u);
            scales[k]      = i32((bk  & 0x0Fu) | (((bk8 >> 4u) & 3u) << 4u)) - 32;
            scales[k + 4u] = i32((bk4 & 0x0Fu) | (((bk8 >> 6u) & 3u) << 4u)) - 32;
        }

        // 8 sub-blocks: oi in 0..1 (outer), si in 0..3 (inner), 32 weights each.
        for (var oi = 0u; oi < 2u; oi = oi + 1u) {
            for (var si = 0u; si < 4u; si = si + 1u) {
                let scale_idx = oi * 4u + si;
                let m         = 1u << scale_idx; // hmask bit selector
                let shift     = si * 2u;         // bit position in qs byte
                let d_scale   = d * f32(scales[scale_idx]);

                for (var l = 0u; l < 32u; l = l + 1u) {
                    // qs byte: block byte at 32 + oi*32 + l
                    let qs_byte  = read_byte(bb + 32u + oi * 32u + l);
                    let low2     = (qs_byte >> shift) & 3u;

                    // hmask byte: block byte at l (same l for all si in this oi)
                    let hmask_byte = read_byte(bb + l);
                    let sub = select(4, 0, (hmask_byte & m) != 0u);
                    let q   = i32(low2) - sub; // range [-4, 3]

                    let w_idx = oi * 128u + si * 32u + l;
                    acc = acc + d_scale * f32(q) * vec[vec_base + w_idx];
                }
            }
        }

        b = b + 64u;
    }

    partials[tid] = acc;
    workgroupBarrier();

    // Parallel tree reduction: 64 → 1.
    if tid < 32u {
        partials[tid] = partials[tid] + partials[tid + 32u];
    }
    workgroupBarrier();
    if tid < 16u {
        partials[tid] = partials[tid] + partials[tid + 16u];
    }
    workgroupBarrier();
    if tid < 8u {
        partials[tid] = partials[tid] + partials[tid + 8u];
    }
    workgroupBarrier();
    if tid < 4u {
        partials[tid] = partials[tid] + partials[tid + 4u];
    }
    workgroupBarrier();
    if tid < 2u {
        partials[tid] = partials[tid] + partials[tid + 2u];
    }
    workgroupBarrier();
    if tid < 1u {
        partials[tid] = partials[tid] + partials[tid + 1u];
    }
    workgroupBarrier();

    if tid == 0u {
        output[batch * params.num_rows + row] = partials[0u];
    }
}
