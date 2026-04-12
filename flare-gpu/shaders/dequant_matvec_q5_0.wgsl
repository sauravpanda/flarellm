// Fused Q5_0 dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  dequant(raw[row, j]) * vec[b * in_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// Q5_0 block layout (22 bytes per block, 32 weights per block):
//   bytes  0-1:  scale (f16 LE)
//   bytes  2-5:  qh (u32 LE) — high bits for all 32 weights
//                  bits  0-15: high bit for weight j        (j in 0..16, low nibbles)
//                  bits 16-31: high bit for weight j+16     (j in 0..16, high nibbles)
//   bytes  6-21: qs[16] — 4-bit nibbles, 2 weights per byte
//                  byte k: lo nibble = weight k, hi nibble = weight k+16
//
// 22 bytes is not u32-aligned.  The Rust caller pads the raw byte buffer to the
// nearest 4-byte multiple before upload.  Weights are accessed via byte-offset
// helpers that extract individual bytes from the u32 storage array.
//
// Weight reconstruction:
//   for j in 0..16:
//     lo_nibble = qs[j] & 0x0F
//     hi_nibble = (qs[j] >> 4) & 0x0F
//     xh_0 = (qh >> j)      & 1
//     xh_1 = (qh >> (j+16)) & 1
//     x0 = i32(lo_nibble | (xh_0 << 4)) − 16   → weight[j]
//     x1 = i32(hi_nibble | (xh_1 << 4)) − 16   → weight[j+16]
//     w[j]    = scale * x0
//     w[j+16] = scale * x1
//
// One workgroup per (output row, batch item). 64 threads stride over blocks;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — packed Q5_0 weight data [num_rows * num_blocks_per_row * 22 bytes, padded]
//   binding 1: vec    — f32 input matrix        [batch_size * num_blocks_per_row * 32]
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
fn dequant_matvec_q5_0(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let row   = wid.x;
    let batch = wid.y;
    if row >= params.num_rows || batch >= params.batch_size {
        return;
    }

    let tid = lid.x;
    let in_cols = params.num_blocks_per_row * 32u;
    var acc: f32 = 0.0;

    // Each thread strides across blocks in this row.
    var b: u32 = tid;
    loop {
        if b >= params.num_blocks_per_row {
            break;
        }

        // Absolute byte offset of this block in the raw buffer (22 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 22u;
        // Corresponding start position in the input vector for this batch item.
        let vec_base = batch * in_cols + b * 32u;

        // Scale: f16 LE at bytes 0-1 of the block.
        let scale_bits = read_byte(bb) | (read_byte(bb + 1u) << 8u);
        let scale = unpack2x16float(scale_bits).x;

        // qh: u32 LE at bytes 2-5.
        let qh = read_byte(bb + 2u)
               | (read_byte(bb + 3u) << 8u)
               | (read_byte(bb + 4u) << 16u)
               | (read_byte(bb + 5u) << 24u);

        // qs[16]: bytes 6-21 of the block. Each byte holds two nibbles:
        //   lo nibble → weight j, hi nibble → weight j+16 (for j in 0..16).
        for (var j = 0u; j < 16u; j = j + 1u) {
            let qs_byte = read_byte(bb + 6u + j);

            let lo_nibble = qs_byte & 0x0Fu;
            let hi_nibble = (qs_byte >> 4u) & 0x0Fu;

            let xh_0 = (qh >> j) & 1u;
            let xh_1 = (qh >> (j + 16u)) & 1u;

            let x0 = i32(lo_nibble | (xh_0 << 4u)) - 16;
            let x1 = i32(hi_nibble | (xh_1 << 4u)) - 16;

            acc = acc + scale * f32(x0) * vec[vec_base + j];
            acc = acc + scale * f32(x1) * vec[vec_base + j + 16u];
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
