// Fused Q5_1 dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  dequant(raw[row, j]) * vec[b * in_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// Q5_1 block layout (24 bytes = 6 u32 per block, 32 weights per block):
//   bytes  0-1:  d (f16 LE) — scale
//   bytes  2-3:  m (f16 LE) — min offset
//   bytes  4-7:  qh (u32 LE) — high bits for all 32 weights
//                  bits  0-15: high bit for weight j        (j in 0..16, low nibbles)
//                  bits 16-31: high bit for weight j+16     (j in 0..16, high nibbles)
//   bytes  8-23: qs[16] — 4-bit nibbles, 2 weights per byte
//                  byte k: lo nibble = weight k, hi nibble = weight k+16
//
// 24 bytes is u32-aligned, so no padding is required.  Direct u32 reads are used.
//
// Weight reconstruction:
//   for j in 0..16:
//     lo_nibble = qs[j] & 0x0F
//     hi_nibble = (qs[j] >> 4) & 0x0F
//     xh_0 = (qh >> j)      & 1
//     xh_1 = (qh >> (j+16)) & 1
//     q5_0 = lo_nibble | (xh_0 << 4)   → range [0, 31]
//     q5_1 = hi_nibble | (xh_1 << 4)   → range [0, 31]
//     w[j]    = d * f32(q5_0) + m
//     w[j+16] = d * f32(q5_1) + m
//
// u32 layout of a block at raw[base .. base+6]:
//   raw[base+0]: bits[15:0]=d, bits[31:16]=m  → unpack2x16float gives (d, m)
//   raw[base+1]: qh
//   raw[base+2..5]: qs[16] as 4 bytes per u32, 4 qs entries per u32
//
// One workgroup per (output row, batch item). 64 threads stride over blocks;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — packed Q5_1 weight data [num_rows * num_blocks_per_row * 6 u32]
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

@compute @workgroup_size(64)
fn dequant_matvec_q5_1(
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

        // Base offset in the u32 array (6 u32 per block).
        let base = (row * params.num_blocks_per_row + b) * 6u;
        // Corresponding start position in the input vector for this batch item.
        let vec_base = batch * in_cols + b * 32u;

        // d and m packed as two f16 in raw[base+0]: unpack gives (d, m).
        let dm = unpack2x16float(raw[base]);
        let d  = dm.x;
        let m  = dm.y;

        // qh at raw[base+1].
        let qh = raw[base + 1u];

        // qs[16] at raw[base+2..5]: 4 bytes per u32, packed as 2 nibbles per byte.
        for (var j = 0u; j < 16u; j = j + 1u) {
            // qs[j] is byte j within raw[base+2 .. base+5].
            let qs_word = raw[base + 2u + j / 4u];
            let qs_byte = (qs_word >> ((j % 4u) * 8u)) & 0xFFu;

            let lo_nibble = qs_byte & 0x0Fu;
            let hi_nibble = (qs_byte >> 4u) & 0x0Fu;

            let xh_0 = (qh >> j) & 1u;
            let xh_1 = (qh >> (j + 16u)) & 1u;

            let q5_0 = f32(lo_nibble | (xh_0 << 4u));
            let q5_1 = f32(hi_nibble | (xh_1 << 4u));

            let w0 = d * q5_0 + m;
            let w1 = d * q5_1 + m;

            acc = acc + w0 * vec[vec_base + j];
            acc = acc + w1 * vec[vec_base + j + 16u];
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
