// Fused Q5_K dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  dequant(raw[row, j]) * vec[b * in_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// Q5_K block layout (176 bytes = 44 u32 per block, 256 weights per block):
//   u32[0]      : d (f16 LE, lower 16 bits) + dmin (f16 LE, upper 16 bits)
//   u32[1..4]   : scales[12] — 8 (scale, min) pairs packed in 12 bytes
//   u32[4..12]  : qh[32]     — 1 high bit per weight (2 × 128-weight groups)
//   u32[12..44] : ql[128]    — low 4 bits per weight; ql[j] & 0xF → weight j lo nibble,
//                              ql[j] >> 4 → weight j+128 lo nibble
//
// Scale decoding (identical to Q4_K):
//   sc[i]     = scales_raw[i]   & 0x3F
//   mn[i]     = scales_raw[i+4] & 0x3F
//   sc[i+4]   = (scales_raw[i] >> 6) | ((scales_raw[i+8] & 0x0F) << 2)
//   mn[i+4]   = (scales_raw[i+4] >> 6) | ((scales_raw[i+8] >> 4) << 2)
//
// Weight reconstruction (for j in 0..128):
//   qh_lo_bit = (qh[j/8] >> (j%8)) & 1
//   qh_hi_bit = (qh[j/8 + 16] >> (j%8)) & 1
//   q_lo = (ql[j] & 0x0F) | (qh_lo_bit << 4)   — 5-bit value
//   q_hi = ((ql[j] >> 4) & 0x0F) | (qh_hi_bit << 4)
//   w[j]     = d * sc[j/32]   * q_lo - dmin * mn[j/32]
//   w[j+128] = d * sc[j/32+4] * q_hi - dmin * mn[j/32+4]
//
// One workgroup per (output row, batch item). 64 threads stride across blocks;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings (standard 4-entry layout: 2 read-only, 1 read-write, 1 uniform):
//   binding 0: raw    — packed Q5_K weight data [num_rows * num_blocks_per_row * 44]
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

@compute @workgroup_size(64)
fn dequant_matvec_q5k(
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

    // Each thread strides across blocks in this row (64-thread stride).
    var b: u32 = tid;
    loop {
        if b >= params.num_blocks_per_row {
            break;
        }

        // Base offset for this block in the raw u32 array (44 u32 per block).
        let raw_base = (row * params.num_blocks_per_row + b) * 44u;
        // Corresponding start position in the input matrix for this batch item.
        let vec_base = batch * in_cols + b * 256u;

        // d and dmin packed as two f16 in the first u32.
        let dm   = unpack2x16float(raw[raw_base]);
        let d    = dm.x;
        let dmin = dm.y;

        // Decode 8 scale/min pairs from scales_raw[0..12] = raw[raw_base+1 .. raw_base+4].
        var sc: array<u32, 8>;
        var mn: array<u32, 8>;
        for (var i = 0u; i < 4u; i = i + 1u) {
            let sr_i  = (raw[raw_base + 1u] >> (i * 8u)) & 0xFFu;
            let sr_i4 = (raw[raw_base + 2u] >> (i * 8u)) & 0xFFu;
            let sr_i8 = (raw[raw_base + 3u] >> (i * 8u)) & 0xFFu;
            sc[i]      = sr_i & 0x3Fu;
            mn[i]      = sr_i4 & 0x3Fu;
            sc[i + 4u] = (sr_i >> 6u) | ((sr_i8 & 0x0Fu) << 2u);
            mn[i + 4u] = (sr_i4 >> 6u) | ((sr_i8 >> 4u) << 2u);
        }

        // Process ql[128] + qh[32]: each ql byte yields one lo-nibble (weight j)
        // and one hi-nibble (weight j+128); qh contributes the 5th bit.
        for (var j = 0u; j < 128u; j = j + 1u) {
            // ql byte j: u32[12 + j/4], byte (j%4)
            let ql_u32  = raw[raw_base + 12u + j / 4u];
            let ql_byte = (ql_u32 >> ((j % 4u) * 8u)) & 0xFFu;
            let lo_nibble = ql_byte & 0x0Fu;
            let hi_nibble = (ql_byte >> 4u) & 0x0Fu;

            // qh bit for weight j (lo): qh byte at index j/8 → u32[4 + j/32], byte (j/8)%4.
            let qh_lo_u32  = raw[raw_base + 4u + j / 32u];
            let qh_lo_byte = (qh_lo_u32 >> (((j / 8u) % 4u) * 8u)) & 0xFFu;
            let qh_lo_bit  = (qh_lo_byte >> (j % 8u)) & 1u;

            // qh bit for weight j+128 (hi): qh byte at j/8+16 → u32[8 + j/32], same byte pos.
            let qh_hi_u32  = raw[raw_base + 8u + j / 32u];
            let qh_hi_byte = (qh_hi_u32 >> (((j / 8u) % 4u) * 8u)) & 0xFFu;
            let qh_hi_bit  = (qh_hi_byte >> (j % 8u)) & 1u;

            // Assemble 5-bit quantized values.
            let q_lo = lo_nibble | (qh_lo_bit << 4u);
            let q_hi = hi_nibble | (qh_hi_bit << 4u);
            let sub  = j / 32u;

            let w_lo = d * f32(sc[sub])      * f32(q_lo) - dmin * f32(mn[sub]);
            let w_hi = d * f32(sc[sub + 4u]) * f32(q_hi) - dmin * f32(mn[sub + 4u]);

            acc = acc + w_lo * vec[vec_base + j];
            acc = acc + w_hi * vec[vec_base + j + 128u];
        }

        b = b + 64u;
    }

    partials[tid] = acc;
    workgroupBarrier();

    // Parallel tree reduction: 64 → 1.
    if tid < 32u { partials[tid] = partials[tid] + partials[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { partials[tid] = partials[tid] + partials[tid + 16u]; }
    workgroupBarrier();
    if tid <  8u { partials[tid] = partials[tid] + partials[tid +  8u]; }
    workgroupBarrier();
    if tid <  4u { partials[tid] = partials[tid] + partials[tid +  4u]; }
    workgroupBarrier();
    if tid <  2u { partials[tid] = partials[tid] + partials[tid +  2u]; }
    workgroupBarrier();
    if tid <  1u { partials[tid] = partials[tid] + partials[tid +  1u]; }
    workgroupBarrier();

    if tid == 0u {
        output[batch * params.num_rows + row] = partials[0u];
    }
}
