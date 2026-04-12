// Fused Q4_K dequantize + matrix-vector multiply on GPU.
//
// Computes: output[row] = Σ_j  dequant(raw[row, j]) * vec[j]
//
// Q4_K block layout (144 bytes = 36 u32 per block, 256 weights per block):
//   u32[0]     : d (f16 LE, lower 16 bits) + dmin (f16 LE, upper 16 bits)
//   u32[1..4]  : scales[12] — 8 (scale, min) pairs packed in 12 bytes
//   u32[4..36] : qs[128] — 4-bit nibbles; low nibbles → weights 0..127,
//                           high nibbles → weights 128..255
//
// Scale decoding (same as dequant_q4k.wgsl):
//   sc[i]     = scales_raw[i]   & 0x3F
//   mn[i]     = scales_raw[i+4] & 0x3F
//   sc[i+4]   = (scales_raw[i] >> 6) | ((scales_raw[i+8] & 0x0F) << 2)
//   mn[i+4]   = (scales_raw[i+4] >> 6) | ((scales_raw[i+8] >> 4) << 2)
//
// Weight reconstruction: w[j]     = d * sc[j/32]   * lo_nibble - dmin * mn[j/32]
//                         w[j+128] = d * sc[j/32+4] * hi_nibble - dmin * mn[j/32+4]
//
// One workgroup per output row. 64 threads split the blocks in the row;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings match the standard 4-entry layout (2 read-only, 1 read-write, 1 uniform):
//   binding 0: raw   — packed Q4_K weight data [num_rows * num_blocks_per_row * 36]
//   binding 1: vec   — f32 input vector        [num_blocks_per_row * 256]
//   binding 2: output — f32 result             [num_rows]
//   binding 3: params — uniform

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows: u32,
    num_blocks_per_row: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

/// Workgroup-shared partial sums for the tree reduction.
var<workgroup> partials: array<f32, 64>;

@compute @workgroup_size(64)
fn dequant_matvec_q4k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if row >= params.num_rows {
        return;
    }

    let tid = lid.x;
    var acc: f32 = 0.0;

    // Each thread strides across blocks in this row.
    var b: u32 = tid;
    loop {
        if b >= params.num_blocks_per_row {
            break;
        }

        // Base offset for this block in the raw u32 array (36 u32 per block).
        let raw_base = (row * params.num_blocks_per_row + b) * 36u;
        // Corresponding start position in the input vector (256 elements per block).
        let vec_base = b * 256u;

        // d and dmin packed as two f16 in the first u32 of the block.
        let dm = unpack2x16float(raw[raw_base]);
        let d = dm.x;
        let dmin = dm.y;

        // Decode 8 scale/min pairs from scales_raw[0..12].
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

        // Process qs[128]: each byte yields two 4-bit nibbles → two weights.
        // low nibbles give weights 0..127, high nibbles give weights 128..255.
        for (var j = 0u; j < 128u; j = j + 1u) {
            let q_u32    = raw[raw_base + 4u + j / 4u];
            let byte_val = (q_u32 >> ((j % 4u) * 8u)) & 0xFFu;
            let lo = f32(byte_val & 0x0Fu);
            let hi = f32((byte_val >> 4u) & 0x0Fu);
            let sub = j / 32u;

            let w_lo = d * f32(sc[sub])      * lo - dmin * f32(mn[sub]);
            let w_hi = d * f32(sc[sub + 4u]) * hi - dmin * f32(mn[sub + 4u]);

            acc = acc + w_lo * vec[vec_base + j];
            acc = acc + w_hi * vec[vec_base + j + 128u];
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
        output[row] = partials[0u];
    }
}
