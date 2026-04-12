// Fused Q8_1 dequantize + matrix-vector multiply on GPU.
//
// Computes: output[row] = Σ_j  dequant(raw[row, j]) * vec[j]
//
// Q8_1 block layout (36 bytes = 9 u32 per block, 32 weights per block):
//   u32[0]     : d (f16 LE, lower 16 bits) + s (f16 LE sum, upper 16 bits — unused)
//   u32[1..9]  : qs[32] — signed int8 values, 4 per u32 in little-endian order
//
// Weight reconstruction: w[i] = d * qs[i], where qs[i] is a signed int8 in [-128, 127].
//
// Sign extension: extract byte as u32, shift left 24, then arithmetic-right-shift 24.
//
// One workgroup per output row. 64 threads split the blocks in the row;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings (standard 4-entry layout: 2 read-only, 1 read-write, 1 uniform):
//   binding 0: raw    — packed Q8_1 weight data [num_rows * num_blocks_per_row * 9]
//   binding 1: vec    — f32 input vector        [num_blocks_per_row * 32]
//   binding 2: output — f32 result              [num_rows]
//   binding 3: params — uniform

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows:           u32,
    num_blocks_per_row: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

/// Workgroup-shared partial sums for the tree reduction.
var<workgroup> partials: array<f32, 64>;

@compute @workgroup_size(64)
fn dequant_matvec_q8_1(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
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

        // Base offset for this block in the raw u32 array (9 u32 per block).
        let raw_base = (row * params.num_blocks_per_row + b) * 9u;
        // Corresponding start position in the input vector (32 elements per block).
        let vec_base = b * 32u;

        // d from lower 16 bits of u32[0]; s (upper 16 bits) is ignored.
        let d = unpack2x16float(raw[raw_base]).x;

        // Process qs[32]: 8 u32s each containing 4 signed bytes.
        for (var qi = 0u; qi < 8u; qi = qi + 1u) {
            let q_u32 = raw[raw_base + 1u + qi];
            for (var bi = 0u; bi < 4u; bi = bi + 1u) {
                let byte_u = (q_u32 >> (bi * 8u)) & 0xFFu;
                // Sign-extend from 8 bits to 32 bits via arithmetic right shift.
                let qs_i = bitcast<i32>(byte_u << 24u) >> 24;
                let w_i  = d * f32(qs_i);
                let pos  = qi * 4u + bi;
                acc = acc + w_i * vec[vec_base + pos];
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
        output[row] = partials[0u];
    }
}
