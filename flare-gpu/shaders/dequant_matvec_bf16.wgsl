// Fused BF16 dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  bf16_to_f32(raw[row, j]) * vec[b * num_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// BF16 layout (2 bytes per weight, no block structure):
//   Each weight is a 16-bit bfloat16 value stored in little-endian order.
//   BF16 shares the same exponent and sign bits as F32; conversion is a
//   left-shift by 16: f32_bits = bf16_bits << 16.
//
// For a matrix of shape [num_rows, num_cols]:
//   - Total bytes = num_rows × num_cols × 2
//   - The shader param `num_blocks_per_row` carries `num_cols` (weights_per_block = 1)
//   - Byte offset of weight (row, j) = (row × num_cols + j) × 2
//
// 2 bytes per weight is not always u32-aligned.  The Rust caller pads the raw
// byte buffer to the nearest 4-byte multiple before upload.  Individual bytes
// are extracted from u32 words via the `read_byte()` helper.
//
// One workgroup per (output row, batch item).  64 threads stride over columns;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — BF16 weight data [num_rows × num_cols × 2 bytes, padded]
//   binding 1: vec    — f32 input matrix [batch_size × num_cols]
//   binding 2: output — f32 result       [batch_size × num_rows]
//   binding 3: params — uniform

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows:           u32,
    num_blocks_per_row: u32, // = num_cols (one "block" per weight for BF16)
    batch_size:         u32,
}
@group(0) @binding(3) var<uniform> params: Params;

/// Workgroup-shared partial sums for the tree reduction.
var<workgroup> partials: array<f32, 64>;

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

/// Convert a BF16 value (LE u16 bits) to f32.
/// BF16 is the top 16 bits of F32, so conversion is a left-shift by 16.
fn bf16_to_f32(bf16_bits: u32) -> f32 {
    return bitcast<f32>(bf16_bits << 16u);
}

@compute @workgroup_size(64)
fn dequant_matvec_bf16(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let row   = wid.x;
    let batch = wid.y;
    if row >= params.num_rows || batch >= params.batch_size {
        return;
    }

    let tid      = lid.x;
    let num_cols = params.num_blocks_per_row;
    var acc: f32 = 0.0;

    // Each thread strides across columns in this row.
    var j: u32 = tid;
    loop {
        if j >= num_cols {
            break;
        }

        // BF16 weight at (row, j): byte offset = (row * num_cols + j) * 2
        let byte_off  = (row * num_cols + j) * 2u;
        let lo        = read_byte(byte_off);
        let hi        = read_byte(byte_off + 1u);
        let bf16_bits = lo | (hi << 8u);
        let w         = bf16_to_f32(bf16_bits);

        acc = acc + w * vec[batch * num_cols + j];

        j = j + 64u;
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
