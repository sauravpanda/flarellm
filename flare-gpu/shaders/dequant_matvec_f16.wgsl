// Fused F16 dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  f16_to_f32(raw[row, j]) * vec[b * num_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// F16 layout (2 bytes per weight, no block structure):
//   Each weight is a 16-bit IEEE 754 half-precision float in little-endian byte order.
//   Conversion to F32 uses the WGSL built-in unpack2x16float(), which interprets
//   the lower 16 bits of a u32 as an F16 value and returns the F32 equivalent.
//
// For a matrix of shape [num_rows, num_cols]:
//   - Total bytes = num_rows × num_cols × 2
//   - The shader param `num_blocks_per_row` carries `num_cols` (weights_per_block = 1)
//   - Byte offset of weight (row, j) = (row × num_cols + j) × 2
//
// 2 bytes per weight may not be u32-aligned for odd num_cols.  The Rust caller
// pads the raw byte buffer to the nearest 4-byte multiple before upload.
// Individual bytes are extracted from u32 words via the `read_byte()` helper.
//
// One workgroup per (output row, batch item).  64 threads stride over columns;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — F16 weight data [num_rows × num_cols × 2 bytes, padded]
//   binding 1: vec    — f32 input matrix [batch_size × num_cols]
//   binding 2: output — f32 result       [batch_size × num_rows]
//   binding 3: params — uniform

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows:           u32,
    num_blocks_per_row: u32, // = num_cols (one "block" per weight for F16)
    batch_size:         u32,
}
@group(0) @binding(3) var<uniform> params: Params;

/// Workgroup-shared partial sums for the tree reduction.
var<workgroup> partials: array<f32, 64>;

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

/// Convert an F16 value (LE u16 bits packed in low 16 bits of a u32) to f32.
/// unpack2x16float interprets bits[0:15] as an F16 and returns it as f32.
fn f16_to_f32(f16_bits: u32) -> f32 {
    return unpack2x16float(f16_bits).x;
}

@compute @workgroup_size(64)
fn dequant_matvec_f16(
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

        // F16 weight at (row, j): byte offset = (row * num_cols + j) * 2
        let byte_off  = (row * num_cols + j) * 2u;
        let lo        = read_byte(byte_off);
        let hi        = read_byte(byte_off + 1u);
        let f16_bits  = lo | (hi << 8u);
        let w         = f16_to_f32(f16_bits);

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
