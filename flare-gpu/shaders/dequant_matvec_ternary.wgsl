// Fused ternary (BitNet b1.58) dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = sum_j  ternary(raw[row, j]) * vec[b * cols + j]
// for each batch item b in [0, batch_size) and output row in [0, num_rows).
//
// Ternary encoding: 2 bits per weight, 4 weights per byte (LSB-first).
//   00 = 0, 01 = +1, 10 = -1, 11 = unused (treated as 0)
//
// No floating-point multiplication for weight application: pure add/sub.
// This makes the kernel integer-only for the weight decoding path.
//
// One byte encodes 4 weights. Bytes per row = ceil(cols / 4).
// Raw buffer is padded to u32 alignment by the Rust caller.
//
// Bindings (standard 4-entry layout):
//   binding 0: raw    - packed ternary weight data [num_rows * bytes_per_row, padded]
//   binding 1: vec    - f32 input matrix           [batch_size * cols]
//   binding 2: output - f32 result                 [batch_size * num_rows]
//   binding 3: params - uniform

@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> vec_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows:       u32,
    cols:           u32,
    batch_size:     u32,
    bytes_per_row:  u32,
}
@group(0) @binding(3) var<uniform> params: Params;

// Workgroup-shared partial sums for tree reduction.
var<workgroup> partials: array<f32, 64>;

// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn dequant_matvec_ternary(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let row = wid.x;
    let batch = wid.y;
    let tid = lid.x;

    if row >= params.num_rows || batch >= params.batch_size {
        return;
    }

    let row_byte_offset = row * params.bytes_per_row;
    let vec_offset = batch * params.cols;

    var acc: f32 = 0.0;

    // Each thread processes a strided subset of columns.
    // Process 4 weights per byte.
    let total_bytes = params.bytes_per_row;

    // Stride over bytes: each thread handles bytes tid, tid+64, tid+128, ...
    var byte_idx = tid;
    loop {
        if byte_idx >= total_bytes {
            break;
        }

        let packed_byte = read_byte(row_byte_offset + byte_idx);
        let base_col = byte_idx * 4u;

        // Unpack 4 ternary weights from this byte
        for (var shift = 0u; shift < 4u; shift = shift + 1u) {
            let col = base_col + shift;
            if col >= params.cols {
                break;
            }
            let bits = (packed_byte >> (shift * 2u)) & 3u;
            if bits == 1u {
                // +1: add input value
                acc = acc + vec_input[vec_offset + col];
            } else if bits == 2u {
                // -1: subtract input value
                acc = acc - vec_input[vec_offset + col];
            }
            // bits == 0 or 3: skip (zero weight)
        }

        byte_idx = byte_idx + 64u;
    }

    // Tree reduction over workgroup shared memory
    partials[tid] = acc;
    workgroupBarrier();

    // Reduce 64 -> 1
    if tid < 32u { partials[tid] = partials[tid] + partials[tid + 32u]; }
    workgroupBarrier();
    if tid < 16u { partials[tid] = partials[tid] + partials[tid + 16u]; }
    workgroupBarrier();
    if tid < 8u { partials[tid] = partials[tid] + partials[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u { partials[tid] = partials[tid] + partials[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u { partials[tid] = partials[tid] + partials[tid + 2u]; }
    workgroupBarrier();
    if tid == 0u {
        output[batch * params.num_rows + row] = partials[0u] + partials[1u];
    }
}
