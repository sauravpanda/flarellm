// Fused IQ4_NL dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  dequant(raw[row, j]) * vec[b * in_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// IQ4_NL block layout (18 bytes per block, 32 weights per block):
//   bytes 0-1:  scale (f16 LE)
//   bytes 2-17: qs[16] — 4-bit nibbles; byte k → (weight[2k], weight[2k+1])
//                         lo nibble = weight[2k], hi nibble = weight[2k+1]
//
// Weight reconstruction: w[i] = scale * kvalues_iq4nl[nibble]
// where kvalues_iq4nl is the 16-entry signed lookup table from llama.cpp.
//
// Unlike Q4_0 (which uses `q - 8`), IQ4_NL uses a neural-network-optimized
// non-uniform codebook for better quality at the same bit width.
//
// 18 bytes is not u32-aligned.  The Rust caller pads the raw byte buffer to
// the nearest 4-byte multiple before upload.  Weights are accessed via
// byte-offset helpers that extract individual bytes from the u32 storage array.
//
// One workgroup per (output row, batch item). 64 threads split the blocks;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — packed IQ4_NL weight data [num_rows * num_blocks_per_row * 18 bytes, padded]
//   binding 1: vec    — f32 input matrix           [batch_size * num_blocks_per_row * 32]
//   binding 2: output — f32 result                 [batch_size * num_rows]
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

/// IQ4_NL lookup table: maps a 4-bit nibble [0, 15] to a signed quantized value.
/// Source: llama.cpp kvalues_iq4nl.
const KVALUES_IQ4NL: array<i32, 16> = array<i32, 16>(
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
);

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn dequant_matvec_iq4nl(
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

        // Absolute byte offset of this block in the raw buffer (18 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 18u;
        // Corresponding start position in the input matrix for this batch item.
        let vec_base = batch * in_cols + b * 32u;

        // Scale: f16 LE at bytes 0-1 of the block.
        let scale_bits = read_byte(bb) | (read_byte(bb + 1u) << 8u);
        let scale = unpack2x16float(scale_bits).x;

        // Process qs[16]: bytes 2-17 of the block.
        for (var k = 0u; k < 16u; k = k + 1u) {
            let byte_val = read_byte(bb + 2u + k);
            let lo_nibble = byte_val & 0x0Fu;
            let hi_nibble = (byte_val >> 4u) & 0x0Fu;
            let w_lo = scale * f32(KVALUES_IQ4NL[lo_nibble]);
            let w_hi = scale * f32(KVALUES_IQ4NL[hi_nibble]);
            acc = acc + w_lo * vec[vec_base + k * 2u];
            acc = acc + w_hi * vec[vec_base + k * 2u + 1u];
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
