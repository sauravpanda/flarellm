// Fused Q4_0 dequantize + matrix-vector multiply on GPU.
//
// Computes: output[row] = Σ_j  dequant(raw[row, j]) * vec[j]
//
// Q4_0 block layout (18 bytes per block, 32 weights per block):
//   bytes 0-1:  scale (f16 LE)
//   bytes 2-17: qs[16] — 4-bit nibbles; byte k → (weight[2k], weight[2k+1])
//                         lo nibble = weight[2k], hi nibble = weight[2k+1]
//
// Weight reconstruction: w[i] = scale * (q - 8), where q ∈ [0, 15].
// The nibbles are unsigned in [0,15]; subtracting 8 centers them at zero.
//
// 18 bytes is not u32-aligned.  The Rust caller pads the raw byte buffer to
// the nearest 4-byte multiple before upload.  Weights are accessed via
// byte-offset helpers that extract individual bytes from the u32 storage array.
//
// One workgroup per output row. 64 threads split the blocks in the row;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings (standard 4-entry layout: 2 read-only, 1 read-write, 1 uniform):
//   binding 0: raw    — packed Q4_0 weight data [num_rows * num_blocks_per_row * 18 bytes, padded]
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

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn dequant_matvec_q4_0(
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

        // Absolute byte offset of this block in the raw buffer (18 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 18u;
        // Corresponding start in the input vector (32 elements per block).
        let vec_base = b * 32u;

        // Scale: f16 LE at bytes 0-1 of the block.
        // Read as a u16 from two consecutive bytes, then reinterpret as f16.
        let scale_bits = read_byte(bb) | (read_byte(bb + 1u) << 8u);
        let scale = unpack2x16float(scale_bits).x;

        // Process qs[16]: bytes 2-17 of the block.
        for (var k = 0u; k < 16u; k = k + 1u) {
            let byte_val = read_byte(bb + 2u + k);
            // Unsigned nibbles centered at 8: subtract 8 → signed range [-8, 7].
            let lo_i = i32(byte_val & 0x0Fu) - 8;
            let hi_i = i32((byte_val >> 4u) & 0x0Fu) - 8;
            let w_lo = scale * f32(lo_i);
            let w_hi = scale * f32(hi_i);
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
        output[row] = partials[0u];
    }
}
