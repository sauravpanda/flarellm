// Fused SiLU-Mul + Q8_0 dequantize matrix-vector multiply.
//
// Combines two operations into a single dispatch:
//   1. SiLU activation + element-wise multiply: intermediate[j] = SiLU(gate[j]) * up[j]
//   2. Q8_0 dequant matvec: output[row] = Σ_j dequant(raw[row, j]) * intermediate[j]
//
// This avoids materialising the intermediate buffer and saves one dispatch per layer
// in the FFN block (silu_mul + dequant_matvec_q8_0 → single fused kernel).
//
// Q8_0 block layout (34 bytes per block, 32 weights per block):
//   bytes  0-1:  scale (f16 LE)
//   bytes  2-33: qs[32] — signed int8 quantized values
//
// Weight reconstruction: w[i] = scale * i32(qs[i])
//
// One workgroup per output row.  64 threads stride over blocks;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings:
//   binding 0: gate   — f32 gate projection output  [in_cols]
//   binding 1: up     — f32 up projection output    [in_cols]
//   binding 2: raw    — packed Q8_0 weight data      [num_rows * num_blocks_per_row * 34 bytes, padded]
//   binding 3: output — f32 result                   [num_rows]
//   binding 4: params — uniform

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read> raw: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows:           u32,
    num_blocks_per_row: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

/// Workgroup-shared partial sums for the tree reduction.
var<workgroup> partials: array<f32, 64>;

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn fused_silu_mul_dequant_matvec_q8_0(
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

        // Absolute byte offset of this block in the raw buffer (34 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 34u;
        // Corresponding start position in the input vectors for this block.
        let vec_base = b * 32u;

        // Scale: f16 LE at bytes 0-1 of the block.
        let scale_bits = read_byte(bb) | (read_byte(bb + 1u) << 8u);
        let scale = unpack2x16float(scale_bits).x;

        // qs[32]: signed int8 values at bytes 2-33 of the block.
        for (var k = 0u; k < 32u; k = k + 1u) {
            let byte_val = read_byte(bb + 2u + k);
            let q = i32(byte_val << 24u) >> 24;

            // Fused SiLU(gate) * up — computed inline instead of from a temp buffer.
            let idx = vec_base + k;
            let g = gate[idx];
            let silu_val = g / (1.0 + exp(-g));
            let input_val = silu_val * up[idx];

            acc = acc + scale * f32(q) * input_val;
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
