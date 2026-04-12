// Fused Q2_K dequantize + matrix-vector multiply on GPU.
//
// Computes: output[row] = Σ_j  dequant(raw[row, j]) * vec[j]
//
// Q2_K block layout (84 bytes = 21 u32 per block, 256 weights per block):
//   u32[0..16]  : qs[64]    — 2-bit quantized values, 4 weights per byte
//   u32[16..20] : scales[16]— per-sub-block nibbles: low nibble = scale, high nibble = min
//   u32[20]     : d (f16 LE, lower 16 bits) + dmin (f16 LE, upper 16 bits)
//
// Weight reconstruction: w[i] = d * scale_nibble * q2 − dmin * min_nibble
//   where:
//     sub          = i / 16             (sub-block index, 0..15)
//     scale_nibble = scales[sub] & 0xF
//     min_nibble   = scales[sub] >> 4
//     q2           = (qs[i/4] >> ((i % 4) * 2)) & 3
//
// One workgroup per output row. 64 threads split the blocks in the row;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings (standard 4-entry layout: 2 read-only, 1 read-write, 1 uniform):
//   binding 0: raw    — packed Q2_K weight data [num_rows * num_blocks_per_row * 21]
//   binding 1: vec    — f32 input vector        [num_blocks_per_row * 256]
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
fn dequant_matvec_q2k(
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

        // Base offset for this block in the raw u32 array (21 u32 per block).
        let raw_base = (row * params.num_blocks_per_row + b) * 21u;
        // Corresponding start in the input vector (256 elements per block).
        let vec_base = b * 256u;

        // d and dmin packed as two f16 in u32[20].
        let dm   = unpack2x16float(raw[raw_base + 20u]);
        let d    = dm.x;
        let dmin = dm.y;

        // Iterate over 16 sub-blocks; each covers 16 weights from 4 qs bytes.
        for (var sub = 0u; sub < 16u; sub = sub + 1u) {
            // scales[16] sit in u32[16..20].
            // Byte `sub` is at u32[16 + sub/4], bit offset (sub%4)*8.
            let scale_byte   = (raw[raw_base + 16u + sub / 4u] >> ((sub % 4u) * 8u)) & 0xFFu;
            let scale_nibble = f32(scale_byte & 0x0Fu);
            let min_nibble   = f32((scale_byte >> 4u) & 0x0Fu);

            // 4 qs bytes for this sub-block (qs byte indices sub*4 .. sub*4+3).
            for (var qi = 0u; qi < 4u; qi = qi + 1u) {
                let k = sub * 4u + qi; // qs byte index (0..63)
                // qs[64] sit in u32[0..16].
                // Byte k is at u32[k/4], bit offset (k%4)*8.
                let qs_val = (raw[raw_base + k / 4u] >> ((k % 4u) * 8u)) & 0xFFu;

                // 4 weights per qs byte; bits 2*bit .. 2*bit+1.
                for (var bit = 0u; bit < 4u; bit = bit + 1u) {
                    let q2  = f32((qs_val >> (bit * 2u)) & 3u);
                    let w   = d * scale_nibble * q2 - dmin * min_nibble;
                    let idx = sub * 16u + qi * 4u + bit;
                    acc = acc + w * vec[vec_base + idx];
                }
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
