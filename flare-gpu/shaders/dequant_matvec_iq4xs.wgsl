// Fused IQ4_XS dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  dequant(raw[row, j]) * vec[b * in_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// IQ4_XS block layout (136 bytes per block, 256 weights per block — GGUF type 22):
//   bytes 0-1:   d       — f16 LE super-block scale
//   bytes 2-3:   scales_h — u16 LE, 2 high bits of each group scale (8 × 2-bit)
//   bytes 4-7:   scales_l[4] — 4 × u8, packed 4-bit low nibbles (2 groups per byte)
//   bytes 8-135: qs[128] — 128 × u8, packed 4-bit indices into KVALUES_IQ4NL
//
// For each ib32 ∈ [0, 7] (groups of 32 weights):
//   ls_lo = (scales_l[ib32/2] >> (4 * (ib32 % 2))) & 0xF   — 4 low bits
//   ls_hi = (scales_h >> (2 * ib32)) & 3                    — 2 high bits
//   ls    = ls_lo | (ls_hi << 4)                             — 6-bit scale [0, 63]
//   dl    = d * f32(i32(ls) - 32)
//   For j ∈ [0, 15]:  (qs byte at 8 + 16*ib32 + j)
//     y[j+0]  = dl * KVALUES_IQ4NL[qs[j] & 0xF]
//     y[j+16] = dl * KVALUES_IQ4NL[qs[j] >> 4]
//
// Block size (136 bytes) is u32-aligned — no padding required.
//
// One workgroup per (output row, batch item). 64 threads split the 8 groups;
// a tree reduction accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — packed IQ4_XS weight data
//   binding 1: vec    — f32 input matrix  [batch_size * num_blocks_per_row * 256]
//   binding 2: output — f32 result        [batch_size * num_rows]
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

var<workgroup> partials: array<f32, 64>;

// kvalues_iq4nl[16]: dequantization lookup table for IQ4_NL / IQ4_XS.
// Source: llama.cpp ggml-common.h
const KVALUES_IQ4NL: array<f32, 16> = array<f32, 16>(
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0,
       1.0,   13.0,  25.0,  38.0,  53.0,  69.0,  89.0, 113.0
);

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

/// Read a u16 (LE) from two consecutive bytes at byte_offset.
fn read_u16(byte_offset: u32) -> u32 {
    return read_byte(byte_offset) | (read_byte(byte_offset + 1u) << 8u);
}

@compute @workgroup_size(64)
fn dequant_matvec_iq4xs(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let row   = wid.x;
    let batch = wid.y;
    if row >= params.num_rows || batch >= params.batch_size {
        return;
    }

    let tid     = lid.x;
    let in_cols = params.num_blocks_per_row * 256u; // 256 weights per block
    var acc: f32 = 0.0;

    // Each block is 136 bytes; threads stride across blocks.
    var b: u32 = tid;
    loop {
        if b >= params.num_blocks_per_row {
            break;
        }

        // Byte offset of this block (136 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 136u;

        // d: f16 LE at bytes 0-1.
        let d_bits = read_byte(bb) | (read_byte(bb + 1u) << 8u);
        let d = unpack2x16float(d_bits).x;

        // scales_h: u16 LE at bytes 2-3.
        let scales_h = read_u16(bb + 2u);

        // vec base for this block and batch.
        let vec_base = batch * in_cols + b * 256u;

        // Process 8 groups of 32 weights (ib32 = 0..7).
        for (var ib32 = 0u; ib32 < 8u; ib32 = ib32 + 1u) {
            // scales_l nibble: byte 4 + ib32/2, select lo or hi nibble.
            let sl_byte = read_byte(bb + 4u + ib32 / 2u);
            let ls_lo = (sl_byte >> ((ib32 % 2u) * 4u)) & 0xFu;
            let ls_hi = (scales_h >> (ib32 * 2u)) & 3u;
            let ls = ls_lo | (ls_hi << 4u);
            let dl = d * f32(i32(ls) - 32);

            // qs base: byte 8 + 16*ib32.
            let qs_base = bb + 8u + 16u * ib32;
            let vec_group = vec_base + ib32 * 32u;

            for (var j = 0u; j < 16u; j = j + 1u) {
                let byte_val = read_byte(qs_base + j);
                let lo_idx = byte_val & 0xFu;
                let hi_idx = byte_val >> 4u;
                acc = acc + dl * KVALUES_IQ4NL[lo_idx] * vec[vec_group + j];
                acc = acc + dl * KVALUES_IQ4NL[hi_idx] * vec[vec_group + j + 16u];
            }
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
    if tid < 8u  { partials[tid] = partials[tid] + partials[tid + 8u]; }
    workgroupBarrier();
    if tid < 4u  { partials[tid] = partials[tid] + partials[tid + 4u]; }
    workgroupBarrier();
    if tid < 2u  { partials[tid] = partials[tid] + partials[tid + 2u]; }
    workgroupBarrier();
    if tid < 1u  { partials[tid] = partials[tid] + partials[tid + 1u]; }
    workgroupBarrier();

    if tid == 0u {
        output[batch * params.num_rows + row] = partials[0u];
    }
}
