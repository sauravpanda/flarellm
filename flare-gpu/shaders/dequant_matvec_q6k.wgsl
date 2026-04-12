// Fused Q6_K dequantize + matrix-vector multiply on GPU.
//
// Computes: output[row] = Σ_j  dequant(raw[row, j]) * vec[j]
//
// Q6_K block layout (210 bytes per block, 256 weights per block):
//   bytes   0-127: ql[128] — lower 4 bits of 6-bit weights (two interleaved groups of 64)
//   bytes 128-191: qh[64]  — upper 2 bits of 6-bit weights
//   bytes 192-207: scales[16] — signed i8 scale per sub-block
//   bytes 208-209: d (f16 LE)
//
// 210 bytes is not u32-aligned.  The Rust caller pads the raw byte buffer to the
// nearest 4-byte multiple before upload.  Weights are accessed via byte-offset
// helpers that extract individual bytes from the u32 storage array.
//
// Weight reconstruction (mirrors dequant_q6k.wgsl):
//   For half in 0..2 (each covering 128 output elements):
//     ql_off = half*64, qh_off = half*32, sc_off = half*8, y_off = half*128
//     For l in 0..32:
//       ql_a = ql[ql_off+l], ql_b = ql[ql_off+l+32], qh_b = qh[qh_off+l]
//       q1 = (ql_a & 0xF)  | ((qh_b & 3)       << 4) − 32
//       q2 = (ql_b & 0xF)  | (((qh_b >> 2) & 3) << 4) − 32
//       q3 =  (ql_a >> 4)  | (((qh_b >> 4) & 3) << 4) − 32
//       q4 =  (ql_b >> 4)  | (((qh_b >> 6) & 3) << 4) − 32
//       is = l / 16
//       sc1..sc4 = sign_extend(scales[sc_off+is+{0,2,4,6}])
//       acc += d*sc1*q1 * vec[b*256 + y_off+l]
//            + d*sc2*q2 * vec[b*256 + y_off+l+32]
//            + d*sc3*q3 * vec[b*256 + y_off+l+64]
//            + d*sc4*q4 * vec[b*256 + y_off+l+96]
//
// One workgroup per output row. 64 threads stride across blocks;
// a tree reduction over workgroup-shared memory accumulates partial sums.
//
// Bindings (standard 4-entry layout: 2 read-only, 1 read-write, 1 uniform):
//   binding 0: raw    — packed Q6_K weight data [num_rows * num_blocks_per_row * 210 bytes, padded]
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

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

/// Reinterpret an 8-bit unsigned value as a signed 8-bit integer (i32).
fn sign_extend_byte(b: u32) -> i32 {
    return bitcast<i32>(b << 24u) >> 24u;
}

@compute @workgroup_size(64)
fn dequant_matvec_q6k(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let row = wid.x;
    if row >= params.num_rows {
        return;
    }

    let tid = lid.x;
    var acc: f32 = 0.0;

    // Each thread strides across blocks in this row (64-thread stride).
    var b: u32 = tid;
    loop {
        if b >= params.num_blocks_per_row {
            break;
        }

        // Byte offset for this block's start in the raw data (210 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 210u;
        // Start position in the input vector (256 elements per block).
        let vec_base = b * 256u;

        // d at bytes 208-209 of this block (LE f16).
        let d_packed = read_byte(bb + 208u) | (read_byte(bb + 209u) << 8u);
        let d = unpack2x16float(d_packed).x;

        // Process both halves (0 and 1), each covering 128 output elements.
        for (var half = 0u; half < 2u; half = half + 1u) {
            let ql_off = half * 64u;
            let qh_off = half * 32u;
            let sc_off = half * 8u;
            let y_off  = half * 128u;

            for (var l = 0u; l < 32u; l = l + 1u) {
                let ql_a = read_byte(bb + ql_off + l);
                let ql_b = read_byte(bb + ql_off + l + 32u);
                let qh_b = read_byte(bb + 128u + qh_off + l);

                // Assemble 6-bit unsigned values, then offset-binary subtract 32.
                let q1 = i32((ql_a & 0x0Fu) | ((qh_b & 3u) << 4u)) - 32;
                let q2 = i32((ql_b & 0x0Fu) | (((qh_b >> 2u) & 3u) << 4u)) - 32;
                let q3 = i32((ql_a >> 4u)   | (((qh_b >> 4u) & 3u) << 4u)) - 32;
                let q4 = i32((ql_b >> 4u)   | (((qh_b >> 6u) & 3u) << 4u)) - 32;

                // Signed i8 scales (two scale values per 32-element group, 4 groups per half).
                let is = l / 16u;
                let sc1 = sign_extend_byte(read_byte(bb + 192u + sc_off + is));
                let sc2 = sign_extend_byte(read_byte(bb + 192u + sc_off + is + 2u));
                let sc3 = sign_extend_byte(read_byte(bb + 192u + sc_off + is + 4u));
                let sc4 = sign_extend_byte(read_byte(bb + 192u + sc_off + is + 6u));

                acc = acc + d * f32(sc1) * f32(q1) * vec[vec_base + y_off + l];
                acc = acc + d * f32(sc2) * f32(q2) * vec[vec_base + y_off + l + 32u];
                acc = acc + d * f32(sc3) * f32(q3) * vec[vec_base + y_off + l + 64u];
                acc = acc + d * f32(sc4) * f32(q4) * vec[vec_base + y_off + l + 96u];
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
    if tid <  8u { partials[tid] = partials[tid] + partials[tid +  8u]; }
    workgroupBarrier();
    if tid <  4u { partials[tid] = partials[tid] + partials[tid +  4u]; }
    workgroupBarrier();
    if tid <  2u { partials[tid] = partials[tid] + partials[tid +  2u]; }
    workgroupBarrier();
    if tid <  1u { partials[tid] = partials[tid] + partials[tid +  1u]; }
    workgroupBarrier();

    if tid == 0u {
        output[row] = partials[0u];
    }
}
