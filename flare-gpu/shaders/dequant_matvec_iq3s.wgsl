// Fused IQ3_S dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  dequant(raw[row, j]) * vec[b * in_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// IQ3_S block layout (110 bytes per block, 256 weights — GGUF type 26):
//   bytes 0-1:     d         — f16 LE super-block scale
//   bytes 2-65:    qs[64]    — 64 × u8, low 8 bits of 9-bit grid indices
//   bytes 66-73:   qh[8]     — 8 × u8, high bit for each pair of grid indices
//   bytes 74-105:  signs[32] — 32 × u8, one byte of sign bits per 8-weight sub-group
//   bytes 106-109: scales[4] — 4 × u8, packed 4-bit scales (two per byte)
//
// For each ib32 ∈ [0, 7] (groups of 32 weights):
//   nibble   = scales[ib32/2] >> (4*(ib32%2)) & 0xF     — 4-bit scale (0-15)
//   db       = d * (1 + 2 * nibble)                      — scale factor
//   qh_byte  = qh[ib32]
//   For l ∈ [0, 3]:
//     grid_idx1 = qs[8*ib32 + 2*l + 0] | (((qh_byte >> (2*l  )) & 1) << 8)
//     grid_idx2 = qs[8*ib32 + 2*l + 1] | (((qh_byte >> (2*l+1)) & 1) << 8)
//     signs_byte = signs[4*ib32 + l]
//     For j ∈ [0, 3]:
//       w[j]   = db * f32(iq3s_grid[grid_idx1] byte j)   * sign(signs_byte, j)
//       w[j+4] = db * f32(iq3s_grid[grid_idx2] byte j)   * sign(signs_byte, j+4)
//
// Lookup table: IQ3S_GRID[512] (uint32_t, 4 bytes each = 4 weight values).
// Sign extraction: bit j of signs_byte → +1.0 (0) or -1.0 (1).
//
// 110 bytes is not u32-aligned.  The Rust caller pads to the nearest 4-byte
// multiple before upload.
//
// One workgroup per (output row, batch item). 64 threads split the 8 groups;
// a tree reduction accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — packed IQ3_S weight data (padded to 4-byte multiple)
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

// iq3s_grid[512] uint32 entries (4 weight bytes each).
// Source: llama.cpp ggml-common.h iq3s_grid table.
const IQ3S_GRID: array<u32, 512> = array<u32, 512>(
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu,
    0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u,
    0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u,
    0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u,
    0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u,
    0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu,
    0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u,
    0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u,
    0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u,
    0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu,
    0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u,
    0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu,
    0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u,
    0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u,
    0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u,
    0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u,
    0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u,
    0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u,
    0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u,
    0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu,
    0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u,
    0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u,
    0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u,
    0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u,
    0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u,
    0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u,
    0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u,
    0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u,
    0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u,
    0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u,
    0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u,
    0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u,
    0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u,
    0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u,
    0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu,
    0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u,
    0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u,
    0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u,
    0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u,
    0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u,
    0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u,
    0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u,
    0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u,
    0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u,
    0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u,
    0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu,
    0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u,
    0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u,
    0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu,
    0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u,
    0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u,
    0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u,
    0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u,
    0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u,
    0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u,
    0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u,
    0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u,
    0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu,
    0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u,
    0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u,
    0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u,
    0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u,
    0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u,
    0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u,
    0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u
);

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

/// Read byte j of iq3s_grid[idx] (j in [0, 3]).
fn grid3s_byte(idx: u32, j: u32) -> u32 {
    return (IQ3S_GRID[idx] >> (j * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn dequant_matvec_iq3s(
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

    // Each block is 110 bytes; threads stride across blocks.
    var b: u32 = tid;
    loop {
        if b >= params.num_blocks_per_row {
            break;
        }

        // Byte offset of this block (110 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 110u;

        // d: f16 LE at bytes 0-1.
        let d_bits = read_byte(bb) | (read_byte(bb + 1u) << 8u);
        let d = unpack2x16float(d_bits).x;

        // vec base for this block and batch.
        let vec_base = batch * in_cols + b * 256u;

        // Process 8 groups of 32 weights (ib32 = 0..7).
        for (var ib32 = 0u; ib32 < 8u; ib32 = ib32 + 1u) {
            // Scale nibble from scales[ib32/2], low or high nibble.
            let scale_byte = read_byte(bb + 106u + ib32 / 2u);
            let nibble = (scale_byte >> ((ib32 % 2u) * 4u)) & 0xFu;
            let db = d * f32(1u + 2u * nibble);

            // qh byte for this group: byte at offset 66 + ib32.
            let qh_byte = read_byte(bb + 66u + ib32);

            let vec_ib32_base = vec_base + ib32 * 32u;

            // 4 sub-groups (l=0..3), each producing 8 output weights.
            for (var l = 0u; l < 4u; l = l + 1u) {
                // Low 8 bits from qs[8*ib32 + 2*l] at bb+2+8*ib32+2*l
                let qs_off  = bb + 2u + 8u * ib32 + 2u * l;
                let qs1_lo  = read_byte(qs_off);
                let qs2_lo  = read_byte(qs_off + 1u);

                // High bit (bit 8) from qh_byte bits 2*l and 2*l+1.
                let grid_idx1 = qs1_lo | (((qh_byte >> (2u * l    )) & 1u) << 8u);
                let grid_idx2 = qs2_lo | (((qh_byte >> (2u * l + 1u)) & 1u) << 8u);

                // Sign byte: signs[4*ib32 + l] at bb+74+4*ib32+l.
                let signs_byte = read_byte(bb + 74u + 4u * ib32 + l);

                let vec_sub = vec_ib32_base + l * 8u;

                // First 4 weights from grid1.
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let gb   = grid3s_byte(grid_idx1, j);
                    let sign = select(1.0, -1.0, (signs_byte & (1u << j)) != 0u);
                    acc = acc + db * f32(gb) * sign * vec[vec_sub + j];
                }
                // Next 4 weights from grid2.
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let gb   = grid3s_byte(grid_idx2, j);
                    let sign = select(1.0, -1.0, (signs_byte & (1u << (j + 4u))) != 0u);
                    acc = acc + db * f32(gb) * sign * vec[vec_sub + 4u + j];
                }
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
