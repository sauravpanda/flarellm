// Fused IQ3_XXS dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  dequant(raw[row, j]) * vec[b * in_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// IQ3_XXS block layout (98 bytes per block, 256 weights per block — GGUF type 18):
//   bytes 0-1:   d  — f16 LE super-block scale
//   bytes 2-65:  qs[64] — 64 × uint8, indices into iq3xxs_grid
//     Two consecutive indices per 8-weight sub-group (grid1, grid2)
//   bytes 66-97: scales_and_signs[32] — 8 × uint32 LE, one per ib32 group
//     aux32: bits[31:28] = 4-bit sub-scale, bits[27:0] = 4×7-bit sign indices
//
// For each ib32 ∈ [0, 7]:
//   aux32     = u32 LE at bytes 66+4*ib32
//   dl        = d * (0.5 + f32(aux32 >> 28)) * 0.5
//   For l ∈ [0, 3]:
//     qs1     = qs[8*ib32 + 2*l + 0] = grid index for first 4 weights
//     qs2     = qs[8*ib32 + 2*l + 1] = grid index for next 4 weights
//     sign_7bit = (aux32 >> (7*l)) & 127
//     signs_8bit = KSIGNS_IQ2XS[sign_7bit] → 8 sign bits
//     For j ∈ [0, 3]:
//       weight[j]   = dl * f32(iq3xxs_grid[qs1] byte j)   * sign(signs_8bit, j)
//       weight[j+4] = dl * f32(iq3xxs_grid[qs2] byte j)   * sign(signs_8bit, j+4)
//
// Lookup tables embedded as WGSL const arrays (sourced from llama.cpp ggml-common.h):
//   IQ3XXS_GRID  — iq3xxs_grid[256] (uint32_t entries, 4 weight bytes each)
//   KSIGNS_IQ2XS — ksigns_iq2xs[128] packed 4-per-u32
//
// 98 bytes is not u32-aligned.  The Rust caller pads the raw buffer to the
// nearest 4-byte multiple before upload.
//
// One workgroup per (output row, batch item). 64 threads split the 8 super-block
// groups; a tree reduction accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — packed IQ3_XXS weight data (padded to 4-byte multiple)
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

// iq3xxs_grid[256] uint32 entries (4 weight bytes each).
// Source: llama.cpp ggml-common.h iq3xxs_grid table.
const IQ3XXS_GRID: array<u32, 256> = array<u32, 256>(
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu,
    0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu,
    0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u,
    0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u,
    0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu,
    0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u,
    0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu,
    0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu,
    0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu,
    0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu,
    0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u,
    0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u,
    0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu,
    0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu,
    0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u,
    0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u,
    0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu,
    0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu,
    0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u,
    0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu,
    0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu,
    0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu,
    0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu,
    0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu,
    0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u,
    0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u,
    0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu,
    0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u,
    0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u,
    0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu,
    0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u,
    0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu,
    0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u
);

// ksigns_iq2xs[128]: converts 7-bit sign index → 8-bit sign mask.
// Source: llama.cpp ggml-common.h. Packed 4 bytes per u32 (LE).
const KSIGNS_IQ2XS: array<u32, 32> = array<u32, 32>(
    0x03828100u, 0x87060584u, 0x8b0a0988u, 0x0f8e8d0cu,
    0x93121190u, 0x17969514u, 0x1b9a9918u, 0x9f1e1d9cu,
    0xa32221a0u, 0x27a6a524u, 0x2baaa928u, 0xaf2e2dacu,
    0x33b2b130u, 0xb73635b4u, 0xbb3a39b8u, 0x3fbebd3cu,
    0xc34241c0u, 0x47c6c544u, 0x4bcac948u, 0xcf4e4dccu,
    0x53d2d150u, 0xd75655d4u, 0xdb5a59d8u, 0x5fdedd5cu,
    0x63e2e160u, 0xe76665e4u, 0xeb6a69e8u, 0x6feeed6cu,
    0xf37271f0u, 0x77f6f574u, 0x7bfaf978u, 0xff7e7dfcu
);

/// Extract one byte at the given absolute byte offset from the u32 storage array.
fn read_byte(byte_offset: u32) -> u32 {
    return (raw[byte_offset / 4u] >> ((byte_offset % 4u) * 8u)) & 0xFFu;
}

/// Read a u32 (LE) from four consecutive bytes at byte_offset.
fn read_u32(byte_offset: u32) -> u32 {
    return read_byte(byte_offset)
         | (read_byte(byte_offset + 1u) << 8u)
         | (read_byte(byte_offset + 2u) << 16u)
         | (read_byte(byte_offset + 3u) << 24u);
}

/// Look up ksigns_iq2xs[i] (i in [0, 127]).
fn ksign(i: u32) -> u32 {
    let word = KSIGNS_IQ2XS[i / 4u];
    return (word >> ((i % 4u) * 8u)) & 0xFFu;
}

/// Read byte j of iq3xxs_grid[idx] (j in [0, 3]).
fn grid3_byte(idx: u32, j: u32) -> u32 {
    return (IQ3XXS_GRID[idx] >> (j * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn dequant_matvec_iq3xxs(
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

    // Each block is 98 bytes; threads stride across blocks.
    var b: u32 = tid;
    loop {
        if b >= params.num_blocks_per_row {
            break;
        }

        // Byte offset of this block in the raw buffer (98 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 98u;

        // d: f16 LE at bytes 0-1 of the block.
        let d_bits = read_byte(bb) | (read_byte(bb + 1u) << 8u);
        let d = unpack2x16float(d_bits).x;

        // vec base for this block and batch.
        let vec_base = batch * in_cols + b * 256u;

        // Process 8 groups of 32 weights (ib32 = 0..7).
        for (var ib32 = 0u; ib32 < 8u; ib32 = ib32 + 1u) {
            // aux32 at bytes 66+4*ib32 of the block.
            let aux32 = read_u32(bb + 66u + 4u * ib32);
            let dl = d * (0.5 + f32(aux32 >> 28u)) * 0.5;

            let vec_ib32_base = vec_base + ib32 * 32u;

            // 4 sub-groups (l=0..3), each producing 8 output weights.
            for (var l = 0u; l < 4u; l = l + 1u) {
                // Grid indices from qs: byte offset = 2 + 8*ib32 + 2*l
                let qs_off = bb + 2u + 8u * ib32 + 2u * l;
                let qs1 = read_byte(qs_off);
                let qs2 = read_byte(qs_off + 1u);

                let sign_7bit  = (aux32 >> (7u * l)) & 127u;
                let signs_8bit = ksign(sign_7bit);

                let vec_sub = vec_ib32_base + l * 8u;

                // First 4 weights from grid1.
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let gb   = grid3_byte(qs1, j);
                    let sign = select(1.0, -1.0, (signs_8bit & (1u << j)) != 0u);
                    acc = acc + dl * f32(gb) * sign * vec[vec_sub + j];
                }
                // Next 4 weights from grid2.
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let gb   = grid3_byte(qs2, j);
                    let sign = select(1.0, -1.0, (signs_8bit & (1u << (j + 4u))) != 0u);
                    acc = acc + dl * f32(gb) * sign * vec[vec_sub + 4u + j];
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
