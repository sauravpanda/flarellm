// Fused IQ2_XS dequantize + batched matrix-vector multiply on GPU.
//
// Computes: output[b * num_rows + row] = Σ_j  dequant(raw[row, j]) * vec[b * in_cols + j]
// for each batch item b ∈ [0, batch_size) and output row ∈ [0, num_rows).
//
// IQ2_XS block layout (74 bytes per block, 256 weights per block — GGUF type 17):
//   bytes 0-1:   d  — f16 LE super-block scale
//   bytes 2-65:  qs[32] — 32 × uint16 LE
//     Each uint16: bits[8:0] = 9-bit grid index (0-511), bits[15:9] = 7-bit sign index
//   bytes 66-73: scales[8] — 8 × uint8, one per ib32 group
//     Per ib32: lo nibble = dl1 scale, hi nibble = dl2 scale
//
// For each ib32 ∈ [0, 7]:
//   scale_byte = scales[ib32]
//   dl1 = d * (0.5 + f32(scale_byte & 0xf)) * 0.25   (used for qs[0], qs[1])
//   dl2 = d * (0.5 + f32(scale_byte >> 4)) * 0.25    (used for qs[2], qs[3])
//   For j ∈ [0, 1]:  dl = dl1 if j==0 else dl2
//     For k ∈ [0, 1]:  entry = qs[4*ib32 + 2*j + k]
//       grid_idx   = entry & 0x1FF
//       sign_7bit  = entry >> 9
//       signs_8bit = KSIGNS_IQ2XS[sign_7bit] → 8 sign bits
//       For i ∈ [0, 7]: weight = dl * grid_byte(grid_idx, i) * sign(signs_8bit, i)
//
// Lookup tables embedded as WGSL const arrays (sourced from llama.cpp ggml-common.h):
//   IQ2XS_GRID_LO / IQ2XS_GRID_HI  — low/high u32 halves of iq2xs_grid[512] (uint64_t)
//   KSIGNS_IQ2XS                    — ksigns_iq2xs[128] packed 4-per-u32
//
// 74 bytes is not u32-aligned.  The Rust caller pads the raw buffer to the
// nearest 4-byte multiple before upload.
//
// One workgroup per (output row, batch item). 64 threads split the 8 super-block
// groups; a tree reduction accumulates partial sums.
//
// Bindings:
//   binding 0: raw    — packed IQ2_XS weight data (padded to 4-byte multiple)
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

// iq2xs_grid[512] uint64 entries split into low / high u32 halves.
// Source: llama.cpp ggml-common.h iq2xs_grid table.
const IQ2XS_GRID_LO: array<u32, 512> = array<u32, 512>(
    0x08080808u, 0x0808082bu, 0x08081919u, 0x08082b08u,
    0x08082b2bu, 0x08190819u, 0x08191908u, 0x0819192bu,
    0x08192b19u, 0x082b0808u, 0x082b082bu, 0x082b1919u,
    0x082b2b08u, 0x19080819u, 0x19081908u, 0x1908192bu,
    0x19082b19u, 0x19190808u, 0x1919082bu, 0x19191919u,
    0x19192b08u, 0x192b0819u, 0x192b1908u, 0x2b080808u,
    0x2b08082bu, 0x2b081919u, 0x2b082b08u, 0x2b190819u,
    0x2b191908u, 0x2b192b19u, 0x2b2b0808u, 0x08080819u,
    0x08081908u, 0x0808192bu, 0x08082b19u, 0x08190808u,
    0x0819082bu, 0x08191919u, 0x08192b08u, 0x08192b2bu,
    0x082b0819u, 0x082b1908u, 0x19080808u, 0x1908082bu,
    0x19081919u, 0x19082b08u, 0x19190819u, 0x19191908u,
    0x192b0808u, 0x192b2b08u, 0x2b080819u, 0x2b081908u,
    0x2b190808u, 0x08080808u, 0x0808082bu, 0x08081919u,
    0x08082b08u, 0x08190819u, 0x08191908u, 0x082b0808u,
    0x19080819u, 0x19081908u, 0x19190808u, 0x19191919u,
    0x2b080808u, 0x2b082b2bu, 0x08080819u, 0x08081908u,
    0x0808192bu, 0x08082b19u, 0x08190808u, 0x0819082bu,
    0x08191919u, 0x08192b08u, 0x082b0819u, 0x082b1908u,
    0x19080808u, 0x1908082bu, 0x19081919u, 0x19082b08u,
    0x19190819u, 0x19191908u, 0x1919192bu, 0x192b0808u,
    0x2b080819u, 0x2b081908u, 0x2b190808u, 0x08080808u,
    0x0808082bu, 0x08081919u, 0x08082b08u, 0x08190819u,
    0x08191908u, 0x082b0808u, 0x19080819u, 0x19081908u,
    0x19190808u, 0x192b0819u, 0x2b080808u, 0x08080819u,
    0x08081908u, 0x08190808u, 0x082b192bu, 0x19080808u,
    0x1908082bu, 0x2b081908u, 0x08080808u, 0x0808082bu,
    0x08081919u, 0x08082b08u, 0x08082b2bu, 0x08190819u,
    0x08191908u, 0x082b0808u, 0x082b1919u, 0x19080819u,
    0x19081908u, 0x19190808u, 0x19192b08u, 0x2b080808u,
    0x2b2b0808u, 0x2b2b2b2bu, 0x08080819u, 0x08081908u,
    0x08190808u, 0x19080808u, 0x2b080819u, 0x2b082b19u,
    0x08080808u, 0x082b0808u, 0x082b2b08u, 0x2b19192bu,
    0x2b2b0808u, 0x08080819u, 0x08081908u, 0x0808192bu,
    0x08082b19u, 0x08190808u, 0x0819082bu, 0x08191919u,
    0x08192b08u, 0x082b0819u, 0x082b1908u, 0x19080808u,
    0x1908082bu, 0x19081919u, 0x19082b08u, 0x19190819u,
    0x19191908u, 0x192b0808u, 0x192b2b2bu, 0x2b080819u,
    0x2b081908u, 0x2b190808u, 0x08080808u, 0x0808082bu,
    0x08081919u, 0x08082b08u, 0x08190819u, 0x08191908u,
    0x082b0808u, 0x19080819u, 0x19081908u, 0x19190808u,
    0x2b080808u, 0x2b191908u, 0x2b19192bu, 0x08080819u,
    0x08081908u, 0x0808192bu, 0x08190808u, 0x19080808u,
    0x192b0808u, 0x08080808u, 0x0808082bu, 0x08081919u,
    0x08082b08u, 0x08190819u, 0x08191908u, 0x082b0808u,
    0x19080819u, 0x19081908u, 0x19082b19u, 0x19190808u,
    0x192b1908u, 0x2b080808u, 0x08080819u, 0x08081908u,
    0x08190808u, 0x19080808u, 0x08080808u, 0x08191908u,
    0x19082b19u, 0x08080819u, 0x08081908u, 0x08190808u,
    0x0819082bu, 0x19080808u, 0x19191908u, 0x2b08192bu,
    0x08080808u, 0x08081919u, 0x192b192bu, 0x19190819u,
    0x2b2b2b19u, 0x08080808u, 0x0808082bu, 0x08081919u,
    0x08082b08u, 0x08082b2bu, 0x08190819u, 0x08191908u,
    0x082b0808u, 0x19080819u, 0x19081908u, 0x19190808u,
    0x2b080808u, 0x2b2b0808u, 0x08080819u, 0x08081908u,
    0x08190808u, 0x19080808u, 0x19082b08u, 0x192b1919u,
    0x08080808u, 0x082b082bu, 0x2b080808u, 0x2b2b2b08u,
    0x08080819u, 0x08081908u, 0x08190808u, 0x082b2b19u,
    0x19080808u, 0x08080808u, 0x19080819u, 0x1919082bu,
    0x2b192b19u, 0x08080819u, 0x08192b2bu, 0x2b2b192bu,
    0x08080808u, 0x08082b08u, 0x08082b2bu, 0x082b0808u,
    0x19191919u, 0x2b082b08u, 0x2b2b082bu, 0x192b2b08u,
    0x2b190808u, 0x08082b08u, 0x082b0808u, 0x2b08082bu,
    0x2b082b08u, 0x2b082b2bu, 0x08080819u, 0x08081908u,
    0x0808192bu, 0x08082b19u, 0x08190808u, 0x0819082bu,
    0x08191919u, 0x08192b08u, 0x082b0819u, 0x082b1908u,
    0x19080808u, 0x1908082bu, 0x19081919u, 0x19082b08u,
    0x19082b2bu, 0x19190819u, 0x19191908u, 0x192b0808u,
    0x192b1919u, 0x2b080819u, 0x2b081908u, 0x2b190808u,
    0x08080808u, 0x0808082bu, 0x08081919u, 0x08082b08u,
    0x08190819u, 0x08191908u, 0x082b0808u, 0x19080819u,
    0x19081908u, 0x19190808u, 0x2b080808u, 0x2b081919u,
    0x2b2b082bu, 0x08080819u, 0x08081908u, 0x08190808u,
    0x0819082bu, 0x082b2b19u, 0x19080808u, 0x08080808u,
    0x0808082bu, 0x08081919u, 0x08082b08u, 0x08190819u,
    0x08191908u, 0x08192b19u, 0x082b0808u, 0x19080819u,
    0x19081908u, 0x19190808u, 0x2b080808u, 0x2b191908u,
    0x08080819u, 0x08081908u, 0x08190808u, 0x082b1908u,
    0x19080808u, 0x2b192b2bu, 0x08080808u, 0x08082b2bu,
    0x19081908u, 0x19190808u, 0x08080819u, 0x08081908u,
    0x08190808u, 0x19080808u, 0x19081919u, 0x19191908u,
    0x192b082bu, 0x08080808u, 0x08190819u, 0x19081908u,
    0x19190808u, 0x192b2b19u, 0x08081908u, 0x08080808u,
    0x0808082bu, 0x08081919u, 0x08082b08u, 0x08190819u,
    0x08191908u, 0x082b0808u, 0x082b2b08u, 0x19080819u,
    0x19081908u, 0x19190808u, 0x2b080808u, 0x08080819u,
    0x08081908u, 0x08190808u, 0x08191919u, 0x19080808u,
    0x1908082bu, 0x08080808u, 0x19081908u, 0x2b2b2b2bu,
    0x08080819u, 0x08081908u, 0x08190808u, 0x082b0819u,
    0x19080808u, 0x192b0808u, 0x2b080819u, 0x2b2b0819u,
    0x08080808u, 0x08082b08u, 0x2b080808u, 0x2b082b08u,
    0x082b0819u, 0x192b2b08u, 0x2b2b0819u, 0x08080808u,
    0x08191908u, 0x19080819u, 0x19190808u, 0x2b192b19u,
    0x08192b2bu, 0x19080808u, 0x1908082bu, 0x2b081919u,
    0x08080819u, 0x08081908u, 0x08190808u, 0x19080808u,
    0x19191908u, 0x192b082bu, 0x2b08192bu, 0x2b2b2b19u,
    0x08080808u, 0x082b1908u, 0x19082b2bu, 0x2b19082bu,
    0x08080808u, 0x0819192bu, 0x08190808u, 0x19080808u,
    0x19081919u, 0x2b2b1908u, 0x08080819u, 0x192b2b2bu,
    0x082b1919u, 0x0808192bu, 0x19191908u, 0x192b082bu,
    0x08080808u, 0x0808082bu, 0x08081919u, 0x08082b08u,
    0x08190819u, 0x08191908u, 0x082b0808u, 0x082b2b2bu,
    0x19080819u, 0x19081908u, 0x19190808u, 0x2b080808u,
    0x2b08082bu, 0x2b2b2b08u, 0x2b2b2b2bu, 0x08080819u,
    0x08081908u, 0x0808192bu, 0x08190808u, 0x19080808u,
    0x19190819u, 0x19192b19u, 0x08080808u, 0x082b0808u,
    0x2b080808u, 0x2b08082bu, 0x2b2b0808u, 0x2b2b2b08u,
    0x08080819u, 0x08081908u, 0x08190808u, 0x0819082bu,
    0x08191919u, 0x19080808u, 0x192b0808u, 0x2b082b19u,
    0x08080808u, 0x19081908u, 0x2b2b1919u, 0x08192b08u,
    0x192b2b2bu, 0x08080808u, 0x08082b08u, 0x082b1919u,
    0x19192b2bu, 0x2b080808u, 0x2b08082bu, 0x2b2b2b08u,
    0x0808192bu, 0x082b082bu, 0x2b080808u, 0x2b082b08u,
    0x2b19192bu, 0x2b2b2b08u, 0x08080819u, 0x08081908u,
    0x08190808u, 0x19080808u, 0x1919192bu, 0x2b081908u,
    0x08080808u, 0x082b082bu, 0x192b1908u, 0x1919192bu,
    0x2b082b19u, 0x08080808u, 0x08081919u, 0x19081908u,
    0x19190808u, 0x19192b08u, 0x082b2b19u, 0x2b190808u,
    0x2b19082bu, 0x19080819u, 0x19190819u, 0x2b2b192bu,
    0x19082b19u, 0x08191919u, 0x192b0808u, 0x08080808u,
    0x0808082bu, 0x08082b08u, 0x08082b2bu, 0x082b0808u,
    0x082b2b2bu, 0x2b2b0808u, 0x19190819u, 0x19192b19u,
    0x2b2b192bu, 0x08080808u, 0x0808082bu, 0x08082b08u,
    0x082b2b2bu, 0x2b080808u, 0x2b2b0808u, 0x19080808u,
    0x2b191919u, 0x192b1919u, 0x2b192b08u, 0x08082b2bu,
    0x082b0808u, 0x082b082bu, 0x082b2b08u, 0x2b2b0808u,
    0x2b2b2b08u, 0x08081908u, 0x2b081908u, 0x2b08192bu,
    0x082b2b08u, 0x082b2b2bu, 0x2b190819u, 0x2b2b2b2bu
);

const IQ2XS_GRID_HI: array<u32, 512> = array<u32, 512>(
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080808u,
    0x08080808u, 0x08080808u, 0x08080808u, 0x08080819u,
    0x08080819u, 0x08080819u, 0x08080819u, 0x08080819u,
    0x08080819u, 0x08080819u, 0x08080819u, 0x08080819u,
    0x08080819u, 0x08080819u, 0x08080819u, 0x08080819u,
    0x08080819u, 0x08080819u, 0x08080819u, 0x08080819u,
    0x08080819u, 0x08080819u, 0x08080819u, 0x08080819u,
    0x08080819u, 0x0808082bu, 0x0808082bu, 0x0808082bu,
    0x0808082bu, 0x0808082bu, 0x0808082bu, 0x0808082bu,
    0x0808082bu, 0x0808082bu, 0x0808082bu, 0x0808082bu,
    0x0808082bu, 0x0808082bu, 0x08081908u, 0x08081908u,
    0x08081908u, 0x08081908u, 0x08081908u, 0x08081908u,
    0x08081908u, 0x08081908u, 0x08081908u, 0x08081908u,
    0x08081908u, 0x08081908u, 0x08081908u, 0x08081908u,
    0x08081908u, 0x08081908u, 0x08081908u, 0x08081908u,
    0x08081908u, 0x08081908u, 0x08081908u, 0x08081919u,
    0x08081919u, 0x08081919u, 0x08081919u, 0x08081919u,
    0x08081919u, 0x08081919u, 0x08081919u, 0x08081919u,
    0x08081919u, 0x08081919u, 0x08081919u, 0x0808192bu,
    0x0808192bu, 0x0808192bu, 0x0808192bu, 0x0808192bu,
    0x0808192bu, 0x0808192bu, 0x08082b08u, 0x08082b08u,
    0x08082b08u, 0x08082b08u, 0x08082b08u, 0x08082b08u,
    0x08082b08u, 0x08082b08u, 0x08082b08u, 0x08082b08u,
    0x08082b08u, 0x08082b08u, 0x08082b08u, 0x08082b08u,
    0x08082b08u, 0x08082b08u, 0x08082b19u, 0x08082b19u,
    0x08082b19u, 0x08082b19u, 0x08082b19u, 0x08082b19u,
    0x08082b2bu, 0x08082b2bu, 0x08082b2bu, 0x08082b2bu,
    0x08082b2bu, 0x08190808u, 0x08190808u, 0x08190808u,
    0x08190808u, 0x08190808u, 0x08190808u, 0x08190808u,
    0x08190808u, 0x08190808u, 0x08190808u, 0x08190808u,
    0x08190808u, 0x08190808u, 0x08190808u, 0x08190808u,
    0x08190808u, 0x08190808u, 0x08190808u, 0x08190808u,
    0x08190808u, 0x08190808u, 0x08190819u, 0x08190819u,
    0x08190819u, 0x08190819u, 0x08190819u, 0x08190819u,
    0x08190819u, 0x08190819u, 0x08190819u, 0x08190819u,
    0x08190819u, 0x08190819u, 0x08190819u, 0x0819082bu,
    0x0819082bu, 0x0819082bu, 0x0819082bu, 0x0819082bu,
    0x0819082bu, 0x08191908u, 0x08191908u, 0x08191908u,
    0x08191908u, 0x08191908u, 0x08191908u, 0x08191908u,
    0x08191908u, 0x08191908u, 0x08191908u, 0x08191908u,
    0x08191908u, 0x08191908u, 0x08191919u, 0x08191919u,
    0x08191919u, 0x08191919u, 0x0819192bu, 0x0819192bu,
    0x0819192bu, 0x08192b08u, 0x08192b08u, 0x08192b08u,
    0x08192b08u, 0x08192b08u, 0x08192b08u, 0x08192b08u,
    0x08192b19u, 0x08192b19u, 0x08192b19u, 0x08192b2bu,
    0x08192b2bu, 0x082b0808u, 0x082b0808u, 0x082b0808u,
    0x082b0808u, 0x082b0808u, 0x082b0808u, 0x082b0808u,
    0x082b0808u, 0x082b0808u, 0x082b0808u, 0x082b0808u,
    0x082b0808u, 0x082b0808u, 0x082b0819u, 0x082b0819u,
    0x082b0819u, 0x082b0819u, 0x082b0819u, 0x082b0819u,
    0x082b082bu, 0x082b082bu, 0x082b082bu, 0x082b082bu,
    0x082b1908u, 0x082b1908u, 0x082b1908u, 0x082b1908u,
    0x082b1908u, 0x082b1919u, 0x082b1919u, 0x082b1919u,
    0x082b1919u, 0x082b192bu, 0x082b192bu, 0x082b192bu,
    0x082b2b08u, 0x082b2b08u, 0x082b2b08u, 0x082b2b08u,
    0x082b2b08u, 0x082b2b08u, 0x082b2b08u, 0x082b2b19u,
    0x082b2b19u, 0x082b2b2bu, 0x082b2b2bu, 0x082b2b2bu,
    0x082b2b2bu, 0x082b2b2bu, 0x19080808u, 0x19080808u,
    0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u,
    0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u,
    0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u,
    0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u,
    0x19080808u, 0x19080808u, 0x19080808u, 0x19080808u,
    0x19080819u, 0x19080819u, 0x19080819u, 0x19080819u,
    0x19080819u, 0x19080819u, 0x19080819u, 0x19080819u,
    0x19080819u, 0x19080819u, 0x19080819u, 0x19080819u,
    0x19080819u, 0x1908082bu, 0x1908082bu, 0x1908082bu,
    0x1908082bu, 0x1908082bu, 0x1908082bu, 0x19081908u,
    0x19081908u, 0x19081908u, 0x19081908u, 0x19081908u,
    0x19081908u, 0x19081908u, 0x19081908u, 0x19081908u,
    0x19081908u, 0x19081908u, 0x19081908u, 0x19081908u,
    0x19081919u, 0x19081919u, 0x19081919u, 0x19081919u,
    0x19081919u, 0x19081919u, 0x1908192bu, 0x1908192bu,
    0x1908192bu, 0x1908192bu, 0x19082b08u, 0x19082b08u,
    0x19082b08u, 0x19082b08u, 0x19082b08u, 0x19082b08u,
    0x19082b08u, 0x19082b19u, 0x19082b19u, 0x19082b19u,
    0x19082b19u, 0x19082b19u, 0x19082b2bu, 0x19190808u,
    0x19190808u, 0x19190808u, 0x19190808u, 0x19190808u,
    0x19190808u, 0x19190808u, 0x19190808u, 0x19190808u,
    0x19190808u, 0x19190808u, 0x19190808u, 0x19190819u,
    0x19190819u, 0x19190819u, 0x19190819u, 0x19190819u,
    0x19190819u, 0x1919082bu, 0x1919082bu, 0x1919082bu,
    0x19191908u, 0x19191908u, 0x19191908u, 0x19191908u,
    0x19191908u, 0x19191908u, 0x19191908u, 0x19191908u,
    0x19191919u, 0x19191919u, 0x19191919u, 0x19191919u,
    0x1919192bu, 0x1919192bu, 0x1919192bu, 0x19192b08u,
    0x19192b08u, 0x19192b08u, 0x19192b08u, 0x19192b08u,
    0x19192b19u, 0x19192b19u, 0x19192b19u, 0x19192b2bu,
    0x192b0808u, 0x192b0808u, 0x192b0808u, 0x192b0808u,
    0x192b0808u, 0x192b0808u, 0x192b0808u, 0x192b0808u,
    0x192b0819u, 0x192b082bu, 0x192b082bu, 0x192b082bu,
    0x192b1908u, 0x192b1908u, 0x192b1919u, 0x192b1919u,
    0x192b1919u, 0x192b1919u, 0x192b2b08u, 0x192b2b08u,
    0x192b2b19u, 0x192b2b2bu, 0x192b2b2bu, 0x192b2b2bu,
    0x2b080808u, 0x2b080808u, 0x2b080808u, 0x2b080808u,
    0x2b080808u, 0x2b080808u, 0x2b080808u, 0x2b080808u,
    0x2b080808u, 0x2b080808u, 0x2b080808u, 0x2b080808u,
    0x2b080808u, 0x2b080808u, 0x2b080808u, 0x2b080819u,
    0x2b080819u, 0x2b080819u, 0x2b080819u, 0x2b080819u,
    0x2b080819u, 0x2b080819u, 0x2b08082bu, 0x2b08082bu,
    0x2b08082bu, 0x2b08082bu, 0x2b08082bu, 0x2b08082bu,
    0x2b081908u, 0x2b081908u, 0x2b081908u, 0x2b081908u,
    0x2b081908u, 0x2b081908u, 0x2b081908u, 0x2b081908u,
    0x2b081919u, 0x2b081919u, 0x2b081919u, 0x2b08192bu,
    0x2b08192bu, 0x2b082b08u, 0x2b082b08u, 0x2b082b08u,
    0x2b082b08u, 0x2b082b08u, 0x2b082b08u, 0x2b082b08u,
    0x2b082b19u, 0x2b082b2bu, 0x2b082b2bu, 0x2b082b2bu,
    0x2b082b2bu, 0x2b082b2bu, 0x2b190808u, 0x2b190808u,
    0x2b190808u, 0x2b190808u, 0x2b190808u, 0x2b190808u,
    0x2b190819u, 0x2b190819u, 0x2b190819u, 0x2b19082bu,
    0x2b19082bu, 0x2b191908u, 0x2b191908u, 0x2b191908u,
    0x2b191908u, 0x2b191908u, 0x2b191919u, 0x2b191919u,
    0x2b191919u, 0x2b19192bu, 0x2b192b08u, 0x2b192b08u,
    0x2b192b19u, 0x2b192b2bu, 0x2b192b2bu, 0x2b2b0808u,
    0x2b2b0808u, 0x2b2b0808u, 0x2b2b0808u, 0x2b2b0808u,
    0x2b2b0808u, 0x2b2b0808u, 0x2b2b0819u, 0x2b2b0819u,
    0x2b2b0819u, 0x2b2b082bu, 0x2b2b082bu, 0x2b2b082bu,
    0x2b2b082bu, 0x2b2b082bu, 0x2b2b082bu, 0x2b2b1908u,
    0x2b2b1908u, 0x2b2b192bu, 0x2b2b192bu, 0x2b2b2b08u,
    0x2b2b2b08u, 0x2b2b2b08u, 0x2b2b2b08u, 0x2b2b2b08u,
    0x2b2b2b08u, 0x2b2b2b19u, 0x2b2b2b19u, 0x2b2b2b19u,
    0x2b2b2b2bu, 0x2b2b2b2bu, 0x2b2b2b2bu, 0x2b2b2b2bu
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

/// Read a u16 (LE) from two consecutive bytes at byte_offset.
fn read_u16(byte_offset: u32) -> u32 {
    return read_byte(byte_offset) | (read_byte(byte_offset + 1u) << 8u);
}

/// Look up ksigns_iq2xs[i] (i in [0, 127]).
fn ksign(i: u32) -> u32 {
    let word = KSIGNS_IQ2XS[i / 4u];
    return (word >> ((i % 4u) * 8u)) & 0xFFu;
}

/// Read byte j of iq2xs_grid[idx] (j in [0, 7]).
fn grid_byte(idx: u32, j: u32) -> u32 {
    if j < 4u {
        return (IQ2XS_GRID_LO[idx] >> (j * 8u)) & 0xFFu;
    } else {
        return (IQ2XS_GRID_HI[idx] >> ((j - 4u) * 8u)) & 0xFFu;
    }
}

@compute @workgroup_size(64)
fn dequant_matvec_iq2xs(
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

    // Each block is 74 bytes; threads stride across blocks.
    var b: u32 = tid;
    loop {
        if b >= params.num_blocks_per_row {
            break;
        }

        // Byte offset of this block in the raw buffer (74 bytes per block).
        let bb = (row * params.num_blocks_per_row + b) * 74u;

        // d: f16 LE at bytes 0-1 of the block.
        let d_bits = read_byte(bb) | (read_byte(bb + 1u) << 8u);
        let d = unpack2x16float(d_bits).x;

        // vec base for this block and batch.
        let vec_base = batch * in_cols + b * 256u;

        // Process 8 groups of 32 weights (ib32 = 0..7).
        for (var ib32 = 0u; ib32 < 8u; ib32 = ib32 + 1u) {
            // scale byte at bytes 66+ib32 of the block.
            let scale_byte = read_byte(bb + 66u + ib32);
            let dl1 = d * (0.5 + f32(scale_byte & 0xFu)) * 0.25;
            let dl2 = d * (0.5 + f32(scale_byte >> 4u)) * 0.25;

            // qs base: 4 uint16 entries per ib32, each 2 bytes, starting at block byte 2.
            // Byte offset of qs[4*ib32]: bb + 2 + 8*ib32
            let qs_base = bb + 2u + 8u * ib32;

            let vec_ib32_base = vec_base + ib32 * 32u;

            // j=0 → dl1, entries qs[0] and qs[1]
            // j=1 → dl2, entries qs[2] and qs[3]
            for (var j = 0u; j < 2u; j = j + 1u) {
                let dl = select(dl2, dl1, j == 0u);
                for (var k = 0u; k < 2u; k = k + 1u) {
                    let qs16 = read_u16(qs_base + (j * 2u + k) * 2u);
                    let grid_idx  = qs16 & 0x1FFu;
                    let sign_7bit = qs16 >> 9u;
                    let signs_8bit = ksign(sign_7bit);

                    let vec_sub = vec_ib32_base + j * 16u + k * 8u;
                    for (var i = 0u; i < 8u; i = i + 1u) {
                        let gb   = grid_byte(grid_idx, i);
                        let sign = select(1.0, -1.0, (signs_8bit & (1u << i)) != 0u);
                        acc = acc + dl * f32(gb) * sign * vec[vec_sub + i];
                    }
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
