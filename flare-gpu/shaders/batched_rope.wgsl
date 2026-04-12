// Batched Rotary Position Embedding (RoPE).
//
// Applies RoPE to a batch of token vectors. Each token t in 0..seq_len
// gets position (start_pos + t).
//
// The input is laid out as [seq_len * num_heads * head_dim], row-major.
// Each thread handles one (element, head, token) rotation pair:
//   idx = t * num_heads * head_dim + h * head_dim + i
//   idx + half = t * num_heads * head_dim + h * head_dim + i + half_dim
//
// Bindings (3-entry layout: 1 read-only, 1 read-write, 1 uniform):
//   binding 0: inp    — f32 input  [seq_len * num_heads * head_dim]
//   binding 1: output — f32 result [seq_len * num_heads * head_dim]
//   binding 2: params — uniform
//
// Dispatch: [seq_len * num_heads * head_dim / 2, 1, 1].

@group(0) @binding(0) var<storage, read>       inp:    array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    num_heads: u32,
    head_dim:  u32,
    seq_len:   u32,
    start_pos: u32,
    theta:     f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn batched_rope(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let half = params.head_dim / 2u;
    let stride = params.num_heads * params.head_dim;

    // Thread gid.x covers one rotation pair: element i within head h of token t.
    let i = gid.x % half;
    let h = (gid.x / half) % params.num_heads;
    let t = gid.x / (params.num_heads * half);

    if t >= params.seq_len {
        return;
    }

    let pos  = params.start_pos + t;
    let freq = 1.0 / pow(params.theta, f32(2u * i) / f32(params.head_dim));
    let angle = f32(pos) * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    let base = t * stride + h * params.head_dim;
    let idx0  = base + i;
    let idx1  = base + i + half;

    let v0 = inp[idx0];
    let v1 = inp[idx1];
    output[idx0] = v0 * cos_val - v1 * sin_val;
    output[idx1] = v0 * sin_val + v1 * cos_val;
}
