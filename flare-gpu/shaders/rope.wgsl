// Rotary Position Embedding (RoPE).

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;

struct Params {
    head_dim: u32,
    position: u32,
    theta: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn rope_embedding(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let head_dim = params.head_dim;
    let half = head_dim / 2u;
    let pos = params.position;

    let i = idx % half;
    let freq = 1.0 / pow(params.theta, f32(2u * i) / f32(head_dim));
    let angle = f32(pos) * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);

    // Apply to Q
    let q0 = q[idx];
    let q1 = q[idx + half];
    q[idx]        = q0 * cos_val - q1 * sin_val;
    q[idx + half] = q0 * sin_val + q1 * cos_val;

    // Apply to K
    let k0 = k[idx];
    let k1 = k[idx + half];
    k[idx]        = k0 * cos_val - k1 * sin_val;
    k[idx + half] = k0 * sin_val + k1 * cos_val;
}
