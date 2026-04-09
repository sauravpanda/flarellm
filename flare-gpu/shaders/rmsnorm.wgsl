// Fused RMSNorm + residual add.
// Each workgroup processes one row of the input.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
    hidden_dim: u32,
    eps: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn fused_rmsnorm_residual(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let row = gid.x;
    let dim = params.hidden_dim;

    // Compute sum of squares
    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let val = input[row * dim + i] + residual[row * dim + i];
        sum_sq = sum_sq + val * val;
    }
    let rms = sqrt(sum_sq / f32(dim) + params.eps);

    // Normalize and scale
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        let val = input[row * dim + i] + residual[row * dim + i];
        output[row * dim + i] = (val / rms) * weight[i];
    }
}
