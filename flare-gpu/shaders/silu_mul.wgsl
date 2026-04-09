// SiLU activation fused with element-wise multiply.
// Used in the FFN: output = SiLU(gate) * up

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn silu_mul(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if (idx >= params.size) {
        return;
    }

    let x = gate[idx];
    let silu = x / (1.0 + exp(-x)); // SiLU(x) = x * sigmoid(x)
    output[idx] = silu * up[idx];
}
