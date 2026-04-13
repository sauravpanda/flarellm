// In-place residual connection: x[i] += residual[i]
//
// Bindings:
//   binding 0: residual — f32 [dim], read-only
//   binding 1: x        — f32 [dim], read-write (modified in place)
//   binding 2: params   — uniform
//
// Dispatch: [ceil(dim / 256), 1, 1]

@group(0) @binding(0) var<storage, read>       residual: array<f32>;
@group(0) @binding(1) var<storage, read_write> x:        array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn add_residual(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if idx >= params.size {
        return;
    }
    x[idx] = x[idx] + residual[idx];
}
