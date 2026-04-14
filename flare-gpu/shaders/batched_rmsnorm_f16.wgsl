// Batched RMSNorm with f16 norm weights.
//
// Identical to batched_rmsnorm.wgsl but reads norm weights from f16 storage,
// halving weight memory bandwidth.  Requires the `shader-f16` WGSL extension.
//
// Input and output remain f32 — only the weight read benefits from f16.
// This is useful when norm weights are stored in half precision to save memory.
//
// Bindings (standard 4-entry layout: 2 read-only, 1 read-write, 1 uniform):
//   binding 0: input  — f32 batch [batch * dim], row-major
//   binding 1: weight — f16 norm weights [dim]
//   binding 2: output — f32 result [batch * dim], row-major
//   binding 3: params — uniform
//
// Dispatch: [batch, 1, 1].

enable f16;

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    dim:   u32,
    batch: u32,
    eps:   f32,
}
@group(0) @binding(3) var<uniform> params: Params;

/// Workgroup-shared partial sums for the tree reduction.
var<workgroup> partials: array<f32, 64>;

@compute @workgroup_size(64)
fn batched_rmsnorm_f16(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let tok = wid.x;
    if tok >= params.batch {
        return;
    }

    let tid  = lid.x;
    let base = tok * params.dim;

    // Each thread strides across the row accumulating its partial sum of squares.
    var ss: f32 = 0.0;
    var i: u32 = tid;
    loop {
        if i >= params.dim {
            break;
        }
        let v = inp[base + i];
        ss = ss + v * v;
        i = i + 64u;
    }

    partials[tid] = ss;
    workgroupBarrier();

    // Parallel tree reduction: 64 -> 1.
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

    let rms_inv = 1.0 / sqrt(partials[0u] / f32(params.dim) + params.eps);

    // Write normalised + scaled outputs. Weight is read as f16, widened to f32.
    var j: u32 = tid;
    loop {
        if j >= params.dim {
            break;
        }
        output[base + j] = inp[base + j] * rms_inv * f32(weight[j]);
        j = j + 64u;
    }
}
