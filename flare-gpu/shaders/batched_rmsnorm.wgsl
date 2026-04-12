// Batched RMSNorm.
//
// Computes for each token t in 0..batch:
//   rms   = sqrt(mean_squares(input[t, :]) + eps)
//   output[t, i] = input[t, i] / rms * weight[i]
//
// Bindings (standard 4-entry layout: 2 read-only, 1 read-write, 1 uniform):
//   binding 0: input  — f32 batch [batch * dim], row-major
//   binding 1: weight — f32 norm weights [dim]
//   binding 2: output — f32 result [batch * dim], row-major
//   binding 3: params — uniform
//
// One workgroup per batch item. 64 threads compute partial sums via tree reduction
// to obtain the mean-square; then each thread writes its normalised + scaled outputs.
// Dispatch: [batch, 1, 1].

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
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
fn batched_rmsnorm(
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

    let rms_inv = 1.0 / sqrt(partials[0u] / f32(params.dim) + params.eps);

    // Write normalised + scaled outputs.
    var j: u32 = tid;
    loop {
        if j >= params.dim {
            break;
        }
        output[base + j] = inp[base + j] * rms_inv * weight[j];
        j = j + 64u;
    }
}
