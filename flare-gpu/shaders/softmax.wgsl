// Stable softmax over a flat f32 array.
// Subtracts the max before exponentiation to avoid overflow.
// One workgroup processes the entire array; dispatch exactly (1, 1, 1).

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.size;

    // --- Pass 1: find the maximum value (for numerical stability) ------------
    var local_max: f32 = -3.402823e+38;  // -FLT_MAX
    for (var i: u32 = tid; i < n; i = i + 256u) {
        local_max = max(local_max, input[i]);
    }
    scratch[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max.
    var stride: u32 = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let global_max = scratch[0];
    workgroupBarrier();

    // --- Pass 2: compute exp(x - max) and partial sums ----------------------
    var local_sum: f32 = 0.0;
    for (var i: u32 = tid; i < n; i = i + 256u) {
        let e = exp(input[i] - global_max);
        output[i] = e;                       // store intermediate exp values
        local_sum = local_sum + e;
    }
    scratch[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum.
    stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            scratch[tid] = scratch[tid] + scratch[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let global_sum = scratch[0];
    workgroupBarrier();

    // --- Pass 3: normalize ---------------------------------------------------
    for (var i: u32 = tid; i < n; i = i + 256u) {
        output[i] = output[i] / global_sum;
    }
}
