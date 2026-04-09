// Stable softmax over a 1D array.
// output[i] = exp(input[i] - max) / sum(exp(input - max))

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    size: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

@compute @workgroup_size(256)
fn softmax(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_size) wg_size: vec3<u32>,
) {
    let tid = lid.x;
    let size = params.size;

    // Phase 1: Find max (thread 0 for simplicity — works for small vocabs)
    if (tid == 0u) {
        var m: f32 = -1e30;
        for (var i: u32 = 0u; i < size; i = i + 1u) {
            m = max(m, input[i]);
        }
        shared_max = m;
    }

    workgroupBarrier();

    // Phase 2: Compute exp(x - max) and partial sums
    if (tid == 0u) {
        var s: f32 = 0.0;
        for (var i: u32 = 0u; i < size; i = i + 1u) {
            let e = exp(input[i] - shared_max);
            output[i] = e;
            s = s + e;
        }
        shared_sum = s;
    }

    workgroupBarrier();

    // Phase 3: Normalize
    for (var i: u32 = tid; i < size; i = i + wg_size.x) {
        output[i] = output[i] / shared_sum;
    }
}
