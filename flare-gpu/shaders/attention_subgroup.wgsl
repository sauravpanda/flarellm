// Subgroup-optimized scaled dot-product attention for a single head.
// Computes: softmax(Q @ K^T / sqrt(head_dim)) @ V
//
// Uses subgroup (warp-level) reductions for the softmax max-finding and
// sum-of-exponentials phases, replacing the single-threaded bottleneck in the
// baseline shader with fully parallel cross-lane operations.
//
// Requires the WebGPU `subgroups` feature (Chrome 144+, wgpu `SUBGROUP`).
//
// Bindings and params are identical to attention.wgsl so the two shaders are
// interchangeable at runtime.

enable subgroups;

@group(0) @binding(0) var<storage, read> q: array<f32>;       // [head_dim]
@group(0) @binding(1) var<storage, read> k_cache: array<f32>; // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(2) var<storage, read> v_cache: array<f32>; // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [head_dim]

struct Params {
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    num_kv_heads: u32,
    kv_head_idx: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory for attention scores (one per sequence position)
var<workgroup> scores: array<f32, 4096>; // max supported seq_len

// Cross-subgroup reduction scratch: one slot per subgroup in the workgroup.
// 256 / min_subgroup_size(4) = 64 slots is the theoretical max needed.
var<workgroup> sg_scratch: array<f32, 64>;

// Final reduced values broadcast to all threads.
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn attention_scores(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let tid = lid.x;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    let kv_stride = params.num_kv_heads * head_dim;
    let kv_head_base = params.kv_head_idx * head_dim;

    // The index of this thread's subgroup within the workgroup.
    let sg_index = tid / sg_size;
    // Number of active subgroups in the workgroup.
    let num_subgroups = WG_SIZE / sg_size;

    // -----------------------------------------------------------------------
    // Phase 1: Compute Q @ K^T scores
    // -----------------------------------------------------------------------
    for (var t: u32 = tid; t < seq_len; t = t + WG_SIZE) {
        var dot: f32 = 0.0;
        let k_offset = t * kv_stride + kv_head_base;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q[d] * k_cache[k_offset + d];
        }
        scores[t] = dot * params.scale;
    }

    workgroupBarrier();

    // -----------------------------------------------------------------------
    // Phase 2: Find max for stable softmax using subgroup reductions
    // -----------------------------------------------------------------------
    var local_max: f32 = -1e30;
    for (var t: u32 = tid; t < seq_len; t = t + WG_SIZE) {
        local_max = max(local_max, scores[t]);
    }

    // Intra-subgroup max reduction.
    local_max = subgroupMax(local_max);

    // Cross-subgroup: each subgroup leader writes its max to shared memory.
    if sg_id == 0u {
        sg_scratch[sg_index] = local_max;
    }
    workgroupBarrier();

    // First subgroup reduces across all subgroup maxes.
    if sg_index == 0u {
        var cross_val: f32 = -1e30;
        if tid < num_subgroups {
            cross_val = sg_scratch[tid];
        }
        cross_val = subgroupMax(cross_val);
        if sg_id == 0u {
            shared_max = cross_val;
        }
    }
    workgroupBarrier();

    // -----------------------------------------------------------------------
    // Phase 3: Exp and sum using subgroup reductions
    // -----------------------------------------------------------------------
    var local_sum: f32 = 0.0;
    for (var t: u32 = tid; t < seq_len; t = t + WG_SIZE) {
        let e = exp(scores[t] - shared_max);
        scores[t] = e;
        local_sum = local_sum + e;
    }

    // Intra-subgroup sum reduction.
    local_sum = subgroupAdd(local_sum);

    // Cross-subgroup sum reduction.
    if sg_id == 0u {
        sg_scratch[sg_index] = local_sum;
    }
    workgroupBarrier();

    if sg_index == 0u {
        var cross_sum: f32 = 0.0;
        if tid < num_subgroups {
            cross_sum = sg_scratch[tid];
        }
        cross_sum = subgroupAdd(cross_sum);
        if sg_id == 0u {
            shared_sum = cross_sum;
        }
    }
    workgroupBarrier();

    // -----------------------------------------------------------------------
    // Phase 4: Normalize scores
    // -----------------------------------------------------------------------
    for (var t: u32 = tid; t < seq_len; t = t + WG_SIZE) {
        scores[t] = scores[t] / shared_sum;
    }

    workgroupBarrier();

    // -----------------------------------------------------------------------
    // Phase 5: Weighted sum of values -> output
    // -----------------------------------------------------------------------
    for (var d: u32 = tid; d < head_dim; d = d + WG_SIZE) {
        var acc: f32 = 0.0;
        for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
            acc = acc + scores[t] * v_cache[t * kv_stride + kv_head_base + d];
        }
        output[d] = acc;
    }
}
