// Single-head scaled dot-product attention.
// Computes: softmax((Q @ K^T) * scale) @ V
// Q, K, V are [seq_len, head_dim]; output is [seq_len, head_dim].

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
    seq_len: u32,
    head_dim: u32,
    scale: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

// Workgroup-shared tile memory for tiled score computation.
const TILE_SIZE: u32 = 16u;
var<workgroup> tile_q: array<array<f32, 16>, 16>;
var<workgroup> tile_k: array<array<f32, 16>, 16>;

// Scratch for attention scores, softmax reduction, and weighted V accumulation.
// Max supported seq_len is 4096 for this scratch buffer.
var<workgroup> scores: array<f32, 4096>;
var<workgroup> shared_max: f32;
var<workgroup> shared_sum: f32;

// ---------------------------------------------------------------------------
// Pass 1: compute attention scores for one query row.
// Dispatched as (seq_len, ceil(seq_len/16), 1) workgroups of size (16, 16).
// Each workgroup computes a 1 x TILE_SIZE strip of (Q @ K^T) * scale for a
// single query row, but we flatten the full computation into pass 2 below.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Full fused kernel: one workgroup per query row.
// workgroup_size(256) — each thread helps with reductions and V accumulation.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn attention(@builtin(local_invocation_id) lid: vec3<u32>,
             @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;                  // query row this workgroup handles
    let tid = lid.x;                  // thread index within workgroup
    let seq = params.seq_len;
    let dim = params.head_dim;

    // --- Step 1: compute raw scores = Q[row] . K[j] for all j ---------------
    // Each thread covers a subset of columns j.
    for (var j: u32 = tid; j < seq; j = j + 256u) {
        var dot: f32 = 0.0;
        for (var d: u32 = 0u; d < dim; d = d + 1u) {
            dot = dot + q[row * dim + d] * k[j * dim + d];
        }
        scores[j] = dot * params.scale;
    }
    workgroupBarrier();

    // --- Step 2: stable softmax — find max -----------------------------------
    var local_max: f32 = -3.402823e+38;  // -FLT_MAX
    for (var j: u32 = tid; j < seq; j = j + 256u) {
        local_max = max(local_max, scores[j]);
    }
    // Simple workgroup reduction via shared memory (reuse scores scratch).
    // Store per-thread max at offset seq (safe — scratch is 4096, seq <= 4096,
    // and we only use indices [seq .. seq+255]).
    scores[seq + tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max among 256 threads.
    var stride: u32 = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            scores[seq + tid] = max(scores[seq + tid], scores[seq + tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        shared_max = scores[seq];
    }
    workgroupBarrier();

    // --- Step 3: exp(score - max) and partial sums ---------------------------
    var local_sum: f32 = 0.0;
    for (var j: u32 = tid; j < seq; j = j + 256u) {
        let e = exp(scores[j] - shared_max);
        scores[j] = e;
        local_sum = local_sum + e;
    }
    scores[seq + tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum.
    stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            scores[seq + tid] = scores[seq + tid] + scores[seq + tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if tid == 0u {
        shared_sum = scores[seq];
    }
    workgroupBarrier();

    // Normalize scores in-place.
    for (var j: u32 = tid; j < seq; j = j + 256u) {
        scores[j] = scores[j] / shared_sum;
    }
    workgroupBarrier();

    // --- Step 4: output = scores @ V  [1, seq] x [seq, dim] -> [1, dim] -----
    // Each thread computes a subset of output dimensions.
    for (var d: u32 = tid; d < dim; d = d + 256u) {
        var acc: f32 = 0.0;
        for (var j: u32 = 0u; j < seq; j = j + 1u) {
            acc = acc + scores[j] * v[j * dim + d];
        }
        output[row * dim + d] = acc;
    }
}
