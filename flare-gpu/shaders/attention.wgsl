// Scaled dot-product attention for a single head.
// Computes: softmax(Q @ K^T / sqrt(head_dim)) @ V
//
// This shader computes attention scores (Q @ K^T) for one query position
// against all KV cache positions. Used during autoregressive generation
// where we compute attention for the newest token against the full cache.
//
// The K/V buffers may be either:
//   • Per-head slices  [seq_len, head_dim]:              num_kv_heads=1, kv_head_idx=0
//   • Full-layer cache [seq_len, num_kv_heads, head_dim]: actual num_kv_heads / kv_head_idx
//
// Indexing: k_cache[t * num_kv_heads * head_dim + kv_head_idx * head_dim + d]

@group(0) @binding(0) var<storage, read> q: array<f32>;       // [head_dim]
@group(0) @binding(1) var<storage, read> k_cache: array<f32>; // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(2) var<storage, read> v_cache: array<f32>; // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [head_dim]

struct Params {
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    num_kv_heads: u32, // 1 for per-head slice layout, actual value for full-layer layout
    kv_head_idx: u32,  // which KV head to attend to
}
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory for attention scores (one per sequence position)
var<workgroup> scores: array<f32, 4096>; // max supported seq_len
var<workgroup> max_score: f32;
var<workgroup> sum_exp: f32;

@compute @workgroup_size(256)
fn attention_scores(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    let kv_stride = params.num_kv_heads * head_dim;
    let kv_head_base = params.kv_head_idx * head_dim;

    // Phase 1: Compute Q @ K^T scores
    // Each thread handles one or more sequence positions
    for (var t: u32 = tid; t < seq_len; t = t + 256u) {
        var dot: f32 = 0.0;
        let k_offset = t * kv_stride + kv_head_base;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q[d] * k_cache[k_offset + d];
        }
        scores[t] = dot * params.scale;
    }

    workgroupBarrier();

    // Phase 2: Find max for stable softmax (single thread for simplicity)
    if (tid == 0u) {
        var m: f32 = -1e30;
        for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
            m = max(m, scores[t]);
        }
        max_score = m;
    }

    workgroupBarrier();

    // Phase 3: Exp and sum
    if (tid == 0u) {
        var s: f32 = 0.0;
        for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
            scores[t] = exp(scores[t] - max_score);
            s = s + scores[t];
        }
        sum_exp = s;
    }

    workgroupBarrier();

    // Phase 4: Normalize
    for (var t: u32 = tid; t < seq_len; t = t + 256u) {
        scores[t] = scores[t] / sum_exp;
    }

    workgroupBarrier();

    // Phase 5: Weighted sum of values -> output
    // Each thread computes one or more output dimensions
    for (var d: u32 = tid; d < head_dim; d = d + 256u) {
        var acc: f32 = 0.0;
        for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
            acc = acc + scores[t] * v_cache[t * kv_stride + kv_head_base + d];
        }
        output[d] = acc;
    }
}
