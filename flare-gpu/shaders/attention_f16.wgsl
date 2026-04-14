// Scaled dot-product attention with f16 KV cache.
//
// Identical to attention.wgsl but reads K/V from f16 storage buffers,
// halving KV cache memory bandwidth.  Requires the `shader-f16` WGSL
// extension (WebGPU Chrome 120+, native Vulkan/Metal via wgpu).
//
// Q and output remain f32 — only the KV cache reads benefit from f16,
// which is the memory-bound bottleneck during autoregressive generation.
//
// Indexing: k_cache[t * num_kv_heads * head_dim + kv_head_idx * head_dim + d]

enable f16;

@group(0) @binding(0) var<storage, read> q: array<f32>;         // [head_dim]
@group(0) @binding(1) var<storage, read> k_cache: array<f16>;   // [seq_len, num_kv_heads, head_dim]
@group(0) @binding(2) var<storage, read> v_cache: array<f16>;   // [seq_len, num_kv_heads, head_dim]
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
var<workgroup> max_score: f32;
var<workgroup> sum_exp: f32;

@compute @workgroup_size(256)
fn attention_scores_f16(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let seq_len = params.seq_len;
    let head_dim = params.head_dim;
    let kv_stride = params.num_kv_heads * head_dim;
    let kv_head_base = params.kv_head_idx * head_dim;

    // Phase 1: Compute Q @ K^T scores
    // Each thread handles one or more sequence positions.
    // K is read as f16, widened to f32 for the dot product.
    for (var t: u32 = tid; t < seq_len; t = t + 256u) {
        var dot: f32 = 0.0;
        let k_offset = t * kv_stride + kv_head_base;
        for (var d: u32 = 0u; d < head_dim; d = d + 1u) {
            dot = dot + q[d] * f32(k_cache[k_offset + d]);
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
    // V is read as f16, widened to f32 for accumulation.
    for (var d: u32 = tid; d < head_dim; d = d + 256u) {
        var acc: f32 = 0.0;
        for (var t: u32 = 0u; t < seq_len; t = t + 1u) {
            acc = acc + scores[t] * f32(v_cache[t * kv_stride + kv_head_base + d]);
        }
        output[d] = acc;
    }
}
