// Batched causal prefill attention on GPU.
//
// Computes causal self-attention for every token position in a prefill
// sequence in a single dispatch, replacing the O(seq_len²) CPU loop in
// forward_prefill().
//
// Data layout (flat f32 arrays):
//   q_all   [seq_len * num_heads    * head_dim] — queries for all positions and heads
//   k_all   [seq_len * num_kv_heads * head_dim] — keys   for all positions and kv-heads
//   v_all   [seq_len * num_kv_heads * head_dim] — values for all positions and kv-heads
//   output  [seq_len * num_heads    * head_dim] — attention output (same layout as q_all)
//
//   Element at (pos, head, dim):  array[pos * H * D + head * D + dim]
//   where H = num_heads (or num_kv_heads for K/V) and D = head_dim.
//
// Dispatch: [seq_len, num_heads, 1] workgroups of size [256, 1, 1].
//   wid.x = query_pos   (0..seq_len)
//   wid.y = head_idx    (0..num_heads)
//
// Causal masking: workgroup (query_pos, head_idx) attends to key/value
//   positions 0..=query_pos only.
//
// Grouped-query attention: kv_head = head_idx % num_kv_heads.
//
// Supports optional Gemma-2-style logit soft-cap:
//   score = tanh(raw_score / softcap) * softcap   (when softcap > 0)
//
// Maximum supported sequence length: 4096
//   (limited by workgroup shared memory for the attention score array).

@group(0) @binding(0) var<storage, read>       q_all:  array<f32>; // [seq_len * num_heads * head_dim]
@group(0) @binding(1) var<storage, read>       k_all:  array<f32>; // [seq_len * num_kv_heads * head_dim]
@group(0) @binding(2) var<storage, read>       v_all:  array<f32>; // [seq_len * num_kv_heads * head_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [seq_len * num_heads * head_dim]

struct Params {
    seq_len:       u32,
    num_heads:     u32,
    num_kv_heads:  u32,
    head_dim:      u32,
    scale:         f32,  // 1.0 / sqrt(head_dim)
    attn_softcap:  f32,  // Gemma-2 logit soft-cap; 0.0 = disabled
}
@group(0) @binding(4) var<uniform> params: Params;

/// Per-workgroup attention score buffer.
/// One entry per sequence position; max 4096 positions.
var<workgroup> scores:    array<f32, 4096>;
var<workgroup> max_score: f32;
var<workgroup> sum_exp:   f32;

@compute @workgroup_size(256)
fn prefill_attention(
    @builtin(local_invocation_id)  lid:     vec3<u32>,
    @builtin(workgroup_id)         wid:     vec3<u32>,
) {
    let query_pos  = wid.x;
    let head_idx   = wid.y;
    let tid        = lid.x;

    let seq_len      = params.seq_len;
    let num_heads    = params.num_heads;
    let num_kv_heads = params.num_kv_heads;
    let head_dim     = params.head_dim;

    // Position query_pos attends to keys at positions 0..=query_pos.
    let attend_len = query_pos + 1u;

    // Shared memory holds at most 4096 scores.
    // (All threads in the workgroup see the same query_pos, so all exit.)
    if attend_len > 4096u {
        return;
    }

    // Grouped-query attention: map attention head → KV head.
    let kv_head = head_idx % num_kv_heads;

    // Flat strides.
    let q_stride  = num_heads    * head_dim;
    let kv_stride = num_kv_heads * head_dim;

    // Base index into q_all for this (query_pos, head_idx).
    let q_base = query_pos * q_stride + head_idx * head_dim;

    // --------------------------------------------------------------------
    // Phase 1: QK^T
    // Each thread computes the dot product for one or more key positions.
    // --------------------------------------------------------------------
    for (var t = tid; t < attend_len; t = t + 256u) {
        var dot = 0.0;
        let k_base = t * kv_stride + kv_head * head_dim;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            dot += q_all[q_base + d] * k_all[k_base + d];
        }
        var score = dot * params.scale;
        // Optional logit soft-cap (Gemma-2).
        if params.attn_softcap > 0.0 {
            score = tanh(score / params.attn_softcap) * params.attn_softcap;
        }
        scores[t] = score;
    }
    workgroupBarrier();

    // --------------------------------------------------------------------
    // Phase 2: Stable softmax (single thread — QK^T above is the bottleneck)
    // --------------------------------------------------------------------
    if tid == 0u {
        var m = -1e30;
        for (var t = 0u; t < attend_len; t = t + 1u) {
            m = max(m, scores[t]);
        }
        max_score = m;

        var s = 0.0;
        for (var t = 0u; t < attend_len; t = t + 1u) {
            scores[t] = exp(scores[t] - max_score);
            s += scores[t];
        }
        sum_exp = s;
    }
    workgroupBarrier();

    // Parallel normalisation.
    let inv_sum = 1.0 / sum_exp;
    for (var t = tid; t < attend_len; t = t + 256u) {
        scores[t] = scores[t] * inv_sum;
    }
    workgroupBarrier();

    // --------------------------------------------------------------------
    // Phase 3: Weighted sum of values.
    // Each thread computes one or more output dimensions.
    // --------------------------------------------------------------------
    let out_base = query_pos * q_stride + head_idx * head_dim;
    for (var d = tid; d < head_dim; d = d + 256u) {
        var acc = 0.0;
        for (var t = 0u; t < attend_len; t = t + 1u) {
            let v_base = t * kv_stride + kv_head * head_dim;
            acc += scores[t] * v_all[v_base + d];
        }
        output[out_base + d] = acc;
    }
}
