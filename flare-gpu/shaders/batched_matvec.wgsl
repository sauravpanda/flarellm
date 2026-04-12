// Batched matrix-vector multiply.
//
// Computes: output[b * out_rows + i] = Σ_j weight[i * in_cols + j] * input[b * in_cols + j]
//
// i.e. for each batch item b, multiply the shared weight matrix W[out_rows × in_cols]
// by the input row input_batch[b, :] and store the result in output[b, :].
//
// Bindings (standard 4-entry layout):
//   binding 0: weight — f32 matrix [out_rows * in_cols], row-major
//   binding 1: input  — f32 batch  [batch * in_cols], row-major
//   binding 2: output — f32 result [batch * out_rows], row-major
//   binding 3: params — uniform
//
// One workgroup per (output_row, batch_item) pair; 64-thread inner dot-product
// with parallel tree reduction.
// Dispatch: [out_rows, batch, 1].

@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    out_rows: u32,
    in_cols:  u32,
    batch:    u32,
}
@group(0) @binding(3) var<uniform> params: Params;

/// Workgroup-shared partial sums for the tree reduction.
var<workgroup> partials: array<f32, 64>;

@compute @workgroup_size(64)
fn batched_matvec(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let row       = wid.x;
    let batch_idx = wid.y;

    if row >= params.out_rows || batch_idx >= params.batch {
        return;
    }

    let tid = lid.x;
    var acc: f32 = 0.0;

    let weight_base = row * params.in_cols;
    let input_base  = batch_idx * params.in_cols;

    // Each thread strides across in_cols with a step of 64.
    var j: u32 = tid;
    loop {
        if j >= params.in_cols {
            break;
        }
        acc = acc + weight[weight_base + j] * inp[input_base + j];
        j = j + 64u;
    }

    partials[tid] = acc;
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

    if tid == 0u {
        // Row-major output: batch_idx selects the row, row selects the column.
        output[batch_idx * params.out_rows + row] = partials[0u];
    }
}
