// Subgroup-optimized batched matrix-vector multiply.
//
// Computes: output[b * out_rows + i] = Sum_j weight[i * in_cols + j] * input[b * in_cols + j]
//
// Uses subgroupAdd for the dot-product reduction, replacing the explicit
// tree reduction with barrier-free warp-level operations.
//
// Requires the WebGPU `subgroups` feature (Chrome 144+, wgpu `SUBGROUP`).
//
// Bindings and params are identical to batched_matvec.wgsl.

enable subgroups;

@group(0) @binding(0) var<storage, read> weight: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    out_rows: u32,
    in_cols:  u32,
    batch:    u32,
}
@group(0) @binding(3) var<uniform> params: Params;

// Cross-subgroup reduction scratch.  64 / min_subgroup_size(4) = 16 max.
var<workgroup> sg_scratch: array<f32, 16>;

@compute @workgroup_size(64)
fn batched_matvec(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
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

    // Intra-subgroup reduction (barrier-free).
    acc = subgroupAdd(acc);

    // Cross-subgroup reduction via shared memory.
    let sg_index = tid / sg_size;
    let num_subgroups = 64u / sg_size;

    if sg_id == 0u {
        sg_scratch[sg_index] = acc;
    }
    workgroupBarrier();

    if sg_index == 0u {
        var cross_sum: f32 = 0.0;
        if tid < num_subgroups {
            cross_sum = sg_scratch[tid];
        }
        cross_sum = subgroupAdd(cross_sum);
        if sg_id == 0u {
            output[batch_idx * params.out_rows + row] = cross_sum;
        }
    }
}
