// Convert f32 data to f16 in a destination buffer.
//
// Used to write f32 K/V projection outputs into the f16 KV cache.
// Requires the `shader-f16` WGSL extension.
//
// Bindings:
//   binding 0: input  — f32 source [count]
//   binding 1: output — f16 destination (written at element offset)
//   binding 2: params — uniform { count, dst_offset }
//
// Dispatch: [ceil(count / 256), 1, 1]

enable f16;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;

struct Params {
    count: u32,      // number of elements to convert
    dst_offset: u32, // element offset into the destination buffer
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn f32_to_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    if idx >= params.count {
        return;
    }
    output[params.dst_offset + idx] = f16(input[idx]);
}
