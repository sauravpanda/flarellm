// Tiled matrix multiply — the core LLM inference kernel.
// Tile size chosen for WebGPU workgroup limits (256 max invocations).

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Params {
    M: u32,
    N: u32,
    K: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

const TILE_SIZE: u32 = 16u;
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn matmul(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.x;
    let col = gid.y;
    var sum: f32 = 0.0;

    let num_tiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let a_col = t * TILE_SIZE + lid.y;
        let b_row = t * TILE_SIZE + lid.x;

        if (row < params.M && a_col < params.K) {
            tile_a[lid.x][lid.y] = a[row * params.K + a_col];
        } else {
            tile_a[lid.x][lid.y] = 0.0;
        }

        if (b_row < params.K && col < params.N) {
            tile_b[lid.x][lid.y] = b[b_row * params.N + col];
        } else {
            tile_b[lid.x][lid.y] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tile_a[lid.x][k] * tile_b[k][lid.y];
        }

        workgroupBarrier();
    }

    if (row < params.M && col < params.N) {
        result[row * params.N + col] = sum;
    }
}
