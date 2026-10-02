//! Real adapter regressions: run with `cargo test -p flarellm-gpu --test decode_errors -- --ignored`.
use flare_core::model::{Model, RawLayerWeights, RawWeight, WeightFormat};
use flare_core::tensor::Tensor;
use flare_gpu::WebGpuBackend;

#[path = "../../flare-core/tests/common/mod.rs"]
mod common;

fn resident_model(gpu: WebGpuBackend, mut model: Model) -> Model {
    let dim = model.config().hidden_dim;
    let inter = model.config().intermediate_dim;
    let raw = |t: &Tensor, cols: usize| RawWeight {
        data: t
            .data()
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect(),
        format: WeightFormat::F16,
        num_rows: t.numel() / cols,
        blocks_per_row: cols,
    };
    let weights = model
        .weights()
        .layers
        .iter()
        .map(|l| RawLayerWeights {
            wq: raw(&l.wq, dim),
            wk: raw(&l.wk, dim),
            wv: raw(&l.wv, dim),
            wo: raw(&l.wo, dim),
            w_gate: raw(&l.w_gate, dim),
            w_up: raw(&l.w_up, dim),
            w_down: raw(&l.w_down, inter),
        })
        .collect();
    model.set_raw_weights(weights);
    model.set_backend(Box::new(gpu));
    model.upload_weights_to_gpu();
    model
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn device_loss_returns_error_and_discards_gpu_context() {
    // A destroyed device must reject decode and discard any partially written KV.
    let gpu = pollster::block_on(WebGpuBackend::new()).expect("GPU adapter required");
    let device = gpu.device().clone();
    let mut model = resident_model(gpu, common::make_model());
    device.destroy();
    let error = pollster::block_on(model.try_forward_async(2, 0)).unwrap_err();
    assert!(error.to_string().contains("GPU decode failed"));
    assert_eq!(model.kv_cache().position(), 0);
    assert_eq!(model.backend().name(), "cpu");
    let actual = pollster::block_on(model.try_forward_async(2, 0)).unwrap();
    let expected = common::make_model().forward(2, 0);
    assert_eq!(actual.data(), expected.data());
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn invalid_pipeline_or_binding_rejects_decode_and_recovers_on_cpu() {
    let gpu = pollster::block_on(WebGpuBackend::new()).expect("GPU adapter required");
    // With wgpu 24, native f16/subgroup shader parsing fails. On devices
    // without these features, the tiny fixture's four-element heads instead
    // produce an unaligned second-head storage binding. Either is a real
    // validation failure: it must be surfaced, never sampled as zero logits.
    let mut model = resident_model(gpu, common::make_model());
    let error = pollster::block_on(model.try_forward_async(2, 0)).unwrap_err();
    assert!(error.to_string().contains("Validation Error"), "{error}");
    assert_eq!(model.kv_cache().position(), 0);
    assert_eq!(model.backend().name(), "cpu");
    let actual = pollster::block_on(model.try_forward_async(2, 0)).unwrap();
    let expected = common::make_model().forward(2, 0);
    assert_eq!(actual.data(), expected.data());
}
