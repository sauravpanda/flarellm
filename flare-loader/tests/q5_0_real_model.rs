//! Integration test: verify Q5_0 dequantization against known reference values
//! extracted from SmolLM2-135M-Instruct Q4_K_M GGUF using the gguf Python library.

#[test]
fn test_q5_0_real_block() {
    use flare_loader::quantize::dequant_q5_0_block;

    // Raw Q5_0 block from blk.0.attn_q.weight, first block
    // Extracted from SmolLM2-135M-Instruct-Q4_K_M.gguf
    let block: [u8; 22] = [
        79, 172, 45, 92, 249, 189, 49, 206, 241, 43, 0, 2, 175, 70, 14, 239, 3, 26, 103, 13, 242,
        24,
    ];

    // Reference values from Python gguf library (llama.cpp compatible dequant)
    let expected: [f32; 32] = [
        -0.067322, 0.134644, -0.067322, -0.740540, 1.077148, -0.134644, 0.067322, 0.673218,
        0.134644, 0.067322, -0.201965, -0.673218, -0.471252, 0.201965, -0.134644, 0.538574,
        -0.201965, 0.269287, 0.067322, -0.134644, -0.000000, -0.000000, -0.673218, -0.269287,
        -0.000000, 0.134644, -0.000000, -0.067322, -0.403931, -0.000000, 0.067322, -0.067322,
    ];

    let mut output = [0.0f32; 32];
    dequant_q5_0_block(&block, &mut output);

    for (i, (&got, &exp)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.01,
            "Q5_0 block value [{i}]: got {got:.6}, expected {exp:.6}"
        );
    }
}
