//! Integration test: verify Q4_K dequantization against known reference values.

#[test]
fn test_q4k_real_block() {
    use flare_loader::quantize::dequant_q4k_block;

    // Raw Q4_K block from blk.2.ffn_down.weight (first block)
    // Extracted from SmolLM2-135M-Instruct-Q4_K_M.gguf
    let block: [u8; 144] = [
        221, 20, 56, 33, 241, 237, 254, 191, 174, 238, 229, 176, 32, 196, 249, 13, 70, 80, 201,
        104, 170, 182, 214, 154, 243, 121, 135, 119, 218, 203, 207, 148, 102, 215, 148, 175, 167,
        107, 91, 3, 155, 121, 189, 120, 119, 181, 120, 198, 118, 163, 11, 87, 54, 67, 149, 69, 134,
        94, 128, 151, 135, 51, 104, 102, 184, 134, 148, 90, 72, 166, 163, 165, 82, 246, 245, 177,
        86, 132, 106, 85, 149, 184, 233, 89, 94, 4, 182, 153, 254, 183, 215, 122, 186, 150, 165,
        197, 136, 216, 187, 186, 120, 174, 194, 174, 149, 157, 199, 171, 230, 199, 168, 192, 106,
        174, 28, 132, 43, 171, 249, 130, 140, 90, 152, 75, 106, 221, 218, 10, 103, 104, 86, 122,
        106, 76, 60, 140, 88, 27, 63, 158, 139, 28, 5, 144,
    ];

    // Reference values from Python (llama.cpp compatible dequant)
    let expected_0_8: [f32; 8] = [
        -0.119799, -0.468872, 0.054738, -0.003441, 0.112917, -0.119799, -0.119799, 0.112917,
    ];
    let expected_128_136: [f32; 8] = [
        -0.087681, -0.084119, -0.059185, -0.080557, -0.066309, -0.062747, -0.055623, -0.069871,
    ];

    let mut output = [0.0f32; 256];
    dequant_q4k_block(&block, &mut output);

    for (i, &exp) in expected_0_8.iter().enumerate() {
        assert!(
            (output[i] - exp).abs() < 0.01,
            "Q4_K value [{i}]: got {:.6}, expected {:.6}",
            output[i],
            exp
        );
    }

    for (i, &exp) in expected_128_136.iter().enumerate() {
        assert!(
            (output[128 + i] - exp).abs() < 0.01,
            "Q4_K value [{}]: got {:.6}, expected {:.6}",
            128 + i,
            output[128 + i],
            exp
        );
    }
}
