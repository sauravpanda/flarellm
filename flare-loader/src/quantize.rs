/// Quantization format identifiers matching GGUF type IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Unknown(u32),
}

impl QuantFormat {
    pub fn from_gguf_type(type_id: u32) -> Self {
        match type_id {
            0 => QuantFormat::F32,
            1 => QuantFormat::F16,
            2 => QuantFormat::Q4_0,
            3 => QuantFormat::Q4_1,
            6 => QuantFormat::Q5_0,
            7 => QuantFormat::Q5_1,
            8 => QuantFormat::Q8_0,
            9 => QuantFormat::Q8_1,
            10 => QuantFormat::Q2K,
            11 => QuantFormat::Q3K,
            12 => QuantFormat::Q4K,
            13 => QuantFormat::Q5K,
            14 => QuantFormat::Q6K,
            other => QuantFormat::Unknown(other),
        }
    }

    /// Bits per weight for this format.
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            QuantFormat::F32 => 32.0,
            QuantFormat::F16 => 16.0,
            QuantFormat::Q4_0 | QuantFormat::Q4_1 | QuantFormat::Q4K => 4.5,
            QuantFormat::Q5_0 | QuantFormat::Q5_1 | QuantFormat::Q5K => 5.5,
            QuantFormat::Q8_0 | QuantFormat::Q8_1 => 8.5,
            QuantFormat::Q2K => 2.6,
            QuantFormat::Q3K => 3.4,
            QuantFormat::Q6K => 6.6,
            QuantFormat::Unknown(_) => 32.0, // assume worst case
        }
    }

    /// Block size for quantized formats (number of weights per block).
    pub fn block_size(&self) -> usize {
        match self {
            QuantFormat::F32 | QuantFormat::F16 => 1,
            QuantFormat::Q4_0 | QuantFormat::Q4_1 => 32,
            QuantFormat::Q5_0 | QuantFormat::Q5_1 => 32,
            QuantFormat::Q8_0 | QuantFormat::Q8_1 => 32,
            QuantFormat::Q2K
            | QuantFormat::Q3K
            | QuantFormat::Q4K
            | QuantFormat::Q5K
            | QuantFormat::Q6K => 256,
            QuantFormat::Unknown(_) => 1,
        }
    }

    /// Bytes per block for quantized formats.
    pub fn block_bytes(&self) -> usize {
        match self {
            QuantFormat::F32 => 4,
            QuantFormat::F16 => 2,
            QuantFormat::Q4_0 => 18, // 2 (scale) + 16 (nibbles)
            QuantFormat::Q4_1 => 20, // 2 (scale) + 2 (min) + 16 (nibbles)
            QuantFormat::Q5_0 => 22, // 2 (scale) + 4 (high bits) + 16 (nibbles)
            QuantFormat::Q5_1 => 24, // 2 (scale) + 2 (min) + 4 (high bits) + 16 (nibbles)
            QuantFormat::Q8_0 => 34, // 2 (scale) + 32 (int8)
            QuantFormat::Q8_1 => 36, // 2 (scale) + 2 (sum) + 32 (int8)
            QuantFormat::Q2K => 84,
            QuantFormat::Q3K => 110,
            QuantFormat::Q4K => 144, // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)
            QuantFormat::Q5K => 176, // 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (ql)
            QuantFormat::Q6K => 210, // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
            QuantFormat::Unknown(_) => 4,
        }
    }

    /// Bytes required for a given number of elements in this format.
    /// Uses exact block sizes instead of approximate bits_per_weight.
    pub fn bytes_for_elements(&self, elements: u64) -> u64 {
        let bs = self.block_size() as u64;
        let bb = self.block_bytes() as u64;
        if bs == 0 {
            return 0;
        }
        let num_blocks = elements.div_ceil(bs);
        num_blocks * bb
    }
}

/// Dequantize a Q4_0 block: 32 weights packed as 16 bytes + 2 byte scale.
pub fn dequant_q4_0_block(block: &[u8], output: &mut [f32; 32]) {
    // Q4_0: 2 bytes scale (f16) + 16 bytes data (32 nibbles)
    if block.len() < 18 {
        return;
    }
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for i in 0..16 {
        let byte = block[2 + i];
        let lo = (byte & 0x0F) as i8 - 8;
        let hi = ((byte >> 4) & 0x0F) as i8 - 8;
        output[i * 2] = lo as f32 * scale;
        output[i * 2 + 1] = hi as f32 * scale;
    }
}

/// Dequantize a Q8_0 block: 32 weights as 32 int8 values + 2 byte scale.
pub fn dequant_q8_0_block(block: &[u8], output: &mut [f32; 32]) {
    if block.len() < 34 {
        return;
    }
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for i in 0..32 {
        output[i] = block[2 + i] as i8 as f32 * scale;
    }
}

/// Dequantize a Q5_0 block: 32 weights with 5 bits each.
/// Layout: 2 bytes scale (f16) + 4 bytes high-bit mask + 16 bytes low nibbles
/// Total: 22 bytes per block of 32 weights.
///
/// High bit layout matches llama.cpp: bits 0-15 are for weights 0-15 (low nibbles),
/// bits 16-31 are for weights 16-31 (high nibbles).
pub fn dequant_q5_0_block(block: &[u8], output: &mut [f32; 32]) {
    if block.len() < 22 {
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    for j in 0..16 {
        let byte = block[6 + j];
        let lo_nibble = byte & 0x0F;
        let hi_nibble = (byte >> 4) & 0x0F;

        // High bit for weight j (low nibble) is at qh bit j
        let xh_0 = ((qh >> j) & 1) as u8;
        // High bit for weight j+16 (high nibble) is at qh bit j+16
        let xh_1 = ((qh >> (j + 16)) & 1) as u8;

        let x0 = (lo_nibble | (xh_0 << 4)) as i32 - 16;
        let x1 = (hi_nibble | (xh_1 << 4)) as i32 - 16;

        output[j] = x0 as f32 * scale;
        output[j + 16] = x1 as f32 * scale;
    }
}

/// Dequantize a Q6_K block: 256 weights.
/// Layout: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes.
///
/// Follows llama.cpp dequant_row_q6_K exactly: processes two halves of 128
/// elements each, with interleaved ql/qh access in groups of 32.
pub fn dequant_q6k_block(block: &[u8], output: &mut [f32; 256]) {
    if block.len() < 210 {
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    // Process two halves of 128 elements each (matching llama.cpp)
    for half in 0..2 {
        let ql_off = half * 64;
        let qh_off = half * 32;
        let sc_off = half * 8;
        let y_off = half * 128;

        for l in 0..32 {
            let is = l / 16; // sub-block index (0 or 1)

            let q1 = ((ql[ql_off + l] & 0xF) | ((qh[qh_off + l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[ql_off + l + 32] & 0xF) | (((qh[qh_off + l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 3) << 4)) as i32 - 32;

            let sc1 = scales[sc_off + is] as i8 as f32;
            let sc2 = scales[sc_off + is + 2] as i8 as f32;
            let sc3 = scales[sc_off + is + 4] as i8 as f32;
            let sc4 = scales[sc_off + is + 6] as i8 as f32;

            output[y_off + l] = d * sc1 * q1 as f32;
            output[y_off + l + 32] = d * sc2 * q2 as f32;
            output[y_off + l + 64] = d * sc3 * q3 as f32;
            output[y_off + l + 96] = d * sc4 * q4 as f32;
        }
    }
}

/// Dequantize a Q4_K block: 256 weights.
/// Layout: d (f16) + dmin (f16) + scales[12] + qs[128]
pub fn dequant_q4k_block(block: &[u8], output: &mut [f32; 256]) {
    // Q4_K_M: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = 144 bytes
    if block.len() < 144 {
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales_raw = &block[4..16];
    let qs = &block[16..144];

    // Decode the 8 scale/min pairs from 12 bytes
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for i in 0..4 {
        sc[i] = scales_raw[i] & 0x3F;
        mn[i] = scales_raw[i + 4] & 0x3F;
        sc[i + 4] = (scales_raw[i] >> 6) | ((scales_raw[i + 8] & 0x0F) << 2);
        mn[i + 4] = (scales_raw[i + 4] >> 6) | ((scales_raw[i + 8] >> 4) << 2);
    }

    // First 128 values from low nibbles, second 128 from high nibbles
    // (matching llama.cpp layout)
    for j in 0..128 {
        let block_idx = j / 32;
        let low = qs[j] & 0x0F;
        let high = (qs[j] >> 4) & 0x0F;

        output[j] = d * sc[block_idx] as f32 * low as f32 - dmin * mn[block_idx] as f32;
        output[j + 128] =
            d * sc[block_idx + 4] as f32 * high as f32 - dmin * mn[block_idx + 4] as f32;
    }
}

/// Dequantize a Q5_K block: 256 weights.
/// Layout: d (f16) + dmin (f16) + scales[12] + qh[32] + ql[128]
/// Each weight is 5 bits: 4 from ql + 1 from qh.
pub fn dequant_q5k_block(block: &[u8], output: &mut [f32; 256]) {
    // Q5_K: 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (ql) = 176 bytes
    if block.len() < 176 {
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales_raw = &block[4..16];
    let qh = &block[16..48]; // high bits
    let ql = &block[48..176]; // low 4 bits

    // Decode scale/min pairs (same as Q4_K)
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for i in 0..4 {
        sc[i] = scales_raw[i] & 0x3F;
        mn[i] = scales_raw[i + 4] & 0x3F;
        sc[i + 4] = (scales_raw[i] >> 6) | ((scales_raw[i + 8] & 0x0F) << 2);
        mn[i + 4] = (scales_raw[i + 4] >> 6) | ((scales_raw[i + 8] >> 4) << 2);
    }

    // First 128 values from low nibbles, second 128 from high nibbles
    // (matching llama.cpp Q5_K layout)
    for j in 0..128 {
        let block_idx = j / 32;
        let low = ql[j] & 0x0F;
        let high = (ql[j] >> 4) & 0x0F;

        // High bits from qh
        let qh_lo = (qh[j / 8] >> (j % 8)) & 1;
        let qh_hi = (qh[(j + 128) / 8] >> ((j + 128) % 8)) & 1;

        let q_lo = (low | (qh_lo << 4)) as u32;
        let q_hi = (high | (qh_hi << 4)) as u32;

        output[j] = d * sc[block_idx] as f32 * q_lo as f32 - dmin * mn[block_idx] as f32;
        output[j + 128] =
            d * sc[block_idx + 4] as f32 * q_hi as f32 - dmin * mn[block_idx + 4] as f32;
    }
}

/// Convert f16 (as u16 bits) to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut mant = mant;
        let mut exp = exp;
        while (mant & 0x400) == 0 {
            mant <<= 1;
            exp = exp.wrapping_sub(1);
        }
        let exp = (exp.wrapping_add(127).wrapping_sub(15).wrapping_add(1)) & 0xFF;
        let mant = (mant & 0x3FF) << 13;
        return f32::from_bits((sign << 31) | (exp << 23) | mant);
    }

    if exp == 31 {
        let mant = mant << 13;
        return f32::from_bits((sign << 31) | (0xFF << 23) | mant);
    }

    let exp = (exp + 127 - 15) & 0xFF;
    let mant = mant << 13;
    f32::from_bits((sign << 31) | (exp << 23) | mant)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32() {
        // f16 representation of 1.0: sign=0, exp=15, mant=0 => 0x3C00
        let val = f16_to_f32(0x3C00);
        assert!((val - 1.0).abs() < 1e-6);

        // f16 representation of 0.0
        let val = f16_to_f32(0x0000);
        assert_eq!(val, 0.0);

        // f16 representation of -1.0: 0xBC00
        let val = f16_to_f32(0xBC00);
        assert!((val - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_quant_format_from_gguf() {
        assert_eq!(QuantFormat::from_gguf_type(0), QuantFormat::F32);
        assert_eq!(QuantFormat::from_gguf_type(2), QuantFormat::Q4_0);
        assert_eq!(QuantFormat::from_gguf_type(8), QuantFormat::Q8_0);
        assert_eq!(QuantFormat::from_gguf_type(999), QuantFormat::Unknown(999));
    }

    #[test]
    fn test_dequant_q4_0() {
        // Q4_0 block: 2 bytes scale + 16 bytes data (32 nibbles)
        let mut block = vec![0x00, 0x3C]; // f16 1.0 scale
                                          // Pack nibbles: each byte has low and high nibble (8 = zero-centered)
        block.extend_from_slice(&[0x88; 16]);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        // All values should be 0.0 (nibble 8 - 8 = 0, times scale 1.0)
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "q4_0 dequant: expected ~0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q6k_zeroed() {
        // Q6_K block with all zeros should produce all zeros
        let block = vec![0u8; 210];
        let mut output = [0.0f32; 256];
        dequant_q6k_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "q6k dequant: expected ~0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4k_zeroed() {
        // Q4_K block with all zeros should produce all zeros
        let block = vec![0u8; 144];
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "q4k dequant: expected ~0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q5k_zeroed() {
        let block = vec![0u8; 176];
        let mut output = [0.0f32; 256];
        dequant_q5k_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "q5k dequant: expected ~0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q5k_short_block() {
        let block = vec![0u8; 10];
        let mut output = [1.0f32; 256];
        dequant_q5k_block(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q6k_short_block() {
        // Too-short block should zero-fill
        let block = vec![0u8; 10];
        let mut output = [1.0f32; 256];
        dequant_q6k_block(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q8_0() {
        // Build a simple Q8_0 block: scale=1.0 (f16: 0x3C00), values 0..31
        let mut block = vec![0x00, 0x3C]; // f16 1.0
        for i in 0..32u8 {
            block.push(i);
        }
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!((val - i as f32).abs() < 1e-3, "mismatch at {i}");
        }
    }
}
