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

    /// Bytes required for a given number of elements in this format.
    pub fn bytes_for_elements(&self, elements: u64) -> u64 {
        (elements as f64 * self.bits_per_weight() as f64 / 8.0).ceil() as u64
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

/// Dequantize a Q6_K block: 256 weights.
/// Layout: quantized_data[128] + scales[8] (Q8) + d (f16)
/// Each weight is 6 bits, packed in groups.
pub fn dequant_q6k_block(block: &[u8], output: &mut [f32; 256]) {
    // Q6_K block: 128 bytes quant data + 64 bytes high bits + 16 bytes scales + 2 bytes d
    // Total: 210 bytes for 256 weights
    if block.len() < 210 {
        // Fallback: zero fill for malformed blocks
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128]; // low 4 bits
    let qh = &block[128..192]; // high 2 bits
    let scales = &block[192..208]; // 16 int8 scales

    for j in 0..256 {
        let ql_byte = ql[j / 2];
        let low4 = if j % 2 == 0 {
            ql_byte & 0x0F
        } else {
            (ql_byte >> 4) & 0x0F
        };

        let qh_byte = qh[j / 4];
        let high2 = (qh_byte >> ((j % 4) * 2)) & 0x03;

        let q = ((high2 as i32) << 4) | (low4 as i32);
        let q = q - 32; // center around zero

        let scale_idx = j / 16;
        let sc = scales[scale_idx] as i8 as f32;

        output[j] = d * sc * q as f32;
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

    for (j, out) in output.iter_mut().enumerate() {
        let block_idx = j / 32;
        let byte_idx = j / 2;
        let nibble = if j % 2 == 0 {
            qs[byte_idx] & 0x0F
        } else {
            (qs[byte_idx] >> 4) & 0x0F
        };

        let scale = d * sc[block_idx] as f32;
        let min = dmin * mn[block_idx] as f32;
        *out = scale * nibble as f32 - min;
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
