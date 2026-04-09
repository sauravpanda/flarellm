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
            QuantFormat::Q2K | QuantFormat::Q3K | QuantFormat::Q4K
            | QuantFormat::Q5K | QuantFormat::Q6K => 256,
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
    fn test_dequant_q8_0() {
        // Build a simple Q8_0 block: scale=1.0 (f16: 0x3C00), values 0..31
        let mut block = vec![0x00, 0x3C]; // f16 1.0
        for i in 0..32u8 {
            block.push(i);
        }
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        for i in 0..32 {
            assert!((output[i] - i as f32).abs() < 1e-3, "mismatch at {i}");
        }
    }
}
