/// Quantization format identifiers matching GGUF type IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    F32,
    F16,
    BF16,
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
    IQ4NL,
    IQ4XS,
    IQ3S,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    Unknown(u32),
}

impl QuantFormat {
    pub fn from_gguf_type(type_id: u32) -> Self {
        match type_id {
            0 => QuantFormat::F32,
            1 => QuantFormat::F16,
            32 => QuantFormat::BF16,
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
            16 => QuantFormat::IQ2XXS,
            17 => QuantFormat::IQ2XS,
            18 => QuantFormat::IQ3XXS,
            20 => QuantFormat::IQ4NL,
            22 => QuantFormat::IQ4XS,
            26 => QuantFormat::IQ3S,
            other => QuantFormat::Unknown(other),
        }
    }

    /// Bits per weight for this format.
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            QuantFormat::F32 => 32.0,
            QuantFormat::F16 | QuantFormat::BF16 => 16.0,
            QuantFormat::Q4_0 | QuantFormat::Q4_1 | QuantFormat::Q4K => 4.5,
            QuantFormat::Q5_0 | QuantFormat::Q5_1 | QuantFormat::Q5K => 5.5,
            QuantFormat::Q8_0 | QuantFormat::Q8_1 => 8.5,
            QuantFormat::Q2K => 2.6,
            QuantFormat::Q3K => 3.4,
            QuantFormat::Q6K => 6.6,
            QuantFormat::IQ4NL => 4.5, // 4 bits + 2-byte scale per 32 weights
            QuantFormat::IQ4XS => 4.25, // 136 bytes per 256 weights
            QuantFormat::IQ3S => 3.4375, // 110 bytes per 256 weights
            QuantFormat::IQ2XXS => 2.0625, // 66 bytes per 256 weights
            QuantFormat::IQ2XS => 2.3125, // 74 bytes per 256 weights
            QuantFormat::IQ3XXS => 3.0625, // 98 bytes per 256 weights
            QuantFormat::Unknown(_) => 32.0, // assume worst case
        }
    }

    /// Block size for quantized formats (number of weights per block).
    pub fn block_size(&self) -> usize {
        match self {
            QuantFormat::F32 | QuantFormat::F16 | QuantFormat::BF16 => 1,
            QuantFormat::Q4_0 | QuantFormat::Q4_1 | QuantFormat::IQ4NL => 32,
            QuantFormat::IQ4XS | QuantFormat::IQ3S => 256,
            QuantFormat::Q5_0 | QuantFormat::Q5_1 => 32,
            QuantFormat::Q8_0 | QuantFormat::Q8_1 => 32,
            QuantFormat::Q2K
            | QuantFormat::Q3K
            | QuantFormat::Q4K
            | QuantFormat::Q5K
            | QuantFormat::Q6K
            | QuantFormat::IQ2XXS
            | QuantFormat::IQ2XS
            | QuantFormat::IQ3XXS => 256,
            QuantFormat::Unknown(_) => 1,
        }
    }

    /// Bytes per block for quantized formats.
    pub fn block_bytes(&self) -> usize {
        match self {
            QuantFormat::F32 => 4,
            QuantFormat::F16 | QuantFormat::BF16 => 2,
            QuantFormat::Q4_0 | QuantFormat::IQ4NL => 18, // 2 (scale) + 16 (nibbles)
            QuantFormat::Q4_1 => 20,                      // 2 (scale) + 2 (min) + 16 (nibbles)
            QuantFormat::Q5_0 => 22, // 2 (scale) + 4 (high bits) + 16 (nibbles)
            QuantFormat::Q5_1 => 24, // 2 (scale) + 2 (min) + 4 (high bits) + 16 (nibbles)
            QuantFormat::Q8_0 => 34, // 2 (scale) + 32 (int8)
            QuantFormat::Q8_1 => 36, // 2 (scale) + 2 (sum) + 32 (int8)
            QuantFormat::Q2K => 84,
            QuantFormat::Q3K => 110,
            QuantFormat::Q4K => 144, // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)
            QuantFormat::Q5K => 176, // 2 (d) + 2 (dmin) + 12 (scales) + 32 (qh) + 128 (ql)
            QuantFormat::Q6K => 210, // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
            QuantFormat::IQ2XXS => 66, // 2 (d) + 64 (qs[32] × u16) for 256 weights
            QuantFormat::IQ2XS => 74, // 2 (d) + 64 (qs[32] × u16) + 8 (scales[8]) for 256 weights
            QuantFormat::IQ3XXS => 98, // 2 (d) + 64 (qs[64] × u8) + 32 (scales_and_signs[8] × u32)
            QuantFormat::IQ4XS => 136, // 2 (d) + 2 (scales_h) + 4 (scales_l[4]) + 128 (qs[128])
            QuantFormat::IQ3S => 110, // 2 (d) + 64 (qs) + 8 (qh) + 32 (signs) + 4 (scales)
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

/// Lookup table for IQ4_NL dequantization (from llama.cpp).
/// Maps a 4-bit nibble [0, 15] to a signed quantized value.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Dequantize an IQ4_NL block: 32 weights via 16-entry lookup + f16 scale.
/// Layout: 2 bytes scale (f16) + 16 bytes qs (32 packed 4-bit nibbles). 18 bytes total.
pub fn dequant_iq4nl_block(block: &[u8], output: &mut [f32; 32]) {
    if block.len() < 18 {
        return;
    }
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for i in 0..16 {
        let byte = block[2 + i];
        let lo = KVALUES_IQ4NL[(byte & 0x0F) as usize];
        let hi = KVALUES_IQ4NL[((byte >> 4) & 0x0F) as usize];
        output[i * 2] = lo as f32 * scale;
        output[i * 2 + 1] = hi as f32 * scale;
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

/// Dequantize a Q4_1 block: 32 weights with additive bias. 20 bytes total.
///
/// Layout: `d[2]` (f16 scale) + `m[2]` (f16 min) + `qs[16]` (4-bit nibbles).
///
/// For weight `i`: `output[i] = d * q + m` where `q` is the 4-bit value (0..15).
pub fn dequant_q4_1_block(block: &[u8], output: &mut [f32; 32]) {
    if block.len() < 20 {
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    for i in 0..16 {
        let byte = block[4 + i];
        let lo = (byte & 0x0F) as f32;
        let hi = ((byte >> 4) & 0x0F) as f32;
        output[i * 2] = d * lo + m;
        output[i * 2 + 1] = d * hi + m;
    }
}

/// Dequantize a Q8_1 block: 32 int8 weights with scale. 36 bytes total.
///
/// Layout: `d[2]` (f16 scale) + `s[2]` (f16 sum, precomputed — not used for dequant)
/// + `qs[32]` (i8 quantized values).
///
/// For weight `i`: `output[i] = d * qs[i]`.
pub fn dequant_q8_1_block(block: &[u8], output: &mut [f32; 32]) {
    if block.len() < 36 {
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    // block[2..4] is the precomputed sum `s` — not needed for dequantization
    for i in 0..32 {
        output[i] = block[4 + i] as i8 as f32 * d;
    }
}

/// Dequantize a Q3_K block: 256 weights.
/// Layout (110 bytes): `hmask[32]` + `qs[64]` + `scales[12]` + `d[2]`
///
/// - `hmask[32]`: high bit (bit 2) for each weight — 8 packed per byte
/// - `qs[64]`:    low 2 bits — 4 packed per byte with the interleaved scheme below
/// - `scales[12]`: 8 sub-block scales (6-bit, offset -32) packed via llama.cpp transform
/// - `d[2]`:      f16 overall delta
///
/// Weight layout (matches llama.cpp `dequantize_row_q3_K`):
///
/// The 256 weights are split into two groups of 128 (outer loop, `oi = 0,1`).
/// Within each group, four sub-blocks of 32 (`si = 0..3`) process `qs[oi*32..(oi+1)*32]`
/// with bit shift `si*2`. The hmask bitmask cycles: `m = 1 << (oi*4 + si)`.
///
/// High-bit rule: `q -= if (hmask[l] & m) == 0 { 4 } else { 0 }` → range [-4, 3].
pub fn dequant_q3k_block(block: &[u8], output: &mut [f32; 256]) {
    if block.len() < 110 {
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let hmask = &block[0..32]; // high bit: 8 per byte
    let qs = &block[32..96]; // low 2 bits: 4 per byte (64 bytes total)
    let scales_raw = &block[96..108]; // 12 bytes → 8 scales
    let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));

    // Decode 8 scales from 12 bytes using the llama.cpp kmask transform.
    // b[0..11] = scales_raw bytes.
    // scales[i] (0..7) are 6-bit values, then subtract 32 for centering.
    let b = scales_raw;
    let scales: [i32; 8] = [
        ((b[0] & 0x0F) | (((b[8] >> 4) & 3) << 4)) as i32 - 32,
        ((b[1] & 0x0F) | (((b[9] >> 4) & 3) << 4)) as i32 - 32,
        ((b[2] & 0x0F) | (((b[10] >> 4) & 3) << 4)) as i32 - 32,
        ((b[3] & 0x0F) | (((b[11] >> 4) & 3) << 4)) as i32 - 32,
        ((b[4] & 0x0F) | (((b[8] >> 6) & 3) << 4)) as i32 - 32,
        ((b[5] & 0x0F) | (((b[9] >> 6) & 3) << 4)) as i32 - 32,
        ((b[6] & 0x0F) | (((b[10] >> 6) & 3) << 4)) as i32 - 32,
        ((b[7] & 0x0F) | (((b[11] >> 6) & 3) << 4)) as i32 - 32,
    ];

    let mut m: u8 = 1; // sliding hmask bit selector
    for oi in 0..2usize {
        // each outer iteration covers 128 weights and 32 qs bytes
        let qs_group = &qs[oi * 32..(oi + 1) * 32];
        for si in 0..4usize {
            let shift = si * 2;
            let scale_idx = oi * 4 + si;
            let d_scale = d * scales[scale_idx] as f32;
            for l in 0..32usize {
                let low2 = (qs_group[l] >> shift) & 3;
                let sub = if (hmask[l] & m) != 0 { 0i32 } else { 4 };
                let q = low2 as i32 - sub;
                output[oi * 128 + si * 32 + l] = d_scale * q as f32;
            }
            m <<= 1;
        }
    }
}

/// Dequantize a Q6_K block: 256 weights.
/// Layout: `ql[128]` + `qh[64]` + `scales[16]` + `d[2]` = 210 bytes.
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
/// Layout: d (f16) + dmin (f16) + `scales[12]` + `qs[128]`
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
/// Layout: d (f16) + dmin (f16) + `scales[12]` + `qh[32]` + `ql[128]`
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

/// Dequantize a Q2_K block: 256 weights, 84 bytes (matches llama.cpp `block_q2_K`).
///
/// Block layout:
/// - `qs[64]` — 2-bit quantized values, 4 weights per byte
/// - `scales[16]` — per-super-block nibbles: low nibble = scale, high nibble = min
/// - `d[2]` — f16 overall delta
/// - `dmin[2]` — f16 overall min delta
///
/// For weight `i`: `output[i] = d * (scales[i/16] & 0xF) * q2 - dmin * (scales[i/16] >> 4)`
/// where `q2 = (qs[i/4] >> (2*(i%4))) & 3`.
pub fn dequant_q2k_block(block: &[u8], output: &mut [f32; 256]) {
    if block.len() < 84 {
        for v in output.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let qs = &block[0..64];
    let scales = &block[64..80];
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));

    for i in 0..256usize {
        let sub = i / 16;
        let scale_nibble = (scales[sub] & 0x0F) as f32;
        let min_nibble = (scales[sub] >> 4) as f32;
        let shift = (i % 4) * 2;
        let q2 = ((qs[i / 4] >> shift) & 0x3) as f32;
        output[i] = d * scale_nibble * q2 - dmin * min_nibble;
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

/// Convert bfloat16 (as u16 bits) to f32.
///
/// BF16 shares the same exponent and sign layout as F32, but has only 7
/// mantissa bits (vs 23 in F32). Conversion is a simple left-shift by 16.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Maps a 7-bit sign index to an 8-bit sign mask (bit=1 means −1.0).
/// Source: llama.cpp ggml-common.h ksigns_iq2xs table.
pub const KSIGNS_IQ2XS: [u8; 128] = [
    0x00, 0x81, 0x82, 0x03, 0x84, 0x05, 0x06, 0x87, 0x88, 0x09, 0x0a, 0x8b, 0x0c, 0x8d, 0x8e, 0x0f,
    0x90, 0x11, 0x12, 0x93, 0x14, 0x95, 0x96, 0x17, 0x18, 0x99, 0x9a, 0x1b, 0x9c, 0x1d, 0x1e, 0x9f,
    0xa0, 0x21, 0x22, 0xa3, 0x24, 0xa5, 0xa6, 0x27, 0x28, 0xa9, 0xaa, 0x2b, 0xac, 0x2d, 0x2e, 0xaf,
    0x30, 0xb1, 0xb2, 0x33, 0xb4, 0x35, 0x36, 0xb7, 0xb8, 0x39, 0x3a, 0xbb, 0x3c, 0xbd, 0xbe, 0x3f,
    0xc0, 0x41, 0x42, 0xc3, 0x44, 0xc5, 0xc6, 0x47, 0x48, 0xc9, 0xca, 0x4b, 0xcc, 0x4d, 0x4e, 0xcf,
    0x50, 0xd1, 0xd2, 0x53, 0xd4, 0x55, 0x56, 0xd7, 0xd8, 0x59, 0x5a, 0xdb, 0x5c, 0xdd, 0xde, 0x5f,
    0x60, 0xe1, 0xe2, 0x63, 0xe4, 0x65, 0x66, 0xe7, 0xe8, 0x69, 0x6a, 0xeb, 0x6c, 0xed, 0xee, 0x6f,
    0xf0, 0x71, 0x72, 0xf3, 0x74, 0xf5, 0xf6, 0x77, 0x78, 0xf9, 0xfa, 0x7b, 0xfc, 0x7d, 0x7e, 0xff,
];

/// CPU reference dequantizer for IQ2_XS blocks (74 bytes → 256 f32 weights).
///
/// IQ2_XS block layout (GGUF type 17):
///   bytes 0-1:   d (f16 LE)
///   bytes 2-65:  qs\[32\] (u16 LE each): bits 8:0 = grid index, bits 15:9 = sign index
///   bytes 66-73: scales\[8\] (u8): lo nibble=dl1, hi nibble=dl2 per ib32 group
///
/// Grid lookup: 512-entry u64 grid (8 weight bytes per entry).
/// Sign lookup: `KSIGNS_IQ2XS` (8 sign bits per byte).
pub fn dequant_iq2xs_block(block: &[u8], grid: &[u64; 512], output: &mut [f32; 256]) {
    if block.len() < 74 {
        output.fill(0.0);
        return;
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    for ib32 in 0..8usize {
        let scale_byte = block[66 + ib32];
        let dl1 = d * (0.5 + (scale_byte & 0xf) as f32) * 0.25;
        let dl2 = d * (0.5 + (scale_byte >> 4) as f32) * 0.25;

        let qs_base = 2 + ib32 * 8; // byte offset of first u16 in this group
        let out_base = ib32 * 32;

        for j in 0..2usize {
            let dl = if j == 0 { dl1 } else { dl2 };
            for k in 0..2usize {
                let off = qs_base + (j * 2 + k) * 2;
                let qs16 = u16::from_le_bytes([block[off], block[off + 1]]) as usize;
                let grid_idx = qs16 & 0x1FF;
                let sign_7bit = qs16 >> 9;
                let signs_8bit = KSIGNS_IQ2XS[sign_7bit];
                let grid_val = grid[grid_idx];
                for i in 0..8usize {
                    let gb = ((grid_val >> (i * 8)) & 0xFF) as f32;
                    let sign = if (signs_8bit >> i) & 1 != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    output[out_base + j * 16 + k * 8 + i] = dl * gb * sign;
                }
            }
        }
    }
}

/// CPU reference dequantizer for IQ3_XXS blocks (98 bytes → 256 f32 weights).
///
/// IQ3_XXS block layout (GGUF type 18):
///   bytes 0-1:   d (f16 LE)
///   bytes 2-65:  qs\[64\] (u8): grid indices
///   bytes 66-97: scales_and_signs\[32\] (u8, read as 8 × u32 LE):
///                bits 31:28 = sub-scale, bits 27:0 = 4×7-bit sign indices
///
/// Grid lookup: 256-entry u32 grid (4 weight bytes per entry).
/// Sign lookup: `KSIGNS_IQ2XS` (8 sign bits per byte).
pub fn dequant_iq3xxs_block(block: &[u8], grid: &[u32; 256], output: &mut [f32; 256]) {
    if block.len() < 98 {
        output.fill(0.0);
        return;
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    for ib32 in 0..8usize {
        let aux32 = u32::from_le_bytes([
            block[66 + ib32 * 4],
            block[67 + ib32 * 4],
            block[68 + ib32 * 4],
            block[69 + ib32 * 4],
        ]);
        let dl = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        let out_base = ib32 * 32;

        for l in 0..4usize {
            let qs1 = block[2 + ib32 * 8 + l * 2] as usize;
            let qs2 = block[2 + ib32 * 8 + l * 2 + 1] as usize;
            let sign_7bit = ((aux32 >> (7 * l)) & 127) as usize;
            let signs_8bit = KSIGNS_IQ2XS[sign_7bit];
            let g1 = grid[qs1];
            let g2 = grid[qs2];
            for j in 0..4usize {
                let gb1 = ((g1 >> (j * 8)) & 0xFF) as f32;
                let sign1 = if (signs_8bit >> j) & 1 != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + l * 8 + j] = dl * gb1 * sign1;

                let gb2 = ((g2 >> (j * 8)) & 0xFF) as f32;
                let sign2 = if (signs_8bit >> (j + 4)) & 1 != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + l * 8 + j + 4] = dl * gb2 * sign2;
            }
        }
    }
}

/// Dequantize an IQ4_XS block: 256 weights with 4-bit indices into KVALUES_IQ4NL + 6-bit per-group scale.
/// Block layout (136 bytes): 2 (d f16) + 2 (scales_h u16) + 4 (scales_l\[4\]) + 128 (qs\[128\])
/// For each ib32 in 0..8: ls = scales_l nibble | (scales_h 2-bit << 4), dl = d*(ls-32).
pub fn dequant_iq4xs_block(block: &[u8], output: &mut [f32; 256]) {
    if block.len() < 136 {
        output.fill(0.0);
        return;
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let scales_h = u16::from_le_bytes([block[2], block[3]]) as u32;
    for ib32 in 0..8usize {
        let sl_byte = block[4 + ib32 / 2];
        let ls_lo = (sl_byte >> ((ib32 % 2) * 4)) & 0xF;
        let ls_hi = ((scales_h >> (ib32 * 2)) & 3) as u8;
        let ls = ls_lo | (ls_hi << 4);
        let dl = d * (ls as i32 - 32) as f32;
        let out_base = ib32 * 32;
        for j in 0..16usize {
            let byte = block[8 + ib32 * 16 + j];
            output[out_base + j] = dl * KVALUES_IQ4NL[(byte & 0xF) as usize] as f32;
            output[out_base + j + 16] = dl * KVALUES_IQ4NL[(byte >> 4) as usize] as f32;
        }
    }
}

/// Dequantize an IQ3_S block: 256 weights from a 512-entry uint32 grid with 9-bit indices.
/// Block layout (110 bytes): 2 (d f16) + 64 (qs) + 8 (qh) + 32 (signs) + 4 (scales)
/// For each ib32: db = d*(1+2*nibble); grid_idx = qs_low8 | (qh_bit << 8).
pub fn dequant_iq3s_block(block: &[u8], grid: &[u32; 512], output: &mut [f32; 256]) {
    if block.len() < 110 {
        output.fill(0.0);
        return;
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for ib32 in 0..8usize {
        let scale_byte = block[106 + ib32 / 2];
        let nibble = (scale_byte >> ((ib32 % 2) * 4)) & 0xF;
        let db = d * (1.0 + 2.0 * nibble as f32);
        let qh_byte = block[66 + ib32];
        let out_base = ib32 * 32;
        for l in 0..4usize {
            let qs1 = block[2 + 8 * ib32 + 2 * l] as usize;
            let qs2 = block[2 + 8 * ib32 + 2 * l + 1] as usize;
            let grid_idx1 = qs1 | (((qh_byte >> (2 * l)) & 1) as usize) << 8;
            let grid_idx2 = qs2 | (((qh_byte >> (2 * l + 1)) & 1) as usize) << 8;
            let signs_byte = block[74 + 4 * ib32 + l];
            let g1 = grid[grid_idx1];
            let g2 = grid[grid_idx2];
            for j in 0..4usize {
                let gb1 = ((g1 >> (j * 8)) & 0xFF) as f32;
                let sign1 = if (signs_byte >> j) & 1 != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + l * 8 + j] = db * gb1 * sign1;
                let gb2 = ((g2 >> (j * 8)) & 0xFF) as f32;
                let sign2 = if (signs_byte >> (j + 4)) & 1 != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + l * 8 + j + 4] = db * gb2 * sign2;
            }
        }
    }
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

    #[test]
    fn test_dequant_q3k_zeroed() {
        // Q3_K block with all zeros should produce all zeros
        let block = vec![0u8; 110];
        let mut output = [1.0f32; 256];
        dequant_q3k_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "q3k dequant: expected ~0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q3k_short_block() {
        // Too-short block should zero-fill (graceful bounds check)
        let block = vec![0u8; 10];
        let mut output = [1.0f32; 256];
        dequant_q3k_block(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q3k_nonzero() {
        // Build a block where all 256 weights equal 1.
        //
        // For q=1: we need low2=1 and hmask bit SET (no subtraction of 4).
        // low2=1 means qs byte = 0x55 (01 repeated 4×).
        // hmask bit set for all weights: hmask byte = 0xFF for all 32 bytes.
        // scales: set all 8 to 33 raw (33-32=1 as signed).
        //   scale[0..3] = (b[0..3] & 0x0F) | (((b[8..11] >> 4) & 3) << 4) = 33
        //   Simplest: b[0..3] = 1 (lower 4 bits = 1), b[8..11] upper nibble = 2 →
        //   Actually: (b[i]&0x0F) = 1 and (b[i+8]>>4)&3 = 2 → scale = 1|(2<<4) = 33 ✓
        //   scale[4..7] = (b[4..7] & 0x0F) | (((b[8..11] >> 6) & 3) << 4) = 33
        //   (b[i+4]&0x0F) = 1 and (b[i+8]>>6)&3 = 2 → scale = 1|(2<<4) = 33 ✓
        //   So b[0..7] = 1, b[8..11] = 0x20 (upper nibble 0x2, lower nibble 0).
        //   But we also need (b[i+8]>>4)&3 = 2 (bits 4-5 of b[i+8]).
        //   b[8] = 0x20: bits[4..5] = 0x2>>0 = 0x20 >> 4 & 3 = (0x20>>4)&3 = 2 ✓
        //                bits[6..7] = (0x20>>6)&3 = 0. Not 2.
        //   Try b[8..11] = 0x21: upper nibble bits[4..5]=2, bits[6..7]=0. scale[0..3]=33, scale[4..7]=1.
        //   Use b[8..11] = 0xA0: bits[4..5]=(0xA0>>4)&3=0x0A&3=2, bits[6..7]=(0xA0>>6)&3=0x02&3=2.
        //   → scale[0..3]=33, scale[4..7]=33 ✓
        //
        // d = 1.0 (f16: 0x3C00 → bytes [0x00, 0x3C])
        let mut block = vec![0u8; 110];
        // hmask[0..32]: all 0xFF → hmask bit set for all weights
        block[0..32].fill(0xFF);
        // qs[32..96]: all 0x55 → low2=1 for all positions
        block[32..96].fill(0x55);
        // scales_raw[96..108]: b[0..7]=1, b[8..11]=0xA0 → all 8 scales = 33 → signed = 1
        block[96..104].fill(1);
        block[104..108].fill(0xA0);
        // d at bytes 108..110
        block[108] = 0x00;
        block[109] = 0x3C; // f16 1.0

        let mut output = [0.0f32; 256];
        dequant_q3k_block(&block, &mut output);

        for (i, &val) in output.iter().enumerate() {
            assert!((val - 1.0).abs() < 1e-4, "expected 1.0 at {i}, got {val}");
        }
    }

    #[test]
    fn test_dequant_q2k_zeroed() {
        // All-zero block: all weights should be 0.0
        let block = vec![0u8; 84];
        let mut output = [1.0f32; 256];
        dequant_q2k_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "q2k zeroed: expected 0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q2k_short_block() {
        // Too-short block should zero-fill
        let block = vec![0u8; 10];
        let mut output = [1.0f32; 256];
        dequant_q2k_block(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q2k_nonzero() {
        // Build a block where output = d * scale * q2 - dmin * min.
        //
        // Strategy: set dmin=0, d=1.0, scale nibble=2 for all sub-blocks, q2=1 for all weights.
        // Expected: output[i] = 1.0 * 2.0 * 1.0 - 0.0 = 2.0 for all i.
        //
        // Block layout:
        //   qs[0..64]: q2=1 for all 256 weights → each byte = 0x55 (01 repeated 4×)
        //   scales[64..80]: scale nibble=2, min nibble=0 → each byte = 0x02
        //   d at [80..82]: f16 1.0 = 0x3C00 → [0x00, 0x3C]
        //   dmin at [82..84]: f16 0.0 = 0x0000 → [0x00, 0x00]
        let mut block = vec![0u8; 84];
        block[0..64].fill(0x55); // q2=1 for all 4 weights per byte
        block[64..80].fill(0x02); // scale nibble=2, min nibble=0
        block[80] = 0x00;
        block[81] = 0x3C; // f16 1.0 for d
                          // dmin stays 0x0000

        let mut output = [0.0f32; 256];
        dequant_q2k_block(&block, &mut output);

        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - 2.0).abs() < 1e-4,
                "q2k nonzero: expected 2.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_1_zeroed() {
        // All-zero block: all weights should be 0.0 (d=0, m=0, q=0)
        let block = vec![0u8; 20];
        let mut output = [1.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "q4_1 zeroed: expected 0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_1_short_block() {
        let block = vec![0u8; 5];
        let mut output = [1.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q4_1_nonzero() {
        // d=1.0, m=0.0, all nibbles=3 → expected output[i] = 1.0*3 + 0.0 = 3.0
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0 for d
                         // m stays 0.0 (block[2..4] = 0)
        block[4..20].fill(0x33); // low nibble=3, high nibble=3
        let mut output = [0.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - 3.0).abs() < 1e-4,
                "q4_1 nonzero: expected 3.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_1_with_min() {
        // d=1.0, m=1.0, all nibbles=2 → expected output[i] = 1.0*2 + 1.0 = 3.0
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0 for d
        block[2] = 0x00;
        block[3] = 0x3C; // f16 1.0 for m
        block[4..20].fill(0x22); // low nibble=2, high nibble=2
        let mut output = [0.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - 3.0).abs() < 1e-4,
                "q4_1 with min: expected 3.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q8_1_zeroed() {
        // All-zero block: d=0, qs=0 → all outputs = 0.0
        let block = vec![0u8; 36];
        let mut output = [1.0f32; 32];
        dequant_q8_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "q8_1 zeroed: expected 0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q8_1_short_block() {
        let block = vec![0u8; 5];
        let mut output = [1.0f32; 32];
        dequant_q8_1_block(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_q8_1_nonzero() {
        // d=1.0, qs[i]=i as i8 for i in 0..32 → output[i] = i as f32
        let mut block = vec![0u8; 36];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0 for d
                         // block[2..4] is the sum field — leave as 0 (not used for dequant)
        for i in 0..32u8 {
            block[4 + i as usize] = i; // i8 values 0..31
        }
        let mut output = [0.0f32; 32];
        dequant_q8_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - i as f32).abs() < 1e-3,
                "q8_1 nonzero: expected {i}.0 at {i}, got {val}"
            );
        }
    }

    // ── IQ2_XS reference dequant tests ────────────────────────────────────────

    /// Minimal IQ2_XS grid fixture: only entry 0 is set (all bytes = 0x08).
    /// All other entries are zero.
    fn iq2xs_grid_fixture() -> Box<[u64; 512]> {
        let mut g = vec![0u64; 512];
        // grid[0] = 0x0808080808080808 (each of the 8 bytes is 8)
        g[0] = 0x0808080808080808u64;
        g[1] = 0x0808082bu64; // slightly different for sign test
        Box::new(g.try_into().unwrap())
    }

    /// Build a minimal IQ2_XS block (74 bytes):
    ///  - d = 1.0 (f16 0x3C00)
    ///  - all qs = 0 (grid_idx=0, sign_7bit=0)
    ///  - all scales = 0 (dl = 1.0 * (0.5+0) * 0.25 = 0.125)
    fn make_iq2xs_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; 74];
        block[0] = 0x00; // f16 1.0 low byte
        block[1] = 0x3C; // f16 1.0 high byte
                         // qs[0..32] as u16 all = 0 (already zero)
                         // scales[0..8] all = 0 (already zero)
        block
    }

    #[test]
    fn test_dequant_iq2xs_zeroed() {
        let grid = iq2xs_grid_fixture();
        let block = make_iq2xs_zero_block();
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);

        // d=1, scale=0 → dl = 0.125; grid[0] bytes all = 8; sign_7bit=0 → all +1
        // Expected: 0.125 * 8 = 1.0 for all 256 weights
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs zero block: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xs_max_scale() {
        let grid = iq2xs_grid_fixture();
        let mut block = make_iq2xs_zero_block();
        // Set scales[0] = 0xFF → dl1 = 1*(0.5+15)*0.25 = 3.875, dl2 = 3.875
        block[66] = 0xFF;
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);

        // First 32 weights: dl = 3.875; grid[0] all 8; sign=+1 → expected 31.0
        let expected = 3.875 * 8.0;
        for &v in &output[0..32] {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2xs max scale: got {v}, expected {expected}"
            );
        }
        // Remaining groups still have scale 0 → expected 1.0
        for &v in &output[32..256] {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs other groups: got {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xs_sign_inversion() {
        let grid = iq2xs_grid_fixture();
        let mut block = make_iq2xs_zero_block();
        // Set qs[0] u16 = (sign_7bit=1 << 9) | grid_idx=0 = 0x0200
        // KSIGNS_IQ2XS[1] = 0x81 = 0b10000001 → bits 0 and 7 flip sign
        block[2] = 0x00; // low byte of u16
        block[3] = 0x02; // high byte: sign_7bit = 1, grid_idx = 0
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);

        // dl = 0.125; grid[0] all 8
        // KSIGNS_IQ2XS[1] = 0x81: bit0=1 (-1), bits 1-6=0 (+1), bit7=1 (-1)
        assert!((output[0] - (-1.0)).abs() < 1e-5, "bit0 should be -1");
        assert!((output[1] - 1.0).abs() < 1e-5, "bit1 should be +1");
        assert!((output[7] - (-1.0)).abs() < 1e-5, "bit7 should be -1");
    }

    // ── IQ3_XXS reference dequant tests ───────────────────────────────────────

    /// Minimal IQ3_XXS grid fixture: only entry 0 is set (all bytes = 0x04).
    fn iq3xxs_grid_fixture() -> Box<[u32; 256]> {
        let mut g = vec![0u32; 256];
        g[0] = 0x04040404u32; // each byte = 4
        Box::new(g.try_into().unwrap())
    }

    /// Build a minimal IQ3_XXS block (98 bytes):
    ///  - d = 1.0 (f16 0x3C00)
    ///  - all qs = 0, scales_and_signs = 0
    ///  - dl = 1.0 * (0.5 + 0) * 0.5 = 0.25
    fn make_iq3xxs_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; 98];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block
    }

    #[test]
    fn test_dequant_iq3xxs_zeroed() {
        let grid = iq3xxs_grid_fixture();
        let block = make_iq3xxs_zero_block();
        let mut output = [0.0f32; 256];
        dequant_iq3xxs_block(&block, &grid, &mut output);

        // d=1, aux32=0 → dl = 0.5*0.5=0.25; grid[0] bytes all=4; all signs +1
        // Expected: 0.25 * 4 = 1.0 for all 256 weights
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq3xxs zero block: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3xxs_max_subscale() {
        let grid = iq3xxs_grid_fixture();
        let mut block = make_iq3xxs_zero_block();
        // Set bits[31:28] = 0xF (max sub-scale) in scales_and_signs[0]
        // aux32 = 0xF0000000 → dl = 1.0 * (0.5 + 15) * 0.5 = 7.75
        block[69] = 0xF0; // high byte of first aux32 (byte 66+3)
        let mut output = [0.0f32; 256];
        dequant_iq3xxs_block(&block, &grid, &mut output);

        let expected = 7.75 * 4.0; // dl * grid_byte
        for &v in &output[0..32] {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq3xxs max subscale: got {v}, expected {expected}"
            );
        }
        // Groups 1..7 still dl = 0.25 → 1.0
        for &v in &output[32..256] {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq3xxs other groups: got {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3xxs_sign_inversion() {
        let grid = iq3xxs_grid_fixture();
        let mut block = make_iq3xxs_zero_block();
        // Set sign_7bit for l=0 of ib32=0: aux32 bits[6:0] = 1
        // KSIGNS_IQ2XS[1] = 0x81 → bit0 and bit7 are -1, rest +1
        block[66] = 0x01; // aux32 low byte = 1
        let mut output = [0.0f32; 256];
        dequant_iq3xxs_block(&block, &grid, &mut output);

        // dl = 0.25; grid[0] all 4
        // l=0: signs_8bit = KSIGNS_IQ2XS[1] = 0x81
        //   j=0 (bit0=1) → -1; j=1,2,3 (bits1-3=0) → +1
        //   j+4=4 (bit4=0) → +1; ...; j+4=7 (bit7=1) → -1
        assert!((output[0] - (-1.0)).abs() < 1e-5, "iq3xxs j=0 should be -1");
        assert!((output[1] - 1.0).abs() < 1e-5, "iq3xxs j=1 should be +1");
        assert!(
            (output[7] - (-1.0)).abs() < 1e-5,
            "iq3xxs j+4=7 should be -1"
        );
    }

    #[test]
    fn test_dequant_iq2xs_short_block() {
        let grid = iq2xs_grid_fixture();
        let block = vec![0u8; 10]; // too short
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dequant_iq3xxs_short_block() {
        let grid = iq3xxs_grid_fixture();
        let block = vec![0u8; 10]; // too short
        let mut output = [0.0f32; 256];
        dequant_iq3xxs_block(&block, &grid, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    // ── IQ4_XS reference dequant tests ────────────────────────────────────────

    /// Build an IQ4_XS block (136 bytes) with:
    /// - d = 1.0 (f16 0x3C00)
    /// - ib32=0: ls_lo=1 (scales_l[0]=0x01), ls_hi=2 (scales_h=0x0002) → ls=33, dl=1.0
    /// - qs[8..24] = 0x88 → both nibbles=8 → KVALUES_IQ4NL[8]=1 → output[0..32]=1.0
    fn make_iq4xs_unit_block() -> Vec<u8> {
        let mut block = vec![0u8; 136];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[2] = 0x02;
        block[3] = 0x00; // scales_h: ib32=0 hi-bits = 2
        block[4] = 0x01; // scales_l: ib32=0 lo-nibble = 1
        block[8..24].fill(0x88); // nibble 8 → KVALUES_IQ4NL[8] = 1
        block
    }

    #[test]
    fn test_dequant_iq4xs_unit_scale() {
        let block = make_iq4xs_unit_block();
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        // ib32=0: ls=33, dl=1.0; nibble=8 → KVALUES_IQ4NL[8]=1 → 1.0*1=1.0
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq4xs unit: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4xs_zero_dl() {
        // ls=32 → dl=0.0 → all outputs 0 even with non-zero qs
        // ls_lo=0 (scales_l[0]=0x00), ls_hi=2 (scales_h=0x0002) → ls=32
        let mut block = vec![0u8; 136];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[2] = 0x02;
        block[3] = 0x00; // scales_h: ib32=0 hi=2; ls = 0|(2<<4)=32
        block[8..24].fill(0xFF); // max nibbles, but dl=0 → output=0
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "iq4xs zero dl: weight[{i}] = {v}, expected 0.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4xs_negative_scale() {
        // ls=31 → dl = 1*(31-32) = -1.0; nibble=8 → KVALUES_IQ4NL[8]=1 → output=-1.0
        // ls_lo=15 (scales_l[0]=0x0F), ls_hi=1 (scales_h=0x0001) → ls=15|(1<<4)=31
        let mut block = vec![0u8; 136];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[2] = 0x01;
        block[3] = 0x00; // scales_h: ib32=0 hi=1
        block[4] = 0x0F; // scales_l: ib32=0 lo=15
        block[8..24].fill(0x88); // nibble 8 → KVALUES_IQ4NL[8]=1
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-5,
                "iq4xs neg scale: weight[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4xs_short_block() {
        let block = vec![0u8; 10]; // too short
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    // ── IQ3_S reference dequant tests ─────────────────────────────────────────

    /// Minimal IQ3_S grid fixture: entry 0 = 0x05050505 (all bytes 5).
    fn iq3s_grid_fixture() -> Box<[u32; 512]> {
        let mut g = vec![0u32; 512];
        g[0] = 0x05050505u32;
        Box::new(g.try_into().unwrap())
    }

    /// Minimal IQ3_S block (110 bytes): d=1.0, all qs/qh/signs/scales zeroed.
    /// dl = 1.0*(1+2*0) = 1.0; grid[0] all 5; signs all +1 → all outputs = 5.0.
    fn make_iq3s_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; 110];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block
    }

    #[test]
    fn test_dequant_iq3s_zeroed() {
        let grid = iq3s_grid_fixture();
        let block = make_iq3s_zero_block();
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s zero: weight[{i}] = {v}, expected 5.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3s_max_scale() {
        let grid = iq3s_grid_fixture();
        let mut block = make_iq3s_zero_block();
        // nibble=15 → db = 1.0*(1+30) = 31.0; scales[0]=0xFF covers ib32=0 and ib32=1
        block[106] = 0xFF;
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        let expected = 31.0_f32 * 5.0;
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq3s max scale: weight[{i}] got {v}, expected {expected}"
            );
        }
        // Groups 2..7 have nibble=0 → db=1.0 → output=5.0
        for &v in &output[64..256] {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s other groups: got {v}, expected 5.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3s_sign_inversion() {
        let grid = iq3s_grid_fixture();
        let mut block = make_iq3s_zero_block();
        // signs[0] = 0x01: bit0=1 → weight[0] negated; bit1=0 → weight[1] positive
        block[74] = 0x01;
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        assert!(
            (output[0] - (-5.0)).abs() < 1e-5,
            "iq3s sign bit0: got {}",
            output[0]
        );
        assert!(
            (output[1] - 5.0).abs() < 1e-5,
            "iq3s weight1: got {}",
            output[1]
        );
    }

    #[test]
    fn test_dequant_iq3s_short_block() {
        let grid = iq3s_grid_fixture();
        let block = vec![0u8; 10]; // too short
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }
}
