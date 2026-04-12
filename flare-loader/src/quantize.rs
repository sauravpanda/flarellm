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
    IQ2S,
    IQ1S,
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
            19 => QuantFormat::IQ1S,
            21 => QuantFormat::IQ2S,
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
            QuantFormat::IQ2S => 2.5625, // 82 bytes per 256 weights
            QuantFormat::IQ1S => 1.5625, // 50 bytes per 256 weights
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
            | QuantFormat::IQ3XXS
            | QuantFormat::IQ2S
            | QuantFormat::IQ1S => 256,
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
            QuantFormat::IQ2S => 82, // 2 (d) + 32 (qs_lo) + 32 (signs) + 8 (qh) + 8 (scales)
            QuantFormat::IQ1S => 50, // 2 (d) + 32 (qs[32]) + 16 (qh[8] × u16)
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

/// CPU reference dequantizer for IQ2_XXS blocks (66 bytes → 256 f32 weights).
///
/// IQ2_XXS block layout (GGUF type 16):
///   bytes 0-1:   d (f16 LE)
///   bytes 2-65:  qs (8 × 8-byte chunks, one per 32-weight group)
///
/// For each ib32 ∈ \[0, 7\], the 8-byte chunk at offset 2 + 8*ib32 contains:
///   aux0 = u32 LE bytes 0-3: four 8-bit grid indices (one per 8-weight sub-group)
///   aux1 = u32 LE bytes 4-7: packed 4×7-bit sign indices + 4-bit sub-scale in bits 31:28
///   db = d * (0.5 + sub_scale) * 0.25
///   For l ∈ \[0, 3\]: grid_idx=(aux0>>(8*l))&0xFF, signs_7bit=(aux1>>(7*l))&127
///
/// Grid: 256-entry u64 (8 weight bytes each). Sign lookup: `KSIGNS_IQ2XS`.
pub fn dequant_iq2xxs_block(block: &[u8], grid: &[u64; 256], output: &mut [f32; 256]) {
    if block.len() < 66 {
        output.fill(0.0);
        return;
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for ib32 in 0..8usize {
        let base = 2 + ib32 * 8;
        let aux0 = u32::from_le_bytes([
            block[base],
            block[base + 1],
            block[base + 2],
            block[base + 3],
        ]);
        let aux1 = u32::from_le_bytes([
            block[base + 4],
            block[base + 5],
            block[base + 6],
            block[base + 7],
        ]);
        let sub_scale = (aux1 >> 28) as f32;
        let db = d * (0.5 + sub_scale) * 0.25;
        let out_base = ib32 * 32;
        for l in 0..4usize {
            let grid_idx = ((aux0 >> (8 * l)) & 0xFF) as usize;
            let signs_7bit = ((aux1 >> (7 * l)) & 127) as usize;
            let signs_8bit = KSIGNS_IQ2XS[signs_7bit];
            let grid_val = grid[grid_idx];
            let lo32 = (grid_val & 0xFFFFFFFF) as u32;
            let hi32 = (grid_val >> 32) as u32;
            for j in 0..4usize {
                let gb = ((lo32 >> (j * 8)) & 0xFF) as f32;
                let sign = if (signs_8bit >> j) & 1 != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + l * 8 + j] = db * gb * sign;
            }
            for j in 0..4usize {
                let gb = ((hi32 >> (j * 8)) & 0xFF) as f32;
                let sign = if (signs_8bit >> (j + 4)) & 1 != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + l * 8 + j + 4] = db * gb * sign;
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

/// Dequantize an IQ2_S block: 256 weights from a 1024-entry u64 grid with 10-bit indices.
/// Block layout (82 bytes): 2 (d f16) + 32 (qs_lo) + 32 (signs) + 8 (qh) + 8 (scales)
/// For each ib32: db = d*(0.5+nibble)*0.25; grid_idx = qs_lo | (2-bit qh << 8).
pub fn dequant_iq2s_block(block: &[u8], grid: &[u64; 1024], output: &mut [f32; 256]) {
    if block.len() < 82 {
        output.fill(0.0);
        return;
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for ib32 in 0..8usize {
        let scale = block[74 + ib32];
        let db0 = d * (0.5 + (scale & 0xF) as f32) * 0.25;
        let db1 = d * (0.5 + (scale >> 4) as f32) * 0.25;
        let qh_byte = block[66 + ib32];
        let out_base = ib32 * 32;
        for l in 0..4usize {
            let dl = if l < 2 { db0 } else { db1 };
            let qs_lo = block[2 + ib32 * 4 + l] as usize;
            let signs_byte = block[34 + ib32 * 4 + l];
            let grid_hi = ((qh_byte >> (2 * l)) & 3) as usize;
            let grid_idx = qs_lo | (grid_hi << 8);
            let grid_val = grid[grid_idx];
            let lo32 = (grid_val & 0xFFFFFFFF) as u32;
            let hi32 = (grid_val >> 32) as u32;
            for j in 0..4usize {
                let gb = ((lo32 >> (j * 8)) & 0xFF) as f32;
                let sign = if (signs_byte >> j) & 1 != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + l * 8 + j] = dl * gb * sign;
            }
            for j in 0..4usize {
                let gb = ((hi32 >> (j * 8)) & 0xFF) as f32;
                let sign = if (signs_byte >> (j + 4)) & 1 != 0 {
                    -1.0
                } else {
                    1.0
                };
                output[out_base + l * 8 + j + 4] = dl * gb * sign;
            }
        }
    }
}

/// Delta constant for IQ1_S dequantization.
const IQ1S_DELTA: f32 = 0.125;

/// Dequantize an IQ1_S block: 256 weights from a 2048-entry signed-byte grid with 11-bit indices.
/// Block layout (50 bytes): 2 (d f16) + 32 (qs\[32\]) + 16 (qh\[8\] × u16)
/// For each ib32: dl = d*(2*s+1) where s=(qh>>12)&7; delta=±IQ1S_DELTA from bit 15.
/// Grid index = qs\[l\] | (((qh>>(3*l))&7)<<8). Grid bytes are signed (−1, 0, +1).
pub fn dequant_iq1s_block(block: &[u8], grid: &[u64; 2048], output: &mut [f32; 256]) {
    if block.len() < 50 {
        output.fill(0.0);
        return;
    }
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for ib32 in 0..8usize {
        let qh_val = u16::from_le_bytes([block[34 + ib32 * 2], block[35 + ib32 * 2]]) as u32;
        let s = (qh_val >> 12) & 7;
        let dl = d * (2 * s + 1) as f32;
        let delta = if qh_val & 0x8000 != 0 {
            -IQ1S_DELTA
        } else {
            IQ1S_DELTA
        };
        let out_base = ib32 * 32;
        for l in 0..4usize {
            let qs_lo = block[2 + ib32 * 4 + l] as usize;
            let grid_hi = ((qh_val >> (3 * l)) & 7) as usize;
            let grid_idx = qs_lo | (grid_hi << 8);
            let grid_val = grid[grid_idx];
            let lo32 = (grid_val & 0xFFFFFFFF) as u32;
            let hi32 = (grid_val >> 32) as u32;
            for j in 0..4usize {
                let gb_u = (lo32 >> (j * 8)) & 0xFF;
                let gb = gb_u as i8 as f32;
                output[out_base + l * 8 + j] = dl * (gb + delta);
            }
            for j in 0..4usize {
                let gb_u = (hi32 >> (j * 8)) & 0xFF;
                let gb = gb_u as i8 as f32;
                output[out_base + l * 8 + j + 4] = dl * (gb + delta);
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
    fn test_f16_to_f32_negative_zero() {
        // 0x8000 = negative zero in f16 (sign=1, exp=0, mant=0)
        let val = f16_to_f32(0x8000);
        assert_eq!(val, 0.0);
        // Negative zero has its sign bit set
        assert!(val.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_positive_infinity() {
        // 0x7C00 = positive infinity in f16 (exp=31, mant=0)
        let val = f16_to_f32(0x7C00);
        assert!(val.is_infinite());
        assert!(val.is_sign_positive());
    }

    #[test]
    fn test_f16_to_f32_negative_infinity() {
        // 0xFC00 = negative infinity in f16 (sign=1, exp=31, mant=0)
        let val = f16_to_f32(0xFC00);
        assert!(val.is_infinite());
        assert!(val.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_nan() {
        // 0x7E00 = NaN in f16 (exp=31, mant != 0)
        let val = f16_to_f32(0x7E00);
        assert!(val.is_nan());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        // 0x0001 = smallest positive subnormal f16 (exp=0, mant=1)
        let val = f16_to_f32(0x0001);
        assert!(val > 0.0);
        assert!(val < 1e-4);
    }

    #[test]
    fn test_f16_to_f32_max_normal() {
        // 0x7BFF = 65504.0 (max finite f16 value)
        let val = f16_to_f32(0x7BFF);
        assert!((val - 65504.0).abs() < 1.0);
    }

    #[test]
    fn test_f16_to_f32_two() {
        // 0x4000 = 2.0 in f16 (sign=0, exp=16, mant=0)
        let val = f16_to_f32(0x4000);
        assert!((val - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_half() {
        // 0x3800 = 0.5 in f16 (sign=0, exp=14, mant=0)
        let val = f16_to_f32(0x3800);
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_one() {
        // 0x3F80 = 1.0 in bf16 (same bit layout as f32 1.0, upper 16 bits)
        let val = bf16_to_f32(0x3F80);
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_zero() {
        // 0x0000 = 0.0 in bf16
        let val = bf16_to_f32(0x0000);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_bf16_to_f32_negative_one() {
        // 0xBF80 = -1.0 in bf16
        let val = bf16_to_f32(0xBF80);
        assert!((val - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_two() {
        // 0x4000 = 2.0 in bf16
        let val = bf16_to_f32(0x4000);
        assert!((val - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32_infinity() {
        // 0x7F80 = positive infinity in bf16 (exp=255, mant=0)
        let val = bf16_to_f32(0x7F80);
        assert!(val.is_infinite());
        assert!(val.is_sign_positive());
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
    fn test_dequant_q4_0_short_block() {
        // block shorter than 18 bytes → output must not be modified
        let block = vec![0x00, 0x3C, 0x88u8]; // only 3 bytes
        let mut output = [99.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        assert!(
            output.iter().all(|&v| (v - 99.0).abs() < 1e-6),
            "short block must leave output unchanged"
        );
    }

    #[test]
    fn test_dequant_q4_0_zero_scale() {
        // scale = 0.0 → all outputs must be 0.0 regardless of nibble values
        let mut block = vec![0x00u8, 0x00]; // f16 0.0
        block.extend_from_slice(&[0xF0u8; 16]); // all max nibbles
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "zero scale: expected 0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_0_nonzero_nibbles() {
        // scale=1.0, block[2]=0xF0: lo nibble=0 → 0-8=-8, hi nibble=0xF=15 → 15-8=7
        let mut block = vec![0x00u8, 0x3C]; // f16 1.0
        block.push(0xF0); // lo=0, hi=15
        block.extend_from_slice(&[0x88u8; 15]);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        assert!(
            (output[0] - (-8.0)).abs() < 1e-4,
            "lo nibble=0: expected -8.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 7.0).abs() < 1e-4,
            "hi nibble=15: expected 7.0, got {}",
            output[1]
        );
        // remaining bytes are 0x88 → nibble 8-8=0
        for (i, &v) in output[2..].iter().enumerate() {
            assert!(v.abs() < 1e-5, "expected 0.0 at {}, got {}", i + 2, v);
        }
    }

    #[test]
    fn test_dequant_q4_0_negative_scale() {
        // scale=-1.0 (f16 0xBC00), block[2]=0xF9: lo=9→9-8=1, hi=0xF=15→7
        // output[0]= 1*(-1.0)=-1.0; output[1]= 7*(-1.0)=-7.0
        let mut block = vec![0x00u8, 0xBC]; // f16 -1.0
        block.push(0xF9); // lo=9, hi=15
        block.extend_from_slice(&[0x88u8; 15]);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        assert!(
            (output[0] - (-1.0)).abs() < 1e-4,
            "expected -1.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - (-7.0)).abs() < 1e-4,
            "expected -7.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_dequant_q4_0_all_nibbles_min() {
        // all qs=0x00: nibble 0-8=-8; scale=1.0 → all outputs=-8.0
        let mut block = vec![0x00u8, 0x3C]; // f16 1.0
        block.extend_from_slice(&[0x00u8; 16]);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - (-8.0)).abs() < 1e-4,
                "all-zero nibble: expected -8.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_0_max_nibble() {
        // all qs=0xFF: nibble 15-8=7; scale=2.0 (f16 0x4000) → all outputs=14.0
        let mut block = vec![0x00u8, 0x40]; // f16 2.0
        block.extend_from_slice(&[0xFFu8; 16]);
        let mut output = [0.0f32; 32];
        dequant_q4_0_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - 14.0).abs() < 1e-4,
                "max nibble: expected 14.0 at {i}, got {val}"
            );
        }
    }

    // ── Q5_0 tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_dequant_q5_0_short_block() {
        // block < 22 bytes → output zero-filled
        let block = vec![0x00u8, 0x3C, 0x00, 0x00]; // only 4 bytes
        let mut output = [9.0f32; 32];
        dequant_q5_0_block(&block, &mut output);
        assert!(
            output.iter().all(|&v| v == 0.0),
            "short block must zero output"
        );
    }

    #[test]
    fn test_dequant_q5_0_zero_scale() {
        // scale=0.0 → all outputs 0.0
        let mut block = vec![0x00u8; 22]; // scale=0.0 (default)
        block[6] = 0xFF; // nonzero nibbles
        let mut output = [0.0f32; 32];
        dequant_q5_0_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(v.abs() < 1e-6, "zero scale: expected 0.0 at {i}, got {v}");
        }
    }

    #[test]
    fn test_dequant_q5_0_zero_qh() {
        // qh=0, scale=1.0, block[6]=0x10 (lo=0, hi=1)
        // weight 0: (0|(0<<4))-16=-16 → output[0]=-16.0
        // weight 16: (1|(0<<4))-16=-15 → output[16]=-15.0
        let mut block = vec![0x00u8, 0x3C]; // scale=1.0
        block.extend_from_slice(&[0u8; 4]); // qh=0
        block.push(0x10); // data[0]: lo=0, hi=1
        block.extend_from_slice(&[0x88u8; 15]); // remaining: nibble 8|(0<<4)-16=-8
        let mut output = [0.0f32; 32];
        dequant_q5_0_block(&block, &mut output);
        assert!(
            (output[0] - (-16.0)).abs() < 1e-4,
            "expected -16.0, got {}",
            output[0]
        );
        assert!(
            (output[16] - (-15.0)).abs() < 1e-4,
            "expected -15.0, got {}",
            output[16]
        );
    }

    #[test]
    fn test_dequant_q5_0_high_bit_weight0() {
        // qh=0x00000001 (bit 0 set) → high bit for weight 0
        // block[6]=0x00 (lo nibble=0), scale=1.0
        // weight 0: (0|(1<<4))-16 = 16-16=0 → output[0]=0.0
        let mut block = vec![0x00u8, 0x3C]; // scale=1.0
        block.push(0x01); // qh low byte: bit 0=1
        block.extend_from_slice(&[0u8; 3]); // qh remaining
        block.push(0x00); // data[0]: lo=0
        block.extend_from_slice(&[0x88u8; 15]);
        let mut output = [0.0f32; 32];
        dequant_q5_0_block(&block, &mut output);
        assert!(output[0].abs() < 1e-4, "expected 0.0, got {}", output[0]);
    }

    #[test]
    fn test_dequant_q5_0_max_5bit() {
        // qh=0xFFFFFFFF, all nibbles=0xF → 5-bit = 0xF|(1<<4)=31-16=15; scale=1.0 → all 15.0
        let mut block = vec![0x00u8, 0x3C]; // scale=1.0
        block.extend_from_slice(&[0xFFu8; 4]); // qh=0xFFFFFFFF
        block.extend_from_slice(&[0xFFu8; 16]); // all max nibbles
        let mut output = [0.0f32; 32];
        dequant_q5_0_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 15.0).abs() < 1e-4,
                "max 5-bit: expected 15.0 at {i}, got {v}"
            );
        }
    }

    #[test]
    fn test_dequant_q5_0_min_5bit() {
        // qh=0, all nibbles=0x00 → 5-bit = 0-16=-16; scale=1.0 → all -16.0
        let mut block = vec![0x00u8, 0x3C]; // scale=1.0
        block.extend_from_slice(&[0u8; 4]); // qh=0
        block.extend_from_slice(&[0u8; 16]); // all zero nibbles
        let mut output = [0.0f32; 32];
        dequant_q5_0_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-16.0)).abs() < 1e-4,
                "min 5-bit: expected -16.0 at {i}, got {v}"
            );
        }
    }

    #[test]
    fn test_dequant_q5_0_negative_scale() {
        // scale=-1.0, qh=0, block[6]=0x00 → weight 0 = -16; output[0] = (-16)*(-1.0)=16.0
        let mut block = vec![0x00u8, 0xBC]; // scale=-1.0
        block.extend_from_slice(&[0u8; 4]); // qh=0
        block.push(0x00); // data[0]: lo=0
        block.extend_from_slice(&[0x88u8; 15]);
        let mut output = [0.0f32; 32];
        dequant_q5_0_block(&block, &mut output);
        assert!(
            (output[0] - 16.0).abs() < 1e-4,
            "negative scale: expected 16.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q5_0_high_nibble_high_bit() {
        // High bit for weight j+16 is at qh bit j+16
        // Set qh bit 16 (j=0 for high nibble): qh = 0x00010000
        // block[6]=0x00 (hi nibble=0), scale=1.0
        // weight 16: (0|(1<<4))-16 = 0 → output[16]=0.0
        let mut block = vec![0x00u8, 0x3C]; // scale=1.0
        block.push(0x00); // qh byte 0
        block.push(0x00); // qh byte 1
        block.push(0x01); // qh byte 2: bit 16 of u32 = byte[2] bit 0
        block.push(0x00); // qh byte 3
        block.push(0x00); // data[0]: lo=0, hi=0
        block.extend_from_slice(&[0x88u8; 15]);
        let mut output = [0.0f32; 32];
        dequant_q5_0_block(&block, &mut output);
        assert!(
            output[16].abs() < 1e-4,
            "high nibble high bit: expected 0.0, got {}",
            output[16]
        );
    }

    #[test]
    fn test_dequant_q8_0_short_block() {
        // block < 34 bytes → output must not be modified
        let block = vec![0x00u8, 0x3C, 0x01, 0x02]; // only 4 bytes
        let mut output = [55.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        assert!(
            output.iter().all(|&v| (v - 55.0).abs() < 1e-6),
            "short block must leave output unchanged"
        );
    }

    #[test]
    fn test_dequant_q8_0_zero_scale() {
        // scale=0.0 → all outputs 0.0 regardless of int8 values
        let mut block = vec![0x00u8, 0x00]; // f16 0.0
        block.extend_from_slice(&[0x7Fu8; 32]); // all i8=127
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "zero scale: expected 0.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q8_0_positive_values() {
        // scale=1.0, first byte i8=42 → output[0]=42.0
        let mut block = vec![0x00u8, 0x3C]; // f16 1.0
        block.push(42);
        block.extend_from_slice(&[0u8; 31]);
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        assert!(
            (output[0] - 42.0).abs() < 1e-4,
            "expected 42.0, got {}",
            output[0]
        );
        for (i, &v) in output[1..].iter().enumerate() {
            assert!(v.abs() < 1e-5, "expected 0.0 at {}, got {}", i + 1, v);
        }
    }

    #[test]
    fn test_dequant_q8_0_negative_values() {
        // scale=1.0, byte=0x80 → i8=-128 → output[0]=-128.0
        let mut block = vec![0x00u8, 0x3C]; // f16 1.0
        block.push(0x80); // i8 = -128
        block.extend_from_slice(&[0u8; 31]);
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        assert!(
            (output[0] - (-128.0)).abs() < 1e-4,
            "expected -128.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q8_0_mixed_signs() {
        // scale=1.0, qs[0]=1 (i8=1), qs[1]=0xFF (i8=-1)
        let mut block = vec![0x00u8, 0x3C]; // f16 1.0
        block.push(0x01);
        block.push(0xFF); // i8 = -1
        block.extend_from_slice(&[0u8; 30]);
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        assert!(
            (output[0] - 1.0).abs() < 1e-4,
            "expected 1.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - (-1.0)).abs() < 1e-4,
            "expected -1.0, got {}",
            output[1]
        );
    }

    #[test]
    fn test_dequant_q8_0_negative_scale() {
        // scale=-2.0 (f16 0xC000), qs[0]=2 (i8) → output[0]=2*(-2.0)=-4.0
        let mut block = vec![0x00u8, 0xC0]; // f16 -2.0
        block.push(0x02);
        block.extend_from_slice(&[0u8; 31]);
        let mut output = [0.0f32; 32];
        dequant_q8_0_block(&block, &mut output);
        assert!(
            (output[0] - (-4.0)).abs() < 1e-4,
            "expected -4.0, got {}",
            output[0]
        );
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
    fn test_dequant_q4k_short_block() {
        // block < 144 bytes → zero-fill output
        let block = vec![0u8; 100];
        let mut output = [1.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        assert!(
            output.iter().all(|&v| v == 0.0),
            "short block must zero output"
        );
    }

    #[test]
    fn test_dequant_q4k_unit_scale_zero_min() {
        // d=1.0, dmin=0.0, sc[0]=2 (scales_raw[0]=0x02), qs[0]=0x04 (lo=4)
        // output[0] = 1*2*4 - 0 = 8.0
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3C; // d = f16 1.0
                         // dmin = 0.0 (default)
        block[4] = 0x02; // scales_raw[0]: sc[0] = 2
        block[16] = 0x04; // qs[0] = lo nibble 4
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        assert!(
            (output[0] - 8.0).abs() < 1e-4,
            "expected 8.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q4k_nonzero_dmin() {
        // d=1.0, dmin=1.0, sc[0]=2 (scales_raw[0]=0x02), mn[0]=3 (scales_raw[4]=0x03)
        // qs[0]=0x00 (lo=0) → output[0] = 1*2*0 - 1*3 = -3.0
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[2] = 0x00;
        block[3] = 0x3C; // dmin = 1.0
        block[4] = 0x02; // scales_raw[0]: sc[0] = 2
        block[8] = 0x03; // scales_raw[4]: mn[0] = 3
                         // qs[0] = 0 → lo nibble = 0
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        assert!(
            (output[0] - (-3.0)).abs() < 1e-4,
            "expected -3.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q4k_scale_upper_bits() {
        // Test sc[i+4] decoding: sc[4] = (scales_raw[0]>>6) | ((scales_raw[8]&0x0F)<<2)
        // scales_raw[0]=0x40 → sc[0]=0, upper 2 bits → 1; scales_raw[8]=0x01 → (0x01&0x0F)<<2=4
        // sc[4] = 1 | 4 = 5
        // d=1.0, dmin=0.0, qs[0]=0x30 (hi nibble=3) → output[128] = 1*5*3 = 15.0
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[4] = 0x40; // scales_raw[0]: upper 2 bits = 01 → contributes 1 to sc[4]
        block[12] = 0x01; // scales_raw[8]: low 4 bits = 1 → contributes 4 to sc[4]
        block[16] = 0x30; // qs[0]: hi nibble = 3
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        assert!(
            (output[128] - 15.0).abs() < 1e-4,
            "upper-bits sc[4]: expected 15.0, got {}",
            output[128]
        );
    }

    #[test]
    fn test_dequant_q4k_negative_d() {
        // d=-1.0, dmin=0.0, sc[0]=3 (scales_raw[0]=0x03), qs[0]=0x02 (lo=2)
        // output[0] = (-1)*3*2 - 0 = -6.0
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0xBC; // d = f16 -1.0
        block[4] = 0x03; // scales_raw[0]: sc[0] = 3
        block[16] = 0x02; // qs[0] lo nibble = 2
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        assert!(
            (output[0] - (-6.0)).abs() < 1e-4,
            "negative d: expected -6.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q4k_second_block_idx() {
        // sc[1] from scales_raw[1]=0x04 → sc[1]=4; d=1.0, dmin=0.0
        // qs[32]=0x05 (lo nibble=5, block_idx=32/32=1)
        // output[32] = 1*4*5 = 20.0
        let mut block = vec![0u8; 144];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[5] = 0x04; // scales_raw[1]: sc[1] = 4
        block[48] = 0x05; // qs[32] lo nibble = 5
        let mut output = [0.0f32; 256];
        dequant_q4k_block(&block, &mut output);
        assert!(
            (output[32] - 20.0).abs() < 1e-4,
            "second block_idx: expected 20.0, got {}",
            output[32]
        );
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
    fn test_dequant_q5k_unit_scale_zero_qh() {
        // d=1.0, dmin=0.0, sc[0]=1 (scales_raw[0]=0x01), qh all 0 → q_lo=low nibble only
        // ql[48]=0x03 → lo=3 → output[0] = 1*1*3 = 3.0
        let mut block = vec![0u8; 176];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[4] = 0x01; // scales_raw[0]: sc[0]=1
                         // qh[16..48] = 0 (all high bits zero, default)
        block[48] = 0x03; // ql[0]: lo nibble=3
        let mut output = [0.0f32; 256];
        dequant_q5k_block(&block, &mut output);
        assert!(
            (output[0] - 3.0).abs() < 1e-4,
            "unit scale zero qh: expected 3.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q5k_high_bit_qh() {
        // Set qh[0] bit 0 to give weight 0 a 5th bit: q_lo = 0|(1<<4)=16
        // sc[0]=1, d=1.0, dmin=0 → output[0] = 1*1*16 = 16.0
        let mut block = vec![0u8; 176];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[4] = 0x01; // sc[0]=1
        block[16] = 0x01; // qh[0] bit 0 = 1 → high bit for weight 0
                          // ql[48] = 0x00 → lo nibble = 0
        let mut output = [0.0f32; 256];
        dequant_q5k_block(&block, &mut output);
        assert!(
            (output[0] - 16.0).abs() < 1e-4,
            "high bit qh: expected 16.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q5k_nonzero_dmin() {
        // d=1.0, dmin=1.0, sc[0]=2, mn[0]=3, ql[48]=0x01 (lo=1)
        // output[0] = 1*2*1 - 1*3 = -1.0
        let mut block = vec![0u8; 176];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[2] = 0x00;
        block[3] = 0x3C; // dmin = 1.0
        block[4] = 0x02; // scales_raw[0]: sc[0]=2
        block[8] = 0x03; // scales_raw[4]: mn[0]=3
        block[48] = 0x01; // ql[0]: lo nibble=1
        let mut output = [0.0f32; 256];
        dequant_q5k_block(&block, &mut output);
        assert!(
            (output[0] - (-1.0)).abs() < 1e-4,
            "nonzero dmin: expected -1.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q5k_negative_d() {
        // d=-1.0, sc[0]=2, ql[48]=0x03 (lo=3) → output[0] = (-1)*2*3 = -6.0
        let mut block = vec![0u8; 176];
        block[0] = 0x00;
        block[1] = 0xBC; // d = -1.0
        block[4] = 0x02; // sc[0]=2
        block[48] = 0x03; // ql[0]: lo=3
        let mut output = [0.0f32; 256];
        dequant_q5k_block(&block, &mut output);
        assert!(
            (output[0] - (-6.0)).abs() < 1e-4,
            "negative d: expected -6.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q5k_second_block_idx() {
        // sc[1] from scales_raw[1]=0x03 → sc[1]=3; d=1.0, dmin=0.0
        // ql[48+32]=ql[80]=0x05 (lo=5, block_idx=32/32=1) → output[32] = 1*3*5 = 15.0
        let mut block = vec![0u8; 176];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[5] = 0x03; // scales_raw[1]: sc[1]=3
        block[80] = 0x05; // ql[32]: lo=5
        let mut output = [0.0f32; 256];
        dequant_q5k_block(&block, &mut output);
        assert!(
            (output[32] - 15.0).abs() < 1e-4,
            "second block_idx: expected 15.0, got {}",
            output[32]
        );
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
    fn test_dequant_q6k_nonzero_scale() {
        // d=1.0 at bytes 208-209; scales[192]=1 (i8=1); ql[0]=0x01, qh[128]=0 (no high bits)
        // half=0, l=0, is=0: q1=(0x01&0xF|(0&3)<<4)-32=1-32=-31; sc1=scales[0]=1
        // output[0] = 1.0 * 1 * (-31) = -31.0
        let mut block = vec![0u8; 210];
        block[208] = 0x00;
        block[209] = 0x3C; // d = 1.0
        block[192] = 0x01; // scales[0] = i8(1)
        block[0] = 0x01; // ql[0] lo nibble = 1
        let mut output = [0.0f32; 256];
        dequant_q6k_block(&block, &mut output);
        assert!(
            (output[0] - (-31.0)).abs() < 1e-4,
            "nonzero scale: expected -31.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q6k_qh_high_bits() {
        // d=1.0, scales[192]=1; ql[0]=0x00, qh[128]=0x03 → q1=(0|(3<<4))-32=48-32=16
        // output[0] = 1.0 * 1 * 16 = 16.0
        let mut block = vec![0u8; 210];
        block[208] = 0x00;
        block[209] = 0x3C; // d = 1.0
        block[192] = 0x01; // scales[0] = 1
                           // ql[0] = 0 → low nibble=0
        block[128] = 0x03; // qh[0]: bits 1:0 = 3 → contributes 3<<4 = 48 to q1
        let mut output = [0.0f32; 256];
        dequant_q6k_block(&block, &mut output);
        assert!(
            (output[0] - 16.0).abs() < 1e-4,
            "qh high bits: expected 16.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q6k_negative_scale() {
        // d=1.0, scales[192]=0xFF (i8=-1); ql[0]=0x01, qh[128]=0 → q1=-31
        // output[0] = 1.0 * (-1) * (-31) = 31.0
        let mut block = vec![0u8; 210];
        block[208] = 0x00;
        block[209] = 0x3C; // d = 1.0
        block[192] = 0xFF; // scales[0] = i8(-1)
        block[0] = 0x01; // ql[0] lo nibble = 1
        let mut output = [0.0f32; 256];
        dequant_q6k_block(&block, &mut output);
        assert!(
            (output[0] - 31.0).abs() < 1e-4,
            "negative scale: expected 31.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q6k_second_half() {
        // half=1: ql_off=64, qh_off=32, sc_off=8, y_off=128
        // d=1.0, scales[192+8]=scales[200]=2 (i8); ql[64]=0x01, qh[128+32]=qh[160]=0
        // l=0, is=0: q1=(ql[64]&0xF|(qh[160]&3)<<4)-32 = 1-32=-31
        // sc1=scales[sc_off+is]=scales[8]=2 → output[128] = 1.0*2*(-31)=-62.0
        let mut block = vec![0u8; 210];
        block[208] = 0x00;
        block[209] = 0x3C; // d = 1.0
        block[200] = 0x02; // scales[8] = 2
        block[64] = 0x01; // ql[64]: lo nibble=1
        let mut output = [0.0f32; 256];
        dequant_q6k_block(&block, &mut output);
        assert!(
            (output[128] - (-62.0)).abs() < 1e-4,
            "second half: expected -62.0, got {}",
            output[128]
        );
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
    fn test_dequant_q3k_hmask_set() {
        // hmask bit 0 SET → sub=0 for weight 0, q = low2 - 0 = low2
        // scales[0]=1 (b[0]=1, b[8]=0xA0), qs[32]=0x03 (low2=3), d=1.0
        // output[0] = 1.0 * 1 * 3 = 3.0
        let mut block = vec![0u8; 110];
        block[0] = 0xFF; // hmask[0]: bit 0 set
        block[32] = 0x03; // qs[0] low2=3 at shift 0
        block[96] = 0x01; // b[0]=1
        block[104] = 0xA0; // b[8]=0xA0 → scales[0] = (1|(2<<4))-32 = 1
        block[108] = 0x00;
        block[109] = 0x3C; // d = 1.0
        let mut output = [0.0f32; 256];
        dequant_q3k_block(&block, &mut output);
        assert!(
            (output[0] - 3.0).abs() < 1e-4,
            "hmask set: expected 3.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q3k_hmask_clear() {
        // hmask bit 0 CLEAR → sub=4 for weight 0, q = low2 - 4
        // scales[0]=1, qs[32]=0x02 (low2=2), d=1.0
        // output[0] = 1.0 * 1 * (2-4) = -2.0
        let mut block = vec![0u8; 110];
        block[0] = 0xFE; // hmask[0]: bit 0 CLEAR
        block[32] = 0x02; // qs[0] low2=2
        block[96] = 0x01; // b[0]=1
        block[104] = 0xA0; // b[8]=0xA0 → scales[0]=1
        block[108] = 0x00;
        block[109] = 0x3C; // d = 1.0
        let mut output = [0.0f32; 256];
        dequant_q3k_block(&block, &mut output);
        assert!(
            (output[0] - (-2.0)).abs() < 1e-4,
            "hmask clear: expected -2.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q3k_negative_scale() {
        // scales[0]=-1: b[0]=0x0F (lo nibble=15), b[8]=0x10 ((>>4)&3=1 → 1<<4=16)
        // (15|16)-32 = 31-32 = -1
        // hmask[0] bit 0 set, qs[32]=0x01 (low2=1), d=1.0
        // q=1, output[0] = 1.0 * (-1) * 1 = -1.0
        let mut block = vec![0u8; 110];
        block[0] = 0xFF; // hmask[0]: bit 0 set
        block[32] = 0x01; // qs[0] low2=1
        block[96] = 0x0F; // b[0]=0x0F (lo nibble=15)
        block[104] = 0x10; // b[8]=0x10 → (>>4)&3=1
        block[108] = 0x00;
        block[109] = 0x3C; // d = 1.0
        let mut output = [0.0f32; 256];
        dequant_q3k_block(&block, &mut output);
        assert!(
            (output[0] - (-1.0)).abs() < 1e-4,
            "negative scale: expected -1.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q3k_second_half() {
        // oi=1, si=0: scale_idx=4, m=0x10 (bit 4 of hmask)
        // scales[4] = (b[4]&0x0F | ((b[8]>>6)&3)<<4) - 32
        // b[4]=2, b[8]=0x80 → (2 | (2<<4))-32 = 34-32 = 2
        // hmask[0]=0xFF (bit 4 set → sub=0), block[64]=0x03 (oi=1 qs_group[0]=3)
        // output[128] = 1.0 * 2 * 3 = 6.0
        let mut block = vec![0u8; 110];
        block[0] = 0xFF; // hmask[0]: all bits set (including bit 4 for oi=1,si=0)
        block[64] = 0x03; // qs for oi=1, l=0: low2=3
        block[100] = 0x02; // b[4]=2
        block[104] = 0x80; // b[8]=0x80 → (>>6)&3=2 (for scales[4]) and (>>4)&3=8&3=0 (scales[0..3])
        block[108] = 0x00;
        block[109] = 0x3C; // d = 1.0
        let mut output = [0.0f32; 256];
        dequant_q3k_block(&block, &mut output);
        assert!(
            (output[128] - 6.0).abs() < 1e-4,
            "second half: expected 6.0, got {}",
            output[128]
        );
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
    fn test_dequant_q2k_negative_d() {
        // d=-1.0, dmin=0.0, scales[64]=0x01 (scale=1, min=0), qs[0]=0x01 (q2=1 at shift 0)
        // output[0] = (-1)*1*1 - 0 = -1.0
        let mut block = vec![0u8; 84];
        block[0] = 0x01; // qs[0]: q2=1 at shift 0
        block[64] = 0x01; // scales[0]: lo nibble=1 (scale), hi nibble=0 (min)
        block[80] = 0x00;
        block[81] = 0xBC; // d = f16 -1.0
        let mut output = [0.0f32; 256];
        dequant_q2k_block(&block, &mut output);
        assert!(
            (output[0] - (-1.0)).abs() < 1e-4,
            "negative d: expected -1.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q2k_nonzero_dmin() {
        // d=0.0, dmin=1.0, scales[64]=0x30 (lo=0, hi=3), qs irrelevant (q2*0=0)
        // output[0] = 0 - 1*3 = -3.0
        let mut block = vec![0u8; 84];
        block[64] = 0x30; // scales[0]: lo nibble=0 (scale), hi nibble=3 (min)
                          // d stays 0.0
        block[82] = 0x00;
        block[83] = 0x3C; // dmin = 1.0
        let mut output = [0.0f32; 256];
        dequant_q2k_block(&block, &mut output);
        assert!(
            (output[0] - (-3.0)).abs() < 1e-4,
            "nonzero dmin: expected -3.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q2k_max_q2() {
        // All qs=0xFF: each 2-bit field=3 (max). d=1.0, scale=1, dmin=0
        // output[0] = 1*1*3 = 3.0
        let mut block = vec![0u8; 84];
        block[0..64].fill(0xFF); // all q2=3
        block[64] = 0x01; // scales[0]: scale=1, min=0
        block[80] = 0x00;
        block[81] = 0x3C; // d = 1.0
        let mut output = [0.0f32; 256];
        dequant_q2k_block(&block, &mut output);
        assert!(
            (output[0] - 3.0).abs() < 1e-4,
            "max q2: expected 3.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q2k_second_subblock() {
        // i=64 → sub=64/16=4 → uses scales[4]; qs[64/4]=qs[16], shift=(64%4)*2=0
        // scales[68]=0x02 (scale=2), d=1.0, dmin=0, qs[16]=0x01 (q2=1)
        // output[64] = 1*2*1 = 2.0
        let mut block = vec![0u8; 84];
        block[16] = 0x01; // qs[16]: q2=1 at shift 0 (for i=64)
        block[68] = 0x02; // scales[4]: lo=2 (scale), hi=0 (min)
        block[80] = 0x00;
        block[81] = 0x3C; // d = 1.0
        let mut output = [0.0f32; 256];
        dequant_q2k_block(&block, &mut output);
        assert!(
            (output[64] - 2.0).abs() < 1e-4,
            "second subblock: expected 2.0, got {}",
            output[64]
        );
    }

    #[test]
    fn test_dequant_q2k_hi_nibble_min() {
        // scales[0]=0x20 (lo=0, hi=2): min_nibble=2
        // d=1.0, dmin=1.0, qs[0]=0 (q2=0) → output[0] = 1*0*0 - 1*2 = -2.0
        let mut block = vec![0u8; 84];
        block[64] = 0x20; // scales[0]: scale=0, min=2
        block[80] = 0x00;
        block[81] = 0x3C; // d = 1.0
        block[82] = 0x00;
        block[83] = 0x3C; // dmin = 1.0
        let mut output = [0.0f32; 256];
        dequant_q2k_block(&block, &mut output);
        assert!(
            (output[0] - (-2.0)).abs() < 1e-4,
            "hi nibble min: expected -2.0, got {}",
            output[0]
        );
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
    fn test_dequant_q4_1_negative_d() {
        // d=-1.0 (0xBC00), m=0.0, all nibbles=3 → output = -1.0*3 + 0.0 = -3.0
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0xBC; // f16 -1.0 for d
        block[4..20].fill(0x33); // nibble=3
        let mut output = [0.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - (-3.0)).abs() < 1e-4,
                "q4_1 neg_d: expected -3.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_1_negative_m() {
        // d=1.0, m=-2.0 (0xC000), all nibbles=0 → output = 1.0*0 + (-2.0) = -2.0
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0 for d
        block[2] = 0x00;
        block[3] = 0xC0; // f16 -2.0 for m
        block[4..20].fill(0x00); // all nibbles=0
        let mut output = [0.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - (-2.0)).abs() < 1e-4,
                "q4_1 neg_m: expected -2.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_1_max_nibble() {
        // d=1.0, m=0.0, all nibbles=15 → output = 15.0
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[4..20].fill(0xFF); // both nibbles=15
        let mut output = [0.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - 15.0).abs() < 1e-4,
                "q4_1 max_nibble: expected 15.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_1_min_nibble_with_m() {
        // d=1.0, m=5.0 (0x4500), all nibbles=0 → output = 1.0*0 + 5.0 = 5.0
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0 for d
        block[2] = 0x00;
        block[3] = 0x45; // f16 5.0 for m
        block[4..20].fill(0x00); // all nibbles=0
        let mut output = [0.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - 5.0).abs() < 1e-4,
                "q4_1 min_nibble_m: expected 5.0 at {i}, got {val}"
            );
        }
    }

    #[test]
    fn test_dequant_q4_1_split_nibbles() {
        // d=1.0, m=0.0, block[4]=0x12 → lo nibble=2, hi nibble=1
        // Q4_1 layout: output[i*2]=d*lo+m, output[i*2+1]=d*hi+m
        // → output[0]=2.0, output[1]=1.0
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[4] = 0x12; // lo nibble=2, hi nibble=1
        let mut output = [0.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        assert!(
            (output[0] - 2.0).abs() < 1e-4,
            "q4_1 split: output[0] = {}, expected 2.0",
            output[0]
        );
        assert!(
            (output[1] - 1.0).abs() < 1e-4,
            "q4_1 split: output[1] = {}, expected 1.0",
            output[1]
        );
    }

    #[test]
    fn test_dequant_q4_1_zero_d_nonzero_m() {
        // d=0.0, m=4.0 (0x4400), all nibbles=7 → output = 0*7 + 4.0 = 4.0
        let mut block = vec![0u8; 20];
        // d stays 0.0 (bytes 0-1 = 0)
        block[2] = 0x00;
        block[3] = 0x44; // f16 4.0 for m
        block[4..20].fill(0x77); // nibbles=7, but d=0 so irrelevant
        let mut output = [0.0f32; 32];
        dequant_q4_1_block(&block, &mut output);
        for (i, &val) in output.iter().enumerate() {
            assert!(
                (val - 4.0).abs() < 1e-4,
                "q4_1 zero_d: expected 4.0 at {i}, got {val}"
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

    #[test]
    fn test_dequant_q8_1_negative_d() {
        // d=-1.0, qs[0]=5 (i8) → output[0] = 5 * (-1.0) = -5.0
        let mut block = vec![0u8; 36];
        block[0] = 0x00;
        block[1] = 0xBC; // f16 -1.0
        block[4] = 5; // i8 = 5
        let mut output = [0.0f32; 32];
        dequant_q8_1_block(&block, &mut output);
        assert!(
            (output[0] - (-5.0)).abs() < 1e-4,
            "negative d: expected -5.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q8_1_i8_min() {
        // d=1.0, qs[0]=0x80 (i8=-128) → output[0]=-128.0
        let mut block = vec![0u8; 36];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[4] = 0x80; // i8 = -128
        let mut output = [0.0f32; 32];
        dequant_q8_1_block(&block, &mut output);
        assert!(
            (output[0] - (-128.0)).abs() < 1e-4,
            "i8 min: expected -128.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q8_1_i8_max() {
        // d=1.0, qs[0]=0x7F (i8=127) → output[0]=127.0
        let mut block = vec![0u8; 36];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[4] = 0x7F; // i8 = 127
        let mut output = [0.0f32; 32];
        dequant_q8_1_block(&block, &mut output);
        assert!(
            (output[0] - 127.0).abs() < 1e-4,
            "i8 max: expected 127.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q8_1_s_field_ignored() {
        // The sum field (bytes 2-3) must not affect dequantization
        // d=1.0, s=0xBEEF (non-zero), qs[0]=3 → output[0]=3.0
        let mut block = vec![0u8; 36];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[2] = 0xEF; // s field (not used)
        block[3] = 0xBE;
        block[4] = 3; // i8 = 3
        let mut output = [0.0f32; 32];
        dequant_q8_1_block(&block, &mut output);
        assert!(
            (output[0] - 3.0).abs() < 1e-4,
            "s field ignored: expected 3.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_dequant_q8_1_mixed_signs() {
        // d=2.0, qs[0]=1 (i8), qs[1]=0xFF (i8=-1) → output[0]=2.0, output[1]=-2.0
        let mut block = vec![0u8; 36];
        block[0] = 0x00;
        block[1] = 0x40; // f16 2.0
        block[4] = 0x01; // i8 = 1
        block[5] = 0xFF; // i8 = -1
        let mut output = [0.0f32; 32];
        dequant_q8_1_block(&block, &mut output);
        assert!(
            (output[0] - 2.0).abs() < 1e-4,
            "mixed signs: expected 2.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - (-2.0)).abs() < 1e-4,
            "mixed signs: expected -2.0, got {}",
            output[1]
        );
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

    // ── IQ3_XXS extended tests ────────────────────────────────────────────────

    fn iq3xxs_grid_full_fixture() -> Box<[u32; 256]> {
        let mut g = vec![0u32; 256];
        g[0] = 0x04040404u32; // all bytes = 4
        g[1] = 0x08080808u32; // all bytes = 8
        g[2] = 0x02020202u32; // all bytes = 2
        g[3] = 0x01010101u32; // all bytes = 1
        Box::new(g.try_into().unwrap())
    }

    #[test]
    fn test_dequant_iq3xxs_all_l_values() {
        // Use different grid entries for each l via qs bytes
        let grid = iq3xxs_grid_full_fixture();
        let mut block = make_iq3xxs_zero_block();
        // ib32=0: qs bytes 2..9 = [0,1,2,3,0,0,0,0] → l=0→grid[0], l=1→grid[1], l=2→grid[2], l=3→grid[3]
        block[2] = 0; // l=0 qs1
        block[3] = 0; // l=0 qs2 (not used for diff grid)
        block[4] = 1; // l=1 qs1
        block[5] = 0;
        block[6] = 2; // l=2 qs1
        block[7] = 0;
        block[8] = 3; // l=3 qs1
        block[9] = 0;
        let mut output = [0.0f32; 256];
        dequant_iq3xxs_block(&block, &grid, &mut output);
        // dl = 1*(0.5+0)*0.5 = 0.25; sign_7bit=0 → KSIGNS_IQ2XS[0]=0x00 (all +)
        // l=0: grid[0] bytes=4 → 0.25*4=1.0; qs2=0 → also grid[0]
        // l=1: grid[1] bytes=8 → 0.25*8=2.0; qs2=0 → grid[0] → 1.0 (second 4)
        // l=2: grid[2] bytes=2 → 0.25*2=0.5; qs2=0 → grid[0] → 1.0
        // l=3: grid[3] bytes=1 → 0.25*1=0.25; qs2=0 → grid[0] → 1.0
        let expected_l = [1.0f32, 2.0, 0.5, 0.25];
        for (l, &exp) in expected_l.iter().enumerate() {
            for (j, &v) in output[l * 8..l * 8 + 4].iter().enumerate() {
                assert!(
                    (v - exp).abs() < 1e-5,
                    "iq3xxs l={l} j={j}: got {v}, expected {exp}"
                );
            }
            // second 4 weights from qs2=0 → grid[0] → 1.0
            for (j, &v) in output[l * 8 + 4..l * 8 + 8].iter().enumerate() {
                assert!(
                    (v - 1.0).abs() < 1e-5,
                    "iq3xxs l={l} qs2 j={j}: got {v}, expected 1.0"
                );
            }
        }
    }

    #[test]
    fn test_dequant_iq3xxs_sign_bit_pattern() {
        // KSIGNS_IQ2XS[0x7F] = 0xFF → all 8 sign bits set → all weights negative
        let grid = iq3xxs_grid_full_fixture();
        let mut block = make_iq3xxs_zero_block();
        // l=0 sign_7bit = 127 (0x7F): set aux32 bits 6:0 = 0x7F
        // aux32 = 0x0000007F → bytes at 66..69 = [0x7F, 0x00, 0x00, 0x00]
        block[66] = 0x7F;
        let mut output = [0.0f32; 256];
        dequant_iq3xxs_block(&block, &grid, &mut output);
        // KSIGNS_IQ2XS[127] = 0xFF → all bits set → all 8 weights of l=0 are negative
        // dl=0.25, grid[0]=4 → -0.25*4 = -1.0
        for (j, &v) in output[..8].iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-5,
                "iq3xxs all signs: j={j} got {v}, expected -1.0"
            );
        }
        // l=1..3 signs_7bit=0 → positive → 1.0
        for (j, &v) in output[8..32].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq3xxs l>0: j={j} got {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3xxs_last_ib32() {
        // Last ib32 group: sub-scale at bits 31:28 of scales_and_signs[7]
        let grid = iq3xxs_grid_full_fixture();
        let mut block = make_iq3xxs_zero_block();
        // scales_and_signs[7] at bytes 94-97 (66 + 7*4): sub_scale=8 → bits 31:28=0x8 → byte[97]=0x80
        block[69 + 7 * 4] = 0x80; // high byte of aux32[7]
        let mut output = [0.0f32; 256];
        dequant_iq3xxs_block(&block, &grid, &mut output);
        // dl = 1*(0.5+8)*0.5 = 4.25; grid[0]=4 → 4.25*4=17.0
        let expected = (0.5f32 + 8.0) * 0.5 * 4.0;
        for (i, &v) in output[224..].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq3xxs last ib32: weight[{i}] = {v}, expected {expected}"
            );
        }
        // First 224 weights use sub_scale=0 → dl=0.25 → 1.0
        for (i, &v) in output[..224].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq3xxs last ib32 rest: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3xxs_negative_d() {
        // d = -1.0 → dl = -1*(0.5+0)*0.5 = -0.25; grid[0] bytes=4 → output = -0.25*4 = -1.0
        let grid = iq3xxs_grid_fixture();
        let mut block = make_iq3xxs_zero_block();
        block[0] = 0x00;
        block[1] = 0xBC; // f16 -1.0
        let mut output = [0.0f32; 256];
        dequant_iq3xxs_block(&block, &grid, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-5,
                "iq3xxs neg d: weight[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3xxs_second_ib32() {
        // ib32=1: aux32 at block[70..73]; sub_scale=3 in bits 31:28 → block[73]=0x30
        // dl = 1*(0.5+3)*0.5 = 1.75; grid[0] bytes=4 → output[32..64] = 1.75*4 = 7.0
        // ib32=0 sub_scale=0 → dl=0.25 → output[0..32] = 1.0
        let grid = iq3xxs_grid_fixture();
        let mut block = make_iq3xxs_zero_block();
        block[73] = 0x30; // aux32[1] bits 31:28 = 3
        let mut output = [0.0f32; 256];
        dequant_iq3xxs_block(&block, &grid, &mut output);
        let expected_ib1 = (0.5f32 + 3.0) * 0.5 * 4.0; // 7.0
        for (i, &v) in output[32..64].iter().enumerate() {
            assert!(
                (v - expected_ib1).abs() < 1e-4,
                "iq3xxs 2nd ib32: output[{}] = {v}, expected {expected_ib1}",
                i + 32
            );
        }
        for (i, &v) in output[0..32].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq3xxs 2nd ib32 group0: output[{i}] = {v}, expected 1.0"
            );
        }
    }

    // ── IQ2_XXS tests ─────────────────────────────────────────────────────────

    fn iq2xxs_grid_fixture() -> Box<[u64; 256]> {
        let mut g = vec![0u64; 256];
        // entry 0: all bytes = 8 → 0x0808080808080808
        g[0] = 0x0808080808080808u64;
        // entry 1: all bytes = 4
        g[1] = 0x0404040404040404u64;
        // entry 2: all bytes = 2
        g[2] = 0x0202020202020202u64;
        // entry 3: alternating 8/4 pattern
        g[3] = 0x0408040804080408u64;
        Box::new(g.try_into().unwrap())
    }

    fn make_iq2xxs_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; 66];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0 (f16)
                         // aux0=0, aux1=0 for all groups → grid_idx=0, signs_7bit=0, sub_scale=0
        block
    }

    #[test]
    fn test_dequant_iq2xxs_zeroed() {
        let grid = iq2xxs_grid_fixture();
        let block = make_iq2xxs_zero_block();
        let mut output = [0.0f32; 256];
        dequant_iq2xxs_block(&block, &grid, &mut output);
        // sub_scale=0 → db=1*(0.5+0)*0.25=0.125; grid[0]=8; signs all +1 → 0.125*8=1.0
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xxs zero: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xxs_max_subscale() {
        let grid = iq2xxs_grid_fixture();
        let mut block = make_iq2xxs_zero_block();
        // sub_scale=15: aux1 bits 31:28=0xF → byte at offset 9 (base+7) = 0xF0
        // block[2+7]=0xF0: ib32=0, aux1 high byte = 0xF0
        block[9] = 0xF0;
        let mut output = [0.0f32; 256];
        dequant_iq2xxs_block(&block, &grid, &mut output);
        // db = 1*(0.5+15)*0.25 = 3.875; grid[0]=8; weight = 3.875*8 = 31.0
        let expected = (0.5f32 + 15.0) * 0.25 * 8.0;
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2xxs max sub_scale: weight[{i}] = {v}, expected {expected}"
            );
        }
        // Remaining groups sub_scale=0 → 1.0
        for (i, &v) in output[32..].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xxs max sub_scale rest: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xxs_sign_inversion() {
        let grid = iq2xxs_grid_fixture();
        let mut block = make_iq2xxs_zero_block();
        // signs_7bit=1 for l=0: set aux1 bits 6:0 = 1 → block[2+4]=0x01
        // KSIGNS_IQ2XS[1] = 0x81 → bit0=1 (neg), bit7=1 (neg), rest +1
        block[6] = 0x01; // aux1 low byte for ib32=0
        let mut output = [0.0f32; 256];
        dequant_iq2xxs_block(&block, &grid, &mut output);
        // db=0.125; grid[0]=8 for all bytes
        // l=0: j=0 (bit0=1) → -1.0; j=1..6 → +1.0; j=7 (bit7=1) → -1.0
        assert!(
            (output[0] - (-1.0)).abs() < 1e-5,
            "iq2xxs sign j=0: got {}",
            output[0]
        );
        assert!(
            (output[1] - 1.0).abs() < 1e-5,
            "iq2xxs sign j=1: got {}",
            output[1]
        );
        assert!(
            (output[7] - (-1.0)).abs() < 1e-5,
            "iq2xxs sign j=7: got {}",
            output[7]
        );
    }

    #[test]
    fn test_dequant_iq2xxs_all_l_values() {
        // Test all 4 l-values via different grid indices in aux0
        // aux0 = [0, 1, 2, 3] bytes → (0) | (1<<8) | (2<<16) | (3<<24) = 0x03020100
        let grid = iq2xxs_grid_fixture();
        let mut block = make_iq2xxs_zero_block();
        // aux0 for ib32=0 at bytes 2..5
        block[2] = 0x00; // l=0 grid_idx = 0
        block[3] = 0x01; // l=1 grid_idx = 1
        block[4] = 0x02; // l=2 grid_idx = 2
        block[5] = 0x03; // l=3 grid_idx = 3
        let mut output = [0.0f32; 256];
        dequant_iq2xxs_block(&block, &grid, &mut output);
        // db=0.125; signs all + (signs_7bit=0)
        // l=0: grid[0]=8 → 0.125*8=1.0
        // l=1: grid[1]=4 → 0.125*4=0.5
        // l=2: grid[2]=2 → 0.125*2=0.25
        // l=3: grid[3] alternating 8/4 bytes → 0.125*8=1.0 or 0.125*4=0.5
        let expected_l = [1.0f32, 0.5, 0.25];
        for (l, &exp) in expected_l.iter().enumerate() {
            for (j, &v) in output[l * 8..(l + 1) * 8].iter().enumerate() {
                assert!(
                    (v - exp).abs() < 1e-5,
                    "iq2xxs l={l} j={j}: got {v}, expected {exp}"
                );
            }
        }
        // l=3: grid[3] alternating 4/8 per byte (0x0408040804080408)
        // lo32=0x04080408: bytes 0=0x08, 1=0x04, 2=0x08, 3=0x04
        // hi32=0x04080408: same pattern
        let exp_l3 = [1.0f32, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5];
        for (j, (&v, &exp)) in output[24..32].iter().zip(exp_l3.iter()).enumerate() {
            assert!(
                (v - exp).abs() < 1e-5,
                "iq2xxs l=3 j={j}: got {v}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xxs_last_ib32() {
        let grid = iq2xxs_grid_fixture();
        let mut block = make_iq2xxs_zero_block();
        // ib32=7 aux1 high byte at block[2+7*8+7] = block[65]: sub_scale=7 → 0x70
        block[65] = 0x70;
        let mut output = [0.0f32; 256];
        dequant_iq2xxs_block(&block, &grid, &mut output);
        // db = 1*(0.5+7)*0.25 = 1.875; grid[0]=8 → 1.875*8=15.0
        let expected = (0.5f32 + 7.0) * 0.25 * 8.0;
        for (i, &v) in output[224..].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2xxs last ib32: weight[{i}] = {v}, expected {expected}"
            );
        }
        // First 224 weights sub_scale=0 → 1.0
        for (i, &v) in output[..224].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xxs last ib32 rest: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xxs_short_block() {
        let grid = iq2xxs_grid_fixture();
        let block = vec![0u8; 10]; // too short
        let mut output = [0.0f32; 256];
        dequant_iq2xxs_block(&block, &grid, &mut output);
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

    // ── IQ2_S reference dequant tests ─────────────────────────────────────────

    /// Minimal IQ2_S grid fixture: entry 0 = 0x0808080808080808 (all bytes 8).
    fn iq2s_grid_fixture() -> Box<[u64; 1024]> {
        let mut g = vec![0u64; 1024];
        g[0] = 0x0808080808080808u64; // each byte = 8
        Box::new(g.try_into().unwrap())
    }

    /// Minimal IQ2_S block (82 bytes): d=1.0, all zeroed except d.
    /// scale=0 → dl = 1.0*(0.5+0)*0.25 = 0.125; grid[0] all 8; signs all 0 (+1).
    /// Expected: 0.125 * 8 = 1.0 for all 256 weights.
    fn make_iq2s_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; 82];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block
    }

    #[test]
    fn test_dequant_iq2s_zeroed() {
        let grid = iq2s_grid_fixture();
        let block = make_iq2s_zero_block();
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        // dl = 1.0*(0.5+0)*0.25 = 0.125; grid[0] all 8; signs all +1
        // Expected: 0.125 * 8 = 1.0
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2s zero: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2s_max_scale() {
        let grid = iq2s_grid_fixture();
        let mut block = make_iq2s_zero_block();
        // lo nibble=15 → dl = 1.0*(0.5+15)*0.25 = 3.875; covers l=0,1 (first 16 weights of ib32=0)
        block[74] = 0x0F; // scales[0]: lo=15
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        let expected = (0.5_f32 + 15.0) * 0.25 * 8.0; // dl * grid_byte
        for (i, &v) in output[..16].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2s max scale: weight[{i}] = {v}, expected {expected}"
            );
        }
        // l=2,3 use hi nibble=0 → dl=0.125, output=1.0
        for &v in &output[16..32] {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2s hi nibble 0: got {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2s_sign_inversion() {
        let grid = iq2s_grid_fixture();
        let mut block = make_iq2s_zero_block();
        // signs[0] at byte 34 = 0x01: bit0=1 → weight[0] negated
        block[34] = 0x01;
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        // dl=0.125; grid[0][byte 0]=8; sign bit0=1 → -1.0
        assert!(
            (output[0] - (-1.0)).abs() < 1e-5,
            "iq2s sign bit0: got {}",
            output[0]
        );
        assert!(
            (output[1] - 1.0).abs() < 1e-5,
            "iq2s weight1: got {}",
            output[1]
        );
    }

    #[test]
    fn test_dequant_iq2s_short_block() {
        let grid = iq2s_grid_fixture();
        let block = vec![0u8; 10]; // too short
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    /// Extended IQ2_S grid fixture with entries at high indices for qh-path testing.
    fn iq2s_grid_full_fixture() -> Box<[u64; 1024]> {
        let mut g = vec![0u64; 1024];
        // entry 0: all bytes = 8 (default)
        g[0] = 0x0808080808080808u64;
        // entry 768 = 0x300 (qh bits 1:0 = 3, qs_lo = 0): all bytes = 4
        g[768] = 0x0404040404040404u64;
        // entry 256 = 0x100 (qh bits 1:0 = 1, qs_lo = 0): all bytes = 16
        g[256] = 0x1010101010101010u64;
        // entry 512 = 0x200 (qh bits 1:0 = 2, qs_lo = 0): all bytes = 12
        g[512] = 0x0C0C0C0C0C0C0C0Cu64;
        Box::new(g.try_into().unwrap())
    }

    #[test]
    fn test_dequant_iq2s_high_grid_index_qh() {
        // Verify 10-bit grid index: grid_idx = qs_lo | (((qh_byte >> (2*l)) & 3) << 8)
        // Set qh[0] bits 1:0 = 3 → l=0 uses grid_idx = 768 (all bytes = 4)
        let grid = iq2s_grid_full_fixture();
        let mut block = vec![0u8; 82];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
                         // qs_lo[0] = 0 (l=0 of ib32=0)
                         // qh[0] = 0x03 → bits 1:0 = 3 → grid_hi for l=0 is 3
        block[66] = 0x03; // qh[0]
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        // dl = 1.0*(0.5+0)*0.25 = 0.125; grid[768] bytes = 4; sign = +1
        let expected_l0 = 0.125f32 * 4.0;
        for (i, &v) in output[..8].iter().enumerate() {
            assert!(
                (v - expected_l0).abs() < 1e-5,
                "iq2s qh high: l=0 weight[{i}] = {v}, expected {expected_l0}"
            );
        }
        // l=1 uses qh bits 3:2 = 0 → grid_idx = 0 → bytes = 8 → weight = 0.125*8 = 1.0
        for (i, &v) in output[8..16].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2s qh high: l=1 weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2s_qh_all_l_values() {
        // Test all 4 l values in qh index selection for ib32=0
        // qh[0] = 0b_11_10_01_00 (from LSB to MSB: l0=0, l1=1, l2=2, l3=3)
        // but 8 bits: bits 1:0=l0, bits 3:2=l1, bits 5:4=l2, bits 7:6=l3
        // So qh_byte = (0<<0) | (1<<2) | (2<<4) | (3<<6) = 0 + 4 + 32 + 192 = 228 = 0xE4
        let grid = iq2s_grid_full_fixture();
        let mut block = vec![0u8; 82];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block[66] = 0xE4; // qh[0]: l0→0, l1→1, l2→2, l3→3
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        // dl for l=0,1 uses lo nibble of scales[0]=0 → db0 = 0.125
        // dl for l=2,3 uses hi nibble of scales[0]=0 → db1 = 0.125
        // l=0: grid[0] bytes=8 → 0.125*8=1.0
        // l=1: grid[256] bytes=16 → 0.125*16=2.0
        // l=2: grid[512] bytes=12 → 0.125*12=1.5
        // l=3: grid[768] bytes=4 → 0.125*4=0.5
        let expected = [1.0f32, 2.0, 1.5, 0.5];
        for (l, &exp) in expected.iter().enumerate() {
            for (j, &v) in output[l * 8..(l + 1) * 8].iter().enumerate() {
                assert!(
                    (v - exp).abs() < 1e-5,
                    "iq2s qh l={l} j={j}: got {v}, expected {exp}"
                );
            }
        }
    }

    #[test]
    fn test_dequant_iq2s_hi_nibble_scale() {
        // Hi nibble of scales[0] controls l=2,3; verify it works independently
        let grid = iq2s_grid_fixture();
        let mut block = make_iq2s_zero_block();
        // scales[0] = 0xF0: lo=0 → db0=0.125, hi=15 → db1 = 1*(0.5+15)*0.25 = 3.875
        block[74] = 0xF0;
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        let expected_lo = 0.125f32 * 8.0; // l=0,1 → 1.0
        let expected_hi = (0.5f32 + 15.0) * 0.25 * 8.0; // l=2,3 → 31.0
        for (i, &v) in output[..16].iter().enumerate() {
            assert!(
                (v - expected_lo).abs() < 1e-4,
                "iq2s hi-nibble scale: l=0,1 weight[{i}] = {v}, expected {expected_lo}"
            );
        }
        for (i, &v) in output[16..32].iter().enumerate() {
            assert!(
                (v - expected_hi).abs() < 1e-4,
                "iq2s hi-nibble scale: l=2,3 weight[{i}] = {v}, expected {expected_hi}"
            );
        }
        // weights 32..256 should still be 1.0 (scales[1..8] = 0)
        for (i, &v) in output[32..].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2s hi-nibble: rest weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2s_all_sign_bits() {
        // signs[0] = 0xFF: all 8 weights in l=0 of ib32=0 are negated
        let grid = iq2s_grid_fixture();
        let mut block = make_iq2s_zero_block();
        block[34] = 0xFF; // signs for l=0 of ib32=0: all bits set
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        // All 8 weights of l=0 should be -1.0
        for (i, &v) in output[..8].iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-5,
                "iq2s all signs: weight[{i}] = {v}, expected -1.0"
            );
        }
        // Remaining 24 weights of ib32=0 should be +1.0 (unsigned)
        for (i, &v) in output[8..32].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2s all signs: rest weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2s_sign_individual_bits() {
        // Verify each sign bit independently: signs[0] with alternating bits 0b01010101 = 0x55
        let grid = iq2s_grid_fixture();
        let mut block = make_iq2s_zero_block();
        block[34] = 0x55; // bits 0,2,4,6 set → weights 0,2,4,6 of l=0 negated
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        for (j, &v) in output[..8].iter().enumerate() {
            let expected = if j % 2 == 0 { -1.0f32 } else { 1.0 };
            assert!(
                (v - expected).abs() < 1e-5,
                "iq2s sign bits: weight[{j}] = {v}, expected {expected}",
            );
        }
    }

    #[test]
    fn test_dequant_iq2s_scale_nibble_zero() {
        // scale nibble=0 for all ib32: dl = 1.0*(0.5+0)*0.25 = 0.125 → weight = 1.0
        let grid = iq2s_grid_fixture();
        let block = make_iq2s_zero_block(); // scales all 0
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2s nibble0: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2s_last_ib32() {
        // Test that the last ib32 group (ib32=7) is processed correctly
        let grid = iq2s_grid_fixture();
        let mut block = make_iq2s_zero_block();
        // scales[7] lo nibble = 7 → db0 = 1*(0.5+7)*0.25 = 1.875 → weight = 1.875*8 = 15.0
        block[81] = 0x07;
        let mut output = [0.0f32; 256];
        dequant_iq2s_block(&block, &grid, &mut output);
        let expected = (0.5f32 + 7.0) * 0.25 * 8.0; // 15.0
        for (i, &v) in output[224..240].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2s last ib32: weight[{i}] = {v}, expected {expected}"
            );
        }
        // Rest of ib32=7 (l=2,3) should use hi nibble=0 → 1.0
        for (i, &v) in output[240..256].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2s last ib32 hi: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    // ─── IQ1_S tests ─────────────────────────────────────────────────────────

    /// Build a minimal IQ1_S grid: entry 0 = all +1 bytes (0x01010101_01010101).
    fn iq1s_grid_fixture() -> Box<[u64; 2048]> {
        let mut g = vec![0u64; 2048];
        // entry 0: all bytes = 0x01 (+1)
        g[0] = 0x0101010101010101u64;
        // entry 1: all bytes = 0xFF (-1)
        g[1] = 0xFFFFFFFFFFFFFFFFu64;
        // entry 2: alternating 0x01/0xFF
        g[2] = 0xFF01FF01FF01FF01u64;
        g.try_into().unwrap()
    }

    /// Build a zeroed IQ1_S block (50 bytes) that maps to grid entry 0 with s=0, delta=+0.125.
    fn make_iq1s_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; 50];
        // d = 1.0 in f16 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        // qs[0..32] = 0 → grid index low 8 bits = 0
        // qh[0..8] = 0x0000 → s=0, delta=+, high grid bits=0 → grid_idx=0
        block
    }

    #[test]
    fn test_dequant_iq1s_zeroed() {
        let grid = iq1s_grid_fixture();
        let block = make_iq1s_zero_block();
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // d=1.0, s=0 → dl=1*(2*0+1)=1, delta=+0.125, grid=+1 → weight=1*(1+0.125)=1.125
        let expected = 1.0 * (1.0 + IQ1S_DELTA);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "iq1s zero: weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1s_scale_factor() {
        let grid = iq1s_grid_fixture();
        let mut block = make_iq1s_zero_block();
        // Set qh[0] with s=3 (bits 14:12 = 0b011 → 0x3000) and positive delta (bit15=0)
        let qh_val: u16 = 0x3000; // s=3
        block[34] = (qh_val & 0xFF) as u8;
        block[35] = (qh_val >> 8) as u8;
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // dl = 1 * (2*3+1) = 7, delta = +0.125, grid[0]=+1 → weight = 7*(1+0.125)=7.875
        let expected_dl = 7.0f32;
        let expected_w = expected_dl * (1.0 + IQ1S_DELTA);
        // Only first 32 weights (ib32=0) are affected by qh[0]; rest use qh=0 → dl=1
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - expected_w).abs() < 1e-5,
                "iq1s scale: weight[{i}] = {v}, expected {expected_w}"
            );
        }
        let other_expected = 1.0 * (1.0 + IQ1S_DELTA);
        for (i, &v) in output[32..].iter().enumerate() {
            assert!(
                (v - other_expected).abs() < 1e-5,
                "iq1s scale rest: weight[{i}] = {v}, expected {other_expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1s_negative_delta() {
        let grid = iq1s_grid_fixture();
        let mut block = make_iq1s_zero_block();
        // Set bit 15 of qh[0] to get negative delta
        let qh_val: u16 = 0x8000; // s=0, delta sign bit set
        block[34] = (qh_val & 0xFF) as u8;
        block[35] = (qh_val >> 8) as u8;
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // dl = 1*(2*0+1) = 1, delta = -0.125, grid[0]=+1 → weight=1*(1-0.125)=0.875
        let expected = 1.0 * (1.0 - IQ1S_DELTA);
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "iq1s neg delta: weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1s_grid_entry_one() {
        let grid = iq1s_grid_fixture();
        let mut block = make_iq1s_zero_block();
        // Set qs[0] = 1 → grid entry 1 (all bytes = 0xFF = -1 in signed)
        block[2] = 1;
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // For l=0 of ib32=0: grid[1] = all -1, dl=1, delta=+0.125 → weight=1*(-1+0.125)=-0.875
        let expected_first8 = 1.0 * (-1.0 + IQ1S_DELTA);
        // Other sub-groups still use entry 0 → weight = 1*(1+0.125)=1.125
        let expected_rest = 1.0 * (1.0 + IQ1S_DELTA);
        for (i, &v) in output[..8].iter().enumerate() {
            assert!(
                (v - expected_first8).abs() < 1e-5,
                "iq1s grid1: weight[{i}] = {v}, expected {expected_first8}"
            );
        }
        for (i, &v) in output[8..32].iter().enumerate() {
            assert!(
                (v - expected_rest).abs() < 1e-5,
                "iq1s grid1 rest: weight[{i}] = {v}, expected {expected_rest}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1s_short_block() {
        let grid = iq1s_grid_fixture();
        let block = vec![0u8; 10]; // too short
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        assert!(output.iter().all(|&v| v == 0.0));
    }

    /// Extended IQ1_S grid with entries at high indices for 11-bit qh-path testing.
    fn iq1s_grid_full_fixture() -> Box<[u64; 2048]> {
        let mut g = vec![0u64; 2048];
        // entry 0: all bytes = 0x01 (+1)
        g[0] = 0x0101010101010101u64;
        // entry 256 (high bits l=1 = 1, qs_lo = 0): all bytes = 0x00 (0)
        g[256] = 0x0000000000000000u64;
        // entry 512 (high bits l=2 = 2, qs_lo = 0): all bytes = 0xFF (-1)
        g[512] = 0xFFFFFFFFFFFFFFFFu64;
        // entry 768 (high bits l=3 = 3, qs_lo = 0): alternating 0x01/0xFF
        g[768] = 0xFF01FF01FF01FF01u64;
        // entry 1792 (high bits = 7, qs_lo = 0): all bytes = 0x01 (+1) — max high bits
        g[1792] = 0x0101010101010101u64;
        Box::new(g.try_into().unwrap())
    }

    #[test]
    fn test_dequant_iq1s_high_grid_index_qh() {
        // Verify 11-bit grid index: qs_lo | (((qh >> (3*l)) & 7) << 8)
        // l=0: high bits = 7 → grid_idx = 0 | (7 << 8) = 1792 (all bytes = +1)
        let grid = iq1s_grid_full_fixture();
        let mut block = make_iq1s_zero_block();
        // qh[0] bits 2:0 = 7 → l=0 uses grid_idx = 1792
        // s=0 (bits 14:12 = 0), positive delta (bit15=0)
        let qh_val: u16 = 0x0007; // bits 2:0 = 7
        block[34] = (qh_val & 0xFF) as u8;
        block[35] = (qh_val >> 8) as u8;
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // dl = 1*(2*0+1)=1, delta=+0.125, grid[1792]=+1 → weight = 1*(1+0.125) = 1.125
        let expected = 1.0f32 * (1.0 + IQ1S_DELTA);
        for (i, &v) in output[..8].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "iq1s high qh l=0: weight[{i}] = {v}, expected {expected}"
            );
        }
        // l=1..3 use high bits 0 → grid_idx = 0 → same expected
        for (i, &v) in output[8..32].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "iq1s high qh l=1..3: weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1s_all_l_values() {
        // All 4 l-values in one block: l0→entry 0, l1→entry 256, l2→entry 512, l3→entry 768
        // qh bits: l0 high=0, l1 high=1, l2 high=2, l3 high=3
        // qh_val = (0<<0) | (1<<3) | (2<<6) | (3<<9) = 0 + 8 + 128 + 1536 = 0x0688
        let grid = iq1s_grid_full_fixture();
        let mut block = make_iq1s_zero_block();
        let qh_val: u16 = 0x0688;
        block[34] = (qh_val & 0xFF) as u8;
        block[35] = (qh_val >> 8) as u8;
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // dl=1*(2*0+1)=1, delta=+0.125
        // l=0: grid[0] = +1 → weight = 1*(1 + 0.125) = 1.125
        let exp_l0 = 1.0f32 * (1.0 + IQ1S_DELTA);
        // l=1: grid[256] = 0 → weight = 1*(0 + 0.125) = 0.125
        let exp_l1 = 1.0f32 * (0.0 + IQ1S_DELTA);
        // l=2: grid[512] = -1 → weight = 1*(-1 + 0.125) = -0.875
        let exp_l2 = 1.0f32 * (-1.0 + IQ1S_DELTA);
        // l=3: grid[768] alternating 0x01/0xFF → +1 or -1 depending on byte position
        let expected = [exp_l0, exp_l1, exp_l2];
        for (l, &exp) in expected.iter().enumerate() {
            for (j, &v) in output[l * 8..(l + 1) * 8].iter().enumerate() {
                assert!(
                    (v - exp).abs() < 1e-5,
                    "iq1s all l: l={l} j={j} got {v}, expected {exp}"
                );
            }
        }
        // l=3: alternating pattern — byte j of grid[768] = 0xFF01FF01... so:
        // lo32 bytes 0,2 = 0x01 (+1), bytes 1,3 = 0xFF (-1)
        // hi32 bytes 0,2 = 0x01 (+1), bytes 1,3 = 0xFF (-1) (same pattern)
        let exp_l3_even = 1.0f32 * (1.0 + IQ1S_DELTA);
        let exp_l3_odd = 1.0f32 * (-1.0 + IQ1S_DELTA);
        for (j, &v) in output[24..32].iter().enumerate() {
            let exp = if j % 2 == 0 { exp_l3_even } else { exp_l3_odd };
            assert!(
                (v - exp).abs() < 1e-5,
                "iq1s all l: l=3 j={j} got {v}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1s_max_scale() {
        // s=7 → dl = d*(2*7+1) = d*15 = 15.0
        let grid = iq1s_grid_full_fixture();
        let mut block = make_iq1s_zero_block();
        // s=7: bits 14:12 = 0b111 → add 0x7000 to qh_val
        let qh_val: u16 = 0x7000;
        block[34] = (qh_val & 0xFF) as u8;
        block[35] = (qh_val >> 8) as u8;
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // dl = 1*(2*7+1) = 15, delta = +0.125, grid[0]=+1 → weight = 15*(1+0.125) = 16.875
        let expected = 15.0f32 * (1.0 + IQ1S_DELTA);
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq1s max scale: weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1s_scale_and_negative_delta() {
        // Combined: s=3, negative delta
        let grid = iq1s_grid_full_fixture();
        let mut block = make_iq1s_zero_block();
        // s=3 (bits 14:12 = 0b011 → 0x3000) + delta sign bit 15 → 0xB000
        let qh_val: u16 = 0xB000;
        block[34] = (qh_val & 0xFF) as u8;
        block[35] = (qh_val >> 8) as u8;
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // dl = 1*(2*3+1) = 7, delta = -0.125, grid[0]=+1 → weight = 7*(1-0.125) = 6.125
        let expected = 7.0f32 * (1.0 - IQ1S_DELTA);
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq1s scale+neg delta: weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1s_zero_grid_byte() {
        // Grid entry with 0 bytes → weight = dl * (0 + delta)
        let grid = iq1s_grid_full_fixture();
        let mut block = make_iq1s_zero_block();
        // l=0 → qs_lo[0]=0, high bits for l=0 = 1 → grid_idx = 256 → all bytes = 0
        let qh_val: u16 = 0x0001; // bits 2:0 = 1
        block[34] = (qh_val & 0xFF) as u8;
        block[35] = (qh_val >> 8) as u8;
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // dl=1, delta=+0.125, grid[256]=0 → weight = 1*(0 + 0.125) = 0.125
        let expected = 1.0f32 * (0.0 + IQ1S_DELTA);
        for (i, &v) in output[..8].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-5,
                "iq1s zero grid: weight[{i}] = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_dequant_iq1s_last_ib32() {
        // Test ib32=7 (last group): qh[7] at bytes 48-49
        let grid = iq1s_grid_full_fixture();
        let mut block = make_iq1s_zero_block();
        // qh[7]: s=5 → bits 14:12 = 0b101 → 0x5000; positive delta
        let qh_val: u16 = 0x5000;
        block[48] = (qh_val & 0xFF) as u8;
        block[49] = (qh_val >> 8) as u8;
        let mut output = [0.0f32; 256];
        dequant_iq1s_block(&block, &grid, &mut output);
        // Last 32 weights (ib32=7): dl = 1*(2*5+1) = 11, delta=+0.125, grid[0]=+1 → 11*1.125=12.375
        let expected_last = 11.0f32 * (1.0 + IQ1S_DELTA);
        for (i, &v) in output[224..].iter().enumerate() {
            assert!(
                (v - expected_last).abs() < 1e-4,
                "iq1s last ib32: weight[{i}] = {v}, expected {expected_last}"
            );
        }
        // First 224 weights use qh=0 → dl=1, same base expected
        let expected_rest = 1.0f32 * (1.0 + IQ1S_DELTA);
        for (i, &v) in output[..224].iter().enumerate() {
            assert!(
                (v - expected_rest).abs() < 1e-5,
                "iq1s last ib32 rest: weight[{i}] = {v}, expected {expected_rest}"
            );
        }
    }

    // ── IQ2_XS extended tests ─────────────────────────────────────────────────

    /// Extended IQ2_XS grid fixture: entries 0, 1, and 256 populated.
    /// - grid\[0\]   = 0x0808080808080808 (all bytes = 8)
    /// - grid\[1\]   = 0x0808082b (bytes: 0x2b,0x08,0x08,0x08,0x00,...)
    /// - grid\[256\] = 0x0404040404040404 (all bytes = 4, tests 9-bit grid index)
    fn iq2xs_grid_full_fixture() -> Box<[u64; 512]> {
        let mut g = vec![0u64; 512];
        g[0] = 0x0808080808080808u64;
        g[1] = 0x0808082bu64;
        g[256] = 0x0404040404040404u64; // all bytes = 4; used for high-bit grid index tests
        Box::new(g.try_into().unwrap())
    }

    #[test]
    fn test_dequant_iq2xs_high_bit_grid_index() {
        // qs16 = 256 = 0x0100 → grid_idx = 256 & 0x1FF = 256 (bit 8 set), sign_7bit = 0
        let grid = iq2xs_grid_full_fixture();
        let mut block = make_iq2xs_zero_block();
        // ib32=0, j=0, k=0: off=2-3, qs16=0x0100
        block[2] = 0x00;
        block[3] = 0x01;
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);
        // grid\[256\] all bytes=4; dl=0.125; signs all + → 0.125*4=0.5
        for (i, &v) in output[0..8].iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-5,
                "iq2xs high_bit: weight[{i}] = {v}, expected 0.5"
            );
        }
        // Other sub-groups use qs16=0 → grid\[0\]=8 → 0.125*8=1.0
        for (i, &v) in output[8..256].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs high_bit rest: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xs_full_sign_byte() {
        // KSIGNS_IQ2XS[127] = 0xFF → all 8 sign bits set → all 8 weights negative
        let grid = iq2xs_grid_full_fixture();
        let mut block = make_iq2xs_zero_block();
        // sign_7bit=127, grid_idx=0 → qs16 = (127 << 9) | 0 = 0xFE00
        block[2] = 0x00;
        block[3] = 0xFE;
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);
        // dl=0.125, grid\[0\] all bytes=8 → sign=-1 → -0.125*8 = -1.0
        for (i, &v) in output[0..8].iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-5,
                "iq2xs full_sign: weight[{i}] = {v}, expected -1.0"
            );
        }
        // Other sub-groups: sign_7bit=0 → KSIGNS_IQ2XS\[0\]=0x00 → positive → 1.0
        for (i, &v) in output[8..256].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs full_sign rest: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xs_split_scale_nibbles() {
        // scale_byte lo=5, hi=10 → dl1=(0.5+5)*0.25=1.375; dl2=(0.5+10)*0.25=2.625
        // Verifies j=0 uses lo-nibble scale and j=1 uses hi-nibble scale.
        let grid = iq2xs_grid_full_fixture();
        let mut block = make_iq2xs_zero_block();
        // scales\[0\] = lo=5, hi=10 → 0xA5
        block[66] = 0xA5;
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);
        let dl1 = (0.5f32 + 5.0) * 0.25;
        let dl2 = (0.5f32 + 10.0) * 0.25;
        // j=0 (output\[0..16\]): dl1 * grid\[0\]=8
        for (i, &v) in output[0..16].iter().enumerate() {
            let expected = dl1 * 8.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2xs split j=0: weight[{i}] = {v}, expected {expected}"
            );
        }
        // j=1 (output\[16..32\]): dl2 * grid\[0\]=8
        for (i, &v) in output[16..32].iter().enumerate() {
            let expected = dl2 * 8.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2xs split j=1: weight[{i}] = {v}, expected {expected}"
            );
        }
        // Remaining 7 groups: scale=0 → dl=0.125 → 1.0
        for (i, &v) in output[32..256].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs split rest: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xs_hi_nibble_max() {
        // scale_byte lo=0, hi=15 → dl1=0.125; dl2=(0.5+15)*0.25=3.875
        // Verifies the hi-nibble=15 path separately from the existing 0xFF test.
        let grid = iq2xs_grid_full_fixture();
        let mut block = make_iq2xs_zero_block();
        // scales\[0\] = 0xF0: lo=0, hi=15
        block[66] = 0xF0;
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);
        let dl2 = (0.5f32 + 15.0) * 0.25;
        // j=0 (output\[0..16\]): dl1=0.125 * 8 = 1.0
        for (i, &v) in output[0..16].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs hi_nibble j=0: weight[{i}] = {v}, expected 1.0"
            );
        }
        // j=1 (output\[16..32\]): dl2 * 8 = 31.0
        for (i, &v) in output[16..32].iter().enumerate() {
            let expected = dl2 * 8.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2xs hi_nibble j=1: weight[{i}] = {v}, expected {expected}"
            );
        }
        // Remaining 7 groups: scale=0 → 1.0
        for (i, &v) in output[32..256].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs hi_nibble rest: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xs_all_four_subgroups() {
        // Exercises all 4 sub-groups (j=0 k=0, j=0 k=1, j=1 k=0, j=1 k=1) in ib32=0.
        // scale lo=0, hi=2 → dl1=0.125, dl2=0.625
        // k=0 → grid\[0\]=all 8; k=1 → grid\[256\]=all 4 (via qs16=256=0x0100)
        let grid = iq2xs_grid_full_fixture();
        let mut block = make_iq2xs_zero_block();
        // scales\[0\] = lo=0, hi=2 → 0x20
        block[66] = 0x20;
        // j=0, k=0 (off 2-3): qs16=0 (already zero) → grid\[0\]
        // j=0, k=1 (off 4-5): qs16=256=0x0100 → grid\[256\]
        block[4] = 0x00;
        block[5] = 0x01;
        // j=1, k=0 (off 6-7): qs16=0 (already zero) → grid\[0\]
        // j=1, k=1 (off 8-9): qs16=256=0x0100 → grid\[256\]
        block[8] = 0x00;
        block[9] = 0x01;
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);
        let dl1 = (0.5f32 + 0.0) * 0.25; // 0.125
        let dl2 = (0.5f32 + 2.0) * 0.25; // 0.625
                                         // j=0 k=0 → output\[0..8\]: dl1 * 8 = 1.0
        for (i, &v) in output[0..8].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs subgrp j=0 k=0: weight[{i}] = {v}, expected 1.0"
            );
        }
        // j=0 k=1 → output\[8..16\]: dl1 * 4 = 0.5
        for (i, &v) in output[8..16].iter().enumerate() {
            let expected = dl1 * 4.0;
            assert!(
                (v - expected).abs() < 1e-5,
                "iq2xs subgrp j=0 k=1: weight[{i}] = {v}, expected {expected}"
            );
        }
        // j=1 k=0 → output\[16..24\]: dl2 * 8 = 5.0
        for (i, &v) in output[16..24].iter().enumerate() {
            let expected = dl2 * 8.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2xs subgrp j=1 k=0: weight[{i}] = {v}, expected {expected}"
            );
        }
        // j=1 k=1 → output\[24..32\]: dl2 * 4 = 2.5
        for (i, &v) in output[24..32].iter().enumerate() {
            let expected = dl2 * 4.0;
            assert!(
                (v - expected).abs() < 1e-5,
                "iq2xs subgrp j=1 k=1: weight[{i}] = {v}, expected {expected}"
            );
        }
        // Remaining 7 groups: scale=0 → dl=0.125 → 1.0
        for (i, &v) in output[32..256].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs subgrp rest: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xs_last_ib32() {
        // ib32=7: scales byte at offset 73; verifies last group is decoded correctly.
        // scale lo=7, hi=0 → dl1=(0.5+7)*0.25=1.875; dl2=0.125
        let grid = iq2xs_grid_full_fixture();
        let mut block = make_iq2xs_zero_block();
        // scales\[7\] at byte 73: lo=7, hi=0 → 0x07
        block[73] = 0x07;
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);
        let dl1 = (0.5f32 + 7.0) * 0.25; // 1.875
                                         // ib32=7: out_base=224
                                         // j=0 (output\[224..240\]): dl1 * grid\[0\]=8 = 15.0
        for (i, &v) in output[224..240].iter().enumerate() {
            let expected = dl1 * 8.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "iq2xs last ib32 j=0: weight[{i}] = {v}, expected {expected}"
            );
        }
        // j=1 (output\[240..256\]): dl2=0.125 * 8 = 1.0
        for (i, &v) in output[240..256].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs last ib32 j=1: weight[{i}] = {v}, expected 1.0"
            );
        }
        // First 224 weights: scale=0 → dl=0.125 → 1.0
        for (i, &v) in output[..224].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xs last ib32 rest: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xs_negative_d() {
        // d = -1.0 (0xBC00), scale=0 → dl = -1*(0.5+0)*0.25 = -0.125
        // grid[0] all bytes=8 → output = -0.125 * 8 = -1.0 for all weights
        let grid = iq2xs_grid_fixture();
        let mut block = make_iq2xs_zero_block();
        block[0] = 0x00;
        block[1] = 0xBC; // f16 -1.0
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-5,
                "iq2xs neg d: weight[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xs_dl2_hi_nibble() {
        // Verify the j=1 path uses the high nibble of scale_byte for dl2.
        // scale_byte = 0xF0 → lo=0, hi=15 → dl1=0.125, dl2=3.875
        // ib32=0: j=0 (output[0..16]) uses dl1 → 0.125*8=1.0
        //         j=1 (output[16..32]) uses dl2 → 3.875*8=31.0
        let grid = iq2xs_grid_fixture();
        let mut block = make_iq2xs_zero_block();
        block[66] = 0xF0; // hi nibble=15, lo=0
        let mut output = [0.0f32; 256];
        dequant_iq2xs_block(&block, &grid, &mut output);
        let dl1_expected = 0.125f32 * 8.0; // 1.0
        let dl2_expected = (0.5f32 + 15.0) * 0.25 * 8.0; // 31.0
        for (i, &v) in output[0..16].iter().enumerate() {
            assert!(
                (v - dl1_expected).abs() < 1e-4,
                "iq2xs dl1: output[{i}] = {v}, expected {dl1_expected}"
            );
        }
        for (i, &v) in output[16..32].iter().enumerate() {
            assert!(
                (v - dl2_expected).abs() < 1e-4,
                "iq2xs dl2: output[{}] = {v}, expected {dl2_expected}",
                i + 16
            );
        }
    }

    #[test]
    fn test_dequant_iq2xxs_negative_d() {
        // d = -1.0 (0xBC00), all sub_scale=0 → db = -0.125
        // grid[0] all bytes=8 → output = -0.125 * 8 = -1.0 for all weights
        let grid = iq2xxs_grid_fixture();
        let mut block = make_iq2xxs_zero_block();
        block[0] = 0x00;
        block[1] = 0xBC; // f16 -1.0
        let mut output = [0.0f32; 256];
        dequant_iq2xxs_block(&block, &grid, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-5,
                "iq2xxs neg d: weight[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq2xxs_second_ib32() {
        // ib32=1 with sub_scale=3 → db = 1*(0.5+3)*0.25 = 0.875
        // grid[0] all bytes=8 → output[32..64] = 0.875*8 = 7.0
        // ib32=0 sub_scale=0 → output[0..32] = 1.0
        let grid = iq2xxs_grid_fixture();
        let mut block = make_iq2xxs_zero_block();
        // ib32=1 base = 2 + 1*8 = 10; aux1 at bytes 14..17; hi nibble of byte 17 = sub_scale
        // aux1 byte 3 (offset 10+7=17): bits 31:28 = sub_scale = 3 → 0x30
        block[17] = 0x30;
        let mut output = [0.0f32; 256];
        dequant_iq2xxs_block(&block, &grid, &mut output);
        let expected_ib1 = (0.5f32 + 3.0) * 0.25 * 8.0; // 7.0
        for (i, &v) in output[32..64].iter().enumerate() {
            assert!(
                (v - expected_ib1).abs() < 1e-4,
                "iq2xxs 2nd ib32: output[{}] = {v}, expected {expected_ib1}",
                i + 32
            );
        }
        // First group sub_scale=0 → 1.0
        for (i, &v) in output[0..32].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq2xxs 2nd ib32 group0: output[{i}] = {v}, expected 1.0"
            );
        }
    }

    // ── IQ3_S extended tests ──────────────────────────────────────────────────

    /// Extended IQ3_S grid fixture with entries 0–3 and 256.
    /// - grid\[0\]   = 0x05050505 (all bytes = 5)
    /// - grid\[1\]   = 0x0A0A0A0A (all bytes = 10)
    /// - grid\[2\]   = 0x04040404 (all bytes = 4)
    /// - grid\[3\]   = 0x07070707 (all bytes = 7)
    /// - grid\[256\] = 0x02020202 (all bytes = 2, used for 9-bit high-bit index tests)
    fn iq3s_grid_full_fixture() -> Box<[u32; 512]> {
        let mut g = vec![0u32; 512];
        g[0] = 0x05050505u32;
        g[1] = 0x0A0A0A0Au32;
        g[2] = 0x04040404u32;
        g[3] = 0x07070707u32;
        g[256] = 0x02020202u32; // for 9-bit grid index test (qh high-bit set)
        Box::new(g.try_into().unwrap())
    }

    #[test]
    fn test_dequant_iq3s_high_bit_qh() {
        // qh_byte bit 0 set for ib32=0 l=0 → grid_idx1 = 0 | (1<<8) = 256
        // grid\[256\] all bytes = 2; db=1*(1+0)=1 → output\[0..4\] = 2.0
        let grid = iq3s_grid_full_fixture();
        let mut block = make_iq3s_zero_block();
        // qh\[0\] at byte 66: bit 0 = high bit of grid_idx1 for l=0
        block[66] = 0x01;
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        // l=0 g1: grid\[256\] all bytes=2 → 2.0
        for (i, &v) in output[0..4].iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "iq3s high_bit_qh l=0 g1: weight[{i}] = {v}, expected 2.0"
            );
        }
        // l=0 g2 (qs2=0, qh bit 1=0): grid\[0\]=5 → 5.0
        for (i, &v) in output[4..8].iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s high_bit_qh l=0 g2: weight[{i}] = {v}, expected 5.0"
            );
        }
        // Remaining groups (l=1,2,3 and ib32=1..7): qh=0, grid\[0\]=5 → 5.0
        for (i, &v) in output[8..256].iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s high_bit_qh rest: weight[{i}] = {v}, expected 5.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3s_g2_grid_path() {
        // qs2 ≠ qs1 for l=0, ib32=0: verifies g2 uses its own independent grid lookup.
        // qs1=0→grid\[0\]=5; qs2=1→grid\[1\]=10; db=1.0
        let grid = iq3s_grid_full_fixture();
        let mut block = make_iq3s_zero_block();
        // qs1 for l=0: block\[2\] = 0 (already zero)
        // qs2 for l=0: block\[3\] = 1
        block[3] = 1;
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        // l=0 g1 (qs1=0): grid\[0\] all 5 → 5.0
        for (i, &v) in output[0..4].iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s g2_path g1: weight[{i}] = {v}, expected 5.0"
            );
        }
        // l=0 g2 (qs2=1): grid\[1\] all 10 → 10.0
        for (i, &v) in output[4..8].iter().enumerate() {
            assert!(
                (v - 10.0).abs() < 1e-5,
                "iq3s g2_path g2: weight[{i}] = {v}, expected 10.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3s_g2_sign_bits() {
        // signs_byte bits 4-7 control sign for g2 weights (j=0..3).
        // Set signs\[0\] (l=0, ib32=0) = 0xF0 → bits 4-7 all set → g2 all negative.
        let grid = iq3s_grid_full_fixture();
        let mut block = make_iq3s_zero_block();
        // signs\[ib32=0, l=0\] at block\[74\] = 0xF0
        block[74] = 0xF0;
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        // g1 (bits 0-3 = 0): all positive → 5.0
        for (i, &v) in output[0..4].iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s g2_sign g1: weight[{i}] = {v}, expected 5.0"
            );
        }
        // g2 (bits 4-7 = 1): all negative → -5.0
        for (i, &v) in output[4..8].iter().enumerate() {
            assert!(
                (v - (-5.0)).abs() < 1e-5,
                "iq3s g2_sign g2: weight[{i}] = {v}, expected -5.0"
            );
        }
        // Remaining l=1,2,3: signs=0 → 5.0
        for (i, &v) in output[8..32].iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s g2_sign rest: weight[{i}] = {v}, expected 5.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3s_all_four_l_values() {
        // Set qs1 for l=0,1,2,3 to entries 0,1,2,3 → different grid bytes per sub-group.
        let grid = iq3s_grid_full_fixture();
        let mut block = make_iq3s_zero_block();
        // For ib32=0: qs1\[l\] = block\[2 + 2*l\]; qs2\[l\] = block\[3 + 2*l\]
        // l=0: qs1=0 (already zero)
        block[4] = 1; // l=1 qs1 → grid\[1\]=10
        block[6] = 2; // l=2 qs1 → grid\[2\]=4
        block[8] = 3; // l=3 qs1 → grid\[3\]=7
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        // db=1.0; qs2 all 0 → g2=grid\[0\]=5; signs all +
        let expected_g1 = [5.0f32, 10.0, 4.0, 7.0];
        for (l, &exp) in expected_g1.iter().enumerate() {
            // g1 output: output\[l*8 .. l*8+4\]
            for (j, &v) in output[l * 8..l * 8 + 4].iter().enumerate() {
                assert!(
                    (v - exp).abs() < 1e-5,
                    "iq3s all_l g1 l={l} j={j}: got {v}, expected {exp}"
                );
            }
            // g2 output: output\[l*8+4 .. l*8+8\] → grid\[0\]=5
            for (j, &v) in output[l * 8 + 4..l * 8 + 8].iter().enumerate() {
                assert!(
                    (v - 5.0).abs() < 1e-5,
                    "iq3s all_l g2 l={l} j={j}: got {v}, expected 5.0"
                );
            }
        }
    }

    #[test]
    fn test_dequant_iq3s_nibble_pair() {
        // scale_byte at block\[106\]: lo nibble for ib32=0, hi nibble for ib32=1.
        // block\[106\] = 0x73 → ib32=0 nibble=3 (db=7.0), ib32=1 nibble=7 (db=15.0).
        let grid = iq3s_grid_full_fixture();
        let mut block = make_iq3s_zero_block();
        // scales byte 0: lo=3, hi=7
        block[106] = 0x73;
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        // ib32=0: db = 1*(1+6) = 7; grid\[0\]=5 → 35.0
        for (i, &v) in output[0..32].iter().enumerate() {
            assert!(
                (v - 35.0).abs() < 1e-4,
                "iq3s nibble_pair ib32=0: weight[{i}] = {v}, expected 35.0"
            );
        }
        // ib32=1: db = 1*(1+14) = 15; grid\[0\]=5 → 75.0
        for (i, &v) in output[32..64].iter().enumerate() {
            assert!(
                (v - 75.0).abs() < 1e-4,
                "iq3s nibble_pair ib32=1: weight[{i}] = {v}, expected 75.0"
            );
        }
        // ib32=2..7: nibble=0 → db=1.0 → 5.0
        for (i, &v) in output[64..256].iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s nibble_pair rest: weight[{i}] = {v}, expected 5.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3s_last_ib32_scale() {
        // ib32=7: scale at block\[106 + 7/2\] = block\[109\], hi nibble (7%2=1 → shift 4).
        // Set block\[109\] = 0x70 → nibble=7 → db = 1*(1+14) = 15.0; grid\[0\]=5 → 75.0.
        let grid = iq3s_grid_full_fixture();
        let mut block = make_iq3s_zero_block();
        block[109] = 0x70; // hi nibble=7 → ib32=7
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        let expected = 15.0f32 * 5.0;
        // Last 32 weights (ib32=7): out_base=224
        for (i, &v) in output[224..256].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq3s last ib32 scale: weight[{i}] = {v}, expected {expected}"
            );
        }
        // First 224 weights: scale=0 → db=1.0 → 5.0
        for (i, &v) in output[..224].iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s last ib32 scale rest: weight[{i}] = {v}, expected 5.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3s_negative_d() {
        // d = -1.0 → db = -1*(1+2*0) = -1.0; grid[0] bytes=5 → output = -5.0 for all
        let grid = iq3s_grid_fixture();
        let mut block = make_iq3s_zero_block();
        block[0] = 0x00;
        block[1] = 0xBC; // f16 -1.0
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-5.0)).abs() < 1e-5,
                "iq3s neg d: weight[{i}] = {v}, expected -5.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq3s_second_ib32() {
        // ib32=0 scale: block[106] lo nibble = 0 → db=1.0 → output[0..32] = 5.0
        // ib32=1 scale: block[106] hi nibble = 5 → nibble=5 → db=1*(1+10)=11.0 → output[32..64]=55.0
        let grid = iq3s_grid_fixture();
        let mut block = make_iq3s_zero_block();
        block[106] = 0x50; // lo nibble=0 (ib32=0), hi nibble=5 (ib32=1)
        let mut output = [0.0f32; 256];
        dequant_iq3s_block(&block, &grid, &mut output);
        // ib32=0: db=1.0, grid[0]=5 → 5.0
        for (i, &v) in output[0..32].iter().enumerate() {
            assert!(
                (v - 5.0).abs() < 1e-5,
                "iq3s 2nd ib32 group0: output[{i}] = {v}, expected 5.0"
            );
        }
        // ib32=1: db=11.0, grid[0]=5 → 55.0
        let expected = 11.0f32 * 5.0;
        for (i, &v) in output[32..64].iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-4,
                "iq3s 2nd ib32: output[{}] = {v}, expected {expected}",
                i + 32
            );
        }
    }

    // ── IQ4_XS extended tests ─────────────────────────────────────────────────

    /// Helper: build an IQ4_XS block (136 bytes) with d=1.0 and all zeroed fields.
    /// By default: scales_h=0, scales_l=0, qs=0 → ls=0, dl=-32.0, nibble=0→-127.
    /// Callers override specific fields for each test.
    fn make_iq4xs_zero_block() -> Vec<u8> {
        let mut block = vec![0u8; 136];
        block[0] = 0x00;
        block[1] = 0x3C; // d = 1.0
        block
    }

    #[test]
    fn test_dequant_iq4xs_max_ls() {
        // ls_lo=15, ls_hi=3 → ls=63, dl=1*(63-32)=31.0; nibble=8→KVALUES[8]=1 → output=31.0
        let mut block = make_iq4xs_zero_block();
        // scales_h bits 1:0 = 3 for ib32=0 → block[2]=0x03
        block[2] = 0x03;
        block[3] = 0x00;
        // scales_l ib32=0 lo nibble=15 → block[4]=0x0F
        block[4] = 0x0F;
        // qs for ib32=0 (bytes 8..24) = 0x88 → nibble=8 → KVALUES_IQ4NL[8]=1
        block[8..24].fill(0x88);
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - 31.0).abs() < 1e-4,
                "iq4xs max_ls: weight[{i}] = {v}, expected 31.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4xs_scales_h_contribution() {
        // ls_lo=0, ls_hi=1 → ls=16, dl=1*(16-32)=-16.0; nibble=8→1 → output=-16.0.
        // Isolates the scales_h contribution independently of ls_lo.
        let mut block = make_iq4xs_zero_block();
        // scales_h bits 1:0 = 1 for ib32=0 → block[2]=0x01
        block[2] = 0x01;
        block[3] = 0x00;
        block[8..24].fill(0x88);
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                (v - (-16.0)).abs() < 1e-4,
                "iq4xs scales_h: weight[{i}] = {v}, expected -16.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4xs_extreme_nibbles() {
        // nibble=0 → KVALUES_IQ4NL[0]=-127; nibble=15 → KVALUES_IQ4NL[15]=113.
        // dl=1.0 (ls=33). One qs byte = 0xF0: lo nibble=0 (→-127), hi nibble=15 (→113).
        let mut block = make_iq4xs_zero_block();
        // ls=33: scales_h bits 1:0=2 → block[2]=0x02; scales_l ib32=0 lo=1 → block[4]=0x01
        block[2] = 0x02;
        block[4] = 0x01;
        // qs[8] = 0xF0: lo=0 (−127), hi=15 (113)
        block[8] = 0xF0;
        block[9..24].fill(0x88);
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        assert!(
            (output[0] - (-127.0)).abs() < 1e-4,
            "iq4xs extreme lo nibble: got {}, expected -127.0",
            output[0]
        );
        assert!(
            (output[16] - 113.0).abs() < 1e-4,
            "iq4xs extreme hi nibble: got {}, expected 113.0",
            output[16]
        );
    }

    #[test]
    fn test_dequant_iq4xs_scales_l_odd_ib32() {
        // For odd ib32=1, ls_lo comes from the hi nibble of scales_l byte block[4].
        // scales_h=0x000A: ib32=0 ls_hi=2 (ls=32→dl=0), ib32=1 ls_hi=2.
        // block[4]=0x10: ib32=0 lo=0, ib32=1 hi=1 → ib32=1 ls=33, dl=1.0.
        let mut block = make_iq4xs_zero_block();
        block[2] = 0x0A; // scales_h low byte: ib32=0 bits=2, ib32=1 bits=2
        block[3] = 0x00;
        block[4] = 0x10; // ib32=0 lo=0; ib32=1 hi=1
        block[8..24].fill(0x88); // ib32=0 qs (dl=0 → output 0)
        block[24..40].fill(0x88); // ib32=1 qs → nibble=8 → KVALUES[8]=1
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        // ib32=0: dl=0 → 0.0
        for (i, &v) in output[..32].iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "iq4xs odd ib32=0: weight[{i}] = {v}, expected 0.0"
            );
        }
        // ib32=1: dl=1.0 → 1.0
        for (i, &v) in output[32..64].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq4xs odd ib32=1: weight[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4xs_last_ib32() {
        // ib32=7: ls_lo from hi-nibble of block[7], ls_hi from scales_h bits 15:14.
        // scales_h=0xAAAA → all ib32 ls_hi=2; ib32=0..6 ls_lo=0 → ls=32, dl=0.
        // block[7]=0xF0 → ib32=7 ls_lo=15 → ls=47, dl=1*(47-32)=15.0.
        let mut block = make_iq4xs_zero_block();
        block[2] = 0xAA; // scales_h low byte
        block[3] = 0xAA; // scales_h high byte → all ib32 ls_hi=2
        block[7] = 0xF0; // ib32=7 ls_lo = hi nibble = 15
        block[120..136].fill(0x88); // ib32=7 qs: nibble=8 → 1
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        // ib32=0..6: ls=32, dl=0 → 0.0
        for (i, &v) in output[..224].iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "iq4xs last ib32 rest: weight[{i}] = {v}, expected 0.0"
            );
        }
        // ib32=7: dl=15.0 → 15.0 * 1 = 15.0
        for (i, &v) in output[224..256].iter().enumerate() {
            assert!(
                (v - 15.0).abs() < 1e-4,
                "iq4xs last ib32: weight[{i}] = {v}, expected 15.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4xs_two_nibbles_per_byte() {
        // One qs byte encodes two weights: lo nibble → output[j], hi nibble → output[j+16].
        // dl=1.0 (ls=33). qs[8]=0x98: lo=8→KVALUES[8]=1; hi=9→KVALUES[9]=13.
        let mut block = make_iq4xs_zero_block();
        block[2] = 0x02;
        block[4] = 0x01; // ls=33, dl=1.0 for ib32=0
        block[8] = 0x98; // lo nibble=8, hi nibble=9
        block[9..24].fill(0x88);
        let mut output = [0.0f32; 256];
        dequant_iq4xs_block(&block, &mut output);
        assert!(
            (output[0] - 1.0).abs() < 1e-5,
            "iq4xs two_nibbles lo: got {}, expected 1.0",
            output[0]
        );
        assert!(
            (output[16] - 13.0).abs() < 1e-4,
            "iq4xs two_nibbles hi: got {}, expected 13.0",
            output[16]
        );
    }

    // ── IQ4_NL tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_dequant_iq4nl_short_block() {
        let block = vec![0u8; 10]; // too short
        let mut output = [42.0f32; 32]; // non-zero sentinel
        dequant_iq4nl_block(&block, &mut output);
        // short block → output unchanged (sentinel still 42.0)
        assert!(
            output.iter().all(|&v| (v - 42.0).abs() < 1e-5),
            "iq4nl short block: output should be unchanged"
        );
    }

    #[test]
    fn test_dequant_iq4nl_zero_scale() {
        let mut block = vec![0u8; 18];
        // scale = 0.0 (f16 0x0000)
        block[0] = 0x00;
        block[1] = 0x00;
        block[2..18].fill(0x88); // nibble=8 → KVALUES[8]=1, but scale=0 → 0
        let mut output = [0.0f32; 32];
        dequant_iq4nl_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                v.abs() < 1e-5,
                "iq4nl zero scale: output[{i}] = {v}, expected 0.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4nl_unit_scale_nibble8() {
        // scale=1.0, all nibbles=8 → KVALUES_IQ4NL[8]=1 → output=1.0 for all 32.
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[2..18].fill(0x88);
        let mut output = [0.0f32; 32];
        dequant_iq4nl_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq4nl unit nibble8: output[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4nl_min_nibble() {
        // nibble=0 → KVALUES_IQ4NL[0]=-127; scale=1.0 → output=-127.0.
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
                         // all bytes = 0x00 → both nibbles = 0
        let mut output = [0.0f32; 32];
        dequant_iq4nl_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-127.0)).abs() < 1e-4,
                "iq4nl min nibble: output[{i}] = {v}, expected -127.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4nl_max_nibble() {
        // nibble=15 → KVALUES_IQ4NL[15]=113; scale=1.0 → output=113.0.
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[2..18].fill(0xFF); // both nibbles = 15
        let mut output = [0.0f32; 32];
        dequant_iq4nl_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - 113.0).abs() < 1e-4,
                "iq4nl max nibble: output[{i}] = {v}, expected 113.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4nl_negative_scale() {
        // scale=-1.0, nibble=8→KVALUES[8]=1 → output=-1.0.
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0xBC; // f16 -1.0
        block[2..18].fill(0x88);
        let mut output = [0.0f32; 32];
        dequant_iq4nl_block(&block, &mut output);
        for (i, &v) in output.iter().enumerate() {
            assert!(
                (v - (-1.0)).abs() < 1e-5,
                "iq4nl neg scale: output[{i}] = {v}, expected -1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4nl_two_nibbles_differ() {
        // One byte = 0x80: lo=0 → KVALUES[0]=-127; hi=8 → KVALUES[8]=1. scale=1.0.
        // Verifies lo nibble → output[2i] and hi nibble → output[2i+1].
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[2] = 0x80; // lo=0 (-127), hi=8 (1)
        block[3..18].fill(0x88); // remaining all nibble=8 → 1.0
        let mut output = [0.0f32; 32];
        dequant_iq4nl_block(&block, &mut output);
        // output[0]: lo nibble=0 → -127.0
        assert!(
            (output[0] - (-127.0)).abs() < 1e-4,
            "iq4nl two_nibbles lo: got {}, expected -127.0",
            output[0]
        );
        // output[1]: hi nibble=8 → 1.0
        assert!(
            (output[1] - 1.0).abs() < 1e-5,
            "iq4nl two_nibbles hi: got {}, expected 1.0",
            output[1]
        );
        // output[2..32]: all nibble=8 → 1.0
        for (i, &v) in output[2..32].iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "iq4nl two_nibbles rest: output[{i}] = {v}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_dequant_iq4nl_all_kvalues_entries() {
        // Use 8 bytes to exercise all 16 KVALUES_IQ4NL entries in order.
        // Byte k = (k<<4)|(k-1) for k=1..8 gives pairs (0,1),(2,3),...,(14,15).
        // bytes: 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE; remaining=0x00.
        // KVALUES_IQ4NL: [-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113]
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        block[2] = 0x10; // lo=0→-127, hi=1→-104
        block[3] = 0x32; // lo=2→ -83, hi=3→ -65
        block[4] = 0x54; // lo=4→ -49, hi=5→ -35
        block[5] = 0x76; // lo=6→ -22, hi=7→ -10
        block[6] = 0x98; // lo=8→   1, hi=9→  13
        block[7] = 0xBA; // lo=10→ 25, hi=11→ 38
        block[8] = 0xDC; // lo=12→ 53, hi=13→ 69
        block[9] = 0xFE; // lo=14→ 89, hi=15→113
                         // block[10..18] = 0 → nibble=0 → -127.0 for output[20..32]
        let mut output = [0.0f32; 32];
        dequant_iq4nl_block(&block, &mut output);
        let expected_first16: [f32; 16] = [
            -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
            69.0, 89.0, 113.0,
        ];
        for (i, (&v, &exp)) in output[..16].iter().zip(expected_first16.iter()).enumerate() {
            assert!(
                (v - exp).abs() < 1e-4,
                "iq4nl all_kvalues output[{i}]: got {v}, expected {exp}"
            );
        }
    }
}
