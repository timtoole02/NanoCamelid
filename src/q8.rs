use std::{
    env, fmt,
    hint::black_box,
    io::{self, Read, Seek, SeekFrom},
    time::{Duration, Instant},
};

use crate::gguf::{GgufTensorDescriptor, GgufTensorType};

pub const DEFAULT_DOT_BENCH_ITERATIONS: usize = 2_000;
pub const DEFAULT_DOT_BENCH_RUNS: usize = 5;
pub const DOT_KERNEL_ENV: &str = "NANOCAMELID_Q8_DOT_KERNEL";
pub const SDOT_CANDIDATE_ENV: &str = "NANOCAMELID_Q8_DOT_SDOT";
pub const Q8_BLOCK_SIZE: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 2 + Q8_BLOCK_SIZE;
pub const Q4_0_BLOCK_BYTES: usize = 2 + (Q8_BLOCK_SIZE / 2);
pub const QK_K_BLOCK_SIZE: usize = 256;
pub const Q6_K_BLOCK_BYTES: usize = 128 + 64 + 16 + 2;

const BENCH_BLOCKS: usize = 1_024;
const BENCH_ELEMENTS: usize = Q8_BLOCK_SIZE * BENCH_BLOCKS;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q8DotKernel {
    Scalar,
    Neon,
    Sdot,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8DotKernelSelector {
    pub requested: Option<Q8DotKernel>,
    pub selected: Q8DotKernel,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Debug)]
pub struct DotBenchmarkReport {
    pub iterations: usize,
    pub runs: usize,
    pub blocks_per_iteration: usize,
    pub elements_per_iteration: usize,
    pub kernel_selector: Q8DotKernelSelector,
    pub selected: DotTimingSummary,
    pub scalar: DotTimingSummary,
    pub neon: Option<DotTimingSummary>,
    pub sdot: Option<DotTimingSummary>,
}

#[derive(Debug)]
pub struct TimedDot {
    pub checksum: i64,
    pub elapsed: Duration,
}

#[derive(Debug)]
pub struct DotTimingSummary {
    pub checksum: i64,
    pub elapsed_runs: Vec<Duration>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q8_0Block {
    scale_bits: u16,
    values: [i8; Q8_BLOCK_SIZE],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q4_0Block {
    scale_bits: u16,
    values: [u8; Q8_BLOCK_SIZE / 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q6KBlock {
    ql: [u8; 128],
    qh: [u8; 64],
    scales: [i8; 16],
    scale_bits: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Q8_0RowReader {
    pub tensor_name: String,
    pub absolute_offset: u64,
    pub rows: usize,
    pub columns: usize,
    pub blocks_per_row: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Q8RowDotReport {
    pub scalar: i32,
    pub selected: i32,
    pub kernel_selector: Q8DotKernelSelector,
}

#[derive(Debug, Eq, PartialEq)]
pub enum Q8BlockError {
    MisalignedLength { bytes: usize, block_bytes: usize },
    InvalidTensorType { name: String },
    InvalidTensorShape { name: String, dimensions: Vec<u64> },
    ColumnMismatch { lhs: usize, rhs: usize },
    RowOutOfBounds { row: usize, rows: usize },
    ValueTooLarge(&'static str),
    OffsetOverflow,
    Io(String),
}

impl fmt::Display for Q8BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MisalignedLength { bytes, block_bytes } => write!(
                f,
                "Q8_0 byte length {bytes} is not aligned to {block_bytes}-byte blocks"
            ),
            Self::InvalidTensorType { name } => write!(f, "tensor {name} is not Q8_0"),
            Self::InvalidTensorShape { name, dimensions } => {
                write!(f, "tensor {name} has invalid Q8_0 shape {dimensions:?}")
            }
            Self::ColumnMismatch { lhs, rhs } => {
                write!(f, "Q8_0 row column mismatch: lhs={lhs}, rhs={rhs}")
            }
            Self::RowOutOfBounds { row, rows } => {
                write!(f, "Q8_0 row {row} is out of bounds for {rows} rows")
            }
            Self::ValueTooLarge(name) => write!(f, "{name} is too large for this platform"),
            Self::OffsetOverflow => write!(f, "Q8_0 row offset overflow"),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for Q8BlockError {}

impl From<io::Error> for Q8BlockError {
    fn from(err: io::Error) -> Self {
        Self::Io(err.to_string())
    }
}

impl DotBenchmarkReport {
    pub fn scalar_min_ns_per_block(&self) -> f64 {
        ns_per_block(
            self.scalar.min_elapsed(),
            self.iterations,
            self.blocks_per_iteration,
        )
    }

    pub fn scalar_median_ns_per_block(&self) -> f64 {
        ns_per_block(
            self.scalar.median_elapsed(),
            self.iterations,
            self.blocks_per_iteration,
        )
    }

    pub fn neon_min_ns_per_block(&self) -> Option<f64> {
        self.neon.as_ref().map(|neon| {
            ns_per_block(
                neon.min_elapsed(),
                self.iterations,
                self.blocks_per_iteration,
            )
        })
    }

    pub fn neon_median_ns_per_block(&self) -> Option<f64> {
        self.neon.as_ref().map(|neon| {
            ns_per_block(
                neon.median_elapsed(),
                self.iterations,
                self.blocks_per_iteration,
            )
        })
    }

    pub fn neon_min_speedup(&self) -> Option<f64> {
        self.neon_min_ns_per_block()
            .map(|neon_ns| self.scalar_min_ns_per_block() / neon_ns)
    }

    pub fn neon_median_speedup(&self) -> Option<f64> {
        self.neon_median_ns_per_block()
            .map(|neon_ns| self.scalar_median_ns_per_block() / neon_ns)
    }

    pub fn sdot_min_ns_per_block(&self) -> Option<f64> {
        self.sdot.as_ref().map(|sdot| {
            ns_per_block(
                sdot.min_elapsed(),
                self.iterations,
                self.blocks_per_iteration,
            )
        })
    }

    pub fn sdot_median_ns_per_block(&self) -> Option<f64> {
        self.sdot.as_ref().map(|sdot| {
            ns_per_block(
                sdot.median_elapsed(),
                self.iterations,
                self.blocks_per_iteration,
            )
        })
    }

    pub fn sdot_min_speedup(&self) -> Option<f64> {
        self.sdot_min_ns_per_block()
            .map(|sdot_ns| self.scalar_min_ns_per_block() / sdot_ns)
    }

    pub fn sdot_median_speedup(&self) -> Option<f64> {
        self.sdot_median_ns_per_block()
            .map(|sdot_ns| self.scalar_median_ns_per_block() / sdot_ns)
    }

    pub fn sdot_vs_neon_min_speedup(&self) -> Option<f64> {
        self.neon_min_ns_per_block()
            .zip(self.sdot_min_ns_per_block())
            .map(|(neon_ns, sdot_ns)| neon_ns / sdot_ns)
    }

    pub fn sdot_vs_neon_median_speedup(&self) -> Option<f64> {
        self.neon_median_ns_per_block()
            .zip(self.sdot_median_ns_per_block())
            .map(|(neon_ns, sdot_ns)| neon_ns / sdot_ns)
    }
}

impl Q8_0Block {
    pub fn from_bytes(bytes: &[u8; Q8_0_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut values = [0_i8; Q8_BLOCK_SIZE];
        for (value, &byte) in values.iter_mut().zip(&bytes[2..]) {
            *value = i8::from_le_bytes([byte]);
        }

        Self { scale_bits, values }
    }

    pub fn from_parts(scale_bits: u16, values: [i8; Q8_BLOCK_SIZE]) -> Self {
        Self { scale_bits, values }
    }

    pub fn scale_bits(&self) -> u16 {
        self.scale_bits
    }

    pub fn scale_f32(&self) -> f32 {
        f16_bits_to_f32(self.scale_bits)
    }

    pub fn values(&self) -> &[i8; Q8_BLOCK_SIZE] {
        &self.values
    }

    pub fn dot_i32(&self, rhs: &Self) -> i32 {
        dot_i8_scalar(&self.values, &rhs.values)
    }

    pub fn scaled_dot_f32(&self, rhs: &Self) -> f32 {
        self.scale_f32() * rhs.scale_f32() * self.dot_i32(rhs) as f32
    }
}

impl Q4_0Block {
    pub fn from_bytes(bytes: &[u8; Q4_0_BLOCK_BYTES]) -> Self {
        let scale_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        values.copy_from_slice(&bytes[2..]);

        Self { scale_bits, values }
    }

    pub fn from_parts(scale_bits: u16, values: [u8; Q8_BLOCK_SIZE / 2]) -> Self {
        Self { scale_bits, values }
    }

    pub fn scale_bits(&self) -> u16 {
        self.scale_bits
    }

    pub fn scale_f32(&self) -> f32 {
        f16_bits_to_f32(self.scale_bits)
    }

    pub fn packed_values(&self) -> &[u8; Q8_BLOCK_SIZE / 2] {
        &self.values
    }

    pub fn unpack_values(&self) -> [i8; Q8_BLOCK_SIZE] {
        let mut out = [0_i8; Q8_BLOCK_SIZE];
        for (idx, &byte) in self.values.iter().enumerate() {
            out[idx] = ((byte & 0x0f) as i8) - 8;
            out[idx + 16] = ((byte >> 4) as i8) - 8;
        }
        out
    }
}

impl Q6KBlock {
    pub fn from_bytes(bytes: &[u8; Q6_K_BLOCK_BYTES]) -> Self {
        let mut ql = [0_u8; 128];
        let mut qh = [0_u8; 64];
        let mut scales = [0_i8; 16];
        ql.copy_from_slice(&bytes[0..128]);
        qh.copy_from_slice(&bytes[128..192]);
        for (scale, &byte) in scales.iter_mut().zip(&bytes[192..208]) {
            *scale = i8::from_le_bytes([byte]);
        }
        let scale_bits = u16::from_le_bytes([bytes[208], bytes[209]]);

        Self {
            ql,
            qh,
            scales,
            scale_bits,
        }
    }

    pub fn from_parts(ql: [u8; 128], qh: [u8; 64], scales: [i8; 16], scale_bits: u16) -> Self {
        Self {
            ql,
            qh,
            scales,
            scale_bits,
        }
    }

    pub fn scale_bits(&self) -> u16 {
        self.scale_bits
    }

    pub fn scale_f32(&self) -> f32 {
        f16_bits_to_f32(self.scale_bits)
    }

    pub fn dequantize(&self, out: &mut [f32; QK_K_BLOCK_SIZE]) {
        let d = self.scale_f32();
        let mut ql_offset = 0;
        let mut qh_offset = 0;
        let mut scale_offset = 0;

        for n in (0..QK_K_BLOCK_SIZE).step_by(128) {
            for l in 0..32 {
                let is = l / 16;
                let qh = self.qh[qh_offset + l];
                let q1 = ((self.ql[ql_offset + l] & 0x0f) | ((qh & 0x03) << 4)) as i8 - 32;
                let q2 =
                    ((self.ql[ql_offset + l + 32] & 0x0f) | (((qh >> 2) & 0x03) << 4)) as i8 - 32;
                let q3 = ((self.ql[ql_offset + l] >> 4) | (((qh >> 4) & 0x03) << 4)) as i8 - 32;
                let q4 =
                    ((self.ql[ql_offset + l + 32] >> 4) | (((qh >> 6) & 0x03) << 4)) as i8 - 32;

                out[n + l] = d * self.scales[scale_offset + is] as f32 * q1 as f32;
                out[n + l + 32] = d * self.scales[scale_offset + is + 2] as f32 * q2 as f32;
                out[n + l + 64] = d * self.scales[scale_offset + is + 4] as f32 * q3 as f32;
                out[n + l + 96] = d * self.scales[scale_offset + is + 6] as f32 * q4 as f32;
            }

            ql_offset += 64;
            qh_offset += 32;
            scale_offset += 8;
        }
    }
}

pub fn decode_q8_0_blocks(bytes: &[u8]) -> Result<Vec<Q8_0Block>, Q8BlockError> {
    if !bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(Q8BlockError::MisalignedLength {
            bytes: bytes.len(),
            block_bytes: Q8_0_BLOCK_BYTES,
        });
    }

    Ok(bytes
        .chunks_exact(Q8_0_BLOCK_BYTES)
        .map(|chunk| {
            let bytes: &[u8; Q8_0_BLOCK_BYTES] = chunk
                .try_into()
                .expect("chunks_exact guarantees Q8_0 block length");
            Q8_0Block::from_bytes(bytes)
        })
        .collect())
}

pub fn decode_q4_0_blocks(bytes: &[u8]) -> Result<Vec<Q4_0Block>, Q8BlockError> {
    if !bytes.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(Q8BlockError::MisalignedLength {
            bytes: bytes.len(),
            block_bytes: Q4_0_BLOCK_BYTES,
        });
    }

    Ok(bytes
        .chunks_exact(Q4_0_BLOCK_BYTES)
        .map(|chunk| {
            let bytes: &[u8; Q4_0_BLOCK_BYTES] = chunk
                .try_into()
                .expect("chunks_exact guarantees Q4_0 block length");
            Q4_0Block::from_bytes(bytes)
        })
        .collect())
}

pub fn decode_q6_k_blocks(bytes: &[u8]) -> Result<Vec<Q6KBlock>, Q8BlockError> {
    if !bytes.len().is_multiple_of(Q6_K_BLOCK_BYTES) {
        return Err(Q8BlockError::MisalignedLength {
            bytes: bytes.len(),
            block_bytes: Q6_K_BLOCK_BYTES,
        });
    }

    Ok(bytes
        .chunks_exact(Q6_K_BLOCK_BYTES)
        .map(|chunk| {
            let bytes: &[u8; Q6_K_BLOCK_BYTES] = chunk
                .try_into()
                .expect("chunks_exact guarantees Q6_K block length");
            Q6KBlock::from_bytes(bytes)
        })
        .collect())
}

pub fn dot_q4_0_q8_0_scalar(weight: &Q4_0Block, activation: &[i8; Q8_BLOCK_SIZE]) -> i32 {
    let weight_values = weight.unpack_values();
    dot_i8_scalar(&weight_values, activation)
}

pub fn dot_q4_0_q8_0_with_selector(
    weight: &Q4_0Block,
    activation: &[i8; Q8_BLOCK_SIZE],
    kernel_selector: Q8DotKernelSelector,
) -> i32 {
    let weight_values = weight.unpack_values();
    match kernel_selector.selected {
        Q8DotKernel::Scalar => dot_i8_scalar(&weight_values, activation),
        Q8DotKernel::Neon => dot_i8_neon_32_selected(&weight_values, activation),
        Q8DotKernel::Sdot => dot_i8_sdot_32_selected(&weight_values, activation),
    }
}

impl Q8_0RowReader {
    pub fn from_tensor_descriptor(desc: &GgufTensorDescriptor) -> Result<Self, Q8BlockError> {
        if desc.tensor_type != GgufTensorType::Q8_0 {
            return Err(Q8BlockError::InvalidTensorType {
                name: desc.name.clone(),
            });
        }
        if desc.dimensions.len() != 2 || !desc.dimensions[0].is_multiple_of(Q8_BLOCK_SIZE as u64) {
            return Err(Q8BlockError::InvalidTensorShape {
                name: desc.name.clone(),
                dimensions: desc.dimensions.clone(),
            });
        }

        let columns = usize_from_u64(desc.dimensions[0], "Q8_0 columns")?;
        let rows = usize_from_u64(desc.dimensions[1], "Q8_0 rows")?;
        let blocks_per_row = columns / Q8_BLOCK_SIZE;
        let expected_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|blocks| blocks.checked_mul(Q8_0_BLOCK_BYTES))
            .ok_or(Q8BlockError::OffsetOverflow)?;
        if desc.n_bytes != expected_bytes as u64 {
            return Err(Q8BlockError::InvalidTensorShape {
                name: desc.name.clone(),
                dimensions: desc.dimensions.clone(),
            });
        }

        Ok(Self {
            tensor_name: desc.name.clone(),
            absolute_offset: desc.absolute_offset,
            rows,
            columns,
            blocks_per_row,
        })
    }

    pub fn read_row_blocks<R: Read + Seek>(
        &self,
        reader: &mut R,
        row: usize,
    ) -> Result<Vec<Q8_0Block>, Q8BlockError> {
        if row >= self.rows {
            return Err(Q8BlockError::RowOutOfBounds {
                row,
                rows: self.rows,
            });
        }

        let byte_len = self
            .blocks_per_row
            .checked_mul(Q8_0_BLOCK_BYTES)
            .ok_or(Q8BlockError::OffsetOverflow)?;
        let row_offset = row
            .checked_mul(byte_len)
            .ok_or(Q8BlockError::OffsetOverflow)?;
        let offset = self
            .absolute_offset
            .checked_add(row_offset as u64)
            .ok_or(Q8BlockError::OffsetOverflow)?;
        let mut bytes = vec![0; byte_len];
        reader.seek(SeekFrom::Start(offset))?;
        reader.read_exact(&mut bytes)?;
        decode_q8_0_blocks(&bytes)
    }

    pub fn read_row_values<R: Read + Seek>(
        &self,
        reader: &mut R,
        row: usize,
    ) -> Result<Vec<i8>, Q8BlockError> {
        let blocks = self.read_row_blocks(reader, row)?;
        let mut values = Vec::with_capacity(self.columns);
        for block in blocks {
            values.extend_from_slice(block.values());
        }
        Ok(values)
    }
}

pub fn dot_q8_0_rows_i32<R: Read + Seek>(
    reader: &mut R,
    lhs: &Q8_0RowReader,
    lhs_row: usize,
    rhs: &Q8_0RowReader,
    rhs_row: usize,
    kernel_selector: Q8DotKernelSelector,
) -> Result<Q8RowDotReport, Q8BlockError> {
    if lhs.columns != rhs.columns {
        return Err(Q8BlockError::ColumnMismatch {
            lhs: lhs.columns,
            rhs: rhs.columns,
        });
    }

    let lhs_values = lhs.read_row_values(reader, lhs_row)?;
    let rhs_values = rhs.read_row_values(reader, rhs_row)?;
    let scalar = dot_i8_scalar(&lhs_values, &rhs_values);
    let selected = dot_i8_with_selector(&lhs_values, &rhs_values, kernel_selector);

    Ok(Q8RowDotReport {
        scalar,
        selected,
        kernel_selector,
    })
}

pub fn dot_q8_0_blocks_scalar(lhs: &[Q8_0Block], rhs: &[Q8_0Block]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| left.scaled_dot_f32(right))
        .sum()
}

fn usize_from_u64(value: u64, name: &'static str) -> Result<usize, Q8BlockError> {
    usize::try_from(value).map_err(|_| Q8BlockError::ValueTooLarge(name))
}

pub fn bench_dot_runs(iterations: usize, runs: usize) -> DotBenchmarkReport {
    let lhs = deterministic_q8_values(BENCH_ELEMENTS, 17);
    let rhs = deterministic_q8_values(BENCH_ELEMENTS, 91);
    let kernel_selector = Q8DotKernelSelector::from_env();

    let scalar = time_dot_runs(iterations, runs, || {
        dot_i8_scalar(black_box(&lhs), black_box(&rhs))
    });
    let selected = time_dot_runs(iterations, runs, || {
        dot_i8_with_selector(black_box(&lhs), black_box(&rhs), kernel_selector)
    });
    let neon = neon_available().then(|| {
        time_dot_runs(iterations, runs, || {
            dot_i8_neon(black_box(&lhs), black_box(&rhs))
        })
    });
    let sdot = sdot_candidate_enabled().then(|| {
        time_dot_runs(iterations, runs, || {
            dot_i8_sdot(black_box(&lhs), black_box(&rhs))
        })
    });

    DotBenchmarkReport {
        iterations,
        runs,
        blocks_per_iteration: BENCH_BLOCKS,
        elements_per_iteration: BENCH_ELEMENTS,
        kernel_selector,
        selected,
        scalar,
        neon,
        sdot,
    }
}

impl Q8DotKernel {
    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Neon => "neon",
            Self::Sdot => "sdot",
        }
    }
}

impl Q8DotKernelSelector {
    pub fn from_env() -> Self {
        let requested = env::var(DOT_KERNEL_ENV)
            .ok()
            .as_deref()
            .and_then(parse_requested_kernel);

        Self::for_request(
            requested,
            RuntimeFeatures::detect(),
            sdot_candidate_requested(),
        )
    }

    fn for_request(
        requested: Option<Q8DotKernel>,
        features: RuntimeFeatures,
        sdot_candidate_enabled: bool,
    ) -> Self {
        match requested {
            None | Some(Q8DotKernel::Scalar) => Self {
                requested,
                selected: Q8DotKernel::Scalar,
                fallback_reason: None,
            },
            Some(Q8DotKernel::Neon) if features.neon => Self {
                requested,
                selected: Q8DotKernel::Neon,
                fallback_reason: None,
            },
            Some(Q8DotKernel::Neon) => Self {
                requested,
                selected: Q8DotKernel::Scalar,
                fallback_reason: Some("neon_unavailable"),
            },
            Some(Q8DotKernel::Sdot) if !sdot_candidate_enabled => Self {
                requested,
                selected: Q8DotKernel::Scalar,
                fallback_reason: Some("sdot_candidate_not_enabled"),
            },
            Some(Q8DotKernel::Sdot) if features.dotprod => Self {
                requested,
                selected: Q8DotKernel::Sdot,
                fallback_reason: None,
            },
            Some(Q8DotKernel::Sdot) => Self {
                requested,
                selected: Q8DotKernel::Scalar,
                fallback_reason: Some("dotprod_unavailable"),
            },
        }
    }
}

#[derive(Clone, Copy)]
struct RuntimeFeatures {
    neon: bool,
    dotprod: bool,
}

impl RuntimeFeatures {
    fn detect() -> Self {
        Self {
            neon: neon_available(),
            dotprod: dotprod_available(),
        }
    }
}

fn parse_requested_kernel(value: &str) -> Option<Q8DotKernel> {
    match value {
        "scalar" | "SCALAR" => Some(Q8DotKernel::Scalar),
        "neon" | "NEON" => Some(Q8DotKernel::Neon),
        "sdot" | "SDOT" => Some(Q8DotKernel::Sdot),
        _ => None,
    }
}

pub fn dot_i8_scalar(lhs: &[i8], rhs: &[i8]) -> i32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(&left, &right)| i32::from(left) * i32::from(right))
        .sum()
}

pub fn dot_i8_with_selector(lhs: &[i8], rhs: &[i8], selector: Q8DotKernelSelector) -> i32 {
    match selector.selected {
        Q8DotKernel::Scalar => dot_i8_scalar(lhs, rhs),
        Q8DotKernel::Neon => dot_i8_neon(lhs, rhs),
        Q8DotKernel::Sdot => dot_i8_sdot(lhs, rhs),
    }
}

pub fn dotprod_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub fn neon_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub fn sdot_candidate_requested() -> bool {
    env::var(SDOT_CANDIDATE_ENV)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

pub fn sdot_candidate_enabled() -> bool {
    sdot_candidate_requested() && dotprod_available()
}

pub fn dot_i8_neon(lhs: &[i8], rhs: &[i8]) -> i32 {
    assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return dot_i8_neon_selected(lhs, rhs);
        }
    }

    dot_i8_scalar(lhs, rhs)
}

pub fn dot_i8_sdot(lhs: &[i8], rhs: &[i8]) -> i32 {
    assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return dot_i8_sdot_selected(lhs, rhs);
        }
    }

    dot_i8_scalar(lhs, rhs)
}

pub(crate) fn dot_i8_neon_selected(lhs: &[i8], rhs: &[i8]) -> i32 {
    assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: callers use this only after selector/runtime feature validation.
        unsafe { dot_i8_neon_aarch64(lhs, rhs) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i8_scalar(lhs, rhs)
    }
}

pub(crate) fn dot_i8_sdot_selected(lhs: &[i8], rhs: &[i8]) -> i32 {
    assert_eq!(lhs.len(), rhs.len());

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: callers use this only after selector/runtime feature validation.
        unsafe { dot_i8_sdot_aarch64(lhs, rhs) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i8_scalar(lhs, rhs)
    }
}

pub(crate) fn dot_i8_neon_32_selected(lhs: &[i8; Q8_BLOCK_SIZE], rhs: &[i8; Q8_BLOCK_SIZE]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: callers use this only after selector/runtime feature validation.
        unsafe { dot_i8_neon_32_aarch64(lhs, rhs) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i8_scalar(lhs, rhs)
    }
}

pub(crate) fn dot_i8_sdot_32_selected(lhs: &[i8; Q8_BLOCK_SIZE], rhs: &[i8; Q8_BLOCK_SIZE]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: callers use this only after selector/runtime feature validation.
        unsafe { dot_i8_sdot_32_aarch64(lhs, rhs) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_i8_scalar(lhs, rhs)
    }
}

impl DotTimingSummary {
    pub fn total_elapsed(&self) -> Duration {
        self.elapsed_runs.iter().sum()
    }

    pub fn min_elapsed(&self) -> Duration {
        self.elapsed_runs.iter().copied().min().unwrap_or_default()
    }

    pub fn median_elapsed(&self) -> Duration {
        let mut runs = self.elapsed_runs.clone();
        runs.sort_unstable();
        runs.get(runs.len() / 2).copied().unwrap_or_default()
    }
}

fn time_dot_runs(iterations: usize, runs: usize, mut dot: impl FnMut() -> i32) -> DotTimingSummary {
    let mut checksum = None;
    let mut elapsed_runs = Vec::with_capacity(runs);

    for _ in 0..runs {
        let timed = time_dot(iterations, &mut dot);
        match checksum {
            Some(expected) => assert_eq!(timed.checksum, expected),
            None => checksum = Some(timed.checksum),
        }
        elapsed_runs.push(timed.elapsed);
    }

    DotTimingSummary {
        checksum: checksum.unwrap_or_default(),
        elapsed_runs,
    }
}

fn time_dot(iterations: usize, mut dot: impl FnMut() -> i32) -> TimedDot {
    let started = Instant::now();
    let mut checksum = 0_i64;
    for _ in 0..iterations {
        checksum = checksum.wrapping_add(i64::from(black_box(dot())));
    }
    TimedDot {
        checksum: black_box(checksum),
        elapsed: started.elapsed(),
    }
}

fn deterministic_q8_values(len: usize, salt: u32) -> Vec<i8> {
    (0..len)
        .map(|idx| {
            let value = ((idx as u32)
                .wrapping_mul(37)
                .wrapping_add(salt.wrapping_mul(19))
                % 127) as i16
                - 63;
            value as i8
        })
        .collect()
}

fn ns_per_block(elapsed: Duration, iterations: usize, blocks_per_iteration: usize) -> f64 {
    elapsed.as_secs_f64() * 1_000_000_000.0 / (iterations * blocks_per_iteration) as f64
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);

    match exponent {
        0 if fraction == 0 => f32::from_bits(sign),
        0 => {
            let mut mantissa = fraction;
            let mut exponent = -14_i32;
            while (mantissa & 0x0400) == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            mantissa &= 0x03ff;
            let f32_exponent = u32::try_from(exponent + 127).expect("subnormal exponent fits");
            f32::from_bits(sign | (f32_exponent << 23) | (mantissa << 13))
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (fraction << 13)),
        _ => {
            let f32_exponent = u32::from(exponent) + (127 - 15);
            f32::from_bits(sign | (f32_exponent << 23) | (fraction << 13))
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_i8_neon_aarch64(lhs: &[i8], rhs: &[i8]) -> i32 {
    use std::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vdupq_n_s32, vget_high_s8, vget_low_s8, vld1q_s8, vmull_s8,
        vpaddlq_s16,
    };

    let mut acc = unsafe { vdupq_n_s32(0) };
    let mut idx = 0;
    while idx + 16 <= lhs.len() {
        // SAFETY: the loop bound guarantees 16 readable i8 lanes from both slices.
        let left = unsafe { vld1q_s8(lhs.as_ptr().add(idx)) };
        let right = unsafe { vld1q_s8(rhs.as_ptr().add(idx)) };
        let low_products = unsafe { vmull_s8(vget_low_s8(left), vget_low_s8(right)) };
        let high_products = unsafe { vmull_s8(vget_high_s8(left), vget_high_s8(right)) };
        acc = unsafe { vaddq_s32(acc, vpaddlq_s16(low_products)) };
        acc = unsafe { vaddq_s32(acc, vpaddlq_s16(high_products)) };
        idx += 16;
    }

    let mut sum = unsafe { vaddvq_s32(acc) };
    while idx < lhs.len() {
        sum += i32::from(lhs[idx]) * i32::from(rhs[idx]);
        idx += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_i8_sdot_aarch64(lhs: &[i8], rhs: &[i8]) -> i32 {
    use std::arch::{
        aarch64::{vaddvq_s32, vdupq_n_s32, vld1q_s8},
        asm,
    };

    let mut acc = unsafe { vdupq_n_s32(0) };
    let mut idx = 0;
    while idx + 16 <= lhs.len() {
        // SAFETY: the loop bound guarantees 16 readable i8 lanes from both slices.
        let left = unsafe { vld1q_s8(lhs.as_ptr().add(idx)) };
        let right = unsafe { vld1q_s8(rhs.as_ptr().add(idx)) };
        unsafe {
            asm!(
                ".arch_extension dotprod",
                "sdot {acc:v}.4s, {left:v}.16b, {right:v}.16b",
                acc = inout(vreg) acc,
                left = in(vreg) left,
                right = in(vreg) right,
                options(nostack, preserves_flags),
            );
        }
        idx += 16;
    }

    let mut sum = unsafe { vaddvq_s32(acc) };
    while idx < lhs.len() {
        sum += i32::from(lhs[idx]) * i32::from(rhs[idx]);
        idx += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_i8_neon_32_aarch64(lhs: &[i8; Q8_BLOCK_SIZE], rhs: &[i8; Q8_BLOCK_SIZE]) -> i32 {
    use std::arch::aarch64::{
        vaddq_s32, vaddvq_s32, vget_high_s8, vget_low_s8, vld1q_s8, vmull_s8, vpaddlq_s16,
    };

    // SAFETY: Q8_0 blocks are exactly 32 i8 lanes, so two 16-byte vector loads cover the block.
    let left0 = unsafe { vld1q_s8(lhs.as_ptr()) };
    let right0 = unsafe { vld1q_s8(rhs.as_ptr()) };
    let left1 = unsafe { vld1q_s8(lhs.as_ptr().add(16)) };
    let right1 = unsafe { vld1q_s8(rhs.as_ptr().add(16)) };

    let low_products0 = unsafe { vmull_s8(vget_low_s8(left0), vget_low_s8(right0)) };
    let high_products0 = unsafe { vmull_s8(vget_high_s8(left0), vget_high_s8(right0)) };
    let low_products1 = unsafe { vmull_s8(vget_low_s8(left1), vget_low_s8(right1)) };
    let high_products1 = unsafe { vmull_s8(vget_high_s8(left1), vget_high_s8(right1)) };

    let acc0 = unsafe { vaddq_s32(vpaddlq_s16(low_products0), vpaddlq_s16(high_products0)) };
    let acc1 = unsafe { vaddq_s32(vpaddlq_s16(low_products1), vpaddlq_s16(high_products1)) };
    let acc = unsafe { vaddq_s32(acc0, acc1) };
    unsafe { vaddvq_s32(acc) }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_i8_sdot_32_aarch64(lhs: &[i8; Q8_BLOCK_SIZE], rhs: &[i8; Q8_BLOCK_SIZE]) -> i32 {
    use std::arch::{
        aarch64::{vaddvq_s32, vdupq_n_s32, vld1q_s8},
        asm,
    };

    let mut acc = unsafe { vdupq_n_s32(0) };
    // SAFETY: Q8_0 blocks are exactly 32 i8 lanes, so two 16-byte vector loads cover the block.
    let left0 = unsafe { vld1q_s8(lhs.as_ptr()) };
    let right0 = unsafe { vld1q_s8(rhs.as_ptr()) };
    let left1 = unsafe { vld1q_s8(lhs.as_ptr().add(16)) };
    let right1 = unsafe { vld1q_s8(rhs.as_ptr().add(16)) };

    unsafe {
        asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {left0:v}.16b, {right0:v}.16b",
            "sdot {acc:v}.4s, {left1:v}.16b, {right1:v}.16b",
            acc = inout(vreg) acc,
            left0 = in(vreg) left0,
            right0 = in(vreg) right0,
            left1 = in(vreg) left1,
            right1 = in(vreg) right1,
            options(nostack, preserves_flags),
        );
    }

    unsafe { vaddvq_s32(acc) }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::gguf::{GgufTensorDescriptor, GgufTensorType};

    use super::{
        Q4_0_BLOCK_BYTES, Q4_0Block, Q6_K_BLOCK_BYTES, Q6KBlock, Q8_0_BLOCK_BYTES, Q8_0Block,
        Q8_0RowReader, Q8_BLOCK_SIZE, Q8BlockError, Q8DotKernel, Q8DotKernelSelector,
        QK_K_BLOCK_SIZE, RuntimeFeatures, bench_dot_runs, decode_q8_0_blocks, dot_i8_neon,
        dot_i8_neon_32_selected, dot_i8_scalar, dot_i8_sdot, dot_i8_sdot_32_selected,
        dot_q4_0_q8_0_scalar, dot_q4_0_q8_0_with_selector, dot_q8_0_blocks_scalar,
        dot_q8_0_rows_i32, f16_bits_to_f32,
    };

    #[test]
    fn scalar_dot_handles_signed_q8_values() {
        let lhs = [-3, -2, -1, 0, 1, 2, 3];
        let rhs = [4, -5, 6, -7, 8, -9, 10];

        assert_eq!(dot_i8_scalar(&lhs, &rhs), 12);
    }

    #[test]
    fn neon_path_matches_scalar() {
        let lhs: Vec<i8> = (0..257).map(|idx| ((idx * 7) % 127) as i8 - 63).collect();
        let rhs: Vec<i8> = (0..257).map(|idx| ((idx * 11) % 127) as i8 - 63).collect();

        assert_eq!(dot_i8_neon(&lhs, &rhs), dot_i8_scalar(&lhs, &rhs));
    }

    #[test]
    fn sdot_path_matches_scalar() {
        let lhs: Vec<i8> = (0..259).map(|idx| ((idx * 5) % 127) as i8 - 63).collect();
        let rhs: Vec<i8> = (0..259).map(|idx| ((idx * 13) % 127) as i8 - 63).collect();

        assert_eq!(dot_i8_sdot(&lhs, &rhs), dot_i8_scalar(&lhs, &rhs));
    }

    #[test]
    fn fixed_q8_block_simd_paths_match_scalar() {
        let lhs = std::array::from_fn(|idx| ((idx * 7) % 127) as i8 - 63);
        let rhs = std::array::from_fn(|idx| ((idx * 11) % 127) as i8 - 63);
        let scalar = dot_i8_scalar(&lhs, &rhs);

        assert_eq!(dot_i8_neon_32_selected(&lhs, &rhs), scalar);
        assert_eq!(dot_i8_sdot_32_selected(&lhs, &rhs), scalar);
    }

    #[test]
    fn q8_kernel_selector_defaults_to_scalar() {
        let selector = Q8DotKernelSelector::for_request(
            None,
            RuntimeFeatures {
                neon: true,
                dotprod: true,
            },
            false,
        );

        assert_eq!(selector.selected, Q8DotKernel::Scalar);
        assert_eq!(selector.fallback_reason, None);
    }

    #[test]
    fn q8_kernel_selector_falls_back_when_neon_is_unavailable() {
        let selector = Q8DotKernelSelector::for_request(
            Some(Q8DotKernel::Neon),
            RuntimeFeatures {
                neon: false,
                dotprod: true,
            },
            false,
        );

        assert_eq!(selector.selected, Q8DotKernel::Scalar);
        assert_eq!(selector.fallback_reason, Some("neon_unavailable"));
    }

    #[test]
    fn q8_kernel_selector_keeps_sdot_default_off() {
        let selector = Q8DotKernelSelector::for_request(
            Some(Q8DotKernel::Sdot),
            RuntimeFeatures {
                neon: true,
                dotprod: true,
            },
            false,
        );

        assert_eq!(selector.selected, Q8DotKernel::Scalar);
        assert_eq!(selector.fallback_reason, Some("sdot_candidate_not_enabled"));
    }

    #[test]
    fn q8_dot_benchmark_preserves_checksum_parity() {
        let report = bench_dot_runs(2, 2);

        assert_eq!(report.selected.checksum, report.scalar.checksum);
        if let Some(neon) = report.neon {
            assert_eq!(neon.checksum, report.scalar.checksum);
        }
        if let Some(sdot) = report.sdot {
            assert_eq!(sdot.checksum, report.scalar.checksum);
        }
    }

    #[test]
    fn q8_0_block_decodes_gguf_layout() {
        let mut bytes = [0_u8; Q8_0_BLOCK_BYTES];
        bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        for (idx, byte) in bytes[2..].iter_mut().enumerate() {
            *byte = (idx as i8 - 16).to_le_bytes()[0];
        }

        let block = Q8_0Block::from_bytes(&bytes);

        assert_eq!(Q8_BLOCK_SIZE, 32);
        assert_eq!(block.scale_bits(), 0x3800);
        assert_eq!(block.scale_f32(), 0.5);
        assert_eq!(block.values()[0], -16);
        assert_eq!(block.values()[31], 15);
    }

    #[test]
    fn q8_0_block_decoder_rejects_partial_blocks() {
        assert_eq!(
            decode_q8_0_blocks(&[0; Q8_0_BLOCK_BYTES - 1]),
            Err(Q8BlockError::MisalignedLength {
                bytes: Q8_0_BLOCK_BYTES - 1,
                block_bytes: Q8_0_BLOCK_BYTES,
            })
        );
    }

    #[test]
    fn q4_0_block_decodes_low_then_high_nibbles() {
        let mut bytes = [0_u8; Q4_0_BLOCK_BYTES];
        bytes[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        for (idx, byte) in bytes[2..].iter_mut().enumerate() {
            *byte = idx as u8 | ((15 - idx as u8) << 4);
        }

        let block = Q4_0Block::from_bytes(&bytes);
        let values = block.unpack_values();

        assert_eq!(block.scale_bits(), 0x3c00);
        assert_eq!(block.scale_f32(), 1.0);
        assert_eq!(values[0], -8);
        assert_eq!(values[15], 7);
        assert_eq!(values[16], 7);
        assert_eq!(values[31], -8);
    }

    #[test]
    fn q4_0_q8_0_scalar_dot_matches_unpacked_reference() {
        let q4 = Q4_0Block::from_parts(
            0x3c00,
            [
                0x80, 0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7, 0x08, 0x19, 0x2a, 0x3b, 0x4c, 0x5d,
                0x6e, 0x7f,
            ],
        );
        let q8: [i8; Q8_BLOCK_SIZE] = core::array::from_fn(|idx| idx as i8 - 16);
        let unpacked = q4.unpack_values();
        let expected: i32 = unpacked
            .iter()
            .zip(q8.iter())
            .map(|(&left, &right)| i32::from(left) * i32::from(right))
            .sum();

        assert_eq!(dot_q4_0_q8_0_scalar(&q4, &q8), expected);
        for kernel in [Q8DotKernel::Neon, Q8DotKernel::Sdot] {
            assert_eq!(
                dot_q4_0_q8_0_with_selector(
                    &q4,
                    &q8,
                    Q8DotKernelSelector {
                        requested: Some(kernel),
                        selected: kernel,
                        fallback_reason: None,
                    },
                ),
                expected,
                "{kernel:?} Q4/Q8 dot diverged"
            );
        }
    }

    #[test]
    fn q6_k_block_dequantizes_quantized_values_and_scales() {
        let mut ql = [0_u8; 128];
        let mut qh = [0_u8; 64];
        let mut scales = [0_i8; 16];
        scales.fill(1);
        ql[0] = 0x10;
        ql[32] = 0x02;
        qh[0] = 0b11_10_01_00;

        let block = Q6KBlock::from_parts(ql, qh, scales, 0x3c00);
        let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
        block.dequantize(&mut values);

        assert_eq!(Q6_K_BLOCK_BYTES, 210);
        assert_eq!(block.scale_bits(), 0x3c00);
        assert_eq!(values[0], -32.0);
        assert_eq!(values[32], -14.0);
        assert_eq!(values[64], 1.0);
        assert_eq!(values[96], 16.0);
    }

    #[test]
    fn q8_0_block_dot_matches_scaled_scalar_reference() {
        let lhs = Q8_0Block::from_parts(0x4000, [2; Q8_BLOCK_SIZE]);
        let rhs = Q8_0Block::from_parts(0x3800, [-3; Q8_BLOCK_SIZE]);

        assert_eq!(lhs.dot_i32(&rhs), -192);
        assert_eq!(lhs.scaled_dot_f32(&rhs), -192.0);
        assert_eq!(dot_q8_0_blocks_scalar(&[lhs], &[rhs]), -192.0);
    }

    #[test]
    fn q8_0_row_reader_reads_descriptor_owned_tensor_bytes() {
        let desc = GgufTensorDescriptor {
            name: "blk.0.attn_q.weight".to_owned(),
            dimensions: vec![64, 2],
            tensor_type: GgufTensorType::Q8_0,
            relative_offset: 0,
            absolute_offset: 16,
            n_bytes: 4 * Q8_0_BLOCK_BYTES as u64,
        };
        let row_reader = Q8_0RowReader::from_tensor_descriptor(&desc).expect("Q8_0 descriptor");
        let mut bytes = vec![0_u8; desc.absolute_offset as usize];
        for block_idx in 0..4 {
            bytes.extend_from_slice(&0x3c00_u16.to_le_bytes());
            for value_idx in 0..Q8_BLOCK_SIZE {
                bytes.push((block_idx as i8 * 10 + value_idx as i8 - 16).to_le_bytes()[0]);
            }
        }

        let mut cursor = Cursor::new(bytes);
        let row = row_reader
            .read_row_values(&mut cursor, 1)
            .expect("second row");
        let rhs: Vec<i8> = (0..row.len())
            .map(|idx| ((idx * 3) % 31) as i8 - 15)
            .collect();
        let scalar = dot_i8_scalar(&row, &rhs);

        assert_eq!(row_reader.rows, 2);
        assert_eq!(row_reader.columns, 64);
        assert_eq!(row_reader.blocks_per_row, 2);
        assert_eq!(row[0], 4);
        assert_eq!(row[63], 45);
        assert_eq!(dot_i8_neon(&row, &rhs), scalar);
        assert_eq!(dot_i8_sdot(&row, &rhs), scalar);
    }

    #[test]
    fn q8_0_row_dot_reads_real_tensor_bytes_with_scalar_default_parity() {
        let lhs_desc = GgufTensorDescriptor {
            name: "lhs.weight".to_owned(),
            dimensions: vec![64, 2],
            tensor_type: GgufTensorType::Q8_0,
            relative_offset: 0,
            absolute_offset: 32,
            n_bytes: 4 * Q8_0_BLOCK_BYTES as u64,
        };
        let rhs_desc = GgufTensorDescriptor {
            name: "rhs.weight".to_owned(),
            dimensions: vec![64, 2],
            tensor_type: GgufTensorType::Q8_0,
            relative_offset: 4 * Q8_0_BLOCK_BYTES as u64,
            absolute_offset: 32 + 4 * Q8_0_BLOCK_BYTES as u64,
            n_bytes: 4 * Q8_0_BLOCK_BYTES as u64,
        };
        let lhs_reader = Q8_0RowReader::from_tensor_descriptor(&lhs_desc).expect("lhs Q8_0");
        let rhs_reader = Q8_0RowReader::from_tensor_descriptor(&rhs_desc).expect("rhs Q8_0");
        let mut bytes = vec![0_u8; lhs_desc.absolute_offset as usize];
        append_q8_blocks(&mut bytes, 4, 0x3c00, 3);
        append_q8_blocks(&mut bytes, 4, 0x3c00, 17);

        let selector = Q8DotKernelSelector::for_request(
            None,
            RuntimeFeatures {
                neon: true,
                dotprod: true,
            },
            true,
        );
        let mut cursor = Cursor::new(bytes.clone());
        let scalar_default =
            dot_q8_0_rows_i32(&mut cursor, &lhs_reader, 1, &rhs_reader, 0, selector)
                .expect("row dot");

        assert_eq!(scalar_default.kernel_selector.selected, Q8DotKernel::Scalar);
        assert_eq!(scalar_default.selected, scalar_default.scalar);

        for kernel in [Q8DotKernel::Neon, Q8DotKernel::Sdot] {
            let selector = Q8DotKernelSelector::for_request(
                Some(kernel),
                RuntimeFeatures {
                    neon: true,
                    dotprod: true,
                },
                true,
            );
            let mut cursor = Cursor::new(bytes.clone());
            let candidate =
                dot_q8_0_rows_i32(&mut cursor, &lhs_reader, 1, &rhs_reader, 0, selector)
                    .expect("candidate row dot");

            assert_eq!(candidate.selected, scalar_default.scalar, "{kernel:?}");
        }
    }

    #[test]
    fn f16_scale_decoder_covers_q8_scale_edges() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert_eq!(f16_bits_to_f32(0x8000), -0.0);
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xbc00), -1.0);
        assert_eq!(f16_bits_to_f32(0x0400), 0.000061035156);
        assert_eq!(f16_bits_to_f32(0x7c00), f32::INFINITY);
    }

    #[test]
    fn simd_candidates_match_scalar_across_q8_shapes() {
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257] {
            let lhs: Vec<i8> = (0..len)
                .map(|idx| ((idx * 17 + 3) % 127) as i8 - 63)
                .collect();
            let rhs: Vec<i8> = (0..len)
                .map(|idx| ((idx * 19 + 7) % 127) as i8 - 63)
                .collect();
            let scalar = dot_i8_scalar(&lhs, &rhs);

            assert_eq!(dot_i8_neon(&lhs, &rhs), scalar, "len={len}");
            assert_eq!(dot_i8_sdot(&lhs, &rhs), scalar, "len={len}");
        }
    }

    fn append_q8_blocks(bytes: &mut Vec<u8>, blocks: usize, scale_bits: u16, salt: i16) {
        for block_idx in 0..blocks {
            bytes.extend_from_slice(&scale_bits.to_le_bytes());
            for value_idx in 0..Q8_BLOCK_SIZE {
                let value = ((block_idx as i16 * 13 + value_idx as i16 * 7 + salt) % 127) - 63;
                bytes.push((value as i8).to_le_bytes()[0]);
            }
        }
    }
}
