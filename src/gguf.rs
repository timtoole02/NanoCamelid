use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Clone, Debug, PartialEq)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufMetadataValue {
    pub fn display(&self) -> String {
        match self {
            Self::U8(v) => v.to_string(),
            Self::I8(v) => v.to_string(),
            Self::U16(v) => v.to_string(),
            Self::I16(v) => v.to_string(),
            Self::U32(v) => v.to_string(),
            Self::I32(v) => v.to_string(),
            Self::F32(v) => v.to_string(),
            Self::Bool(v) => v.to_string(),
            Self::String(v) => v.clone(),
            Self::Array(v) => format!("[Array; {}]", v.len()),
            Self::U64(v) => v.to_string(),
            Self::I64(v) => v.to_string(),
            Self::F64(v) => v.to_string(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgufTensorType {
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
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ2S,
    IQ4NL,
    IQ3S,
    IQ1S,
    IQ1M,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    Bf16,
    Unknown(u32),
}

impl GgufTensorType {
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::Bf16,
            other => Self::Unknown(other),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ2S => "IQ2_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ3S => "IQ3_S",
            Self::IQ1S => "IQ1_S",
            Self::IQ1M => "IQ1_M",
            Self::IQ4XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::Bf16 => "BF16",
            Self::Unknown(_) => "UNKNOWN",
        }
    }

    pub fn layout(self) -> Option<(u64, u64)> {
        match self {
            Self::F32 => Some((1, 4)),
            Self::F16 => Some((1, 2)),
            Self::Q4_0 => Some((32, 18)),
            Self::Q4_1 => Some((32, 20)),
            Self::Q5_0 => Some((32, 22)),
            Self::Q5_1 => Some((32, 24)),
            Self::Q8_0 => Some((32, 34)),
            Self::Q8_1 => Some((32, 36)),
            Self::Q2K => Some((256, 84)),
            Self::Q3K => Some((256, 110)),
            Self::Q4K => Some((256, 144)),
            Self::Q5K => Some((256, 176)),
            Self::Q6K => Some((256, 210)),
            Self::Q8K => Some((256, 292)),
            Self::IQ2XXS => Some((256, 66)),
            Self::IQ2XS => Some((256, 74)),
            Self::IQ3XXS => Some((256, 98)),
            Self::IQ2S => Some((256, 82)),
            Self::IQ4NL => Some((32, 18)),
            Self::IQ3S => Some((256, 110)),
            Self::IQ1S => Some((256, 50)),
            Self::IQ1M => Some((256, 56)),
            Self::IQ4XS => Some((256, 136)),
            Self::I8 => Some((1, 1)),
            Self::I16 | Self::Bf16 => Some((1, 2)),
            Self::I32 => Some((1, 4)),
            Self::I64 | Self::F64 => Some((1, 8)),
            Self::Unknown(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufTensorDescriptor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgufTensorType,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub n_bytes: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    pub path: PathBuf,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub alignment: u64,
    pub data_start_offset: u64,
    pub metadata: BTreeMap<String, GgufMetadataValue>,
    pub tensors: Vec<GgufTensorDescriptor>,
}

impl GgufFile {
    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::String(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn metadata_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::Bool(v)) => Some(*v),
            Some(GgufMetadataValue::String(v)) => match v.as_str() {
                "true" | "1" => Some(true),
                "false" | "0" => Some(false),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn metadata_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::U32(v)) => Some(*v),
            Some(GgufMetadataValue::U64(v)) => (*v).try_into().ok(),
            Some(GgufMetadataValue::I32(v)) => (*v).try_into().ok(),
            _ => None,
        }
    }

    pub fn metadata_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::F32(v)) => Some(*v),
            Some(GgufMetadataValue::F64(v)) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn metadata_array_strings(&self, key: &str) -> Option<Vec<String>> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::Array(vals)) => vals
                .iter()
                .map(|val| match val {
                    GgufMetadataValue::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    pub fn metadata_array_f32(&self, key: &str) -> Option<Vec<f32>> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::Array(vals)) => vals
                .iter()
                .map(|val| match val {
                    GgufMetadataValue::F32(f) => Some(*f),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    pub fn metadata_array_i32(&self, key: &str) -> Option<Vec<i32>> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::Array(vals)) => vals
                .iter()
                .map(|val| match val {
                    GgufMetadataValue::I32(i) => Some(*i),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    InvalidUtf8,
    InvalidBool(u8),
    InvalidArrayType(u32),
    InvalidAlignment(u64),
    InvalidTensor(String),
    UnsupportedTensorType(u32),
    ValueTooLarge(&'static str),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::InvalidMagic(magic) => write!(f, "invalid GGUF magic: {magic:?}"),
            Self::UnsupportedVersion(version) => write!(f, "unsupported GGUF version: {version}"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 string"),
            Self::InvalidBool(value) => write!(f, "invalid GGUF bool value: {value}"),
            Self::InvalidArrayType(value_type) => {
                write!(f, "invalid GGUF array type: {value_type}")
            }
            Self::InvalidAlignment(alignment) => write!(f, "invalid GGUF alignment: {alignment}"),
            Self::InvalidTensor(message) => write!(f, "invalid GGUF tensor: {message}"),
            Self::UnsupportedTensorType(tensor_type) => {
                write!(f, "unsupported GGUF tensor type: {tensor_type}")
            }
            Self::ValueTooLarge(name) => write!(f, "{name} is too large for this platform"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<io::Error> for GgufError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

#[derive(Debug)]
pub struct GgufSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub alignment: u64,
    pub data_start_offset: u64,
    pub important_metadata: Vec<MetadataEntry>,
    pub tensor_types: Vec<TensorTypeSummary>,
    pub tensors: Vec<GgufTensorDescriptor>,
}

#[derive(Debug)]
pub struct MetadataEntry {
    pub key: String,
    pub value: String,
}

#[derive(Debug)]
pub struct TensorTypeSummary {
    pub name: String,
    pub count: usize,
}

const IMPORTANT_KEYS: &[&str] = &[
    "general.architecture",
    "general.name",
    "general.file_type",
    "general.quantization_version",
    "llama.context_length",
    "llama.embedding_length",
    "llama.block_count",
    "llama.attention.head_count",
    "llama.attention.head_count_kv",
    "llama.rope.freq_base",
    "tokenizer.ggml.model",
];

pub fn inspect(path: &Path) -> Result<GgufSummary, GgufError> {
    let file = read_file(path)?;
    Ok(summarize(&file))
}

pub fn summarize(file: &GgufFile) -> GgufSummary {
    let mut important_metadata = Vec::new();
    for key in IMPORTANT_KEYS {
        if let Some(val) = file.metadata.get(*key) {
            important_metadata.push(MetadataEntry {
                key: key.to_string(),
                value: val.display(),
            });
        }
    }

    let mut type_counts = BTreeMap::new();
    for tensor in &file.tensors {
        *type_counts
            .entry(tensor.tensor_type.name().to_owned())
            .or_insert(0) += 1;
    }

    GgufSummary {
        version: file.version,
        tensor_count: file.tensor_count,
        metadata_count: file.metadata_count,
        alignment: file.alignment,
        data_start_offset: file.data_start_offset,
        important_metadata,
        tensor_types: type_counts
            .into_iter()
            .map(|(name, count)| TensorTypeSummary { name, count })
            .collect(),
        tensors: file.tensors.clone(),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, path::PathBuf};

    use super::{GgufFile, GgufMetadataValue, GgufTensorType, tensor_nbytes};

    fn gguf_with_metadata(key: &str, value: GgufMetadataValue) -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(key.to_owned(), value);
        GgufFile {
            path: PathBuf::from("fixture.gguf"),
            version: 3,
            tensor_count: 0,
            metadata_count: metadata.len() as u64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: Vec::new(),
        }
    }

    #[test]
    fn metadata_bool_reads_native_bool_values() {
        let gguf = gguf_with_metadata("flag", GgufMetadataValue::Bool(true));
        assert_eq!(gguf.metadata_bool("flag"), Some(true));
    }

    #[test]
    fn metadata_bool_accepts_string_compat_values() {
        let gguf = gguf_with_metadata("flag", GgufMetadataValue::String("false".to_owned()));
        assert_eq!(gguf.metadata_bool("flag"), Some(false));
    }

    #[test]
    fn q5_0_and_q5_1_have_distinct_block_sizes() {
        assert_eq!(GgufTensorType::Q5_0.layout(), Some((32, 22)));
        assert_eq!(GgufTensorType::Q5_1.layout(), Some((32, 24)));
        assert_eq!(
            tensor_nbytes("q5_0", &[896, 151_936], GgufTensorType::Q5_0).unwrap(),
            93_592_576
        );
        assert_eq!(
            tensor_nbytes("q5_1", &[896, 151_936], GgufTensorType::Q5_1).unwrap(),
            102_100_992
        );
    }
}

pub fn read_file(path: &Path) -> Result<GgufFile, GgufError> {
    let file = File::open(path)?;
    let file_len = fs::metadata(path)?.len();
    let mut reader = Reader::new(file);

    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic(magic));
    }

    let version = reader.read_u32()?;
    if !(2..=3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = reader.read_u64()?;
    let metadata_count = reader.read_u64()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = reader.read_string()?;
        let value_type = ValueType::from_u32(reader.read_u32()?)?;
        let value = reader.read_value(value_type)?;
        metadata.insert(key, value);
    }

    let alignment = match metadata.get("general.alignment") {
        Some(GgufMetadataValue::U32(v)) => u64::from(*v),
        Some(GgufMetadataValue::U64(v)) => *v,
        Some(GgufMetadataValue::I32(v)) => *v as u64,
        _ => DEFAULT_ALIGNMENT,
    };

    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::InvalidAlignment(alignment));
    }

    let mut raw_tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        let dimensions = usize_from_u32(reader.read_u32()?, "tensor dimensions")?;
        if dimensions == 0 || dimensions > 4 {
            return Err(GgufError::InvalidTensor(format!(
                "tensor {name} has invalid dimension count {dimensions}"
            )));
        }
        let mut dims = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            dims.push(reader.read_u64()?);
        }
        let tensor_type = GgufTensorType::from_u32(reader.read_u32()?);
        let relative_offset = reader.read_u64()?;
        raw_tensors.push((name, dims, tensor_type, relative_offset));
    }

    let data_start_offset = align_to(reader.position(), alignment)?;
    if data_start_offset > file_len {
        return Err(GgufError::InvalidTensor(
            "aligned tensor data start overflows file".to_owned(),
        ));
    }

    let mut tensors = Vec::with_capacity(raw_tensors.len());
    let mut seen_names = BTreeSet::new();
    for (name, dimensions, tensor_type, relative_offset) in raw_tensors {
        if !seen_names.insert(name.clone()) {
            return Err(GgufError::InvalidTensor(format!(
                "duplicate tensor name {name}"
            )));
        }
        let n_bytes = tensor_nbytes(&name, &dimensions, tensor_type)?;
        let absolute_offset = data_start_offset
            .checked_add(relative_offset)
            .ok_or_else(|| {
                GgufError::InvalidTensor(format!("tensor {name} absolute offset overflow"))
            })?;
        tensors.push(GgufTensorDescriptor {
            name,
            dimensions,
            tensor_type,
            relative_offset,
            absolute_offset,
            n_bytes,
        });
    }

    Ok(GgufFile {
        path: path.to_path_buf(),
        version,
        tensor_count,
        metadata_count,
        alignment,
        data_start_offset,
        metadata,
        tensors,
    })
}

struct Reader<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Reader<R> {
    fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    fn position(&self) -> u64 {
        self.pos
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), GgufError> {
        self.inner.read_exact(buf)?;
        self.pos = self
            .pos
            .checked_add(buf.len() as u64)
            .ok_or_else(|| GgufError::InvalidTensor("reader position overflow".to_owned()))?;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        let mut bytes = [0; 1];
        self.read_exact(&mut bytes)?;
        Ok(bytes[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let mut bytes = [0; 2];
        self.read_exact(&mut bytes)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        let mut bytes = [0; 2];
        self.read_exact(&mut bytes)?;
        Ok(i16::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        let mut bytes = [0; 4];
        self.read_exact(&mut bytes)?;
        Ok(f32::from_le_bytes(bytes))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(i64::from_le_bytes(bytes))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let mut bytes = [0; 8];
        self.read_exact(&mut bytes)?;
        Ok(f64::from_le_bytes(bytes))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = usize_from_u64(self.read_u64()?, "string length")?;
        let mut bytes = vec![0; len];
        self.read_exact(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8)
    }

    fn read_value(&mut self, value_type: ValueType) -> Result<GgufMetadataValue, GgufError> {
        match value_type {
            ValueType::U8 => Ok(GgufMetadataValue::U8(self.read_u8()?)),
            ValueType::I8 => Ok(GgufMetadataValue::I8(self.read_i8()?)),
            ValueType::U16 => Ok(GgufMetadataValue::U16(self.read_u16()?)),
            ValueType::I16 => Ok(GgufMetadataValue::I16(self.read_i16()?)),
            ValueType::U32 => Ok(GgufMetadataValue::U32(self.read_u32()?)),
            ValueType::I32 => Ok(GgufMetadataValue::I32(self.read_i32()?)),
            ValueType::F32 => Ok(GgufMetadataValue::F32(self.read_f32()?)),
            ValueType::Bool => {
                let value = self.read_u8()?;
                match value {
                    0 => Ok(GgufMetadataValue::Bool(false)),
                    1 => Ok(GgufMetadataValue::Bool(true)),
                    _ => Err(GgufError::InvalidBool(value)),
                }
            }
            ValueType::String => Ok(GgufMetadataValue::String(self.read_string()?)),
            ValueType::Array => {
                let item_type = ValueType::from_array_type(self.read_u32()?)?;
                let len = usize_from_u64(self.read_u64()?, "array length")?;
                let mut vals = Vec::with_capacity(len);
                for _ in 0..len {
                    vals.push(self.read_value(item_type)?);
                }
                Ok(GgufMetadataValue::Array(vals))
            }
            ValueType::U64 => Ok(GgufMetadataValue::U64(self.read_u64()?)),
            ValueType::I64 => Ok(GgufMetadataValue::I64(self.read_i64()?)),
            ValueType::F64 => Ok(GgufMetadataValue::F64(self.read_f64()?)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_u32(value: u32) -> Result<Self, GgufError> {
        match value {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            other => Err(GgufError::InvalidArrayType(other)),
        }
    }

    fn from_array_type(value: u32) -> Result<Self, GgufError> {
        let value_type = Self::from_u32(value)?;
        match value_type {
            Self::Array => Err(GgufError::InvalidArrayType(value)),
            _ => Ok(value_type),
        }
    }
}

fn align_to(value: u64, alignment: u64) -> Result<u64, GgufError> {
    let add = alignment - 1;
    value
        .checked_add(add)
        .map(|value| value & !add)
        .ok_or_else(|| GgufError::InvalidTensor("alignment overflow".to_owned()))
}

fn usize_from_u32(value: u32, name: &'static str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::ValueTooLarge(name))
}

fn usize_from_u64(value: u64, name: &'static str) -> Result<usize, GgufError> {
    usize::try_from(value).map_err(|_| GgufError::ValueTooLarge(name))
}

fn tensor_nbytes(
    name: &str,
    dimensions: &[u64],
    tensor_type: GgufTensorType,
) -> Result<u64, GgufError> {
    let (block_size, type_size) = tensor_type.layout().ok_or_else(|| {
        GgufError::InvalidTensor(format!(
            "tensor {name} has unknown or removed GGML type {tensor_type:?}"
        ))
    })?;
    let first_dim = *dimensions.first().unwrap_or(&1);
    if !first_dim.is_multiple_of(block_size) {
        return Err(GgufError::InvalidTensor(format!(
            "tensor {name} first dimension {first_dim} is not divisible by block size {block_size}"
        )));
    }
    let mut elements = 1u64;
    for dim in dimensions {
        elements = elements.checked_mul(*dim).ok_or_else(|| {
            GgufError::InvalidTensor(format!("tensor {name} element count overflow"))
        })?;
    }
    elements
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size))
        .ok_or_else(|| GgufError::InvalidTensor(format!("tensor {name} byte size overflow")))
}
