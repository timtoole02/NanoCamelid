use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io::{self, Read},
    path::Path,
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

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
    I8,
    I16,
    I32,
    I64,
    F64,
    Bf16,
    Unknown(u32),
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
            Self::Io(err) => write!(f, "{err}"),
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

pub fn inspect(path: &Path) -> Result<GgufSummary, GgufError> {
    let file = File::open(path)?;
    parse(file)
}

fn parse(reader: impl Read) -> Result<GgufSummary, GgufError> {
    let mut reader = Reader::new(reader);
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
    let mut important_metadata = Vec::new();
    let mut alignment = DEFAULT_ALIGNMENT;

    for _ in 0..metadata_count {
        let key = reader.read_string()?;
        let value_type = ValueType::from_u32(reader.read_u32()?)?;
        let value = reader.read_value(value_type)?;
        let display_value = value.display();
        if key == "general.alignment" {
            alignment = display_value
                .parse::<u64>()
                .map_err(|_| GgufError::InvalidAlignment(0))?;
        }

        if IMPORTANT_KEYS.contains(&key.as_str()) {
            important_metadata.push(MetadataEntry {
                key,
                value: display_value,
            });
        }
    }

    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(GgufError::InvalidAlignment(alignment));
    }

    let mut raw_tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = reader.read_string()?;
        let dimensions = usize_from_u32(reader.read_u32()?, "tensor dimensions")?;
        if dimensions == 0 {
            return Err(GgufError::InvalidTensor(format!(
                "tensor {name} has no dimensions"
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
    let mut tensor_types = BTreeMap::new();
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
        *tensor_types
            .entry(tensor_type.name().to_owned())
            .or_insert(0) += 1;
        tensors.push(GgufTensorDescriptor {
            name,
            dimensions,
            tensor_type,
            relative_offset,
            absolute_offset,
            n_bytes,
        });
    }

    Ok(GgufSummary {
        version,
        tensor_count,
        metadata_count,
        alignment,
        data_start_offset,
        important_metadata,
        tensor_types: tensor_types
            .into_iter()
            .map(|(name, count)| TensorTypeSummary { name, count })
            .collect(),
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
        Ok(i8::from_le_bytes([self.read_u8()?]))
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

    fn read_value(&mut self, value_type: ValueType) -> Result<Value, GgufError> {
        match value_type {
            ValueType::U8 => Ok(Value::Scalar(self.read_u8()?.to_string())),
            ValueType::I8 => Ok(Value::Scalar(self.read_i8()?.to_string())),
            ValueType::U16 => Ok(Value::Scalar(self.read_u16()?.to_string())),
            ValueType::I16 => Ok(Value::Scalar(self.read_i16()?.to_string())),
            ValueType::U32 => Ok(Value::Scalar(self.read_u32()?.to_string())),
            ValueType::I32 => Ok(Value::Scalar(self.read_i32()?.to_string())),
            ValueType::F32 => Ok(Value::Scalar(format_float(self.read_f32()?))),
            ValueType::Bool => {
                let value = self.read_u8()?;
                match value {
                    0 => Ok(Value::Scalar("false".to_owned())),
                    1 => Ok(Value::Scalar("true".to_owned())),
                    _ => Err(GgufError::InvalidBool(value)),
                }
            }
            ValueType::String => Ok(Value::Scalar(self.read_string()?)),
            ValueType::Array => {
                let item_type = ValueType::from_array_type(self.read_u32()?)?;
                let len = usize_from_u64(self.read_u64()?, "array length")?;
                for _ in 0..len {
                    let _ = self.read_value(item_type)?;
                }
                Ok(Value::Array { item_type, len })
            }
            ValueType::U64 => Ok(Value::Scalar(self.read_u64()?.to_string())),
            ValueType::I64 => Ok(Value::Scalar(self.read_i64()?.to_string())),
            ValueType::F64 => Ok(Value::Scalar(format_float(self.read_f64()?))),
        }
    }
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
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
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
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::Bf16 => "BF16",
            Self::Unknown(_) => "UNKNOWN",
        }
    }

    fn layout(self) -> Option<(u64, u64)> {
        match self {
            Self::F32 => Some((1, 4)),
            Self::F16 => Some((1, 2)),
            Self::Q4_0 | Self::Q4_1 => Some((32, 18)),
            Self::Q5_0 | Self::Q5_1 => Some((32, 22)),
            Self::Q8_0 => Some((32, 34)),
            Self::Q8_1 => Some((32, 36)),
            Self::Q2K => Some((256, 84)),
            Self::Q3K => Some((256, 110)),
            Self::Q4K => Some((256, 144)),
            Self::Q5K => Some((256, 176)),
            Self::Q6K => Some((256, 210)),
            Self::Q8K => Some((256, 292)),
            Self::I8 => Some((1, 1)),
            Self::I16 | Self::Bf16 => Some((1, 2)),
            Self::I32 => Some((1, 4)),
            Self::I64 | Self::F64 => Some((1, 8)),
            Self::Unknown(_) => None,
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

enum Value {
    Scalar(String),
    Array { item_type: ValueType, len: usize },
}

impl Value {
    fn display(self) -> String {
        match self {
            Self::Scalar(value) => value,
            Self::Array { item_type, len } => format!("[{item_type:?}; {len}]"),
        }
    }
}

fn format_float(value: impl fmt::Display) -> String {
    value.to_string()
}

fn tensor_nbytes(
    name: &str,
    dimensions: &[u64],
    tensor_type: GgufTensorType,
) -> Result<u64, GgufError> {
    let (block_size, type_size) = tensor_type.layout().ok_or_else(|| match tensor_type {
        GgufTensorType::Unknown(value) => GgufError::UnsupportedTensorType(value),
        _ => GgufError::InvalidTensor(format!("tensor {name} has unsupported layout")),
    })?;
    let first_dim = dimensions.first().copied().unwrap_or(1);
    if !first_dim.is_multiple_of(block_size) {
        return Err(GgufError::InvalidTensor(format!(
            "tensor {name} first dimension {first_dim} is not divisible by block size {block_size}"
        )));
    }
    let elements = dimensions.iter().try_fold(1_u64, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            GgufError::InvalidTensor(format!("tensor {name} element count overflow"))
        })
    })?;
    elements
        .checked_div(block_size)
        .and_then(|blocks| blocks.checked_mul(type_size))
        .ok_or_else(|| GgufError::InvalidTensor(format!("tensor {name} byte size overflow")))
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{GgufTensorType, parse};

    #[test]
    fn parses_minimal_gguf_summary() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());

        write_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        write_string(&mut bytes, "llama");

        write_string(&mut bytes, "llama.context_length");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&2048_u32.to_le_bytes());

        write_string(&mut bytes, "blk.0.attn_q.weight");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u64.to_le_bytes());
        bytes.extend_from_slice(&4096_u64.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());

        let summary = parse(Cursor::new(bytes)).expect("valid minimal gguf");
        assert_eq!(summary.version, 3);
        assert_eq!(summary.tensor_count, 1);
        assert_eq!(summary.metadata_count, 2);
        assert_eq!(summary.alignment, 32);
        assert_eq!(summary.data_start_offset, 192);
        assert_eq!(summary.important_metadata[0].key, "general.architecture");
        assert_eq!(summary.important_metadata[0].value, "llama");
        assert_eq!(summary.important_metadata[1].key, "llama.context_length");
        assert_eq!(summary.important_metadata[1].value, "2048");
        assert_eq!(summary.tensor_types[0].name, "Q8_0");
        assert_eq!(summary.tensor_types[0].count, 1);
        assert_eq!(summary.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(summary.tensors[0].dimensions, [32, 4096]);
        assert_eq!(summary.tensors[0].tensor_type, GgufTensorType::Q8_0);
        assert_eq!(summary.tensors[0].relative_offset, 0);
        assert_eq!(summary.tensors[0].absolute_offset, 192);
        assert_eq!(summary.tensors[0].n_bytes, 139_264);
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
