use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    io::{self, Read},
    path::Path,
};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

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
    pub important_metadata: Vec<MetadataEntry>,
    pub tensor_types: Vec<TensorTypeSummary>,
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

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    InvalidUtf8,
    InvalidBool(u8),
    InvalidArrayType(u32),
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

    for _ in 0..metadata_count {
        let key = reader.read_string()?;
        let value_type = ValueType::from_u32(reader.read_u32()?)?;
        let value = reader.read_value(value_type)?;

        if IMPORTANT_KEYS.contains(&key.as_str()) {
            important_metadata.push(MetadataEntry {
                key,
                value: value.display(),
            });
        }
    }

    let mut tensor_types = BTreeMap::new();
    for _ in 0..tensor_count {
        let _name = reader.read_string()?;
        let dimensions = usize_from_u32(reader.read_u32()?, "tensor dimensions")?;
        for _ in 0..dimensions {
            let _ = reader.read_u64()?;
        }
        let tensor_type = reader.read_u32()?;
        let _offset = reader.read_u64()?;
        *tensor_types
            .entry(tensor_type_name(tensor_type).to_owned())
            .or_insert(0) += 1;
    }

    Ok(GgufSummary {
        version,
        tensor_count,
        metadata_count,
        important_metadata,
        tensor_types: tensor_types
            .into_iter()
            .map(|(name, count)| TensorTypeSummary { name, count })
            .collect(),
    })
}

struct Reader<R> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn new(inner: R) -> Self {
        Self { inner }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), GgufError> {
        self.inner.read_exact(buf)?;
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

fn tensor_type_name(value: u32) -> &'static str {
    match value {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        31 => "Q4_0_4_4",
        32 => "Q4_0_4_8",
        33 => "Q4_0_8_8",
        _ => "UNKNOWN",
    }
}

fn format_float(value: impl fmt::Display) -> String {
    value.to_string()
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

    use super::parse;

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
        assert_eq!(summary.important_metadata[0].key, "general.architecture");
        assert_eq!(summary.important_metadata[0].value, "llama");
        assert_eq!(summary.important_metadata[1].key, "llama.context_length");
        assert_eq!(summary.important_metadata[1].value, "2048");
        assert_eq!(summary.tensor_types[0].name, "Q8_0");
        assert_eq!(summary.tensor_types[0].count, 1);
    }

    fn write_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
}
