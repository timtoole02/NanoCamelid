use std::io::{self, Read, Write};

pub const ACTIVATION_MAGIC: u32 = 0xDEAD_BEEF;
// Speculative verification: same layout as an activation packet, but the
// final worker replies with one prediction per batch position
// (BATCH_FEEDBACK) instead of a single token. Middles relay both verbatim.
pub const ACTIVATION_VERIFY_MAGIC: u32 = 0xDEAD_BEE5;
pub const TOKEN_FEEDBACK_MAGIC: u32 = 0xCAFE_FEED;
pub const BATCH_FEEDBACK_MAGIC: u32 = 0xCAFE_FEE5;
pub const MAX_BATCH_FEEDBACK_TOKENS: usize = 256;
pub const CLUSTER_HELLO_MAGIC: u32 = 0x4E43_4C48; // "NCLH"
pub const CLUSTER_PROTOCOL_VERSION: u32 = 1;
pub const ACTIVATION_HEADER_BYTES: usize = 16;
pub const TOKEN_FEEDBACK_BYTES: usize = 12;
pub const CLUSTER_HELLO_BYTES: usize = 48;
pub const DEFAULT_MAX_ACTIVATION_FLOATS: usize = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActivationHeader {
    pub pos: u32,
    pub seq_len: u32,
    pub float_count: u32,
    /// True when the packet arrived with ACTIVATION_VERIFY_MAGIC: the final
    /// worker must reply with per-position predictions (BATCH_FEEDBACK).
    pub verify: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenFeedback {
    pub token_id: u32,
    pub is_finished: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusterNodeRole {
    Master = 1,
    Middle = 2,
    FinalWorker = 3,
}

impl ClusterNodeRole {
    fn from_wire(value: u32) -> io::Result<Self> {
        match value {
            1 => Ok(Self::Master),
            2 => Ok(Self::Middle),
            3 => Ok(Self::FinalWorker),
            other => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid cluster node role: {other}"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClusterHello {
    pub role: ClusterNodeRole,
    pub block_count: u32,
    pub embedding_length: u32,
    pub context_length: u32,
    pub kv_width: u32,
    pub vocab_size: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
}

pub fn send_cluster_hello<W: Write>(writer: &mut W, hello: ClusterHello) -> io::Result<()> {
    let mut bytes = [0_u8; CLUSTER_HELLO_BYTES];
    let fields = [
        CLUSTER_HELLO_MAGIC,
        CLUSTER_PROTOCOL_VERSION,
        hello.role as u32,
        hello.block_count,
        hello.embedding_length,
        hello.context_length,
        hello.kv_width,
        hello.vocab_size,
        hello.layer_start,
        hello.layer_end,
        hello.expert_count,
        hello.expert_used_count,
    ];
    for (idx, field) in fields.iter().enumerate() {
        let start = idx * 4;
        bytes[start..start + 4].copy_from_slice(&field.to_le_bytes());
    }
    writer.write_all(&bytes)?;
    writer.flush()
}

pub fn recv_cluster_hello<R: Read>(reader: &mut R) -> io::Result<ClusterHello> {
    let mut bytes = [0_u8; CLUSTER_HELLO_BYTES];
    reader.read_exact(&mut bytes)?;

    let read_u32 = |idx: usize| -> u32 {
        let start = idx * 4;
        u32::from_le_bytes(bytes[start..start + 4].try_into().unwrap())
    };

    let magic = read_u32(0);
    if magic != CLUSTER_HELLO_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid cluster hello magic: 0x{magic:08X}"),
        ));
    }

    let version = read_u32(1);
    if version != CLUSTER_PROTOCOL_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported cluster protocol version: {version}"),
        ));
    }

    Ok(ClusterHello {
        role: ClusterNodeRole::from_wire(read_u32(2))?,
        block_count: read_u32(3),
        embedding_length: read_u32(4),
        context_length: read_u32(5),
        kv_width: read_u32(6),
        vocab_size: read_u32(7),
        layer_start: read_u32(8),
        layer_end: read_u32(9),
        expert_count: read_u32(10),
        expert_used_count: read_u32(11),
    })
}

pub fn send_activation_packet<W: Write>(
    writer: &mut W,
    pos: u32,
    seq_len: u32,
    activations: &[f32],
) -> io::Result<()> {
    send_activation_packet_with_magic(writer, ACTIVATION_MAGIC, pos, seq_len, activations)
}

/// A verify packet carries the same payload as an activation packet; the
/// magic tells downstream nodes to answer with per-position predictions.
pub fn send_verify_packet<W: Write>(
    writer: &mut W,
    pos: u32,
    seq_len: u32,
    activations: &[f32],
) -> io::Result<()> {
    send_activation_packet_with_magic(writer, ACTIVATION_VERIFY_MAGIC, pos, seq_len, activations)
}

fn send_activation_packet_with_magic<W: Write>(
    writer: &mut W,
    magic: u32,
    pos: u32,
    seq_len: u32,
    activations: &[f32],
) -> io::Result<()> {
    let float_count = u32::try_from(activations.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "activation vector has too many floats: {}",
                activations.len()
            ),
        )
    })?;

    // Single-write framing: header and payload leave in one write_all so a
    // packet never straddles a flush boundary as two small writes.
    let mut message =
        Vec::with_capacity(ACTIVATION_HEADER_BYTES + std::mem::size_of_val(activations));
    message.extend_from_slice(&magic.to_le_bytes());
    message.extend_from_slice(&pos.to_le_bytes());
    message.extend_from_slice(&seq_len.to_le_bytes());
    message.extend_from_slice(&float_count.to_le_bytes());
    for value in activations {
        message.extend_from_slice(&value.to_le_bytes());
    }
    writer.write_all(&message)?;
    writer.flush()
}

pub fn recv_activation_packet<R: Read>(
    reader: &mut R,
    out_activations: &mut Vec<f32>,
) -> io::Result<ActivationHeader> {
    recv_activation_packet_limited(reader, out_activations, DEFAULT_MAX_ACTIVATION_FLOATS)
}

pub fn recv_activation_packet_limited<R: Read>(
    reader: &mut R,
    out_activations: &mut Vec<f32>,
    max_float_count: usize,
) -> io::Result<ActivationHeader> {
    let mut header = [0_u8; ACTIVATION_HEADER_BYTES];
    reader.read_exact(&mut header)?;

    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    let verify = match magic {
        ACTIVATION_MAGIC => false,
        ACTIVATION_VERIFY_MAGIC => true,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid activation header magic: 0x{magic:08X}"),
            ));
        }
    };

    let pos = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let seq_len = u32::from_le_bytes(header[8..12].try_into().unwrap());
    let float_count = u32::from_le_bytes(header[12..16].try_into().unwrap());
    let float_count_usize = float_count as usize;
    if float_count_usize > max_float_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("activation float_count {float_count_usize} exceeds limit {max_float_count}"),
        ));
    }

    let byte_count = float_count_usize
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "activation size overflow"))?;
    let mut payload = vec![0_u8; byte_count];
    reader.read_exact(&mut payload)?;

    out_activations.clear();
    out_activations.reserve(float_count_usize);
    for chunk in payload.chunks_exact(4) {
        out_activations.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }

    Ok(ActivationHeader {
        pos,
        seq_len,
        float_count,
        verify,
    })
}

/// Per-position predictions for a verify batch: the target's greedy argmax
/// after each of the batch's positions, in order.
pub fn send_batch_feedback<W: Write>(writer: &mut W, tokens: &[u32]) -> io::Result<()> {
    let count = u32::try_from(tokens.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "batch feedback too large"))?;
    let mut message = Vec::with_capacity(8 + tokens.len() * 4);
    message.extend_from_slice(&BATCH_FEEDBACK_MAGIC.to_le_bytes());
    message.extend_from_slice(&count.to_le_bytes());
    for token in tokens {
        message.extend_from_slice(&token.to_le_bytes());
    }
    writer.write_all(&message)?;
    writer.flush()
}

pub fn recv_batch_feedback<R: Read>(reader: &mut R) -> io::Result<Vec<u32>> {
    let mut header = [0_u8; 8];
    reader.read_exact(&mut header)?;
    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    if magic != BATCH_FEEDBACK_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid batch feedback magic: 0x{magic:08X}"),
        ));
    }
    let count = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    if count == 0 || count > MAX_BATCH_FEEDBACK_TOKENS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("batch feedback count {count} outside 1..={MAX_BATCH_FEEDBACK_TOKENS}"),
        ));
    }
    let mut payload = vec![0_u8; count * 4];
    reader.read_exact(&mut payload)?;
    Ok(payload
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

pub fn send_token_feedback<W: Write>(
    writer: &mut W,
    token_id: u32,
    is_finished: bool,
) -> io::Result<()> {
    let mut bytes = [0_u8; TOKEN_FEEDBACK_BYTES];
    bytes[0..4].copy_from_slice(&TOKEN_FEEDBACK_MAGIC.to_le_bytes());
    bytes[4..8].copy_from_slice(&token_id.to_le_bytes());
    bytes[8..12].copy_from_slice(&(is_finished as u32).to_le_bytes());
    writer.write_all(&bytes)?;
    writer.flush()
}

pub fn recv_token_feedback<R: Read>(reader: &mut R) -> io::Result<TokenFeedback> {
    let mut bytes = [0_u8; TOKEN_FEEDBACK_BYTES];
    reader.read_exact(&mut bytes)?;

    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != TOKEN_FEEDBACK_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid token feedback magic: 0x{magic:08X}"),
        ));
    }

    let token_id = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let is_finished = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    match is_finished {
        0 | 1 => Ok(TokenFeedback {
            token_id,
            is_finished: is_finished == 1,
        }),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid token feedback is_finished value: {other}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{
        ACTIVATION_MAGIC, CLUSTER_HELLO_MAGIC, CLUSTER_PROTOCOL_VERSION, ClusterHello,
        ClusterNodeRole, TOKEN_FEEDBACK_MAGIC, recv_activation_packet,
        recv_activation_packet_limited, recv_cluster_hello, recv_token_feedback,
        send_activation_packet, send_cluster_hello, send_token_feedback,
    };

    #[test]
    fn cluster_hello_round_trips() {
        let hello = ClusterHello {
            role: ClusterNodeRole::Middle,
            block_count: 32,
            embedding_length: 4096,
            context_length: 512,
            kv_width: 1024,
            vocab_size: 32000,
            layer_start: 11,
            layer_end: 22,
            expert_count: 8,
            expert_used_count: 2,
        };

        let mut bytes = Vec::new();
        send_cluster_hello(&mut bytes, hello).unwrap();

        let decoded = recv_cluster_hello(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(decoded, hello);
    }

    #[test]
    fn cluster_hello_rejects_bad_magic_version_and_role() {
        let mut bad_magic = Vec::new();
        bad_magic.extend_from_slice(&0_u32.to_le_bytes());
        bad_magic.extend_from_slice(&CLUSTER_PROTOCOL_VERSION.to_le_bytes());
        bad_magic.extend_from_slice(&(ClusterNodeRole::Master as u32).to_le_bytes());
        bad_magic.resize(super::CLUSTER_HELLO_BYTES, 0);
        let err = recv_cluster_hello(&mut Cursor::new(bad_magic)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("invalid cluster hello magic"));

        let mut bad_version = Vec::new();
        bad_version.extend_from_slice(&CLUSTER_HELLO_MAGIC.to_le_bytes());
        bad_version.extend_from_slice(&99_u32.to_le_bytes());
        bad_version.extend_from_slice(&(ClusterNodeRole::Master as u32).to_le_bytes());
        bad_version.resize(super::CLUSTER_HELLO_BYTES, 0);
        let err = recv_cluster_hello(&mut Cursor::new(bad_version)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(
            err.to_string()
                .contains("unsupported cluster protocol version")
        );

        let mut bad_role = Vec::new();
        bad_role.extend_from_slice(&CLUSTER_HELLO_MAGIC.to_le_bytes());
        bad_role.extend_from_slice(&CLUSTER_PROTOCOL_VERSION.to_le_bytes());
        bad_role.extend_from_slice(&99_u32.to_le_bytes());
        bad_role.resize(super::CLUSTER_HELLO_BYTES, 0);
        let err = recv_cluster_hello(&mut Cursor::new(bad_role)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("invalid cluster node role"));
    }

    #[test]
    fn activation_packet_round_trips_little_endian_payload() {
        let activations = [0.0_f32, 1.25, -2.5, f32::INFINITY];
        let mut bytes = Vec::new();
        send_activation_packet(&mut bytes, 17, 23, &activations).unwrap();

        let mut decoded = Vec::new();
        let header = recv_activation_packet(&mut Cursor::new(bytes), &mut decoded).unwrap();

        assert_eq!(header.pos, 17);
        assert_eq!(header.seq_len, 23);
        assert_eq!(header.float_count, activations.len() as u32);
        assert_eq!(decoded, activations);
    }

    #[test]
    fn activation_packet_rejects_bad_magic() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());

        let err = recv_activation_packet(&mut Cursor::new(bytes), &mut Vec::new()).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("invalid activation header magic"));
    }

    #[test]
    fn activation_packet_rejects_oversized_payload() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&ACTIVATION_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&4_u32.to_le_bytes());

        let err = recv_activation_packet_limited(&mut Cursor::new(bytes), &mut Vec::new(), 3)
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("exceeds limit"));
    }

    #[test]
    fn verify_packet_round_trips_with_verify_flag() {
        let activations = [1.0_f32, -2.0];
        let mut bytes = Vec::new();
        super::send_verify_packet(&mut bytes, 7, 2, &activations).unwrap();
        let mut decoded = Vec::new();
        let header = recv_activation_packet(&mut Cursor::new(bytes), &mut decoded).unwrap();
        assert!(header.verify);
        assert_eq!(header.pos, 7);
        assert_eq!(header.seq_len, 2);
        assert_eq!(decoded, activations);

        let mut plain = Vec::new();
        send_activation_packet(&mut plain, 7, 2, &activations).unwrap();
        let header = recv_activation_packet(&mut Cursor::new(plain), &mut decoded).unwrap();
        assert!(!header.verify);
    }

    #[test]
    fn batch_feedback_round_trips_and_rejects_bad_input() {
        let tokens = [5_u32, 0, 31999];
        let mut bytes = Vec::new();
        super::send_batch_feedback(&mut bytes, &tokens).unwrap();
        let decoded = super::recv_batch_feedback(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(decoded, tokens);

        let mut zero = Vec::new();
        zero.extend_from_slice(&super::BATCH_FEEDBACK_MAGIC.to_le_bytes());
        zero.extend_from_slice(&0_u32.to_le_bytes());
        assert!(super::recv_batch_feedback(&mut Cursor::new(zero)).is_err());

        let mut bad_magic = Vec::new();
        bad_magic.extend_from_slice(&0_u32.to_le_bytes());
        bad_magic.extend_from_slice(&1_u32.to_le_bytes());
        assert!(super::recv_batch_feedback(&mut Cursor::new(bad_magic)).is_err());
    }

    #[test]
    fn token_feedback_round_trips() {
        let mut bytes = Vec::new();
        send_token_feedback(&mut bytes, 32001, true).unwrap();

        let feedback = recv_token_feedback(&mut Cursor::new(bytes)).unwrap();
        assert_eq!(feedback.token_id, 32001);
        assert!(feedback.is_finished);
    }

    #[test]
    fn token_feedback_rejects_bad_magic_and_invalid_bool() {
        let mut bad_magic = Vec::new();
        bad_magic.extend_from_slice(&0_u32.to_le_bytes());
        bad_magic.extend_from_slice(&5_u32.to_le_bytes());
        bad_magic.extend_from_slice(&0_u32.to_le_bytes());
        let err = recv_token_feedback(&mut Cursor::new(bad_magic)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);

        let mut invalid_bool = Vec::new();
        invalid_bool.extend_from_slice(&TOKEN_FEEDBACK_MAGIC.to_le_bytes());
        invalid_bool.extend_from_slice(&5_u32.to_le_bytes());
        invalid_bool.extend_from_slice(&2_u32.to_le_bytes());
        let err = recv_token_feedback(&mut Cursor::new(invalid_bool)).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
