//! Wire protocol for the pipeline transport.
//!
//! Every frame is `[type: u8][payload_len: u32 LE][payload]`. Floats are serialized as raw
//! little-endian bytes (`f32::to_le_bytes`), matching the GGUF on-disk convention, so the
//! hidden state that crosses the wire is bit-for-bit what the next stage would have received
//! in a single-node run. This is what keeps the distributed path bit-identical.

use std::io;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const MSG_FORWARD: u8 = 1;
pub const MSG_RESET: u8 = 2;
pub const MSG_HIDDEN: u8 = 3;
pub const MSG_LOGITS: u8 = 4;
pub const MSG_ERROR: u8 = 5;
pub const MSG_HELLO: u8 = 6;
pub const MSG_HELLO_ACK: u8 = 7;

#[derive(Debug, Clone)]
pub enum Message {
    /// Head -> stage: run this stage's layers at `pos` on `hidden`.
    Forward { request_id: u64, pos: u32, hidden: Vec<f32> },
    /// Head -> stage: drop any cached session state for `request_id` (new sequence).
    Reset { request_id: u64 },
    /// Stage -> head: intermediate hidden state (non-tail stage).
    Hidden { request_id: u64, hidden: Vec<f32> },
    /// Stage -> head: final logits (tail stage).
    Logits { request_id: u64, logits: Vec<f32> },
    /// Stage -> head: error description.
    Error { request_id: u64, message: String },
    /// Head -> stage handshake; carries the head's view of the model dims for a sanity check.
    Hello { block_count: u32, embedding_length: u32 },
    /// Stage -> head handshake reply; carries the stage's own dims.
    HelloAck { block_count: u32, embedding_length: u32 },
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}
fn put_f32_vec(buf: &mut Vec<u8>, data: &[f32]) {
    put_u32(buf, data.len() as u32);
    for &x in data {
        buf.extend_from_slice(&x.to_le_bytes());
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.pos + n > self.buf.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "frame payload truncated"));
        }
        let slice = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32_vec(&mut self) -> io::Result<Vec<f32>> {
        let n = self.u32()? as usize;
        let bytes = self.take(n * 4)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
    fn rest(&mut self) -> &'a [u8] {
        let slice = &self.buf[self.pos..];
        self.pos = self.buf.len();
        slice
    }
}

impl Message {
    pub fn type_byte(&self) -> u8 {
        match self {
            Message::Forward { .. } => MSG_FORWARD,
            Message::Reset { .. } => MSG_RESET,
            Message::Hidden { .. } => MSG_HIDDEN,
            Message::Logits { .. } => MSG_LOGITS,
            Message::Error { .. } => MSG_ERROR,
            Message::Hello { .. } => MSG_HELLO,
            Message::HelloAck { .. } => MSG_HELLO_ACK,
        }
    }

    fn encode_payload(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            Message::Forward { request_id, pos, hidden } => {
                put_u64(&mut buf, *request_id);
                put_u32(&mut buf, *pos);
                put_f32_vec(&mut buf, hidden);
            }
            Message::Reset { request_id } => put_u64(&mut buf, *request_id),
            Message::Hidden { request_id, hidden } => {
                put_u64(&mut buf, *request_id);
                put_f32_vec(&mut buf, hidden);
            }
            Message::Logits { request_id, logits } => {
                put_u64(&mut buf, *request_id);
                put_f32_vec(&mut buf, logits);
            }
            Message::Error { request_id, message } => {
                put_u64(&mut buf, *request_id);
                buf.extend_from_slice(message.as_bytes());
            }
            Message::Hello { block_count, embedding_length }
            | Message::HelloAck { block_count, embedding_length } => {
                put_u32(&mut buf, *block_count);
                put_u32(&mut buf, *embedding_length);
            }
        }
        buf
    }

    fn decode(type_byte: u8, payload: &[u8]) -> io::Result<Message> {
        let mut r = Reader::new(payload);
        let msg = match type_byte {
            MSG_FORWARD => Message::Forward {
                request_id: r.u64()?,
                pos: r.u32()?,
                hidden: r.f32_vec()?,
            },
            MSG_RESET => Message::Reset { request_id: r.u64()? },
            MSG_HIDDEN => Message::Hidden {
                request_id: r.u64()?,
                hidden: r.f32_vec()?,
            },
            MSG_LOGITS => Message::Logits {
                request_id: r.u64()?,
                logits: r.f32_vec()?,
            },
            MSG_ERROR => Message::Error {
                request_id: r.u64()?,
                message: String::from_utf8_lossy(r.rest()).into_owned(),
            },
            MSG_HELLO => Message::Hello {
                block_count: r.u32()?,
                embedding_length: r.u32()?,
            },
            MSG_HELLO_ACK => Message::HelloAck {
                block_count: r.u32()?,
                embedding_length: r.u32()?,
            },
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown message type {other}"),
                ))
            }
        };
        Ok(msg)
    }
}

/// Write a single length-prefixed frame and flush it.
pub async fn write_message<W: AsyncWrite + Unpin>(w: &mut W, msg: &Message) -> io::Result<()> {
    let payload = msg.encode_payload();
    let mut header = Vec::with_capacity(5);
    header.push(msg.type_byte());
    header.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    w.write_all(&header).await?;
    w.write_all(&payload).await?;
    w.flush().await?;
    Ok(())
}

/// Read a single length-prefixed frame.
pub async fn read_message<R: AsyncRead + Unpin>(r: &mut R) -> io::Result<Message> {
    let type_byte = r.read_u8().await?;
    let mut len_bytes = [0u8; 4];
    r.read_exact(&mut len_bytes).await?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    let mut payload = vec![0u8; len];
    r.read_exact(&mut payload).await?;
    Message::decode(type_byte, &payload)
}
