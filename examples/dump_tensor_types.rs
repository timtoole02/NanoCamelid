//! Dump every tensor's name and quantization type from a GGUF header.
//! Works on a truncated file as long as the header (metadata + tensor table) is intact —
//! handy for pre-flighting a multi-GB download from a ranged request of the first ~16 MB.
//!
//! Usage: cargo run --example dump_tensor_types -- <model.gguf>

use std::path::Path;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_tensor_types <model.gguf>");
    let gguf = nanocamelid::gguf::read_file(Path::new(&path)).expect("failed to parse GGUF");
    for t in &gguf.tensors {
        println!("{}\t{:?}", t.name, t.tensor_type);
    }
}
