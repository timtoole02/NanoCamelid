//! Pipeline stage server. Loads only this node's layer shard and serves Forward requests:
//! run `forward_layers` over the shard's KV cache, then either return the intermediate
//! hidden state (middle stage) or run `finalize` and return logits (tail stage).

use std::collections::HashMap;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;

use tokio::net::{TcpListener, TcpStream};

use crate::gguf;
use crate::inference::{finalize, forward_layers, LlamaKvCache, LlamaWorkspace};
use crate::model::{LlamaModelConfig, LlamaWeights};
use crate::q4::Q4DotKernelSelector;
use crate::q8::Q8DotKernelSelector;

use super::config::ClusterConfig;
use super::frame::{read_message, write_message, Message};
use super::RopeParams;

struct StageContext {
    config: LlamaModelConfig,
    weights: LlamaWeights,
    start_layer: usize,
    end_layer: usize,
    is_tail: bool,
    rope: RopeParams,
    selector_q8: Q8DotKernelSelector,
    selector_q4: Q4DotKernelSelector,
}

/// Per-request decode state (KV cache + scratch), keyed by request id.
struct Session {
    cache: LlamaKvCache,
    ws: LlamaWorkspace,
}

/// Load this node's shard and serve the stage protocol until killed.
pub async fn run_stage(
    cluster: &ClusterConfig,
    node_name: &str,
    model_path: &str,
) -> Result<(), String> {
    let idx = cluster
        .index_of(node_name)
        .ok_or_else(|| format!("node '{node_name}' not found in cluster config"))?;
    if idx == 0 {
        return Err(
            "node0 is the head: run `generate --distributed` there, not `serve-stage`".to_string(),
        );
    }
    let port = cluster.nodes[idx].port;

    let gguf = gguf::read_file(Path::new(model_path)).map_err(|e| format!("read gguf: {e:?}"))?;
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let ranges = cluster.ranges(config.block_count)?;
    let (start_layer, end_layer) = ranges[idx];
    let is_tail = idx == cluster.nodes.len() - 1;

    eprintln!(
        "[stage {node_name}] loading layers [{start_layer},{end_layer}) of {} (tail={is_tail})...",
        config.block_count
    );
    let weights = LlamaWeights::load_range(
        Path::new(model_path),
        &config,
        &gguf,
        start_layer,
        end_layer,
        false, // only the head embeds
        is_tail,
    )?;

    let ctx = Arc::new(StageContext {
        config,
        weights,
        start_layer,
        end_layer,
        is_tail,
        rope: RopeParams::from_gguf(&gguf),
        selector_q8: Q8DotKernelSelector::from_env_or_auto(),
        selector_q4: Q4DotKernelSelector::from_env_or_auto(),
    });

    let bind = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&bind)
        .await
        .map_err(|e| format!("bind {bind}: {e}"))?;
    eprintln!(
        "[stage {node_name}] ready: kernel q8={} q4={}, listening on {bind}",
        ctx.selector_q8.selected.name(),
        ctx.selector_q4.selected.name()
    );

    // One head connects at a time; handle connections serially.
    loop {
        let (stream, peer) = listener
            .accept()
            .await
            .map_err(|e| format!("accept: {e}"))?;
        eprintln!("[stage {node_name}] connection from {peer}");
        if let Err(e) = handle_connection(stream, &ctx).await {
            eprintln!("[stage {node_name}] connection closed: {e}");
        }
    }
}

async fn handle_connection(mut stream: TcpStream, ctx: &StageContext) -> std::io::Result<()> {
    stream.set_nodelay(true).ok();
    let mut sessions: HashMap<u64, Session> = HashMap::new();
    loop {
        let msg = match read_message(&mut stream).await {
            Ok(m) => m,
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        match msg {
            Message::Hello { .. } => {
                write_message(
                    &mut stream,
                    &Message::HelloAck {
                        block_count: ctx.config.block_count as u32,
                        embedding_length: ctx.config.embedding_length as u32,
                    },
                )
                .await?;
            }
            Message::Reset { request_id } => {
                sessions.remove(&request_id);
            }
            Message::Forward { request_id, pos, mut hidden } => {
                let reply = process_forward(ctx, &mut sessions, request_id, pos, &mut hidden);
                write_message(&mut stream, &reply).await?;
            }
            other => {
                write_message(
                    &mut stream,
                    &Message::Error {
                        request_id: 0,
                        message: format!("unexpected message type {}", other.type_byte()),
                    },
                )
                .await?;
            }
        }
    }
}

fn process_forward(
    ctx: &StageContext,
    sessions: &mut HashMap<u64, Session>,
    request_id: u64,
    pos: u32,
    hidden: &mut Vec<f32>,
) -> Message {
    if hidden.len() != ctx.config.embedding_length {
        return Message::Error {
            request_id,
            message: format!(
                "hidden length {} != embedding_length {}",
                hidden.len(),
                ctx.config.embedding_length
            ),
        };
    }

    let n_local = ctx.end_layer - ctx.start_layer;
    let config = &ctx.config;
    let session = sessions.entry(request_id).or_insert_with(|| Session {
        cache: LlamaKvCache::new(n_local, config.context_length, config.kv_width),
        ws: LlamaWorkspace::new(config),
    });

    forward_layers(
        hidden,
        ctx.start_layer,
        ctx.end_layer,
        pos as usize,
        config,
        &ctx.weights,
        &mut session.cache,
        &mut session.ws,
        ctx.selector_q8,
        ctx.selector_q4,
        ctx.rope.factor,
        ctx.rope.orig_ctx,
        ctx.rope.low,
        ctx.rope.high,
    );

    if ctx.is_tail {
        let logits = finalize(
            hidden,
            config,
            &ctx.weights,
            &mut session.ws,
            ctx.selector_q8,
            ctx.selector_q4,
        );
        Message::Logits {
            request_id,
            logits: logits.to_vec(),
        }
    } else {
        Message::Hidden {
            request_id,
            hidden: hidden.clone(),
        }
    }
}
