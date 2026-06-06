//! Head-node driver for distributed generation. Runs node0's own layer shard locally, then
//! RPCs each downstream stage with the hidden state, and finally samples on the head (so the
//! RNG stream matches a single-node run exactly).

use std::io::Write;
use std::path::Path;

use tokio::net::TcpStream;

use crate::gguf;
use crate::inference::{self, embed, finalize, forward_layers, LlamaKvCache, LlamaWorkspace};
use crate::model::{LlamaModelConfig, LlamaWeights};
use crate::q4::Q4DotKernelSelector;
use crate::q8::Q8DotKernelSelector;
use crate::tokenizer::Tokenizer;

use super::config::ClusterConfig;
use super::frame::{read_message, write_message, Message};
use super::RopeParams;

struct Head {
    config: LlamaModelConfig,
    weights: LlamaWeights,
    head_start: usize,
    head_end: usize,
    head_is_tail: bool,
    cache: LlamaKvCache,
    ws: LlamaWorkspace,
    selector_q8: Q8DotKernelSelector,
    selector_q4: Q4DotKernelSelector,
    rope: RopeParams,
    request_id: u64,
}

impl Head {
    /// Produce logits for `input_token` at `pos`: embed + head layers locally, then walk the
    /// downstream stages. The tail stage returns logits.
    async fn step(
        &mut self,
        workers: &mut [TcpStream],
        input_token: usize,
        pos: usize,
    ) -> Result<Vec<f32>, String> {
        let mut x = embed(input_token, &self.config, &self.weights);
        forward_layers(
            &mut x,
            self.head_start,
            self.head_end,
            pos,
            &self.config,
            &self.weights,
            &mut self.cache,
            &mut self.ws,
            self.selector_q8,
            self.selector_q4,
            self.rope.factor,
            self.rope.orig_ctx,
            self.rope.low,
            self.rope.high,
        );

        if self.head_is_tail {
            // Single-node degenerate case: head owns the output weights too.
            let logits = finalize(
                &x,
                &self.config,
                &self.weights,
                &mut self.ws,
                self.selector_q8,
                self.selector_q4,
            );
            return Ok(logits.to_vec());
        }

        for stream in workers.iter_mut() {
            let msg = Message::Forward {
                request_id: self.request_id,
                pos: pos as u32,
                hidden: x.clone(),
            };
            write_message(stream, &msg).await.map_err(|e| e.to_string())?;
            match read_message(stream).await.map_err(|e| e.to_string())? {
                Message::Hidden { hidden, .. } => x = hidden,
                Message::Logits { logits, .. } => return Ok(logits),
                Message::Error { message, .. } => return Err(format!("stage error: {message}")),
                other => {
                    return Err(format!("unexpected reply type {}", other.type_byte()));
                }
            }
        }
        Err("tail stage did not return logits".to_string())
    }
}

pub async fn run_distributed_generation(
    cluster: &ClusterConfig,
    model_path: &str,
    prompt: &str,
    temp: f32,
    max_tokens: usize,
) -> Result<(), String> {
    let gguf = gguf::read_file(Path::new(model_path)).map_err(|e| format!("read gguf: {e:?}"))?;
    let config = LlamaModelConfig::from_gguf(&gguf)?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;

    let ranges = cluster.ranges(config.block_count)?;
    let (head_start, head_end) = ranges[0];
    let head_is_tail = cluster.nodes.len() == 1;

    println!("Architecture: LLaMA (distributed pipeline)");
    println!("Layers: {} split across {} nodes", config.block_count, cluster.nodes.len());
    for (i, node) in cluster.nodes.iter().enumerate() {
        let (s, e) = ranges[i];
        println!("  {} -> layers [{s},{e}){}", node.name, if i == cluster.nodes.len() - 1 { " + finalize" } else { "" });
    }

    eprintln!("[head] loading layers [{head_start},{head_end}) locally...");
    let weights = LlamaWeights::load_range(
        Path::new(model_path),
        &config,
        &gguf,
        head_start,
        head_end,
        true, // head embeds
        head_is_tail,
    )?;

    let n_local = head_end - head_start;
    let cache = LlamaKvCache::new(n_local, config.context_length, config.kv_width);
    let ws = LlamaWorkspace::new(&config);

    let mut head = Head {
        config,
        weights,
        head_start,
        head_end,
        head_is_tail,
        cache,
        ws,
        selector_q8: Q8DotKernelSelector::from_env_or_auto(),
        selector_q4: Q4DotKernelSelector::from_env_or_auto(),
        rope: RopeParams::from_gguf(&gguf),
        request_id: 1,
    };

    // Connect + handshake with the downstream stages.
    let mut workers: Vec<TcpStream> = Vec::new();
    for node in &cluster.nodes[1..] {
        let addr = format!("{}:{}", node.host, node.port);
        eprintln!("[head] connecting to stage {} at {addr}...", node.name);
        let mut stream = TcpStream::connect(&addr)
            .await
            .map_err(|e| format!("connect {addr}: {e}"))?;
        stream.set_nodelay(true).ok();
        write_message(
            &mut stream,
            &Message::Hello {
                block_count: head.config.block_count as u32,
                embedding_length: head.config.embedding_length as u32,
            },
        )
        .await
        .map_err(|e| e.to_string())?;
        match read_message(&mut stream).await.map_err(|e| e.to_string())? {
            Message::HelloAck { block_count, embedding_length } => {
                if block_count as usize != head.config.block_count
                    || embedding_length as usize != head.config.embedding_length
                {
                    return Err(format!(
                        "stage {} reports different model (block_count={block_count}, embedding_length={embedding_length})",
                        node.name
                    ));
                }
            }
            other => {
                return Err(format!(
                    "stage {} returned bad handshake (type {})",
                    node.name,
                    other.type_byte()
                ))
            }
        }
        workers.push(stream);
    }

    // Fresh sequence: clear any stale session state on the workers.
    for stream in workers.iter_mut() {
        write_message(stream, &Message::Reset { request_id: head.request_id })
            .await
            .map_err(|e| e.to_string())?;
    }

    let prompt_tokens = tokenizer.encode(prompt, true, true)?;
    println!("Prompt tokens: {:?}", prompt_tokens);
    println!("\nGenerating response:\n");

    let mut pos = 0usize;
    let mut logits: Vec<f32> = Vec::new();

    // Prefill
    for &token in &prompt_tokens {
        logits = head.step(&mut workers, token as usize, pos).await?;
        pos += 1;
    }

    // Generation
    let mut generated_count = 0usize;
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut last_printed_len = 0usize;
    let start_gen = std::time::Instant::now();

    loop {
        let next_token = inference::sample_logits(&logits, temp);
        if Some(next_token as u32) == tokenizer.special.eos
            || Some(next_token as u32) == tokenizer.special.eot
            || pos >= head.config.context_length
            || generated_count >= max_tokens
        {
            break;
        }

        generated_tokens.push(next_token as u32);
        if let Ok(full_text) = tokenizer.decode(&generated_tokens, true) {
            if full_text.len() > last_printed_len {
                print!("{}", &full_text[last_printed_len..]);
                std::io::stdout().flush().ok();
                last_printed_len = full_text.len();
            }
        }

        logits = head.step(&mut workers, next_token, pos).await?;
        pos += 1;
        generated_count += 1;
    }

    let elapsed = start_gen.elapsed().as_secs_f64();
    println!(
        "\n\nGenerated {} tokens in {:.2}s ({:.2} tokens/sec)",
        generated_count,
        elapsed,
        generated_count as f64 / elapsed.max(1e-9)
    );
    Ok(())
}
