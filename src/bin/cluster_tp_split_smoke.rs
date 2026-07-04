// Phase 4 P4.1: single-process tensor-parallel split-vs-full parity harness.
// Builds N TP shards from the full dense weights, decodes greedily with both
// the full forward pass and the sharded forward, and compares the token
// stream (hard gate) plus max logit delta per step (reported).
//
// Usage: cluster_tp_split_smoke <model.gguf> [shards] [max_tokens] ["prompt"]
//
// Weights are loaded with the 1x4 swizzle disabled (shards slice row-major
// blocks) and the scalar Q8 kernel pinned on both sides, mirroring the
// cluster harness's determinism choice.

use std::path::Path;
use std::process::ExitCode;

use nanocamelid::{gguf, inference, model, q8, tokenizer, tp};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            println!("result: FAIL");
            ExitCode::FAILURE
        }
    }
}

fn rope_scaling_from_gguf(gguf: &gguf::GgufFile) -> inference::RopeScaling {
    let prefix = gguf
        .metadata_string("general.architecture")
        .and_then(model::metadata_prefix_for_arch)
        .unwrap_or("llama");
    inference::RopeScaling {
        factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.factor")),
        original_context_length: gguf
            .metadata_u32(&format!("{prefix}.rope.scaling.original_context_length"))
            .map(|value| value as f32),
        low_freq_factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.low_freq_factor")),
        high_freq_factor: gguf.metadata_f32(&format!("{prefix}.rope.scaling.high_freq_factor")),
    }
}

fn run() -> Result<(), String> {
    // Must happen before any weight load: shards slice row-major blocks.
    unsafe {
        std::env::set_var("NANOCAMELID_Q4_SWIZZLE_1X4", "0");
        std::env::set_var("NANOCAMELID_Q8_SWIZZLE_1X4", "0");
    }

    let mut args = std::env::args().skip(1);
    let model_path = args.next().ok_or("missing model path")?;
    let shard_count: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(2);
    let max_tokens: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(24);
    let prompt = args
        .next()
        .unwrap_or_else(|| "Explain in one sentence why the sky appears blue.".to_owned());

    let path = Path::new(&model_path);
    let gguf = gguf::read_file(path).map_err(|e| e.to_string())?;
    let mut config = model::LlamaModelConfig::from_gguf(&gguf).map_err(|e| e.to_string())?;
    if config.context_length > 512 {
        config.context_length = 512;
    }
    let tok = tokenizer::Tokenizer::from_gguf(&gguf).map_err(|e| e.to_string())?;
    let rendered = tok.render_chat_prompt(&[tokenizer::ChatMessage {
        role: "user",
        content: &prompt,
    }]);
    let prompt_tokens = tok
        .encode(&rendered.text, rendered.add_special, rendered.parse_special)
        .map_err(|e| e.to_string())?;

    println!("NanoCamelid TP split smoke");
    println!("model: {model_path}");
    println!("shards: {shard_count}");
    println!("prompt_tokens: {}", prompt_tokens.len());

    let weights = model::LlamaWeights::load(path, &config, &gguf)?;
    let options = inference::LlamaRuntimeOptions {
        q8_selector: q8::Q8DotKernelSelector {
            requested: Some(q8::Q8DotKernel::Scalar),
            selected: q8::Q8DotKernel::Scalar,
            fallback_reason: None,
        },
        rope_scaling: rope_scaling_from_gguf(&gguf),
        compute_logits: true,
    };

    // Reference: full single-process forward.
    let mut ref_cache =
        inference::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
    let mut ref_ws = inference::LlamaWorkspace::new(&config);

    // TP: shards plus a full workspace for the replicated embedding/head.
    let mut shards = tp::build_tp_shards(&config, &weights.layers, shard_count)?;
    let mut rt = tp::TpRuntime::new(&config);
    let mut tp_ws = inference::LlamaWorkspace::new(&config);
    let emb = config.embedding_length;

    let mut max_delta = 0.0_f32;
    let mut mismatches = 0usize;
    let mut prompt_phase_mismatches = 0usize;
    let mut generated = Vec::new();
    let mut token = *prompt_tokens.first().ok_or("empty prompt")?;
    let mut pos = 0usize;
    let total_steps = prompt_tokens.len() + max_tokens;

    for step in 0..total_steps {
        // Reference forward.
        inference::forward_pass(
            token as usize,
            pos,
            &config,
            &weights,
            &mut ref_cache,
            &mut ref_ws,
            options,
        );

        // TP forward: replicated embedding lookup, sharded layers,
        // replicated head.
        let emb_start = token as usize * emb;
        tp_ws.hidden.copy_from_slice(&weights.token_embeddings[emb_start..emb_start + emb]);
        tp::tp_forward_token(&mut tp_ws.hidden, &mut shards, &mut rt, pos, options)?;
        inference::compute_logits_from_hidden(
            &config,
            &weights.token_embeddings,
            &weights.output_norm,
            weights.output_projection.as_ref(),
            &mut tp_ws,
            options,
        );

        let delta = ref_ws
            .logits
            .iter()
            .zip(tp_ws.logits.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        max_delta = max_delta.max(delta);

        let ref_next = inference::sample_logits(&ref_ws.logits, 0.0) as u32;
        let tp_next = inference::sample_logits(&tp_ws.logits, 0.0) as u32;
        if ref_next != tp_next {
            // Predictions at prompt positions are teacher-forced and never
            // consumed; only decode-step disagreements break the output
            // stream. Both are reported.
            if step + 1 < prompt_tokens.len() {
                prompt_phase_mismatches += 1;
            } else {
                mismatches += 1;
            }
            println!(
                "MISMATCH step {step} pos {pos} ({}): ref {ref_next} tp {tp_next} (logit delta {delta:.6})",
                if step + 1 < prompt_tokens.len() { "prompt phase" } else { "DECODE" }
            );
        }

        pos += 1;
        token = if step + 1 < prompt_tokens.len() {
            prompt_tokens[step + 1]
        } else {
            generated.push(ref_next);
            ref_next
        };
        if pos >= config.context_length {
            break;
        }
    }

    println!("generated_tokens: {generated:?}");
    println!(
        "generated_text: {:?}",
        tok.decode(&generated, true).unwrap_or_default()
    );
    println!("max_logit_delta: {max_delta:.8}");
    println!("decode_token_mismatches: {mismatches}");
    println!("prompt_phase_mismatches: {prompt_phase_mismatches}");
    println!(
        "json: {{\"benchmark\":\"tp-split-smoke\",\"shards\":{shard_count},\"steps\":{total_steps},\"max_logit_delta\":{max_delta},\"decode_token_mismatches\":{mismatches},\"prompt_phase_mismatches\":{prompt_phase_mismatches}}}"
    );
    if mismatches == 0 {
        println!("result: PASS");
        Ok(())
    } else {
        Err(format!("{mismatches} decode token mismatches"))
    }
}
