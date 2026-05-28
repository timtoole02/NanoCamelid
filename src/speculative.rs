use crate::gguf;
use crate::inference::{
    self, LlamaBatchWorkspace, LlamaKvCache, LlamaRuntimeOptions, LlamaWorkspace,
};
use crate::model::{LlamaModelConfig, LlamaWeights};
use std::path::Path;

pub struct SpeculativeContext {
    pub config: LlamaModelConfig,
    pub weights: LlamaWeights,
    pub cache: LlamaKvCache,
    pub ws: LlamaWorkspace,
    pub batch_ws: LlamaBatchWorkspace,
    pub runtime_options: LlamaRuntimeOptions,
}

impl SpeculativeContext {
    pub fn load(path: &Path, runtime_options: LlamaRuntimeOptions) -> Result<Self, String> {
        let gguf = gguf::read_file(path).map_err(|e| e.to_string())?;
        let mut config = LlamaModelConfig::from_gguf(&gguf).map_err(|e| e.to_string())?;

        // Respect context limit from environment if set, matching main model behavior
        if let Ok(limit_str) = std::env::var("NANOCAMELID_CONTEXT_LIMIT")
            && let Ok(limit) = limit_str.parse::<usize>()
        {
            config.context_length = limit;
        }

        let weights = LlamaWeights::load(path, &config, &gguf)?;
        let cache = LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
        let ws = LlamaWorkspace::new(&config);

        let draft_k = std::env::var("NANOCAMELID_DRAFT_K")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(4);

        let batch_ws = LlamaBatchWorkspace::new(&config, draft_k);

        Ok(Self {
            config,
            weights,
            cache,
            ws,
            batch_ws,
            runtime_options,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SpeculativeStats {
    pub drafted: usize,
    pub accepted: usize,
}

pub struct SpeculativeTarget<'a> {
    pub config: &'a LlamaModelConfig,
    pub weights: &'a LlamaWeights,
    pub cache: &'a mut LlamaKvCache,
    pub ws: &'a mut LlamaWorkspace,
    pub batch_ws: &'a mut LlamaBatchWorkspace,
    pub pos: &'a mut usize,
    pub context_tokens: &'a mut Vec<u32>,
    pub runtime_options: LlamaRuntimeOptions,
}

pub fn speculative_decoding_step(
    target: &mut SpeculativeTarget<'_>,
    draft: &mut SpeculativeContext,
    draft_pos: &mut usize,
    temp: f32,
    draft_k: usize,
    is_stop_token: impl Fn(u32) -> bool,
) -> Result<(Vec<u32>, SpeculativeStats), String> {
    debug_assert_eq!(*target.pos, *draft_pos);
    let start_pos = *target.pos;

    let mut drafted_tokens = Vec::with_capacity(draft_k);
    let mut current_draft_pos = start_pos;

    // 1. Drafting Phase
    while drafted_tokens.len() < draft_k {
        if current_draft_pos >= draft.config.context_length
            || current_draft_pos >= target.config.context_length
        {
            break;
        }

        let draft_token = inference::sample_logits(&draft.ws.logits, temp) as u32;
        drafted_tokens.push(draft_token);

        if is_stop_token(draft_token) {
            break;
        }

        inference::forward_pass(
            draft_token as usize,
            current_draft_pos,
            &draft.config,
            &draft.weights,
            &mut draft.cache,
            &mut draft.ws,
            draft.runtime_options,
        );
        current_draft_pos += 1;
    }

    let num_drafted = drafted_tokens.len();
    if num_drafted == 0 {
        return Ok((
            Vec::new(),
            SpeculativeStats {
                drafted: 0,
                accepted: 0,
            },
        ));
    }

    // 2. Target Ingestion (Batched Forward Pass)
    inference::prefill_pass_batch(
        &drafted_tokens,
        start_pos,
        target.config,
        target.weights,
        target.cache,
        target.batch_ws,
        target.runtime_options,
    );

    // 3. Verification Loop
    let mut accepted_tokens = Vec::with_capacity(num_drafted + 1);
    let mut rejected = false;
    let emb_len = target.config.embedding_length;
    let mut stats = SpeculativeStats {
        drafted: num_drafted,
        accepted: 0,
    };

    for (i, &t_i) in drafted_tokens.iter().enumerate() {
        // For i = 0, target logits are already pre-computed in target_ws.logits from previous turn.
        // For i > 0, extract the logits from the hidden state of token index i - 1 in target batch workspace.
        if i > 0 {
            let hidden_start = (i - 1) * emb_len;
            target
                .ws
                .hidden
                .copy_from_slice(&target.batch_ws.hidden[hidden_start..hidden_start + emb_len]);
            inference::compute_logits_from_hidden(
                target.config,
                &target.weights.token_embeddings,
                &target.weights.output_norm,
                target.weights.output_projection.as_ref(),
                target.ws,
                target.runtime_options,
            );
        }

        let t_target = inference::sample_logits(&target.ws.logits, temp) as u32;

        if t_i == t_target {
            accepted_tokens.push(t_i);
            stats.accepted += 1;

            // If the accepted token is a stop token, we stop verifying immediately
            if is_stop_token(t_i) {
                // Correct KV positions for next generation loop
                *target.pos = start_pos + i + 1;
                *draft_pos = start_pos + i + 1;
                rejected = true; // prevent the trailing token generation
                break;
            }
        } else {
            rejected = true;
            accepted_tokens.push(t_target);

            // Target corrective forward pass
            inference::forward_pass(
                t_target as usize,
                start_pos + i,
                target.config,
                target.weights,
                target.cache,
                target.ws,
                target.runtime_options,
            );

            // Draft corrective forward pass
            inference::forward_pass(
                t_target as usize,
                start_pos + i,
                &draft.config,
                &draft.weights,
                &mut draft.cache,
                &mut draft.ws,
                draft.runtime_options,
            );

            *target.pos = start_pos + i + 1;
            *draft_pos = start_pos + i + 1;
            break;
        }
    }

    // 4. All drafted tokens accepted, generate trailing verified token
    if !rejected {
        let hidden_start = (num_drafted - 1) * emb_len;
        target
            .ws
            .hidden
            .copy_from_slice(&target.batch_ws.hidden[hidden_start..hidden_start + emb_len]);
        inference::compute_logits_from_hidden(
            target.config,
            &target.weights.token_embeddings,
            &target.weights.output_norm,
            target.weights.output_projection.as_ref(),
            target.ws,
            target.runtime_options,
        );

        let t_trailing = inference::sample_logits(&target.ws.logits, temp) as u32;
        accepted_tokens.push(t_trailing);

        inference::forward_pass(
            t_trailing as usize,
            start_pos + num_drafted,
            target.config,
            target.weights,
            target.cache,
            target.ws,
            target.runtime_options,
        );

        inference::forward_pass(
            t_trailing as usize,
            start_pos + num_drafted,
            &draft.config,
            &draft.weights,
            &mut draft.cache,
            &mut draft.ws,
            draft.runtime_options,
        );

        *target.pos = start_pos + num_drafted + 1;
        *draft_pos = start_pos + num_drafted + 1;
    }

    target.context_tokens.extend_from_slice(&accepted_tokens);
    Ok((accepted_tokens, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::QuantizedMatrix;
    use crate::q8::Q8_0Block;
    use crate::q8::Q8DotKernel;
    use crate::q8::Q8DotKernelSelector;

    fn test_config() -> LlamaModelConfig {
        LlamaModelConfig {
            architecture: "llama".to_owned(),
            metadata_prefix: "llama".to_owned(),
            context_length: 32,
            embedding_length: 32,
            block_count: 1,
            feed_forward_length: 32,
            attention_head_count: 1,
            attention_head_count_kv: 1,
            rope_dimension_count: 32,
            rope_freq_base: 10000.0,
            rms_norm_epsilon: 1e-5,
            vocab_size: 4,
            head_dim: 32,
            attention_output_width: 32,
            kv_width: 32,
            expert_count: 0,
            expert_used_count: 0,
        }
    }

    fn zero_q8_matrix(rows: usize, cols: usize) -> QuantizedMatrix {
        QuantizedMatrix::Q8_0(vec![Q8_0Block::from_parts(0, [0; 32]); rows * (cols / 32)])
    }

    fn test_layer(config: &LlamaModelConfig) -> crate::model::LlamaLayerWeights {
        crate::model::LlamaLayerWeights {
            attention_norm: vec![1.0; config.embedding_length],
            wq: zero_q8_matrix(config.embedding_length, config.embedding_length),
            wk: zero_q8_matrix(config.kv_width, config.embedding_length),
            wav: zero_q8_matrix(config.kv_width, config.embedding_length),
            wq_bias: None,
            wk_bias: None,
            wav_bias: None,
            wo: zero_q8_matrix(config.embedding_length, config.embedding_length),
            ffn_norm: vec![1.0; config.embedding_length],
            ffn: crate::model::LlamaFfnWeights::Dense {
                w1: zero_q8_matrix(config.feed_forward_length, config.embedding_length),
                w3: zero_q8_matrix(config.feed_forward_length, config.embedding_length),
                w2: zero_q8_matrix(config.embedding_length, config.feed_forward_length),
            },
        }
    }

    fn test_weights(config: &LlamaModelConfig) -> LlamaWeights {
        LlamaWeights {
            token_embeddings: vec![0.1; config.vocab_size * config.embedding_length],
            output_norm: vec![1.0; config.embedding_length],
            output_projection: None,
            layers: vec![test_layer(config)],
        }
    }

    #[test]
    fn test_speculative_greedy_decoding() {
        let config = test_config();
        let target_weights = test_weights(&config);
        let draft_weights = test_weights(&config);

        let mut target_cache =
            LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
        let mut target_ws = LlamaWorkspace::new(&config);
        let mut target_batch_ws = LlamaBatchWorkspace::new(&config, 4);

        let draft_cache =
            LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
        let draft_ws = LlamaWorkspace::new(&config);
        let draft_batch_ws = LlamaBatchWorkspace::new(&config, 4);

        let sel = Q8DotKernelSelector {
            requested: None,
            selected: Q8DotKernel::Scalar,
            fallback_reason: None,
        };
        let runtime_options = LlamaRuntimeOptions {
            q8_selector: sel,
            rope_scaling: crate::inference::RopeScaling::default(),
            compute_logits: true,
        };

        let mut draft_ctx = SpeculativeContext {
            config: config.clone(),
            weights: draft_weights,
            cache: draft_cache,
            ws: draft_ws,
            batch_ws: draft_batch_ws,
            runtime_options,
        };

        // Prepopulate logits with mock distributions where drafts will match target predictions.
        // Target model prediction at pos 0 is token 1
        target_ws.logits[1] = 10.0;
        draft_ctx.ws.logits[1] = 10.0;

        let mut target_pos = 0;
        let mut draft_pos = 0;
        let mut target_context_tokens = vec![0];

        let is_stop = |t| t == 3;

        let (tokens, stats) = speculative_decoding_step(
            &mut SpeculativeTarget {
                config: &config,
                weights: &target_weights,
                cache: &mut target_cache,
                ws: &mut target_ws,
                batch_ws: &mut target_batch_ws,
                pos: &mut target_pos,
                context_tokens: &mut target_context_tokens,
                runtime_options,
            },
            &mut draft_ctx,
            &mut draft_pos,
            0.0,
            2,
            is_stop,
        )
        .unwrap();

        // Under temp=0.0, greedy sampling chooses the max logit (token 1).
        // Since draft_ws.logits has max at 1, the draft guesses 1.
        // During verification, target_ws.logits also has max at 1. So t_0 (1) == T_0 (1) -> accepted!
        assert_eq!(stats.drafted, 2);
        assert!(stats.accepted > 0);
        assert_eq!(target_pos, draft_pos);
        assert_eq!(target_context_tokens[1], 1);
        assert_eq!(tokens[0], 1);
    }
}
