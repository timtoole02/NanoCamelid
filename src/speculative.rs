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

/// Prompt-lookup drafting: find the longest n-gram suffix of `history`
/// (between `min_ngram` and `max_ngram`) that occurred earlier, preferring
/// the most recent occurrence, and propose the tokens that followed it.
/// Zero extra weights and near-zero cost per draft; proposes nothing on
/// novel text, so the fallback is exactly a plain greedy step.
#[derive(Debug, Clone)]
pub struct NGramDrafter {
    pub max_ngram: usize,
    pub min_ngram: usize,
}

impl Default for NGramDrafter {
    fn default() -> Self {
        Self {
            max_ngram: 4,
            // Two-token patterns (e.g. ", " pairs) recur with unrelated
            // continuations and mostly waste verify rows; three-token
            // matches measure far higher acceptance.
            min_ngram: 3,
        }
    }
}

impl NGramDrafter {
    pub fn draft(&self, history: &[u32], max_tokens: usize) -> Vec<u32> {
        if max_tokens == 0 || self.min_ngram == 0 || history.len() <= self.min_ngram {
            return Vec::new();
        }
        let len = history.len();
        let max_n = self.max_ngram.min(len.saturating_sub(1));
        for n in (self.min_ngram..=max_n).rev() {
            let pattern = &history[len - n..];
            // Most recent earlier occurrence; the window at len-n is the
            // suffix itself and is excluded.
            for start in (0..len - n).rev() {
                if &history[start..start + n] == pattern {
                    let continuation_start = start + n;
                    let continuation_end = (continuation_start + max_tokens).min(len);
                    if continuation_start < continuation_end {
                        return history[continuation_start..continuation_end].to_vec();
                    }
                    break;
                }
            }
        }
        Vec::new()
    }
}

pub struct NGramStepOutcome {
    pub emitted: Vec<u32>,
    pub hit_stop: bool,
}

/// One decode step with n-gram prompt-lookup speculation.
///
/// Entry invariant (same as the plain greedy loop): `target.ws.logits` holds
/// the logits predicting the token at `target.pos`, and
/// `target.context_tokens` holds exactly the `pos` consumed tokens. The step
/// samples the target's own next token t0 from those logits, proposes up to
/// `draft_k` continuation tokens from history, and verifies t0 plus the
/// drafts in one batched forward pass. Every emitted token is the target's
/// own greedy argmax given the accepted prefix, so at temp 0 the emitted
/// stream is identical to the plain loop's. Stop tokens are neither emitted
/// nor consumed, matching the plain loop.
pub fn ngram_speculative_step(
    target: &mut SpeculativeTarget<'_>,
    drafter: &NGramDrafter,
    temp: f32,
    draft_k: usize,
    is_stop_token: impl Fn(u32) -> bool,
) -> Result<(NGramStepOutcome, SpeculativeStats), String> {
    let start_pos = *target.pos;
    let emb_len = target.config.embedding_length;
    let mut stats = SpeculativeStats::default();

    let t0 = inference::sample_logits(&target.ws.logits, temp) as u32;
    if is_stop_token(t0) {
        return Ok((
            NGramStepOutcome {
                emitted: Vec::new(),
                hit_stop: true,
            },
            stats,
        ));
    }

    let mut history = Vec::with_capacity(target.context_tokens.len() + 1);
    history.extend_from_slice(target.context_tokens);
    history.push(t0);
    let mut drafts = if draft_k == 0 {
        Vec::new()
    } else {
        drafter.draft(&history, draft_k)
    };
    // The batch is t0 plus the drafts: bound it by the verify workspace and
    // the remaining context.
    let max_drafts = target
        .batch_ws
        .max_batch
        .saturating_sub(1)
        .min(target.config.context_length.saturating_sub(start_pos + 1));
    drafts.truncate(max_drafts);

    if drafts.is_empty() {
        inference::forward_pass(
            t0 as usize,
            start_pos,
            target.config,
            target.weights,
            target.cache,
            target.ws,
            target.runtime_options,
        );
        target.context_tokens.push(t0);
        *target.pos = start_pos + 1;
        return Ok((
            NGramStepOutcome {
                emitted: vec![t0],
                hit_stop: false,
            },
            stats,
        ));
    }

    stats.drafted = drafts.len();

    // One batched pass over t0 plus the drafts. Row i's hidden state yields
    // the logits that predict the token after batch[i]. KV rows written for
    // a rejected suffix are never attended (attention never looks past the
    // current position) and are overwritten when those positions are reached.
    let mut batch = Vec::with_capacity(drafts.len() + 1);
    batch.push(t0);
    batch.extend_from_slice(&drafts);
    inference::prefill_pass_batch(
        &batch,
        start_pos,
        target.config,
        target.weights,
        target.cache,
        target.batch_ws,
        target.runtime_options,
    );

    let mut emitted = vec![t0];
    for (i, &draft_token) in drafts.iter().enumerate() {
        let hidden_start = i * emb_len;
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
        let t_target = inference::sample_logits(&target.ws.logits, temp) as u32;
        if draft_token == t_target {
            stats.accepted += 1;
        }

        if is_stop_token(t_target) {
            // Stop token: neither emitted nor consumed. batch[..=i] is
            // consumed history (its logits produced this prediction).
            target.context_tokens.extend_from_slice(&emitted);
            *target.pos = start_pos + i + 1;
            return Ok((NGramStepOutcome { emitted, hit_stop: true }, stats));
        }

        if draft_token == t_target {
            emitted.push(draft_token);
        } else {
            emitted.push(t_target);
            // Corrective forward replaces the rejected draft's KV row and
            // leaves ws.logits predicting the following position.
            inference::forward_pass(
                t_target as usize,
                start_pos + i + 1,
                target.config,
                target.weights,
                target.cache,
                target.ws,
                target.runtime_options,
            );
            target.context_tokens.extend_from_slice(&emitted);
            *target.pos = start_pos + i + 2;
            return Ok((
                NGramStepOutcome {
                    emitted,
                    hit_stop: false,
                },
                stats,
            ));
        }
    }

    // All drafts accepted: the final batch row's logits predict the next
    // position and are left in ws.logits for the next step.
    let hidden_start = (batch.len() - 1) * emb_len;
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
    target.context_tokens.extend_from_slice(&emitted);
    *target.pos = start_pos + batch.len();
    Ok((
        NGramStepOutcome {
            emitted,
            hit_stop: false,
        },
        stats,
    ))
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
            rope_style: crate::model::RopeStyle::Norm,
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
    fn ngram_drafts_continuation_of_most_recent_match() {
        let drafter = NGramDrafter::default();
        // Suffix [1, 2, 3, 4] (n=4) matches at the start; continuation is [5, 6, 9].
        let history = vec![1, 2, 3, 4, 5, 6, 9, 9, 1, 2, 3, 4];
        assert_eq!(drafter.draft(&history, 3), vec![5, 6, 9]);
    }

    #[test]
    fn ngram_prefers_longer_patterns_and_recent_matches() {
        let drafter = NGramDrafter::default();
        // [3, 4] occurs twice earlier with different continuations; the most
        // recent occurrence (followed by 8) wins.
        let history = vec![3, 4, 7, 0, 3, 4, 8, 0, 3, 4];
        assert_eq!(drafter.draft(&history, 2), vec![8, 0]);
    }

    #[test]
    fn ngram_returns_empty_when_no_repeat_exists() {
        let drafter = NGramDrafter::default();
        assert!(drafter.draft(&[1, 2, 3, 4, 5], 4).is_empty());
        assert!(drafter.draft(&[1, 2], 4).is_empty());
        assert!(drafter.draft(&[], 4).is_empty());
    }

    #[test]
    fn ngram_caps_at_requested_tokens() {
        let drafter = NGramDrafter {
            max_ngram: 3,
            min_ngram: 2,
        };
        let history = vec![1, 2, 9, 8, 7, 6, 1, 2];
        assert_eq!(drafter.draft(&history, 2), vec![9, 8]);
        assert_eq!(drafter.draft(&history, 10), vec![9, 8, 7, 6, 1, 2]);
    }

    #[test]
    fn ngram_step_keeps_pos_and_context_consistent() {
        let config = test_config();
        let weights = test_weights(&config);
        let mut cache =
            LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
        let mut ws = LlamaWorkspace::new(&config);
        let mut batch_ws = LlamaBatchWorkspace::new(&config, 8);
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
        // Zero weights make every logit equal, so greedy always picks token
        // 0; a history of zeros makes the n-gram drafter propose zeros, which
        // all verify. This exercises the batched-verify bookkeeping.
        ws.logits[0] = 1.0;
        let mut pos = 6;
        let mut context_tokens = vec![0_u32; 6];
        let drafter = NGramDrafter::default();
        let (outcome, stats) = ngram_speculative_step(
            &mut SpeculativeTarget {
                config: &config,
                weights: &weights,
                cache: &mut cache,
                ws: &mut ws,
                batch_ws: &mut batch_ws,
                pos: &mut pos,
                context_tokens: &mut context_tokens,
                runtime_options,
            },
            &drafter,
            0.0,
            4,
            |token| token == 3,
        )
        .unwrap();
        assert!(!outcome.hit_stop);
        assert!(!outcome.emitted.is_empty());
        // Exit invariant: pos matches consumed history and every emitted
        // token was appended to it.
        assert_eq!(pos, context_tokens.len());
        assert_eq!(context_tokens.len(), 6 + outcome.emitted.len());
        assert!(stats.accepted <= stats.drafted);
    }

    #[test]
    fn ngram_step_returns_stop_without_emitting() {
        let config = test_config();
        let weights = test_weights(&config);
        let mut cache =
            LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
        let mut ws = LlamaWorkspace::new(&config);
        let mut batch_ws = LlamaBatchWorkspace::new(&config, 8);
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
        ws.logits[2] = 10.0;
        let mut pos = 3;
        let mut context_tokens = vec![0, 1, 2];
        let drafter = NGramDrafter::default();
        let (outcome, stats) = ngram_speculative_step(
            &mut SpeculativeTarget {
                config: &config,
                weights: &weights,
                cache: &mut cache,
                ws: &mut ws,
                batch_ws: &mut batch_ws,
                pos: &mut pos,
                context_tokens: &mut context_tokens,
                runtime_options,
            },
            &drafter,
            0.0,
            4,
            |token| token == 2,
        )
        .unwrap();
        assert!(outcome.hit_stop);
        assert!(outcome.emitted.is_empty());
        assert_eq!(pos, 3);
        assert_eq!(context_tokens, vec![0, 1, 2]);
        assert_eq!(stats.drafted, 0);
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
