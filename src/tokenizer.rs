use std::collections::{BTreeSet, BinaryHeap, HashMap};

use crate::gguf::GgufFile;

pub type TokenId = u32;

const SPM_SPACE: char = '▁';

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    LlamaSpm,
    Gpt2Bpe,
}

impl TokenizerModel {
    pub fn as_summary_model(self) -> &'static str {
        match self {
            Self::LlamaSpm => "llama_spm",
            Self::Gpt2Bpe => "gpt2_bpe",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Undefined,
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenKind {
    fn from_i32(value: i32) -> Result<Self, String> {
        Ok(match value {
            0 => Self::Undefined,
            1 => Self::Normal,
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            other => return Err(format!("unknown tokenizer token type {other}")),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub id: TokenId,
    pub text: String,
    pub score: f32,
    pub kind: TokenKind,
}

#[derive(Debug, Clone, Default)]
pub struct BpeRegistry {
    ranks: HashMap<(String, String), usize>,
}

impl BpeRegistry {
    fn from_merges(merges: Vec<String>) -> Self {
        let ranks = merges
            .into_iter()
            .enumerate()
            .filter_map(|(rank, merge)| {
                let (left, right) = merge.split_once(' ')?;
                Some(((left.to_string(), right.to_string()), rank))
            })
            .collect();
        Self { ranks }
    }

    pub fn len(&self) -> usize {
        self.ranks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ranks.is_empty()
    }

    fn rank(&self, left: &str, right: &str) -> Option<usize> {
        self.ranks
            .get(&(left.to_string(), right.to_string()))
            .copied()
    }

    fn ranks(&self) -> &HashMap<(String, String), usize> {
        &self.ranks
    }

    fn merge_symbols(&self, mut symbols: Vec<String>) -> Vec<String> {
        while symbols.len() > 1 {
            let mut heap = BinaryHeap::new();
            for idx in 0..symbols.len() - 1 {
                if let Some(rank) = self.rank(&symbols[idx], &symbols[idx + 1]) {
                    heap.push(BpeMergeCandidate { rank, index: idx });
                }
            }

            let Some(best) = heap.pop() else { break };
            let left = symbols[best.index].clone();
            let right = symbols[best.index + 1].clone();
            let mut merged = Vec::with_capacity(symbols.len() - 1);
            let mut idx = 0;
            while idx < symbols.len() {
                if idx + 1 < symbols.len() && symbols[idx] == left && symbols[idx + 1] == right {
                    merged.push(format!("{}{}", symbols[idx], symbols[idx + 1]));
                    idx += 2;
                } else {
                    merged.push(symbols[idx].clone());
                    idx += 1;
                }
            }
            symbols = merged;
        }
        symbols
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BpeMergeCandidate {
    rank: usize,
    index: usize,
}

impl Ord for BpeMergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for BpeMergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    pub bos: Option<TokenId>,
    pub eos: Option<TokenId>,
    pub eot: Option<TokenId>,
    pub eom: Option<TokenId>,
    pub unk: Option<TokenId>,
    pub sep: Option<TokenId>,
    pub pad: Option<TokenId>,
    pub mask: Option<TokenId>,
    pub eog: BTreeSet<TokenId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerConfig {
    pub add_bos: bool,
    pub add_eos: bool,
    pub add_sep: bool,
    pub add_space_prefix: bool,
    pub remove_extra_whitespaces: bool,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub source_name: Option<String>,
    pub model: TokenizerModel,
    pub tokens: Vec<Token>,
    pub token_to_id: HashMap<String, TokenId>,
    pub byte_token_to_id: HashMap<u8, TokenId>,
    pub bpe_ranks: HashMap<(String, String), usize>,
    pub bpe_registry: BpeRegistry,
    pub special: SpecialTokens,
    pub config: TokenizerConfig,
    pub chat_template: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedChatPrompt {
    pub text: String,
    pub add_special: bool,
    pub parse_special: bool,
    pub renderer: &'static str,
}

impl Tokenizer {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, String> {
        let model_name = file
            .metadata_string("tokenizer.ggml.model")
            .ok_or_else(|| "tokenizer.ggml.model missing".to_owned())?;
        let model = match model_name {
            "llama" => TokenizerModel::LlamaSpm,
            "gpt2" => TokenizerModel::Gpt2Bpe,
            other => return Err(format!("unsupported tokenizer model: {other}")),
        };
        if model == TokenizerModel::Gpt2Bpe {
            let pre_tokenizer = file.metadata_string("tokenizer.ggml.pre");
            let missing_llama_bpe = pre_tokenizer.is_none()
                && file.metadata_string("general.architecture") == Some("llama");
            if !missing_llama_bpe
                && !matches!(
                    pre_tokenizer,
                    Some(
                        "llama-bpe"
                            | "qwen2"
                            | "deepseek-r1-qwen"
                            | "smollm"
                            | "smaug-bpe"
                            | "lfm2"
                    )
                )
            {
                return Err(format!(
                    "unsupported GPT-2/BPE pre-tokenizer: {pre_tokenizer:?}"
                ));
            }
        }

        let token_texts = file
            .metadata_array_strings("tokenizer.ggml.tokens")
            .ok_or_else(|| "tokenizer.ggml.tokens missing or invalid".to_owned())?;
        if token_texts.is_empty() {
            return Err("tokenizer.ggml.tokens must not be empty".to_string());
        }

        let scores = file
            .metadata_array_f32("tokenizer.ggml.scores")
            .unwrap_or_else(|| vec![0.0; token_texts.len()]);
        if scores.len() < token_texts.len() {
            return Err(format!(
                "tokenizer.ggml.scores length {} is shorter than token count {}",
                scores.len(),
                token_texts.len()
            ));
        }

        let kinds_raw = file
            .metadata_array_i32("tokenizer.ggml.token_type")
            .unwrap_or_else(|| vec![1; token_texts.len()]);
        if kinds_raw.len() < token_texts.len() {
            return Err(format!(
                "tokenizer.ggml.token_type length {} is shorter than token count {}",
                kinds_raw.len(),
                token_texts.len()
            ));
        }

        let bpe_registry = BpeRegistry::from_merges(
            file.metadata_array_strings("tokenizer.ggml.merges")
                .unwrap_or_default(),
        );
        let bpe_ranks = bpe_registry.ranks().clone();

        let mut tokens = Vec::with_capacity(token_texts.len());
        let mut token_to_id = HashMap::with_capacity(token_texts.len());
        let mut byte_token_to_id = HashMap::new();
        for (idx, text) in token_texts.into_iter().enumerate() {
            let id = idx as TokenId;
            let kind = TokenKind::from_i32(kinds_raw[idx])?;
            if let Some(byte) = parse_byte_token(&text) {
                byte_token_to_id.insert(byte, id);
            }
            token_to_id.insert(text.clone(), id);
            tokens.push(Token {
                id,
                text,
                score: scores[idx],
                kind,
            });
        }

        let default_bos = match model {
            TokenizerModel::LlamaSpm => Some(1),
            TokenizerModel::Gpt2Bpe => token_to_id.get("<|begin_of_text|>").copied(),
        };
        let default_eos = match model {
            TokenizerModel::LlamaSpm => Some(2),
            TokenizerModel::Gpt2Bpe => token_to_id.get("<|end_of_text|>").copied(),
        };
        let default_unk = match model {
            TokenizerModel::LlamaSpm => Some(0),
            TokenizerModel::Gpt2Bpe => None,
        };

        let bos = file
            .metadata_u32("tokenizer.ggml.bos_token_id")
            .or(default_bos);
        let eos = file
            .metadata_u32("tokenizer.ggml.eos_token_id")
            .or(default_eos);
        let unk = file
            .metadata_u32("tokenizer.ggml.unknown_token_id")
            .or(default_unk);
        let eot = file
            .metadata_u32("tokenizer.ggml.eot_token_id")
            .or_else(|| token_to_id.get("<|eot_id|>").copied());
        let eom = file.metadata_u32("tokenizer.ggml.eom_token_id");
        let sep = file
            .metadata_u32("tokenizer.ggml.separator_token_id")
            .or_else(|| file.metadata_u32("tokenizer.ggml.seperator_token_id"));
        let pad = file.metadata_u32("tokenizer.ggml.padding_token_id");
        let mask = file.metadata_u32("tokenizer.ggml.mask_token_id");
        let eog = [eos, eot, eom].into_iter().flatten().collect();

        validate_token_id("bos", bos, tokens.len())?;
        validate_token_id("eos", eos, tokens.len())?;
        validate_token_id("unk", unk, tokens.len())?;
        validate_token_id("eot", eot, tokens.len())?;
        validate_token_id("eom", eom, tokens.len())?;
        validate_token_id("sep", sep, tokens.len())?;
        validate_token_id("pad", pad, tokens.len())?;
        validate_token_id("mask", mask, tokens.len())?;

        Ok(Self {
            source_name: file.metadata_string("general.name").map(str::to_owned),
            model,
            tokens,
            token_to_id,
            byte_token_to_id,
            bpe_ranks,
            bpe_registry,
            special: SpecialTokens {
                bos,
                eos,
                eot,
                eom,
                unk,
                sep,
                pad,
                mask,
                eog,
            },
            config: TokenizerConfig {
                add_bos: file
                    .metadata_bool("tokenizer.ggml.add_bos_token")
                    .unwrap_or(true),
                add_eos: file
                    .metadata_bool("tokenizer.ggml.add_eos_token")
                    .unwrap_or(false),
                add_sep: file
                    .metadata_bool("tokenizer.ggml.add_sep_token")
                    .unwrap_or(false),
                add_space_prefix: file
                    .metadata_bool("tokenizer.ggml.add_space_prefix")
                    .unwrap_or(true),
                remove_extra_whitespaces: file
                    .metadata_bool("tokenizer.ggml.remove_extra_whitespaces")
                    .unwrap_or(false),
            },
            chat_template: file
                .metadata_string("tokenizer.chat_template")
                .map(str::to_owned),
        })
    }

    pub fn token_text(&self, id: Option<TokenId>) -> Option<&str> {
        id.and_then(|id| self.tokens.get(id as usize))
            .map(|token| token.text.as_str())
    }

    pub fn token_id(&self, text: &str) -> Option<TokenId> {
        self.token_to_id.get(text).copied()
    }

    pub fn encode(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<TokenId>, String> {
        let mut out = Vec::new();
        if add_special
            && self.config.add_bos
            && let Some(bos) = self.special.bos
        {
            out.push(bos);
        }

        match self.model {
            TokenizerModel::LlamaSpm => {
                let normalized = self.normalize_spm_text(text, parse_special);
                if !normalized.is_empty() {
                    out.extend(self.encode_piece(&normalized, parse_special)?);
                }
            }
            TokenizerModel::Gpt2Bpe => {
                if !text.is_empty() {
                    out.extend(self.encode_bpe_text(text, parse_special)?);
                }
            }
        }

        if add_special
            && self.config.add_eos
            && let Some(eos) = self.special.eos
        {
            out.push(eos);
        }
        Ok(out)
    }

    pub fn decode(&self, token_ids: &[TokenId], remove_special: bool) -> Result<String, String> {
        if self.model == TokenizerModel::Gpt2Bpe {
            return self.decode_bpe(token_ids, remove_special);
        }

        let mut bytes = Vec::new();
        let mut text = String::new();

        for id in token_ids {
            if remove_special && self.is_special(*id) {
                continue;
            }
            let token = self
                .tokens
                .get(*id as usize)
                .ok_or_else(|| format!("token id {id} out of range"))?;
            if token.kind == TokenKind::Control && remove_special {
                continue;
            }
            if let Some(byte) = parse_byte_token(&token.text) {
                bytes.push(byte);
                continue;
            }
            flush_bytes(&mut bytes, &mut text)?;
            text.push_str(&token.text.replace(SPM_SPACE, " "));
        }
        flush_bytes(&mut bytes, &mut text)?;
        Ok(text)
    }

    pub fn chat_prompt_parse_special(&self) -> bool {
        matches!(self.model, TokenizerModel::Gpt2Bpe)
    }

    pub fn chat_template_format(&self) -> Option<&'static str> {
        self.chat_template.as_deref().map_or_else(
            || {
                if self.has_qwen_im_tokens() {
                    Some("qwen_im_token_fallback")
                } else if self.has_llama3_instruct_tokens() {
                    Some("llama3_instruct")
                } else {
                    self.has_mistral_inst_tokens()
                        .then_some("mistral_inst_token_fallback")
                }
            },
            |template| Some(detect_chat_template_format(template)),
        )
    }

    pub fn render_chat_prompt(&self, messages: &[ChatMessage<'_>]) -> RenderedChatPrompt {
        if let Some(template) = self.chat_template.as_deref() {
            if is_tinyllama_marker_template(template) {
                return RenderedChatPrompt {
                    text: render_tinyllama_marker_prompt(messages, self),
                    add_special: true,
                    parse_special: self.chat_prompt_parse_special(),
                    renderer: "tinyllama_marker",
                };
            }
            if is_llama3_instruct_template(template) {
                return RenderedChatPrompt {
                    text: render_llama3_instruct_prompt(messages),
                    add_special: true,
                    parse_special: self.chat_prompt_parse_special(),
                    renderer: "llama3_instruct",
                };
            }
            if is_qwen_im_template(template) {
                return RenderedChatPrompt {
                    text: render_qwen_im_prompt(messages),
                    add_special: true,
                    parse_special: self.chat_prompt_parse_special(),
                    renderer: "qwen_im",
                };
            }
            if is_gemma_turn_template(template) {
                return RenderedChatPrompt {
                    text: render_gemma_turn_prompt(messages),
                    add_special: true,
                    parse_special: self.chat_prompt_parse_special(),
                    renderer: "gemma_turn",
                };
            }
            if is_deepseek_r1_qwen_template(template) {
                return RenderedChatPrompt {
                    text: render_deepseek_r1_qwen_prompt(messages, self),
                    add_special: false,
                    parse_special: self.chat_prompt_parse_special(),
                    renderer: "deepseek_r1_qwen",
                };
            }
            if is_mistral_inst_template(template) {
                return RenderedChatPrompt {
                    text: render_mistral_inst_prompt(messages),
                    add_special: false,
                    parse_special: self.chat_prompt_parse_special(),
                    renderer: "mistral_inst",
                };
            }
        }
        if self.has_qwen_im_tokens() {
            return RenderedChatPrompt {
                text: render_qwen_im_prompt(messages),
                add_special: true,
                parse_special: self.chat_prompt_parse_special(),
                renderer: "qwen_im_token_fallback",
            };
        }
        if self.chat_template.is_none() && self.has_llama3_instruct_tokens() {
            return RenderedChatPrompt {
                text: render_llama3_instruct_prompt(messages),
                add_special: true,
                parse_special: self.chat_prompt_parse_special(),
                renderer: "llama3_instruct",
            };
        }
        if self.has_mistral_inst_tokens() {
            return RenderedChatPrompt {
                text: render_mistral_inst_prompt(messages),
                add_special: false,
                parse_special: self.chat_prompt_parse_special(),
                renderer: "mistral_inst_token_fallback",
            };
        }

        RenderedChatPrompt {
            text: render_role_colon_prompt(messages),
            add_special: true,
            parse_special: self.chat_prompt_parse_special(),
            renderer: "role_colon_fallback",
        }
    }

    fn has_llama3_instruct_tokens(&self) -> bool {
        self.model == TokenizerModel::Gpt2Bpe
            && self.token_to_id.contains_key("<|start_header_id|>")
            && self.token_to_id.contains_key("<|end_header_id|>")
            && self.token_to_id.contains_key("<|eot_id|>")
    }

    fn has_qwen_im_tokens(&self) -> bool {
        self.model == TokenizerModel::Gpt2Bpe
            && self.token_to_id.contains_key("<|im_start|>")
            && self.token_to_id.contains_key("<|im_end|>")
    }

    fn has_mistral_inst_tokens(&self) -> bool {
        self.model == TokenizerModel::LlamaSpm
            && (self
                .source_name
                .as_deref()
                .is_some_and(|name| name.to_ascii_lowercase().contains("mistral"))
                || (self.token_to_id.contains_key("[INST]")
                    && self.token_to_id.contains_key("[/INST]")))
    }

    fn encode_bpe_text(&self, text: &str, parse_special: bool) -> Result<Vec<TokenId>, String> {
        let mut out = Vec::new();
        let mut byte_start = 0;

        while byte_start < text.len() {
            if parse_special
                && let Some((token_text, token_len)) =
                    self.longest_control_token_at(text, byte_start)
                && let Some(id) = self.token_to_id.get(token_text)
            {
                out.push(*id);
                byte_start += token_len;
                continue;
            }

            let byte_end = if parse_special {
                self.next_control_token_start(text, byte_start)
                    .unwrap_or(text.len())
            } else {
                text.len()
            };

            for segment in bpe_pretokenize(&text[byte_start..byte_end]) {
                self.encode_bpe_segment(segment, &mut out)?;
            }
            byte_start = byte_end;
        }

        Ok(out)
    }

    fn encode_bpe_segment(&self, segment: &str, out: &mut Vec<TokenId>) -> Result<(), String> {
        if segment.is_empty() {
            return Ok(());
        }

        let mut symbols: Vec<String> = segment
            .as_bytes()
            .iter()
            .map(|byte| bpe_byte_to_char(*byte).to_string())
            .collect();

        symbols = self.bpe_registry.merge_symbols(symbols);

        for symbol in symbols {
            let id = self.token_to_id.get(&symbol).copied().ok_or_else(|| {
                format!("GPT-2/BPE token {symbol:?} is missing from tokenizer.ggml.tokens")
            })?;
            out.push(id);
        }
        Ok(())
    }

    fn decode_bpe(&self, token_ids: &[TokenId], remove_special: bool) -> Result<String, String> {
        let mut bytes = Vec::new();
        for id in token_ids {
            if remove_special && self.is_special(*id) {
                continue;
            }
            let token = self
                .tokens
                .get(*id as usize)
                .ok_or_else(|| format!("token id {id} out of range"))?;
            if remove_special && token.kind == TokenKind::Control {
                continue;
            }
            for ch in token.text.chars() {
                if let Some(byte) = bpe_char_to_byte(ch) {
                    bytes.push(byte);
                } else if !remove_special || token.kind != TokenKind::Control {
                    return Err(format!(
                        "GPT-2/BPE token {:?} contains non-byte character {ch:?}",
                        token.text
                    ));
                }
            }
        }

        String::from_utf8(bytes).map_err(|_| "GPT-2/BPE decode produced invalid UTF-8".to_string())
    }

    fn normalize_spm_text(&self, text: &str, parse_special: bool) -> String {
        let mut normalized = String::new();
        if text.is_empty() {
            return normalized;
        }
        if self.config.add_space_prefix
            && !text.starts_with(char::is_whitespace)
            && !(parse_special && self.longest_control_token_at(text, 0).is_some())
        {
            normalized.push(SPM_SPACE);
        }
        for ch in text.chars() {
            if ch == ' ' {
                normalized.push(SPM_SPACE);
            } else {
                normalized.push(ch);
            }
        }
        if parse_special {
            normalized
        } else {
            self.add_dummy_prefix_after_control_tokens(&normalized)
        }
    }

    fn add_dummy_prefix_after_control_tokens(&self, text: &str) -> String {
        if !self.config.add_space_prefix || text.is_empty() {
            return text.to_string();
        }

        let mut normalized = String::with_capacity(text.len());
        let mut byte_start = 0;
        while byte_start < text.len() {
            if let Some((token_text, token_len)) = self.longest_control_token_at(text, byte_start) {
                normalized.push_str(token_text);
                byte_start += token_len;

                let rest = &text[byte_start..];
                let next_is_control = self.longest_control_token_at(text, byte_start).is_some();
                let should_insert_dummy_prefix =
                    self.should_insert_dummy_after_control(token_text, rest, next_is_control);
                if should_insert_dummy_prefix {
                    normalized.push(SPM_SPACE);
                }
                continue;
            }

            let ch = text[byte_start..]
                .chars()
                .next()
                .expect("byte_start is in-bounds");
            normalized.push(ch);
            byte_start += ch.len_utf8();
        }
        normalized
    }

    fn should_insert_dummy_after_control(
        &self,
        token_text: &str,
        rest: &str,
        next_is_control: bool,
    ) -> bool {
        if rest.is_empty() || next_is_control {
            return false;
        }

        if self
            .token_text(self.special.bos)
            .is_some_and(|bos| token_text == bos)
            && rest.starts_with("[INST]")
            && self.token_to_id.contains_key("[INST]")
            && self.token_to_id.contains_key("[/INST]")
        {
            return false;
        }

        if token_text == "[INST]"
            && self.token_to_id.contains_key("[INST]")
            && self.token_to_id.contains_key("[/INST]")
        {
            return true;
        }

        !rest.starts_with(SPM_SPACE)
    }

    fn longest_control_token_at<'a>(
        &'a self,
        text: &str,
        byte_start: usize,
    ) -> Option<(&'a str, usize)> {
        if !text.is_char_boundary(byte_start) {
            return None;
        }

        self.tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Control)
            .filter(|token| text[byte_start..].starts_with(&token.text))
            .max_by_key(|token| token.text.len())
            .map(|token| (token.text.as_str(), token.text.len()))
    }

    fn encode_piece(&self, piece: &str, parse_special: bool) -> Result<Vec<TokenId>, String> {
        if self.bpe_ranks.is_empty() && !parse_special {
            return self.encode_piece_greedy(piece);
        }

        let mut out = Vec::new();
        let mut byte_start = 0;
        while byte_start < piece.len() {
            if parse_special
                && let Some((token_text, token_len)) =
                    self.longest_control_token_at(piece, byte_start)
                && let Some(id) = self.token_to_id.get(token_text)
            {
                out.push(*id);
                byte_start += token_len;
                let rest = &piece[byte_start..];
                let next_is_control = self.longest_control_token_at(piece, byte_start).is_some();
                if self.config.add_space_prefix
                    && self.should_insert_dummy_after_control(token_text, rest, next_is_control)
                    && let Some(dummy_prefix) = self.token_to_id.get(&SPM_SPACE.to_string())
                {
                    out.push(*dummy_prefix);
                }
                continue;
            }

            let byte_end = if parse_special {
                self.next_control_token_start(piece, byte_start)
                    .unwrap_or(piece.len())
            } else {
                piece.len()
            };
            if self.bpe_ranks.is_empty() {
                if parse_special {
                    self.encode_spm_segment(&piece[byte_start..byte_end], &mut out)?;
                } else {
                    out.extend(self.encode_piece_greedy(&piece[byte_start..byte_end])?);
                }
            } else {
                self.encode_spm_segment(&piece[byte_start..byte_end], &mut out)?;
            }
            byte_start = byte_end;
        }
        Ok(out)
    }

    fn next_control_token_start(&self, text: &str, byte_start: usize) -> Option<usize> {
        text[byte_start..]
            .char_indices()
            .map(|(offset, _)| byte_start + offset)
            .find(|idx| self.longest_control_token_at(text, *idx).is_some())
    }

    fn encode_spm_segment(&self, segment: &str, out: &mut Vec<TokenId>) -> Result<(), String> {
        if segment.is_empty() {
            return Ok(());
        }

        let symbols = if self.bpe_ranks.is_empty() {
            self.merge_spm_symbols_by_score(segment)
        } else {
            self.bpe_registry
                .merge_symbols(segment.chars().map(|ch| ch.to_string()).collect())
        };

        let mut unresolved = String::new();
        for symbol in symbols {
            if symbol.contains("▁▁") {
                unresolved.push_str(&symbol);
                continue;
            }

            if let Some(id) = self.token_to_id.get(&symbol).copied() {
                if !unresolved.is_empty() {
                    out.extend(self.encode_piece_greedy(&unresolved)?);
                    unresolved.clear();
                }
                out.push(id);
            } else {
                unresolved.push_str(&symbol);
            }
        }
        if !unresolved.is_empty() {
            out.extend(self.encode_piece_greedy(&unresolved)?);
        }
        Ok(())
    }

    fn merge_spm_symbols_by_score(&self, segment: &str) -> Vec<String> {
        let mut symbols: Vec<String> = segment.chars().map(|ch| ch.to_string()).collect();

        loop {
            let mut best: Option<(f32, usize)> = None;
            for idx in 0..symbols.len().saturating_sub(1) {
                let candidate = format!("{}{}", symbols[idx], symbols[idx + 1]);
                if candidate.contains("▁▁") {
                    continue;
                }
                let Some(id) = self.token_to_id.get(&candidate).copied() else {
                    continue;
                };
                let score = self.tokens[id as usize].score;
                match best {
                    Some((best_score, best_idx))
                        if score < best_score || (score == best_score && idx >= best_idx) => {}
                    _ => best = Some((score, idx)),
                }
            }

            let Some((_, idx)) = best else { break };
            symbols[idx] = format!("{}{}", symbols[idx], symbols[idx + 1]);
            symbols.remove(idx + 1);
        }

        symbols
    }

    fn encode_unknown_symbol_bytes(
        &self,
        symbol: &str,
        out: &mut Vec<TokenId>,
    ) -> Result<(), String> {
        for byte in symbol.as_bytes() {
            let id = self
                .byte_token_to_id
                .get(byte)
                .copied()
                .or(self.special.unk);
            match id {
                Some(id) => out.push(id),
                None => return Err(format!("SPM byte fallback token <0x{byte:02X}> is missing")),
            }
        }
        Ok(())
    }

    fn encode_piece_greedy(&self, piece: &str) -> Result<Vec<TokenId>, String> {
        let chars: Vec<(usize, char)> = piece.char_indices().collect();
        let mut out = Vec::new();
        let mut byte_start = 0;

        while byte_start < piece.len() {
            let mut best: Option<(usize, TokenId, f32)> = None;
            for byte_end in piece[byte_start..]
                .char_indices()
                .skip(1)
                .map(|(offset, _)| byte_start + offset)
                .chain(std::iter::once(piece.len()))
            {
                let candidate = &piece[byte_start..byte_end];
                if candidate.contains("▁▁") {
                    continue;
                }
                if let Some(id) = self.token_to_id.get(candidate) {
                    let score = self.tokens[*id as usize].score;
                    let len = byte_end - byte_start;
                    match best {
                        Some((best_len, _, best_score))
                            if len < best_len || (len == best_len && score <= best_score) => {}
                        _ => best = Some((len, *id, score)),
                    }
                }
            }

            if let Some((len, id, _)) = best {
                out.push(id);
                byte_start += len;
                continue;
            }

            let ch = chars
                .iter()
                .find(|(idx, _)| *idx == byte_start)
                .map(|(_, ch)| *ch)
                .ok_or_else(|| "internal UTF-8 tokenizer cursor error".to_string())?;
            let mut buf = [0u8; 4];
            self.encode_unknown_symbol_bytes(ch.encode_utf8(&mut buf), &mut out)?;
            byte_start += ch.len_utf8();
        }
        Ok(out)
    }

    fn is_special(&self, id: TokenId) -> bool {
        self.special.bos == Some(id)
            || self.special.eos == Some(id)
            || self.special.eot == Some(id)
            || self.special.eom == Some(id)
            || self.special.sep == Some(id)
            || self.special.pad == Some(id)
            || self.special.mask == Some(id)
    }
}

fn bpe_pretokenize(text: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut byte_start = 0;

    while byte_start < text.len() {
        let byte_end = next_llama_bpe_segment_end(text, byte_start);
        segments.push(&text[byte_start..byte_end]);
        byte_start = byte_end;
    }

    segments
}

fn next_llama_bpe_segment_end(text: &str, byte_start: usize) -> usize {
    if let Some(end) = consume_contraction(text, byte_start) {
        return end;
    }
    if let Some(end) = consume_optional_prefix_letters(text, byte_start) {
        return end;
    }

    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if is_number(ch) {
        return consume_digits(text, byte_start, 3);
    }
    if let Some(end) = consume_optional_space_punctuation(text, byte_start) {
        return end;
    }
    if let Some(end) = consume_whitespace_with_newline(text, byte_start) {
        return end;
    }
    if is_whitespace(ch) {
        return consume_whitespace_before_nonspace(text, byte_start);
    }

    byte_start + ch.len_utf8()
}

fn consume_contraction(text: &str, byte_start: usize) -> Option<usize> {
    ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
        .into_iter()
        .find_map(|suffix| {
            text[byte_start..]
                .get(..suffix.len())
                .filter(|candidate| candidate.eq_ignore_ascii_case(suffix))
                .map(|_| byte_start + suffix.len())
        })
}

fn consume_optional_prefix_letters(text: &str, byte_start: usize) -> Option<usize> {
    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if is_letter(ch) {
        return Some(consume_letters(text, byte_start));
    }
    if ch == '\r' || ch == '\n' || is_letter(ch) || is_number(ch) {
        return None;
    }

    let next_idx = byte_start + ch.len_utf8();
    let next = (next_idx < text.len()).then(|| next_char(text, next_idx))??;
    is_letter(next).then(|| consume_letters(text, next_idx))
}

fn consume_optional_space_punctuation(text: &str, byte_start: usize) -> Option<usize> {
    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    let punctuation_start = if ch == ' ' {
        let next_idx = byte_start + ch.len_utf8();
        let next = (next_idx < text.len()).then(|| next_char(text, next_idx))??;
        if is_punctuation_for_bpe(next) {
            next_idx
        } else {
            return None;
        }
    } else if is_punctuation_for_bpe(ch) {
        byte_start
    } else {
        return None;
    };

    let mut idx = punctuation_start;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_punctuation_for_bpe(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if ch != '\n' && ch != '\r' {
            break;
        }
        idx += ch.len_utf8();
    }
    Some(idx)
}

fn consume_whitespace_with_newline(text: &str, byte_start: usize) -> Option<usize> {
    let ch = next_char(text, byte_start).expect("byte_start is in-bounds");
    if !is_whitespace(ch) {
        return None;
    }

    let mut idx = byte_start;
    let mut last_newline_end = None;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_whitespace(ch) {
            break;
        }
        idx += ch.len_utf8();
        if ch == '\n' || ch == '\r' {
            last_newline_end = Some(idx);
        }
    }
    last_newline_end
}

fn consume_whitespace_before_nonspace(text: &str, byte_start: usize) -> usize {
    let whitespace_end = consume_whitespace(text, byte_start);
    if whitespace_end == text.len() {
        return whitespace_end;
    }

    let chars: Vec<(usize, char)> = text[byte_start..whitespace_end]
        .char_indices()
        .map(|(offset, ch)| (byte_start + offset, ch))
        .collect();
    if chars.len() > 1 {
        chars[chars.len() - 1].0
    } else {
        whitespace_end
    }
}

fn next_char(text: &str, byte_start: usize) -> Option<char> {
    text[byte_start..].chars().next()
}

fn is_letter(ch: char) -> bool {
    ch.is_alphabetic()
}

fn is_number(ch: char) -> bool {
    ch.is_numeric()
}

fn is_whitespace(ch: char) -> bool {
    ch.is_whitespace()
}

fn is_punctuation_for_bpe(ch: char) -> bool {
    !is_whitespace(ch) && !is_letter(ch) && !is_number(ch)
}

fn consume_letters(text: &str, byte_start: usize) -> usize {
    let mut idx = byte_start;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_letter(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn consume_digits(text: &str, byte_start: usize, max_digits: usize) -> usize {
    let mut idx = byte_start;
    let mut count = 0;
    while idx < text.len() && count < max_digits {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_number(ch) {
            break;
        }
        idx += ch.len_utf8();
        count += 1;
    }
    idx
}

fn consume_whitespace(text: &str, byte_start: usize) -> usize {
    let mut idx = byte_start;
    while idx < text.len() {
        let ch = next_char(text, idx).expect("idx is in-bounds");
        if !is_whitespace(ch) {
            break;
        }
        idx += ch.len_utf8();
    }
    idx
}

fn bpe_byte_to_char(byte: u8) -> char {
    let byte = u32::from(byte);
    if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte) {
        return char::from_u32(byte).expect("visible byte maps to Unicode scalar");
    }

    let offset = (0..byte)
        .filter(|candidate| {
            !((33..=126).contains(candidate)
                || (161..=172).contains(candidate)
                || (174..=255).contains(candidate))
        })
        .count() as u32;
    char::from_u32(256 + offset).expect("GPT-2 byte fallback maps to Unicode scalar")
}

fn bpe_char_to_byte(ch: char) -> Option<u8> {
    (0..=u8::MAX).find(|byte| bpe_byte_to_char(*byte) == ch)
}

fn validate_token_id(name: &str, id: Option<TokenId>, len: usize) -> Result<(), String> {
    if let Some(id) = id
        && id as usize >= len
    {
        return Err(format!(
            "{name} token id {id} out of range for vocab size {len}"
        ));
    }
    Ok(())
}

fn parse_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

fn flush_bytes(bytes: &mut Vec<u8>, text: &mut String) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    let decoded = String::from_utf8(std::mem::take(bytes))
        .map_err(|_| "byte fallback produced invalid UTF-8".to_string())?;
    text.push_str(&decoded);
    Ok(())
}

fn detect_chat_template_format(template: &str) -> &'static str {
    if is_llama3_instruct_template(template) {
        "llama3_instruct"
    } else if is_qwen_im_template(template) {
        "qwen_im"
    } else if is_gemma_turn_template(template) {
        "gemma_turn"
    } else if is_deepseek_r1_qwen_template(template) {
        "deepseek_r1_qwen"
    } else if is_mistral_inst_template(template) {
        "mistral_inst"
    } else if is_tinyllama_marker_template(template) {
        "tinyllama_marker"
    } else {
        "metadata_unparsed"
    }
}

fn is_tinyllama_marker_template(template: &str) -> bool {
    template.contains("<|user|>")
        && template.contains("<|assistant|>")
        && template.contains("<|system|>")
}

fn is_llama3_instruct_template(template: &str) -> bool {
    template.contains("<|start_header_id|>")
        && template.contains("<|end_header_id|>")
        && template.contains("<|eot_id|>")
}

fn is_qwen_im_template(template: &str) -> bool {
    template.contains("<|im_start|>") && template.contains("<|im_end|>")
}

fn is_gemma_turn_template(template: &str) -> bool {
    template.contains("<start_of_turn>") && template.contains("<end_of_turn>")
}

fn is_deepseek_r1_qwen_template(template: &str) -> bool {
    template.contains("<｜User｜>") && template.contains("<｜Assistant｜>")
}

fn is_mistral_inst_template(template: &str) -> bool {
    template.contains("[INST]") && template.contains("[/INST]")
}

fn render_tinyllama_marker_prompt(messages: &[ChatMessage<'_>], tokenizer: &Tokenizer) -> String {
    let eos = tokenizer
        .token_text(tokenizer.special.eos)
        .unwrap_or("</s>");
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|");
        prompt.push_str(message.role.trim());
        prompt.push_str("|>\n");
        prompt.push_str(message.content);
        prompt.push_str(eos);
        prompt.push('\n');
    }
    if messages
        .last()
        .is_none_or(|message| message.role.trim() != "assistant")
    {
        prompt.push_str("<|assistant|>\n");
    }
    prompt
}

fn render_llama3_instruct_prompt(messages: &[ChatMessage<'_>]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(message.role.trim());
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(message.content);
        prompt.push_str("<|eot_id|>");
    }
    if messages
        .last()
        .is_none_or(|message| message.role.trim() != "assistant")
    {
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    prompt
}

fn render_qwen_im_prompt(messages: &[ChatMessage<'_>]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.trim());
        prompt.push('\n');
        prompt.push_str(message.content);
        prompt.push_str("<|im_end|>\n");
    }
    if messages
        .last()
        .is_none_or(|message| message.role.trim() != "assistant")
    {
        prompt.push_str("<|im_start|>assistant\n");
    }
    prompt
}

fn render_gemma_turn_prompt(messages: &[ChatMessage<'_>]) -> String {
    let mut prompt = String::new();
    for message in messages {
        let role = match message.role.trim() {
            "assistant" => "model",
            role => role,
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(message.content);
        prompt.push_str("<end_of_turn>\n");
    }
    if messages
        .last()
        .is_none_or(|message| message.role.trim() != "assistant")
    {
        prompt.push_str("<start_of_turn>model\n");
    }
    prompt
}

fn render_deepseek_r1_qwen_prompt(messages: &[ChatMessage<'_>], tokenizer: &Tokenizer) -> String {
    let mut prompt = tokenizer
        .special
        .bos
        .and_then(|id| tokenizer.token_text(Some(id)))
        .unwrap_or("")
        .to_owned();
    let mut pending_system = String::new();

    for message in messages {
        let role = message.role.trim();
        let content = message.content.trim();
        match role {
            "system" => {
                if !pending_system.is_empty() {
                    pending_system.push_str("\n\n");
                }
                pending_system.push_str(content);
            }
            "user" => {
                if !pending_system.is_empty() {
                    prompt.push_str(&pending_system);
                    pending_system.clear();
                }
                prompt.push_str("<｜User｜>");
                prompt.push_str(content);
            }
            "assistant" => {
                prompt.push_str("<｜Assistant｜>");
                prompt.push_str(content);
                prompt.push_str("<｜end▁of▁sentence｜>");
            }
            _ => {
                prompt.push_str("<｜User｜>");
                prompt.push_str(content);
            }
        }
    }
    if messages
        .last()
        .is_none_or(|message| message.role.trim() != "assistant")
    {
        prompt.push_str("<｜Assistant｜><think>\n");
    }
    prompt
}

fn render_mistral_inst_prompt(messages: &[ChatMessage<'_>]) -> String {
    let mut prompt = String::new();
    let mut pending_system = String::new();

    for message in messages {
        let role = message.role.trim();
        let content = message.content.trim();
        match role {
            "system" => {
                if !pending_system.is_empty() {
                    pending_system.push_str("\n\n");
                }
                pending_system.push_str(content);
            }
            "user" => {
                let content = if pending_system.is_empty() {
                    content.to_owned()
                } else {
                    let combined = format!("{pending_system}\n\n{content}");
                    pending_system.clear();
                    combined
                };
                prompt.push_str("<s>[INST] ");
                prompt.push_str(content.trim());
                prompt.push_str(" [/INST]");
            }
            "assistant" => {
                prompt.push(' ');
                prompt.push_str(content);
                prompt.push_str("</s>");
            }
            _ => {
                prompt.push_str("<s>[INST] ");
                prompt.push_str(content);
                prompt.push_str(" [/INST]");
            }
        }
    }

    if prompt.is_empty() && !pending_system.is_empty() {
        prompt.push_str("<s>[INST] ");
        prompt.push_str(pending_system.trim());
        prompt.push_str(" [/INST]");
    }

    prompt
}

fn render_role_colon_prompt(messages: &[ChatMessage<'_>]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str(message.role.trim());
        prompt.push_str(": ");
        prompt.push_str(message.content);
        prompt.push('\n');
    }
    prompt
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, HashMap},
        path::PathBuf,
    };

    use crate::gguf::{GgufFile, GgufMetadataValue};

    use super::{
        BpeRegistry, ChatMessage, RenderedChatPrompt, SpecialTokens, Token, TokenKind, Tokenizer,
        TokenizerConfig, TokenizerModel,
    };

    #[test]
    fn tokenizer_reads_native_bool_config_flags() {
        let tokenizer = Tokenizer::from_gguf(&tokenizer_fixture([
            (
                "tokenizer.ggml.add_bos_token",
                GgufMetadataValue::Bool(false),
            ),
            (
                "tokenizer.ggml.add_eos_token",
                GgufMetadataValue::Bool(true),
            ),
            (
                "tokenizer.ggml.add_space_prefix",
                GgufMetadataValue::Bool(false),
            ),
            (
                "tokenizer.ggml.remove_extra_whitespaces",
                GgufMetadataValue::Bool(true),
            ),
        ]))
        .expect("fixture tokenizer should load");

        assert_eq!(tokenizer.model, TokenizerModel::LlamaSpm);
        assert!(!tokenizer.config.add_bos);
        assert!(tokenizer.config.add_eos);
        assert!(!tokenizer.config.add_space_prefix);
        assert!(tokenizer.config.remove_extra_whitespaces);
    }

    #[test]
    fn tokenizer_preserves_string_flag_compatibility() {
        let tokenizer = Tokenizer::from_gguf(&tokenizer_fixture([
            (
                "tokenizer.ggml.add_bos_token",
                GgufMetadataValue::String("0".to_owned()),
            ),
            (
                "tokenizer.ggml.add_space_prefix",
                GgufMetadataValue::String("true".to_owned()),
            ),
        ]))
        .expect("fixture tokenizer should load");

        assert!(!tokenizer.config.add_bos);
        assert!(tokenizer.config.add_space_prefix);
        assert!(!tokenizer.config.add_eos);
    }

    #[test]
    fn tokenizer_accepts_qwen2_bpe_pre_tokenizer() {
        let tokenizer = Tokenizer::from_gguf(&tokenizer_fixture([
            (
                "tokenizer.ggml.model",
                GgufMetadataValue::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.pre",
                GgufMetadataValue::String("qwen2".to_owned()),
            ),
            (
                "tokenizer.ggml.add_bos_token",
                GgufMetadataValue::Bool(false),
            ),
        ]))
        .expect("qwen2 BPE tokenizer should load");

        assert_eq!(tokenizer.model, TokenizerModel::Gpt2Bpe);
        assert!(!tokenizer.config.add_bos);
    }

    #[test]
    fn tokenizer_accepts_missing_llama_bpe_pre_tokenizer_for_llama_arch() {
        let tokenizer = Tokenizer::from_gguf(&tokenizer_fixture([
            (
                "general.architecture",
                GgufMetadataValue::String("llama".to_owned()),
            ),
            (
                "tokenizer.ggml.model",
                GgufMetadataValue::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.tokens",
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::String("<unk>".to_owned()),
                    GgufMetadataValue::String("<|begin_of_text|>".to_owned()),
                    GgufMetadataValue::String("<|end_of_text|>".to_owned()),
                    GgufMetadataValue::String("<|start_header_id|>".to_owned()),
                    GgufMetadataValue::String("<|end_header_id|>".to_owned()),
                    GgufMetadataValue::String("<|eot_id|>".to_owned()),
                    GgufMetadataValue::String("hello".to_owned()),
                ]),
            ),
            (
                "tokenizer.ggml.token_type",
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::I32(2),
                    GgufMetadataValue::I32(3),
                    GgufMetadataValue::I32(3),
                    GgufMetadataValue::I32(3),
                    GgufMetadataValue::I32(3),
                    GgufMetadataValue::I32(3),
                    GgufMetadataValue::I32(1),
                ]),
            ),
            (
                "tokenizer.ggml.scores",
                GgufMetadataValue::Array(vec![
                    GgufMetadataValue::F32(0.0),
                    GgufMetadataValue::F32(0.0),
                    GgufMetadataValue::F32(0.0),
                    GgufMetadataValue::F32(0.0),
                    GgufMetadataValue::F32(0.0),
                    GgufMetadataValue::F32(0.0),
                    GgufMetadataValue::F32(0.0),
                ]),
            ),
            (
                "tokenizer.ggml.add_bos_token",
                GgufMetadataValue::Bool(false),
            ),
        ]))
        .expect("Llama GPT-2/BPE tokenizer without pre metadata should load");

        assert_eq!(tokenizer.model, TokenizerModel::Gpt2Bpe);
        assert_eq!(tokenizer.chat_template_format(), Some("llama3_instruct"));
        assert_eq!(
            tokenizer.render_chat_prompt(&[ChatMessage {
                role: "user",
                content: "hello",
            }]),
            RenderedChatPrompt {
                text: "<|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".to_owned(),
                add_special: true,
                parse_special: true,
                renderer: "llama3_instruct",
            }
        );
        assert!(!tokenizer.config.add_bos);
    }

    #[test]
    fn tokenizer_rejects_missing_bpe_pre_tokenizer_for_non_llama_arch() {
        let err = Tokenizer::from_gguf(&tokenizer_fixture([
            (
                "general.architecture",
                GgufMetadataValue::String("qwen2".to_owned()),
            ),
            (
                "tokenizer.ggml.model",
                GgufMetadataValue::String("gpt2".to_owned()),
            ),
        ]))
        .expect_err("non-Llama GPT-2/BPE tokenizer without pre metadata should fail");

        assert!(err.contains("unsupported GPT-2/BPE pre-tokenizer: None"));
    }

    #[test]
    fn tokenizer_accepts_deepseek_r1_qwen_bpe_pre_tokenizer() {
        let tokenizer = Tokenizer::from_gguf(&tokenizer_fixture([
            (
                "tokenizer.ggml.model",
                GgufMetadataValue::String("gpt2".to_owned()),
            ),
            (
                "tokenizer.ggml.pre",
                GgufMetadataValue::String("deepseek-r1-qwen".to_owned()),
            ),
            (
                "tokenizer.ggml.add_bos_token",
                GgufMetadataValue::Bool(false),
            ),
        ]))
        .expect("DeepSeek R1 Qwen BPE tokenizer should load");

        assert_eq!(tokenizer.model, TokenizerModel::Gpt2Bpe);
        assert!(!tokenizer.config.add_bos);
    }

    #[test]
    fn tokenizer_accepts_small_model_bpe_pre_tokenizers() {
        for pre_tokenizer in ["smollm", "smaug-bpe", "lfm2"] {
            let tokenizer = Tokenizer::from_gguf(&tokenizer_fixture([
                (
                    "tokenizer.ggml.model",
                    GgufMetadataValue::String("gpt2".to_owned()),
                ),
                (
                    "tokenizer.ggml.pre",
                    GgufMetadataValue::String(pre_tokenizer.to_owned()),
                ),
                (
                    "tokenizer.ggml.add_bos_token",
                    GgufMetadataValue::Bool(false),
                ),
            ]))
            .expect("small-model BPE tokenizer should load");

            assert_eq!(tokenizer.model, TokenizerModel::Gpt2Bpe);
            assert!(!tokenizer.config.add_bos);
        }
    }

    #[test]
    fn tokenizer_detects_llama3_chat_template_format() {
        let tokenizer = llama3_test_tokenizer();

        assert_eq!(tokenizer.chat_template_format(), Some("llama3_instruct"));
    }

    #[test]
    fn tokenizer_renders_llama3_single_turn_chat_prompt() {
        let tokenizer = llama3_test_tokenizer();

        assert_eq!(
            tokenizer.render_chat_prompt(&[ChatMessage {
                role: "user",
                content: "Say alpha.",
            }]),
            RenderedChatPrompt {
                text: "<|start_header_id|>user<|end_header_id|>\n\nSay alpha.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n".to_owned(),
                add_special: true,
                parse_special: true,
                renderer: "llama3_instruct",
            }
        );
    }

    #[test]
    fn tokenizer_renders_llama3_chat_prompt_without_extra_generation_header_after_assistant() {
        let tokenizer = llama3_test_tokenizer();

        assert_eq!(
            tokenizer.render_chat_prompt(&[
                ChatMessage {
                    role: "user",
                    content: "Complete cam",
                },
                ChatMessage {
                    role: "assistant",
                    content: "elid",
                },
            ]),
            RenderedChatPrompt {
                text: "<|start_header_id|>user<|end_header_id|>\n\nComplete cam<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nelid<|eot_id|>".to_owned(),
                add_special: true,
                parse_special: true,
                renderer: "llama3_instruct",
            }
        );
    }

    #[test]
    fn tokenizer_renders_qwen_im_chat_prompt() {
        let mut tokenizer = llama3_test_tokenizer();
        tokenizer.chat_template =
            Some("<|im_start|>{{ role }}\n{{ content }}<|im_end|>".to_owned());

        assert_eq!(tokenizer.chat_template_format(), Some("qwen_im"));
        assert_eq!(
            tokenizer.render_chat_prompt(&[ChatMessage {
                role: "user",
                content: "Write fizzbuzz.",
            }]),
            RenderedChatPrompt {
                text: "<|im_start|>user\nWrite fizzbuzz.<|im_end|>\n<|im_start|>assistant\n"
                    .to_owned(),
                add_special: true,
                parse_special: true,
                renderer: "qwen_im",
            }
        );
    }

    #[test]
    fn tokenizer_renders_qwen_im_without_metadata_template_when_tokens_exist() {
        let mut tokenizer = llama3_test_tokenizer();
        tokenizer.chat_template = None;
        tokenizer
            .token_to_id
            .insert("<|im_start|>".to_owned(), 1000);
        tokenizer.token_to_id.insert("<|im_end|>".to_owned(), 1001);

        assert_eq!(
            tokenizer.chat_template_format(),
            Some("qwen_im_token_fallback")
        );
        assert_eq!(
            tokenizer.render_chat_prompt(&[ChatMessage {
                role: "user",
                content: "Write fizzbuzz.",
            }]),
            RenderedChatPrompt {
                text: "<|im_start|>user\nWrite fizzbuzz.<|im_end|>\n<|im_start|>assistant\n"
                    .to_owned(),
                add_special: true,
                parse_special: true,
                renderer: "qwen_im_token_fallback",
            }
        );
    }

    #[test]
    fn tokenizer_renders_gemma_turn_chat_prompt() {
        let mut tokenizer = llama3_test_tokenizer();
        tokenizer.chat_template =
            Some("{{ '<start_of_turn>' + role }}{{ '<end_of_turn>' }}".to_owned());

        assert_eq!(tokenizer.chat_template_format(), Some("gemma_turn"));
        assert_eq!(
            tokenizer.render_chat_prompt(&[ChatMessage {
                role: "user",
                content: "Say hello.",
            }]),
            RenderedChatPrompt {
                text: "<start_of_turn>user\nSay hello.<end_of_turn>\n<start_of_turn>model\n"
                    .to_owned(),
                add_special: true,
                parse_special: true,
                renderer: "gemma_turn",
            }
        );
    }

    #[test]
    fn tokenizer_renders_deepseek_r1_qwen_chat_prompt() {
        let mut tokenizer = llama3_test_tokenizer();
        tokenizer.chat_template =
            Some("{{ bos_token }}{{ '<｜User｜>' + message['content'] }}{{ '<｜Assistant｜><think>\\n' }}".to_owned());

        assert_eq!(tokenizer.chat_template_format(), Some("deepseek_r1_qwen"));
        assert_eq!(
            tokenizer.render_chat_prompt(&[
                ChatMessage {
                    role: "system",
                    content: "be brief",
                },
                ChatMessage {
                    role: "user",
                    content: "Say hello.",
                },
            ]),
            RenderedChatPrompt {
                text: "<|begin_of_text|>be brief<｜User｜>Say hello.<｜Assistant｜><think>\n"
                    .to_owned(),
                add_special: false,
                parse_special: true,
                renderer: "deepseek_r1_qwen",
            }
        );
    }

    #[test]
    fn tokenizer_falls_back_to_role_colon_prompt_for_unparsed_templates() {
        let mut tokenizer = llama3_test_tokenizer();
        tokenizer.chat_template = Some("{{ messages }}".to_owned());

        assert_eq!(
            tokenizer.render_chat_prompt(&[ChatMessage {
                role: "user",
                content: "hello",
            }]),
            RenderedChatPrompt {
                text: "user: hello\n".to_owned(),
                add_special: true,
                parse_special: true,
                renderer: "role_colon_fallback",
            }
        );
        assert_eq!(tokenizer.chat_template_format(), Some("metadata_unparsed"));
    }

    #[test]
    fn tokenizer_renders_mistral_inst_prompt() {
        let mut tokenizer = llama3_test_tokenizer();
        tokenizer.chat_template =
            Some("{{ bos_token }}[INST] {{ message['content'] }} [/INST]".to_owned());

        assert_eq!(
            tokenizer.render_chat_prompt(&[ChatMessage {
                role: "user",
                content: "hello",
            }]),
            RenderedChatPrompt {
                text: "<s>[INST] hello [/INST]".to_owned(),
                add_special: false,
                parse_special: true,
                renderer: "mistral_inst",
            }
        );
        assert_eq!(tokenizer.chat_template_format(), Some("mistral_inst"));
    }

    #[test]
    fn tokenizer_renders_mistral_inst_without_metadata_template_when_tokens_exist() {
        let mut tokenizer = llama3_test_tokenizer();
        tokenizer.model = TokenizerModel::LlamaSpm;
        tokenizer.source_name = Some("mistralai_mistral-7b-instruct-v0.1".to_owned());
        tokenizer.chat_template = None;

        assert_eq!(
            tokenizer.chat_template_format(),
            Some("mistral_inst_token_fallback")
        );
        assert_eq!(
            tokenizer.render_chat_prompt(&[ChatMessage {
                role: "user",
                content: "hello",
            }]),
            RenderedChatPrompt {
                text: "<s>[INST] hello [/INST]".to_owned(),
                add_special: false,
                parse_special: false,
                renderer: "mistral_inst_token_fallback",
            }
        );
    }

    #[test]
    fn tokenizer_renders_mistral_inst_with_system_and_history() {
        let mut tokenizer = llama3_test_tokenizer();
        tokenizer.chat_template = Some("[INST] {{ messages }} [/INST]".to_owned());

        assert_eq!(
            tokenizer.render_chat_prompt(&[
                ChatMessage {
                    role: "system",
                    content: "be brief",
                },
                ChatMessage {
                    role: "user",
                    content: "hello",
                },
                ChatMessage {
                    role: "assistant",
                    content: "hi",
                },
                ChatMessage {
                    role: "user",
                    content: "next",
                },
            ]),
            RenderedChatPrompt {
                text: "<s>[INST] be brief\n\nhello [/INST] hi</s><s>[INST] next [/INST]".to_owned(),
                add_special: false,
                parse_special: true,
                renderer: "mistral_inst",
            }
        );
    }

    fn llama3_test_tokenizer() -> Tokenizer {
        let begin_of_text = "<|begin_of_text|>".to_owned();
        let end_of_text = "<|end_of_text|>".to_owned();
        let start_header = "<|start_header_id|>".to_owned();
        let end_header = "<|end_header_id|>".to_owned();
        let eot = "<|eot_id|>".to_owned();
        let tokens = vec![
            Token {
                id: 0,
                text: begin_of_text.clone(),
                score: 0.0,
                kind: TokenKind::Control,
            },
            Token {
                id: 1,
                text: end_of_text.clone(),
                score: 0.0,
                kind: TokenKind::Control,
            },
            Token {
                id: 2,
                text: start_header.clone(),
                score: 0.0,
                kind: TokenKind::Control,
            },
            Token {
                id: 3,
                text: end_header.clone(),
                score: 0.0,
                kind: TokenKind::Control,
            },
            Token {
                id: 4,
                text: eot.clone(),
                score: 0.0,
                kind: TokenKind::Control,
            },
        ];
        let token_to_id = HashMap::from([
            (begin_of_text, 0),
            (end_of_text, 1),
            (start_header, 2),
            (end_header, 3),
            (eot, 4),
        ]);

        Tokenizer {
            source_name: None,
            model: TokenizerModel::Gpt2Bpe,
            tokens,
            token_to_id,
            byte_token_to_id: HashMap::new(),
            bpe_ranks: HashMap::new(),
            bpe_registry: BpeRegistry::default(),
            special: SpecialTokens {
                bos: Some(0),
                eos: Some(1),
                eot: Some(4),
                ..SpecialTokens::default()
            },
            config: TokenizerConfig {
                add_bos: true,
                add_eos: false,
                add_sep: false,
                add_space_prefix: false,
                remove_extra_whitespaces: false,
            },
            chat_template: Some(
                "<|start_header_id|>{{ role }}<|end_header_id|>{{ content }}<|eot_id|>".to_owned(),
            ),
        }
    }

    fn tokenizer_fixture<const N: usize>(overrides: [(&str, GgufMetadataValue); N]) -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            GgufMetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.model".to_owned(),
            GgufMetadataValue::String("llama".to_owned()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_owned(),
            GgufMetadataValue::Array(vec![
                GgufMetadataValue::String("<unk>".to_owned()),
                GgufMetadataValue::String("<s>".to_owned()),
                GgufMetadataValue::String("</s>".to_owned()),
                GgufMetadataValue::String("▁hello".to_owned()),
                GgufMetadataValue::String("hello".to_owned()),
                GgufMetadataValue::String("▁".to_owned()),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_owned(),
            GgufMetadataValue::Array(vec![
                GgufMetadataValue::F32(0.0),
                GgufMetadataValue::F32(0.0),
                GgufMetadataValue::F32(0.0),
                GgufMetadataValue::F32(10.0),
                GgufMetadataValue::F32(2.0),
                GgufMetadataValue::F32(1.0),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_owned(),
            GgufMetadataValue::Array(vec![
                GgufMetadataValue::I32(2),
                GgufMetadataValue::I32(3),
                GgufMetadataValue::I32(3),
                GgufMetadataValue::I32(1),
                GgufMetadataValue::I32(1),
                GgufMetadataValue::I32(1),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.bos_token_id".to_owned(),
            GgufMetadataValue::U32(1),
        );
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_owned(),
            GgufMetadataValue::U32(2),
        );

        for (key, value) in overrides {
            metadata.insert(key.to_owned(), value);
        }

        GgufFile {
            path: PathBuf::from("tokenizer-fixture.gguf"),
            version: 3,
            tensor_count: 0,
            metadata_count: metadata.len() as u64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: Vec::new(),
        }
    }
}
