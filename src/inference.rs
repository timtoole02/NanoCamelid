use crate::model::{
    LlamaFfnWeights, LlamaLayerWeights, LlamaModelConfig, LlamaWeights, MoeExpertWeights, RopeStyle,
    PageAlignedQ4_0Swizzled1x4, PageAlignedQ8_0Swizzled1x4, QuantizedMatrix,
};
use crate::q8::{
    IQ4NLBlock, Q2KBlock, Q3KBlock, Q4_0Block, Q4_1Block, Q5KBlock, Q6KBlock, Q8_0Block,
    Q8_BLOCK_SIZE, Q8DotKernel, Q8DotKernelSelector, Q8KBlock, QK_K_BLOCK_SIZE,
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    env,
    ffi::c_void,
    hash::{Hash, Hasher},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

pub const MATMUL_MIN_ROWS_ENV: &str = "NANOCAMELID_MATMUL_MIN_ROWS";
pub const Q4_1X4_SDOT_ENV: &str = "NANOCAMELID_Q4_1X4_SDOT";
pub const Q6K_SDOT_ENV: &str = "NANOCAMELID_Q6K_SDOT";
pub const ATTENTION_HEAD_PARALLEL_ENV: &str = "NANOCAMELID_ATTENTION_HEAD_PARALLEL";
pub const KV_CACHE_F16_ENV: &str = "NANOCAMELID_KV_CACHE_F16";
pub const KV_CACHE_Q8_ENV: &str = "NANOCAMELID_KV_CACHE_Q8";
pub const ROPE_CACHE_ENV: &str = "NANOCAMELID_ROPE_CACHE";
pub const TRACE_ENV: &str = "NANOCAMELID_TRACE";
const DEFAULT_MATMUL_MIN_ROWS: usize = 128;
const MAX_STACK_BATCH_SUMS: usize = 64;

fn matmul_min_rows() -> usize {
    static MIN_ROWS: OnceLock<usize> = OnceLock::new();
    *MIN_ROWS.get_or_init(|| {
        env::var(MATMUL_MIN_ROWS_ENV)
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_MATMUL_MIN_ROWS)
    })
}

fn should_parallelize_matmul(rows: usize) -> bool {
    rows >= matmul_min_rows()
}

fn q4_1x4_sdot_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(Q4_1X4_SDOT_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(true)
    })
}

fn q6_k_sdot_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(Q6K_SDOT_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(true)
    })
}

fn attention_head_parallel_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(ATTENTION_HEAD_PARALLEL_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(true)
    })
}

fn kv_cache_f16_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(KV_CACHE_F16_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(false)
    })
}

fn kv_cache_q8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(KV_CACHE_Q8_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(false)
    })
}

fn rope_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(ROPE_CACHE_ENV)
            .ok()
            .map(|value| {
                !matches!(
                    value.as_str(),
                    "0" | "false" | "FALSE" | "off" | "OFF" | "no"
                )
            })
            .unwrap_or(true)
    })
}

fn trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(TRACE_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(false)
    })
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TraceStats {
    pub calls: u64,
    pub total: Duration,
}

fn trace_store() -> &'static Mutex<HashMap<&'static str, TraceStats>> {
    static STORE: OnceLock<Mutex<HashMap<&'static str, TraceStats>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[inline]
fn trace_record(stage: &'static str, elapsed: Duration) {
    if !trace_enabled() {
        return;
    }
    let Ok(mut store) = trace_store().lock() else {
        return;
    };
    let stats = store.entry(stage).or_default();
    stats.calls += 1;
    stats.total += elapsed;
}

pub fn trace_reset() {
    if let Ok(mut store) = trace_store().lock() {
        store.clear();
    }
}

pub fn trace_snapshot() -> Vec<(&'static str, TraceStats)> {
    let Ok(store) = trace_store().lock() else {
        return Vec::new();
    };
    let mut rows = store
        .iter()
        .map(|(&stage, &stats)| (stage, stats))
        .collect::<Vec<_>>();
    rows.sort_by_key(|(_, stats)| std::cmp::Reverse(stats.total));
    rows
}

pub struct WorkerState {
    pub task_id: AtomicU64,
    pub completed_id: AtomicU64,
    pub active: AtomicBool,
}

pub struct SpinThreadPool {
    _workers: Vec<std::thread::JoinHandle<()>>,
    states: Vec<Arc<WorkerState>>,
    _terminated: Arc<AtomicBool>,
    pub thread_count: usize,
}

type MatmulRowFn = unsafe fn(*const c_void, usize, *mut f32);

struct MatmulWork {
    compute_row: *const c_void,
    compute: MatmulRowFn,
    out: *mut f32,
    len: usize,
}

static ACTIVE_WORK: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

fn active_work_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

unsafe fn compute_matmul_row<F>(compute_row: *const c_void, row: usize, out: *mut f32)
where
    F: Fn(usize, &mut f32) + Sync,
{
    let compute_row = unsafe { &*(compute_row as *const F) };
    let out = unsafe { &mut *out };
    compute_row(row, out);
}

fn get_asymmetric_slice(len: usize, thread_idx: usize, thread_count: usize) -> (usize, usize) {
    if thread_count == 3 {
        // Asymmetric partition: Master (thread_idx = 0) gets 10%, Workers 0, 1, 2 get 30% each.
        let master_share = len / 10;
        let remaining = len - master_share;
        let worker_share = remaining / 3;

        if thread_idx == 0 {
            (0, master_share)
        } else {
            let w_idx = thread_idx - 1;
            let start = master_share + w_idx * worker_share;
            let end = if w_idx == 2 {
                len
            } else {
                std::cmp::min(start + worker_share, len)
            };
            (start, end)
        }
    } else {
        // Uniform fallback if thread count is different
        let chunk_size = (len + thread_count) / (thread_count + 1);
        let start = thread_idx * chunk_size;
        let end = std::cmp::min(start + chunk_size, len);
        (start, end)
    }
}

impl SpinThreadPool {
    pub fn new(thread_count: usize) -> Self {
        let terminated = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::with_capacity(thread_count);
        let mut states = Vec::with_capacity(thread_count);

        for i in 0..thread_count {
            let state = Arc::new(WorkerState {
                task_id: AtomicU64::new(0),
                completed_id: AtomicU64::new(0),
                active: AtomicBool::new(false),
            });
            states.push(state.clone());

            let term = terminated.clone();
            let state_clone = state.clone();
            let handle = std::thread::spawn(move || {
                // Pin worker threads strictly to isolated cores using core_affinity
                let core_ids = core_affinity::get_core_ids().unwrap_or_default();
                if i + 1 < core_ids.len() {
                    core_affinity::set_for_current(core_ids[i + 1]);
                }

                // Tight, zero-latency spin loop
                while !term.load(Ordering::Relaxed) {
                    let task = state_clone.task_id.load(Ordering::Acquire);
                    let completed = state_clone.completed_id.load(Ordering::Relaxed);

                    if task > completed {
                        if state_clone.active.load(Ordering::Relaxed) {
                            let work_ptr = ACTIVE_WORK.load(Ordering::Acquire);
                            if !work_ptr.is_null() {
                                let work = unsafe { &*(work_ptr as *const MatmulWork) };
                                let (start, end) =
                                    get_asymmetric_slice(work.len, i + 1, thread_count);
                                if start < end {
                                    for r in start..end {
                                        unsafe {
                                            (work.compute)(work.compute_row, r, work.out.add(r));
                                        }
                                    }
                                }
                            }
                        }
                        state_clone.completed_id.store(task, Ordering::Release);
                    } else {
                        std::hint::spin_loop();
                    }
                }
            });
            workers.push(handle);
        }

        Self {
            _workers: workers,
            states,
            _terminated: terminated,
            thread_count,
        }
    }
}

fn get_spin_pool() -> Option<&'static SpinThreadPool> {
    static INITIALIZED: OnceLock<Option<SpinThreadPool>> = OnceLock::new();
    INITIALIZED
        .get_or_init(|| {
            let enabled = env::var("NANOCAMELID_SPIN_POOL")
                .ok()
                .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
                .unwrap_or_else(|| cfg!(target_arch = "aarch64"));

            if enabled {
                let cores = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);
                let thread_count = if cores > 1 { cores - 1 } else { 1 };
                Some(SpinThreadPool::new(thread_count))
            } else {
                None
            }
        })
        .as_ref()
}

fn for_each_matmul_row<F>(out: &mut [f32], compute_row: F)
where
    F: Fn(usize, &mut f32) + Sync,
{
    if should_parallelize_matmul(out.len()) {
        if let Some(pool) = get_spin_pool() {
            let Ok(_work_guard) = active_work_lock().lock() else {
                out.par_iter_mut()
                    .with_min_len(matmul_min_rows())
                    .enumerate()
                    .for_each(|(r, out_val)| compute_row(r, out_val));
                return;
            };
            let work = MatmulWork {
                compute_row: &compute_row as *const F as *const c_void,
                compute: compute_matmul_row::<F>,
                out: out.as_mut_ptr(),
                len: out.len(),
            };
            ACTIVE_WORK.store(&work as *const _ as *mut c_void, Ordering::Release);

            // Master thread increments TASK_ID to start workers
            static TASK_ID: AtomicU64 = AtomicU64::new(1);
            let current_task = TASK_ID.fetch_add(1, Ordering::Relaxed);

            // Dispatch task to workers
            for state in &pool.states {
                state.active.store(true, Ordering::Relaxed);
                state.task_id.store(current_task, Ordering::Release);
            }

            // Master thread handles chunk 0 directly on the calling thread!
            let (master_start, master_end) = get_asymmetric_slice(out.len(), 0, pool.thread_count);
            for (r, out_val) in out
                .iter_mut()
                .enumerate()
                .take(master_end)
                .skip(master_start)
            {
                compute_row(r, out_val);
            }

            // Master thread waits for all workers to complete in nanoseconds
            for state in &pool.states {
                while state.completed_id.load(Ordering::Acquire) < current_task {
                    std::hint::spin_loop();
                }
                state.active.store(false, Ordering::Relaxed);
            }

            ACTIVE_WORK.store(std::ptr::null_mut(), Ordering::Release);
            return;
        }

        // Rayon fallback
        out.par_iter_mut()
            .with_min_len(matmul_min_rows())
            .enumerate()
            .for_each(|(r, out_val)| compute_row(r, out_val));
    } else {
        out.iter_mut()
            .enumerate()
            .for_each(|(r, out_val)| compute_row(r, out_val));
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RopeScaling {
    pub factor: Option<f32>,
    pub original_context_length: Option<f32>,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct LlamaRuntimeOptions {
    pub q8_selector: Q8DotKernelSelector,
    pub rope_scaling: RopeScaling,
    pub compute_logits: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct BatchMatmulShape {
    pub batch_size: usize,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy)]
pub enum KvCacheSlice<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
    Q8_0(&'a [Q8_0Block]),
}

enum LlamaKvCacheStorage {
    F32 {
        k_cache: Vec<f32>,
        v_cache: Vec<f32>,
    },
    F16 {
        k_cache: Vec<u16>,
        v_cache: Vec<u16>,
    },
    Q8_0 {
        k_cache: Vec<Q8_0Block>,
        v_cache: Vec<Q8_0Block>,
    },
}

pub struct LlamaKvCache {
    storage: LlamaKvCacheStorage,
    pub kv_width: usize,
    pub max_seq_len: usize,
}

impl LlamaKvCache {
    pub fn new(block_count: usize, max_seq_len: usize, kv_width: usize) -> Self {
        let size = block_count * max_seq_len * kv_width;
        let storage = if kv_cache_q8_enabled() && kv_width.is_multiple_of(Q8_BLOCK_SIZE) {
            let block_size = block_count * max_seq_len * (kv_width / Q8_BLOCK_SIZE);
            let zero_block = Q8_0Block::from_parts(0, [0i8; Q8_BLOCK_SIZE]);
            LlamaKvCacheStorage::Q8_0 {
                k_cache: vec![zero_block; block_size],
                v_cache: vec![zero_block; block_size],
            }
        } else if kv_cache_f16_enabled() {
            LlamaKvCacheStorage::F16 {
                k_cache: vec![0; size],
                v_cache: vec![0; size],
            }
        } else {
            LlamaKvCacheStorage::F32 {
                k_cache: vec![0.0; size],
                v_cache: vec![0.0; size],
            }
        };
        Self {
            storage,
            kv_width,
            max_seq_len,
        }
    }

    pub fn store_kv(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        let pos_offset = pos * self.kv_width;
        let start = layer_offset + pos_offset;

        match &mut self.storage {
            LlamaKvCacheStorage::F32 { k_cache, v_cache } => {
                k_cache[start..start + self.kv_width].copy_from_slice(k);
                v_cache[start..start + self.kv_width].copy_from_slice(v);
            }
            LlamaKvCacheStorage::F16 { k_cache, v_cache } => {
                for idx in 0..self.kv_width {
                    k_cache[start + idx] = crate::q8::fast_f32_to_f16(k[idx]);
                    v_cache[start + idx] = crate::q8::fast_f32_to_f16(v[idx]);
                }
            }
            LlamaKvCacheStorage::Q8_0 { k_cache, v_cache } => {
                let blocks_per_width = self.kv_width / Q8_BLOCK_SIZE;
                let block_base = (layer * self.max_seq_len + pos) * blocks_per_width;
                quantize_row_to_q8_0_blocks(
                    k,
                    &mut k_cache[block_base..block_base + blocks_per_width],
                );
                quantize_row_to_q8_0_blocks(
                    v,
                    &mut v_cache[block_base..block_base + blocks_per_width],
                );
            }
        }
    }

    pub fn get_k_cache(&self, layer: usize) -> KvCacheSlice<'_> {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        let range = layer_offset..layer_offset + self.max_seq_len * self.kv_width;
        match &self.storage {
            LlamaKvCacheStorage::F32 { k_cache, .. } => KvCacheSlice::F32(&k_cache[range]),
            LlamaKvCacheStorage::F16 { k_cache, .. } => KvCacheSlice::F16(&k_cache[range]),
            LlamaKvCacheStorage::Q8_0 { k_cache, .. } => {
                let blocks_per_width = self.kv_width / Q8_BLOCK_SIZE;
                let base = layer * self.max_seq_len * blocks_per_width;
                let block_range = base..base + self.max_seq_len * blocks_per_width;
                KvCacheSlice::Q8_0(&k_cache[block_range])
            }
        }
    }

    pub fn get_v_cache(&self, layer: usize) -> KvCacheSlice<'_> {
        let layer_offset = layer * self.max_seq_len * self.kv_width;
        let range = layer_offset..layer_offset + self.max_seq_len * self.kv_width;
        match &self.storage {
            LlamaKvCacheStorage::F32 { v_cache, .. } => KvCacheSlice::F32(&v_cache[range]),
            LlamaKvCacheStorage::F16 { v_cache, .. } => KvCacheSlice::F16(&v_cache[range]),
            LlamaKvCacheStorage::Q8_0 { v_cache, .. } => {
                let blocks_per_width = self.kv_width / Q8_BLOCK_SIZE;
                let base = layer * self.max_seq_len * blocks_per_width;
                let block_range = base..base + self.max_seq_len * blocks_per_width;
                KvCacheSlice::Q8_0(&v_cache[block_range])
            }
        }
    }
}

pub struct LlamaWorkspace {
    pub hidden: Vec<f32>,
    pub residual: Vec<f32>,
    pub norm_x: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_gate_up: Vec<f32>,
    pub x_i8: Vec<i8>,
    pub x_scales: Vec<f32>,
    pub logits: Vec<f32>,
}

pub struct LlamaBatchWorkspace {
    pub max_batch: usize,
    pub hidden: Vec<f32>,
    pub residual: Vec<f32>,
    pub norm_x: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_output: Vec<f32>,
    pub attn_scores: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_gate_up: Vec<f32>,
    pub x_i8: Vec<i8>,
    pub x_scales: Vec<f32>,
}

impl LlamaWorkspace {
    pub fn new(config: &LlamaModelConfig) -> Self {
        let max_size = config
            .vocab_size
            .max(config.feed_forward_length)
            .max(config.embedding_length)
            .max(config.attention_output_width);
        let attn_score_len = if attention_head_parallel_enabled() {
            config.attention_head_count * config.context_length
        } else {
            config.context_length
        };
        Self {
            hidden: vec![0.0; config.embedding_length],
            residual: vec![0.0; config.embedding_length],
            norm_x: vec![0.0; config.embedding_length],
            q: vec![0.0; config.attention_output_width],
            k: vec![0.0; config.kv_width],
            v: vec![0.0; config.kv_width],
            attn_output: vec![0.0; config.attention_output_width],
            attn_scores: vec![0.0; attn_score_len],
            ffn_gate: vec![0.0; config.feed_forward_length],
            ffn_up: vec![0.0; config.feed_forward_length],
            ffn_gate_up: vec![0.0; config.feed_forward_length],
            x_i8: vec![0; max_size],
            x_scales: vec![0.0; max_size / 32 + 1],
            logits: vec![0.0; config.vocab_size],
        }
    }
}

impl LlamaBatchWorkspace {
    pub fn new(config: &LlamaModelConfig, max_batch: usize) -> Self {
        let max_size = config
            .vocab_size
            .max(config.feed_forward_length)
            .max(config.embedding_length)
            .max(config.attention_output_width);
        let attn_score_len = if attention_head_parallel_enabled() {
            max_batch * config.attention_head_count * config.context_length
        } else {
            max_batch * config.context_length
        };
        Self {
            max_batch,
            hidden: vec![0.0; max_batch * config.embedding_length],
            residual: vec![0.0; max_batch * config.embedding_length],
            norm_x: vec![0.0; max_batch * config.embedding_length],
            q: vec![0.0; max_batch * config.attention_output_width],
            k: vec![0.0; max_batch * config.kv_width],
            v: vec![0.0; max_batch * config.kv_width],
            attn_output: vec![0.0; max_batch * config.attention_output_width],
            attn_scores: vec![0.0; attn_score_len],
            ffn_gate: vec![0.0; max_batch * config.feed_forward_length],
            ffn_up: vec![0.0; max_batch * config.feed_forward_length],
            ffn_gate_up: vec![0.0; max_batch * config.feed_forward_length],
            x_i8: vec![0; max_batch * max_size],
            x_scales: vec![0.0; max_batch * (max_size / Q8_BLOCK_SIZE + 1)],
        }
    }
}

pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], epsilon: f32) {
    let mut sum = 0.0_f32;
    for &val in x {
        sum += val * val;
    }
    let scale = 1.0_f32 / (sum / x.len() as f32 + epsilon).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

pub fn apply_rope(
    data: &mut [f32],
    pos: usize,
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    rope_scaling: RopeScaling,
    rope_style: RopeStyle,
) {
    if rope_cache_enabled() {
        apply_rope_cached(
            data,
            pos,
            head_count,
            head_dim,
            rope_dim,
            freq_base,
            rope_scaling,
            rope_style,
        );
        return;
    }

    let half = rope_dim / 2;
    for head in 0..head_count {
        let head_start = head * head_dim;
        for i in 0..half {
            // NORM rotates interleaved pairs (2i, 2i+1); NEOX rotates split-half pairs
            // (i, i+rope_dim/2). Same angle table, different element pairing.
            let (dim0, dim1) = match rope_style {
                RopeStyle::Norm => (head_start + i * 2, head_start + i * 2 + 1),
                RopeStyle::Neox => (head_start + i, head_start + i + half),
            };

            let mut theta = freq_base.powf(-((i * 2) as f32) / rope_dim as f32);

            // Apply Llama 3 / 3.2 scaled RoPE if config is present
            if let (Some(factor), Some(orig_len), Some(low), Some(high)) = (
                rope_scaling.factor,
                rope_scaling.original_context_length,
                rope_scaling.low_freq_factor,
                rope_scaling.high_freq_factor,
            ) {
                let wavelength = (2.0 * std::f32::consts::PI) / theta;
                let low_freq_wavelength = orig_len / low;
                let high_freq_wavelength = orig_len / high;
                if wavelength >= high_freq_wavelength {
                    if wavelength > low_freq_wavelength {
                        theta /= factor;
                    } else {
                        let smooth = (orig_len / wavelength - low) / (high - low);
                        theta = ((1.0 - smooth) * theta / factor) + (smooth * theta);
                    }
                }
            }

            let angle = pos as f32 * theta;
            let (sin, cos) = angle.sin_cos();

            let x0 = data[dim0];
            let x1 = data[dim1];

            data[dim0] = x0 * cos - x1 * sin;
            data[dim1] = x0 * sin + x1 * cos;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq)]
struct RopeCacheKey {
    pos: usize,
    rope_dim: usize,
    freq_base_bits: u32,
    factor_bits: Option<u32>,
    original_context_length_bits: Option<u32>,
    low_freq_factor_bits: Option<u32>,
    high_freq_factor_bits: Option<u32>,
}

impl PartialEq for RopeCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.rope_dim == other.rope_dim
            && self.freq_base_bits == other.freq_base_bits
            && self.factor_bits == other.factor_bits
            && self.original_context_length_bits == other.original_context_length_bits
            && self.low_freq_factor_bits == other.low_freq_factor_bits
            && self.high_freq_factor_bits == other.high_freq_factor_bits
    }
}

impl Hash for RopeCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos.hash(state);
        self.rope_dim.hash(state);
        self.freq_base_bits.hash(state);
        self.factor_bits.hash(state);
        self.original_context_length_bits.hash(state);
        self.low_freq_factor_bits.hash(state);
        self.high_freq_factor_bits.hash(state);
    }
}

type RopeAngleTable = Arc<[(f32, f32)]>;
type RopeCache = Mutex<HashMap<RopeCacheKey, RopeAngleTable>>;

fn rope_cache() -> &'static RopeCache {
    static CACHE: OnceLock<RopeCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn rope_cache_key(
    pos: usize,
    rope_dim: usize,
    freq_base: f32,
    rope_scaling: RopeScaling,
) -> RopeCacheKey {
    RopeCacheKey {
        pos,
        rope_dim,
        freq_base_bits: freq_base.to_bits(),
        factor_bits: rope_scaling.factor.map(f32::to_bits),
        original_context_length_bits: rope_scaling.original_context_length.map(f32::to_bits),
        low_freq_factor_bits: rope_scaling.low_freq_factor.map(f32::to_bits),
        high_freq_factor_bits: rope_scaling.high_freq_factor.map(f32::to_bits),
    }
}

fn cached_rope_angles(
    pos: usize,
    rope_dim: usize,
    freq_base: f32,
    rope_scaling: RopeScaling,
) -> Arc<[(f32, f32)]> {
    let key = rope_cache_key(pos, rope_dim, freq_base, rope_scaling);
    if let Ok(cache) = rope_cache().lock()
        && let Some(angles) = cache.get(&key)
    {
        return Arc::clone(angles);
    }

    let mut angles = Vec::with_capacity(rope_dim / 2);
    for i in 0..(rope_dim / 2) {
        let mut theta = freq_base.powf(-((i * 2) as f32) / rope_dim as f32);
        if let (Some(factor), Some(orig_len), Some(low), Some(high)) = (
            rope_scaling.factor,
            rope_scaling.original_context_length,
            rope_scaling.low_freq_factor,
            rope_scaling.high_freq_factor,
        ) {
            let wavelength = (2.0 * std::f32::consts::PI) / theta;
            let low_freq_wavelength = orig_len / low;
            let high_freq_wavelength = orig_len / high;
            if wavelength >= high_freq_wavelength {
                if wavelength > low_freq_wavelength {
                    theta /= factor;
                } else {
                    let smooth = (orig_len / wavelength - low) / (high - low);
                    theta = ((1.0 - smooth) * theta / factor) + (smooth * theta);
                }
            }
        }
        angles.push((pos as f32 * theta).sin_cos());
    }

    let angles = Arc::<[(f32, f32)]>::from(angles);
    if let Ok(mut cache) = rope_cache().lock() {
        Arc::clone(cache.entry(key).or_insert_with(|| Arc::clone(&angles)))
    } else {
        angles
    }
}

fn apply_rope_cached(
    data: &mut [f32],
    pos: usize,
    head_count: usize,
    head_dim: usize,
    rope_dim: usize,
    freq_base: f32,
    rope_scaling: RopeScaling,
    rope_style: RopeStyle,
) {
    let angles = cached_rope_angles(pos, rope_dim, freq_base, rope_scaling);
    let half = rope_dim / 2;
    for head in 0..head_count {
        let head_start = head * head_dim;
        for (i, &(sin, cos)) in angles.iter().enumerate() {
            // NORM: interleaved pairs (2i, 2i+1); NEOX: split-half pairs (i, i+rope_dim/2).
            let (dim0, dim1) = match rope_style {
                RopeStyle::Norm => (head_start + i * 2, head_start + i * 2 + 1),
                RopeStyle::Neox => (head_start + i, head_start + i + half),
            };
            let x0 = data[dim0];
            let x1 = data[dim1];

            data[dim0] = x0 * cos - x1 * sin;
            data[dim1] = x0 * sin + x1 * cos;
        }
    }
}

pub fn quantize_f32_to_q8_0(x: &[f32], x_i8: &mut [i8], x_scales: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON/ASIMD support.
        unsafe {
            quantize_f32_to_q8_0_neon_max(x, x_i8, x_scales);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        quantize_f32_to_q8_0_scalar(x, x_i8, x_scales);
    }
}

pub fn rms_norm_batch(
    out: &mut [f32],
    x: &[f32],
    weight: &[f32],
    epsilon: f32,
    batch_size: usize,
    dim: usize,
) {
    debug_assert_eq!(out.len(), batch_size * dim);
    debug_assert_eq!(x.len(), batch_size * dim);
    debug_assert_eq!(weight.len(), dim);

    out.par_chunks_mut(dim)
        .zip(x.par_chunks(dim))
        .take(batch_size)
        .for_each(|(out_token, x_token)| {
            rms_norm(out_token, x_token, weight, epsilon);
        });
}

pub fn quantize_f32_to_q8_0_batch(
    x: &[f32],
    x_i8: &mut [i8],
    x_scales: &mut [f32],
    batch_size: usize,
    dim: usize,
) {
    let blocks_per_token = dim / 32;
    debug_assert_eq!(x.len(), batch_size * dim);
    debug_assert_eq!(x_i8.len(), batch_size * dim);
    debug_assert!(x_scales.len() >= batch_size * blocks_per_token);

    x_i8.par_chunks_mut(dim)
        .zip(x_scales.par_chunks_mut(blocks_per_token))
        .zip(x.par_chunks(dim))
        .take(batch_size)
        .for_each(|((out_i8, out_scales), x_token)| {
            quantize_f32_to_q8_0(x_token, out_i8, out_scales);
        });
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn quantize_f32_to_q8_0_scalar(x: &[f32], x_i8: &mut [i8], x_scales: &mut [f32]) {
    let num_blocks = x.len() / 32;
    for (b, scale_out) in x_scales.iter_mut().enumerate().take(num_blocks) {
        let chunk = &x[b * 32..(b + 1) * 32];
        let max_abs = max_abs_scalar(chunk);
        quantize_q8_block_with_max_abs(chunk, &mut x_i8[b * 32..(b + 1) * 32], scale_out, max_abs);
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn max_abs_scalar(chunk: &[f32]) -> f32 {
    let mut max_abs = 0.0_f32;
    for &val in chunk {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    max_abs
}

fn quantize_q8_block_with_max_abs(
    chunk: &[f32],
    out: &mut [i8],
    scale_out: &mut f32,
    max_abs: f32,
) {
    let scale = max_abs / 127.0;
    *scale_out = scale;
    if scale > 0.0 {
        let inv_scale = 1.0 / scale;
        for (dst, &value) in out.iter_mut().zip(chunk) {
            let q = (value * inv_scale).round();
            *dst = q.clamp(-127.0, 127.0) as i8;
        }
    } else {
        out.fill(0);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn quantize_q8_block_neon(src_ptr: *const f32, dst_ptr: *mut i8, inv_scale: f32) {
    use std::arch::aarch64::{
        vcombine_s8, vcombine_s16, vcvtnq_s32_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vqmovn_s16,
        vqmovn_s32, vst1q_s8,
    };

    unsafe {
        let inv_scale_vec = vdupq_n_f32(inv_scale);

        // Process first 16 elements
        let f0 = vld1q_f32(src_ptr);
        let f1 = vld1q_f32(src_ptr.add(4));
        let f2 = vld1q_f32(src_ptr.add(8));
        let f3 = vld1q_f32(src_ptr.add(12));

        let m0 = vmulq_f32(f0, inv_scale_vec);
        let m1 = vmulq_f32(f1, inv_scale_vec);
        let m2 = vmulq_f32(f2, inv_scale_vec);
        let m3 = vmulq_f32(f3, inv_scale_vec);

        let s0 = vcvtnq_s32_f32(m0);
        let s1 = vcvtnq_s32_f32(m1);
        let s2 = vcvtnq_s32_f32(m2);
        let s3 = vcvtnq_s32_f32(m3);

        let h0 = vqmovn_s32(s0);
        let h1 = vqmovn_s32(s1);
        let h2 = vqmovn_s32(s2);
        let h3 = vqmovn_s32(s3);

        let s16_0 = vcombine_s16(h0, h1);
        let s16_1 = vcombine_s16(h2, h3);

        let b0 = vqmovn_s16(s16_0);
        let b1 = vqmovn_s16(s16_1);

        let b_low = vcombine_s8(b0, b1);
        vst1q_s8(dst_ptr, b_low);

        // Process second 16 elements (to make 32 total for the block)
        let f4 = vld1q_f32(src_ptr.add(16));
        let f5 = vld1q_f32(src_ptr.add(20));
        let f6 = vld1q_f32(src_ptr.add(24));
        let f7 = vld1q_f32(src_ptr.add(28));

        let m4 = vmulq_f32(f4, inv_scale_vec);
        let m5 = vmulq_f32(f5, inv_scale_vec);
        let m6 = vmulq_f32(f6, inv_scale_vec);
        let m7 = vmulq_f32(f7, inv_scale_vec);

        let s4 = vcvtnq_s32_f32(m4);
        let s5 = vcvtnq_s32_f32(m5);
        let s6 = vcvtnq_s32_f32(m6);
        let s7 = vcvtnq_s32_f32(m7);

        let h4 = vqmovn_s32(s4);
        let h5 = vqmovn_s32(s5);
        let h6 = vqmovn_s32(s6);
        let h7 = vqmovn_s32(s7);

        let s16_2 = vcombine_s16(h4, h5);
        let s16_3 = vcombine_s16(h6, h7);

        let b2 = vqmovn_s16(s16_2);
        let b3 = vqmovn_s16(s16_3);

        let b_high = vcombine_s8(b2, b3);
        vst1q_s8(dst_ptr.add(16), b_high);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn quantize_f32_to_q8_0_neon_max(x: &[f32], x_i8: &mut [i8], x_scales: &mut [f32]) {
    let num_blocks = x.len() / 32;
    for (b, scale_out) in x_scales.iter_mut().enumerate().take(num_blocks) {
        let offset = b * 32;
        let chunk = &x[offset..offset + 32];
        let max_abs = unsafe { max_abs_32_neon(chunk.as_ptr()) };

        let scale = max_abs / 127.0;
        *scale_out = scale;

        if scale > 0.0 {
            let inv_scale = 1.0 / scale;
            unsafe {
                quantize_q8_block_neon(chunk.as_ptr(), x_i8.as_mut_ptr().add(offset), inv_scale);
            }
        } else {
            x_i8[offset..offset + 32].fill(0);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn max_abs_32_neon(ptr: *const f32) -> f32 {
    use std::arch::aarch64::{vabsq_f32, vld1q_f32, vmaxq_f32, vmaxvq_f32};

    let mut max_abs = unsafe { vabsq_f32(vld1q_f32(ptr)) };
    for offset in [4, 8, 12, 16, 20, 24, 28] {
        let values = unsafe { vabsq_f32(vld1q_f32(ptr.add(offset))) };
        max_abs = unsafe { vmaxq_f32(max_abs, values) };
    }
    unsafe { vmaxvq_f32(max_abs) }
}

unsafe fn activation_block_ptr(x_i8: &[i8], block_idx: usize) -> &[i8; 32] {
    debug_assert!((block_idx + 1) * 32 <= x_i8.len());
    // SAFETY: callers use model-derived dimensions where each activation block is
    // exactly 32 i8 lanes and block_idx is bounded by blocks_per_row.
    unsafe { &*(x_i8.as_ptr().add(block_idx * 32) as *const [i8; 32]) }
}

pub fn matmul_q8_0(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q8_0Block],
    _rows: usize,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / 32;
    match selector.selected {
        Q8DotKernel::Scalar => {
            for_each_matmul_row(out, |r, out_val| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
                    let dot_val = crate::q8::dot_i8_scalar(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
        Q8DotKernel::Neon => {
            for_each_matmul_row(out, |r, out_val| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
                    let dot_val =
                        crate::q8::dot_i8_neon_32_selected(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
        Q8DotKernel::Sdot => {
            for_each_matmul_row(out, |r, out_val| {
                let mut sum = 0.0_f32;
                let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
                for b in 0..blocks_per_row {
                    let w_block = &w_row[b];
                    let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
                    let dot_val =
                        crate::q8::dot_i8_sdot_32_selected(w_block.values(), x_block_vals);
                    sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
                }
                *out_val = sum;
            });
        }
    }
}

pub fn matmul_quantized(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &QuantizedMatrix,
    rows: usize,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    match w {
        QuantizedMatrix::Q8_0(blocks) => {
            matmul_q8_0(out, x_i8, x_scales, blocks, rows, cols, selector)
        }
        QuantizedMatrix::Q8_0Swizzled1x4(matrix) => {
            debug_assert_eq!(matrix.rows, rows);
            debug_assert_eq!(matrix.cols, cols);
            matmul_q8_0_swizzled_1x4(out, x_i8, x_scales, matrix, cols, selector);
        }
        QuantizedMatrix::Q4_0(blocks) => {
            matmul_q4_0(out, x_i8, x_scales, blocks, rows, cols, selector)
        }
        QuantizedMatrix::Q4_1(blocks) => matmul_q4_1(out, x_i8, x_scales, blocks, rows, cols),
        QuantizedMatrix::Q4_0Swizzled1x4(matrix) => {
            debug_assert_eq!(matrix.rows, rows);
            debug_assert_eq!(matrix.cols, cols);
            matmul_q4_0_swizzled_1x4(out, x_i8, x_scales, matrix, cols, selector);
        }
        QuantizedMatrix::Q5_0(blocks) => matmul_q5_0(out, x_i8, x_scales, blocks, rows, cols),
        QuantizedMatrix::Q5_1(blocks) => matmul_q5_1(out, x_i8, x_scales, blocks, rows, cols),
        QuantizedMatrix::Q2K(blocks) => matmul_q2_k(out, x_i8, x_scales, blocks, rows, cols),
        QuantizedMatrix::Q3K(blocks) => matmul_q3_k(out, x_i8, x_scales, blocks, rows, cols),
        QuantizedMatrix::Q4K(blocks) => matmul_q4_k(out, x_i8, x_scales, blocks, rows, cols),
        QuantizedMatrix::Q5K(blocks) => matmul_q5_k(out, x_i8, x_scales, blocks, rows, cols),
        QuantizedMatrix::Q6K(blocks) => matmul_q6_k(out, x_i8, x_scales, blocks, rows, cols),
        QuantizedMatrix::Q8K(blocks) => {
            matmul_q8_k(out, x_i8, x_scales, blocks, rows, cols, selector)
        }
        QuantizedMatrix::IQ4NL(blocks) => matmul_iq4_nl(out, x_i8, x_scales, blocks, rows, cols),
        QuantizedMatrix::F32(weights) => {
            let x_f32 = dequantize_x_q8_0(x_i8, x_scales);
            matmul_f32(out, &x_f32, weights, rows, cols);
        }
    }
}

pub fn matmul_quantized_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &QuantizedMatrix,
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    debug_assert_eq!(out.len(), batch_size * rows);
    debug_assert_eq!(x_i8.len(), batch_size * cols);
    debug_assert!(x_scales.len() >= batch_size * (cols / Q8_BLOCK_SIZE));

    match w {
        QuantizedMatrix::Q8_0(blocks) => {
            matmul_q8_0_batch(out, x_i8, x_scales, blocks, shape, selector)
        }
        QuantizedMatrix::Q8_0Swizzled1x4(matrix) => {
            debug_assert_eq!(matrix.rows, rows);
            debug_assert_eq!(matrix.cols, cols);
            matmul_q8_0_swizzled_1x4_batch(out, x_i8, x_scales, matrix, shape, selector);
        }
        QuantizedMatrix::Q4_0(blocks) => {
            matmul_q4_0_batch(out, x_i8, x_scales, blocks, shape, selector)
        }
        QuantizedMatrix::Q4_1(blocks) => matmul_q4_1_batch(out, x_i8, x_scales, blocks, shape),
        QuantizedMatrix::Q4_0Swizzled1x4(matrix) => {
            debug_assert_eq!(matrix.rows, rows);
            debug_assert_eq!(matrix.cols, cols);
            matmul_q4_0_swizzled_1x4_batch(out, x_i8, x_scales, matrix, shape, selector);
        }
        QuantizedMatrix::Q5_0(blocks) => matmul_q5_0_batch(out, x_i8, x_scales, blocks, shape),
        QuantizedMatrix::Q5_1(blocks) => matmul_q5_1_batch(out, x_i8, x_scales, blocks, shape),
        QuantizedMatrix::Q2K(blocks) => matmul_q2_k_batch(out, x_i8, x_scales, blocks, shape),
        QuantizedMatrix::Q3K(blocks) => matmul_q3_k_batch(out, x_i8, x_scales, blocks, shape),
        QuantizedMatrix::Q4K(blocks) => matmul_q4_k_batch(out, x_i8, x_scales, blocks, shape),
        QuantizedMatrix::Q5K(blocks) => matmul_q5_k_batch(out, x_i8, x_scales, blocks, shape),
        QuantizedMatrix::Q6K(blocks) => matmul_q6_k_batch(out, x_i8, x_scales, blocks, shape),
        QuantizedMatrix::Q8K(blocks) => {
            matmul_q8_k_batch(out, x_i8, x_scales, blocks, shape, selector)
        }
        QuantizedMatrix::IQ4NL(blocks) => matmul_iq4_nl_batch(out, x_i8, x_scales, blocks, shape),
        QuantizedMatrix::F32(weights) => {
            let x_f32 = dequantize_x_q8_0(x_i8, x_scales);
            for token_idx in 0..batch_size {
                let x_offset = token_idx * cols;
                let out_offset = token_idx * rows;
                let x_token = &x_f32[x_offset..x_offset + cols];
                let out_token = &mut out[out_offset..out_offset + rows];
                matmul_f32(out_token, x_token, weights, rows, cols);
            }
        }
    }
}

fn dequantize_x_q8_0(x_i8: &[i8], x_scales: &[f32]) -> Vec<f32> {
    let mut x_f32 = vec![0.0_f32; x_i8.len()];
    for (b, &scale) in x_scales.iter().enumerate() {
        let offset = b * 32;
        if offset + 32 <= x_i8.len() {
            for i in 0..32 {
                x_f32[offset + i] = x_i8[offset + i] as f32 * scale;
            }
        }
    }
    x_f32
}

fn for_each_batch_matmul_row<F>(rows: usize, compute_row: F)
where
    F: Fn(usize) + Send + Sync,
{
    if should_parallelize_matmul(rows) {
        (0..rows)
            .into_par_iter()
            .with_min_len(matmul_min_rows())
            .for_each(compute_row);
    } else {
        (0..rows).for_each(compute_row);
    }
}

#[inline]
unsafe fn write_batch_out(out_addr: usize, token_idx: usize, rows: usize, row: usize, value: f32) {
    // SAFETY: batch matmul workers are partitioned by output row. For a fixed
    // row, each token writes token_idx * rows + row, so no two row workers write
    // the same element.
    unsafe {
        let out = out_addr as *mut f32;
        *out.add(token_idx * rows + row) = value;
    }
}

pub fn matmul_q8_0_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q8_0Block],
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w_row[b];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val = match selector.selected {
                    Q8DotKernel::Scalar => crate::q8::dot_i8_scalar(w_block.values(), x_block_vals),
                    Q8DotKernel::Neon => {
                        crate::q8::dot_i8_neon_32_selected(w_block.values(), x_block_vals)
                    }
                    Q8DotKernel::Sdot => {
                        crate::q8::dot_i8_sdot_32_selected(w_block.values(), x_block_vals)
                    }
                };
                sum += w_block.scale_f32() * x_token_scales[b] * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

pub fn matmul_q4_0_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w_row[b];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val =
                    crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
                sum += w_block.scale_f32() * x_token_scales[b] * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

pub fn matmul_q4_1_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_1Block],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w_row[b];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let (weighted_sum, activation_sum) =
                    crate::q8::dot_q4_1_q8_0_scalar(w_block, x_block_vals);
                sum += x_token_scales[b]
                    * (w_block.scale_f32() * weighted_sum as f32
                        + w_block.min_f32() * activation_sum as f32);
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

pub fn matmul_q5_0_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[crate::q8::Q5_0Block],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w_row[b];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val = crate::q8::dot_q5_0_q8_0_scalar(w_block, x_block_vals);
                sum += w_block.scale_f32() * x_token_scales[b] * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

pub fn matmul_q5_1_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[crate::q8::Q5_1Block],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w_row[b];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let (weighted_sum, activation_sum) =
                    crate::q8::dot_q5_1_q8_0_scalar(w_block, x_block_vals);
                sum += x_token_scales[b]
                    * (w_block.scale_f32() * weighted_sum as f32
                        + w_block.min_f32() * activation_sum as f32);
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

fn matmul_q4_0_swizzled_1x4_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &crate::model::Q4_0Swizzled1x4Matrix,
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let w = &matrix.swizzled_1x4;
    debug_assert_eq!(w.len(), rows * blocks_per_row);
    debug_assert!(rows.is_multiple_of(4));

    if let Some(aligned) = &matrix.page_aligned_1x4 {
        matmul_q4_0_page_aligned_1x4_batch(out, x_i8, x_scales, aligned, shape, selector);
        return;
    }

    if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
        matmul_q4_0_swizzled_1x4_sdot_batch(out, x_i8, x_scales, w, shape, blocks_per_row);
        return;
    }

    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let chunk_idx = r / 4;
        let lane = r % 4;
        let chunk_base = chunk_idx * blocks_per_row * 4;
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w[chunk_base + b * 4 + lane];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val =
                    crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
                sum += w_block.scale_f32() * x_token_scales[b] * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

fn matmul_q4_0_swizzled_1x4_sdot_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    shape: BatchMatmulShape,
    blocks_per_row: usize,
) {
    let rows = shape.rows;
    let chunks = rows / 4;
    let out_addr = out.as_mut_ptr() as usize;
    let ctx = SwizzledBatchChunkContext {
        out_addr,
        x_i8,
        x_scales,
        w,
        shape,
        blocks_per_row,
    };
    if should_parallelize_matmul(rows) {
        (0..chunks).into_par_iter().for_each(|chunk_idx| {
            compute_q4_0_swizzled_1x4_sdot_batch_chunk(chunk_idx, ctx);
        });
    } else {
        (0..chunks).for_each(|chunk_idx| {
            compute_q4_0_swizzled_1x4_sdot_batch_chunk(chunk_idx, ctx);
        });
    }
}

#[derive(Clone, Copy)]
struct SwizzledBatchChunkContext<'a> {
    out_addr: usize,
    x_i8: &'a [i8],
    x_scales: &'a [f32],
    w: &'a [Q4_0Block],
    shape: BatchMatmulShape,
    blocks_per_row: usize,
}

fn compute_q4_0_swizzled_1x4_sdot_batch_chunk(
    chunk_idx: usize,
    ctx: SwizzledBatchChunkContext<'_>,
) {
    let batch_size = ctx.shape.batch_size;
    if batch_size <= MAX_STACK_BATCH_SUMS {
        let mut token_sums = [[0.0_f32; 4]; MAX_STACK_BATCH_SUMS];
        compute_q4_0_swizzled_1x4_sdot_batch_chunk_with_sums(
            chunk_idx,
            ctx,
            &mut token_sums[..batch_size],
        );
    } else {
        let mut token_sums = vec![[0.0_f32; 4]; batch_size];
        compute_q4_0_swizzled_1x4_sdot_batch_chunk_with_sums(chunk_idx, ctx, &mut token_sums);
    }
}

fn compute_q4_0_swizzled_1x4_sdot_batch_chunk_with_sums(
    chunk_idx: usize,
    ctx: SwizzledBatchChunkContext<'_>,
    token_sums: &mut [[f32; 4]],
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = ctx.shape;
    debug_assert_eq!(batch_size, token_sums.len());
    let chunk_base = chunk_idx * ctx.blocks_per_row * 4;
    let row_base = chunk_idx * 4;
    for b in 0..ctx.blocks_per_row {
        let w_base = chunk_base + b * 4;
        let row0 = &ctx.w[w_base];
        let row1 = &ctx.w[w_base + 1];
        let row2 = &ctx.w[w_base + 2];
        let row3 = &ctx.w[w_base + 3];
        let row0_scale = row0.scale_f32();
        let row1_scale = row1.scale_f32();
        let row2_scale = row2.scale_f32();
        let row3_scale = row3.scale_f32();

        #[cfg(target_arch = "aarch64")]
        let unpacked = unsafe {
            let (w0_low, w0_high) = crate::q8::unpack_q4_0_lanes_aarch64(row0);
            let (w1_low, w1_high) = crate::q8::unpack_q4_0_lanes_aarch64(row1);
            let (w2_low, w2_high) = crate::q8::unpack_q4_0_lanes_aarch64(row2);
            let (w3_low, w3_high) = crate::q8::unpack_q4_0_lanes_aarch64(row3);
            crate::q8::Q4_0Unpacked1x4Aarch64 {
                w0_low,
                w0_high,
                w1_low,
                w1_high,
                w2_low,
                w2_high,
                w3_low,
                w3_high,
            }
        };

        for (token_idx, sums) in token_sums.iter_mut().enumerate() {
            let x_offset = token_idx * cols;
            let x_scale = ctx.x_scales[token_idx * ctx.blocks_per_row + b];
            let x_token = &ctx.x_i8[x_offset..x_offset + cols];
            let x_block_vals = unsafe { activation_block_ptr(x_token, b) };

            #[cfg(target_arch = "aarch64")]
            let dots = unsafe {
                crate::q8::dot_q4_0_q8_0_1x4_sdot_preloaded_aarch64(unpacked, x_block_vals)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let dots = [
                crate::q8::dot_q4_0_q8_0_scalar(row0, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row1, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row2, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row3, x_block_vals),
            ];

            sums[0] += row0_scale * x_scale * dots[0] as f32;
            sums[1] += row1_scale * x_scale * dots[1] as f32;
            sums[2] += row2_scale * x_scale * dots[2] as f32;
            sums[3] += row3_scale * x_scale * dots[3] as f32;
        }
    }

    for (token_idx, sums) in token_sums.iter().enumerate() {
        for (lane, &sum) in sums.iter().enumerate() {
            unsafe {
                write_batch_out(ctx.out_addr, token_idx, rows, row_base + lane, sum);
            }
        }
    }
}

fn matmul_q4_0_page_aligned_1x4_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &PageAlignedQ4_0Swizzled1x4,
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    debug_assert_eq!(matrix.blocks_per_row(), blocks_per_row);
    debug_assert_eq!(matrix.chunk_count(), rows / 4);

    if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
        let chunks = rows / 4;
        let out_addr = out.as_mut_ptr() as usize;
        let ctx = PageAlignedBatchChunkContext {
            out_addr,
            x_i8,
            x_scales,
            matrix,
            shape,
            blocks_per_row,
        };
        if should_parallelize_matmul(rows) {
            (0..chunks).into_par_iter().for_each(|chunk_idx| {
                compute_q4_0_page_aligned_1x4_sdot_batch_chunk(chunk_idx, ctx);
            });
        } else {
            (0..chunks).for_each(|chunk_idx| {
                compute_q4_0_page_aligned_1x4_sdot_batch_chunk(chunk_idx, ctx);
            });
        }
        return;
    }

    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let chunk_idx = r / 4;
        let lane = r % 4;
        let chunk = matrix.chunk_ptr(chunk_idx);
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for (b, x_scale) in x_token_scales
                .iter()
                .copied()
                .enumerate()
                .take(blocks_per_row)
            {
                let w_block = unsafe { &*chunk.add(b * 4 + lane) };
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val =
                    crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
                sum += w_block.scale_f32() * x_scale * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

#[derive(Clone, Copy)]
struct PageAlignedBatchChunkContext<'a> {
    out_addr: usize,
    x_i8: &'a [i8],
    x_scales: &'a [f32],
    matrix: &'a PageAlignedQ4_0Swizzled1x4,
    shape: BatchMatmulShape,
    blocks_per_row: usize,
}

fn compute_q4_0_page_aligned_1x4_sdot_batch_chunk(
    chunk_idx: usize,
    ctx: PageAlignedBatchChunkContext<'_>,
) {
    let batch_size = ctx.shape.batch_size;
    if batch_size <= MAX_STACK_BATCH_SUMS {
        let mut token_sums = [[0.0_f32; 4]; MAX_STACK_BATCH_SUMS];
        compute_q4_0_page_aligned_1x4_sdot_batch_chunk_with_sums(
            chunk_idx,
            ctx,
            &mut token_sums[..batch_size],
        );
    } else {
        let mut token_sums = vec![[0.0_f32; 4]; batch_size];
        compute_q4_0_page_aligned_1x4_sdot_batch_chunk_with_sums(chunk_idx, ctx, &mut token_sums);
    }
}

fn compute_q4_0_page_aligned_1x4_sdot_batch_chunk_with_sums(
    chunk_idx: usize,
    ctx: PageAlignedBatchChunkContext<'_>,
    token_sums: &mut [[f32; 4]],
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = ctx.shape;
    debug_assert_eq!(batch_size, token_sums.len());
    let chunk = ctx.matrix.chunk_ptr(chunk_idx);
    let row_base = chunk_idx * 4;
    for b in 0..ctx.blocks_per_row {
        let w_base = b * 4;
        let row0 = unsafe { &*chunk.add(w_base) };
        let row1 = unsafe { &*chunk.add(w_base + 1) };
        let row2 = unsafe { &*chunk.add(w_base + 2) };
        let row3 = unsafe { &*chunk.add(w_base + 3) };
        let row0_scale = row0.scale_f32();
        let row1_scale = row1.scale_f32();
        let row2_scale = row2.scale_f32();
        let row3_scale = row3.scale_f32();

        #[cfg(target_arch = "aarch64")]
        let unpacked = unsafe {
            let (w0_low, w0_high) = crate::q8::unpack_q4_0_lanes_aarch64(row0);
            let (w1_low, w1_high) = crate::q8::unpack_q4_0_lanes_aarch64(row1);
            let (w2_low, w2_high) = crate::q8::unpack_q4_0_lanes_aarch64(row2);
            let (w3_low, w3_high) = crate::q8::unpack_q4_0_lanes_aarch64(row3);
            crate::q8::Q4_0Unpacked1x4Aarch64 {
                w0_low,
                w0_high,
                w1_low,
                w1_high,
                w2_low,
                w2_high,
                w3_low,
                w3_high,
            }
        };

        for (token_idx, sums) in token_sums.iter_mut().enumerate() {
            let x_offset = token_idx * cols;
            let x_scale = ctx.x_scales[token_idx * ctx.blocks_per_row + b];
            let x_token = &ctx.x_i8[x_offset..x_offset + cols];
            let x_block_vals = unsafe { activation_block_ptr(x_token, b) };

            #[cfg(target_arch = "aarch64")]
            let dots = unsafe {
                crate::q8::dot_q4_0_q8_0_1x4_sdot_preloaded_aarch64(unpacked, x_block_vals)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let dots = [
                crate::q8::dot_q4_0_q8_0_scalar(row0, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row1, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row2, x_block_vals),
                crate::q8::dot_q4_0_q8_0_scalar(row3, x_block_vals),
            ];

            sums[0] += row0_scale * x_scale * dots[0] as f32;
            sums[1] += row1_scale * x_scale * dots[1] as f32;
            sums[2] += row2_scale * x_scale * dots[2] as f32;
            sums[3] += row3_scale * x_scale * dots[3] as f32;
        }
    }

    for (token_idx, sums) in token_sums.iter().enumerate() {
        for (lane, &sum) in sums.iter().enumerate() {
            unsafe {
                write_batch_out(ctx.out_addr, token_idx, rows, row_base + lane, sum);
            }
        }
    }
}

pub fn matmul_q6_k_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q6KBlock],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    let q8_blocks_per_token = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    #[cfg(target_arch = "aarch64")]
    if q6_k_sdot_enabled() && std::arch::is_aarch64_feature_detected!("dotprod") {
        for_each_batch_matmul_row(rows, |r| {
            let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
            if batch_size <= MAX_STACK_BATCH_SUMS {
                let mut sums = [0.0_f32; MAX_STACK_BATCH_SUMS];
                let sums = &mut sums[..batch_size];
                for (block_idx, w_block) in w_row.iter().enumerate() {
                    let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                    let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                    let unpacked = unsafe { w_block.unpack_aarch64() };
                    for (token_idx, sum) in sums.iter_mut().enumerate() {
                        let x_offset = token_idx * cols;
                        let scale_offset = token_idx * q8_blocks_per_token;
                        let x_token = &x_i8[x_offset..x_offset + cols];
                        let x_token_scales =
                            &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                        *sum += unsafe {
                            crate::q8::dot_q8_scaled_sdot_preloaded(
                                &unpacked,
                                &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                                &x_token_scales[x_scale_start
                                    ..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                            )
                        };
                    }
                }
                for (token_idx, &sum) in sums.iter().enumerate() {
                    unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
                }
            } else {
                let mut sums = vec![0.0_f32; batch_size];
                for (block_idx, w_block) in w_row.iter().enumerate() {
                    let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                    let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                    let unpacked = unsafe { w_block.unpack_aarch64() };
                    for (token_idx, sum) in sums.iter_mut().enumerate() {
                        let x_offset = token_idx * cols;
                        let scale_offset = token_idx * q8_blocks_per_token;
                        let x_token = &x_i8[x_offset..x_offset + cols];
                        let x_token_scales =
                            &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                        *sum += unsafe {
                            crate::q8::dot_q8_scaled_sdot_preloaded(
                                &unpacked,
                                &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                                &x_token_scales[x_scale_start
                                    ..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                            )
                        };
                    }
                }
                for (token_idx, &sum) in sums.iter().enumerate() {
                    unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
                }
            }
        });
        return;
    }

    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        if batch_size <= MAX_STACK_BATCH_SUMS {
            let mut sums = [0.0_f32; MAX_STACK_BATCH_SUMS];
            let sums = &mut sums[..batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += w_block.dot_q8_scaled(
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        } else {
            let mut sums = vec![0.0_f32; batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += w_block.dot_q8_scaled(
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        }
    });
}

pub fn matmul_q4_k_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[crate::q8::Q4KBlock],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    let q8_blocks_per_token = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        if batch_size <= MAX_STACK_BATCH_SUMS {
            let mut sums = [0.0_f32; MAX_STACK_BATCH_SUMS];
            let sums = &mut sums[..batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                let unpacked = w_block.unpack();
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += crate::q8::Q4KBlock::dot_q8_scaled_preloaded(
                        &unpacked,
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        } else {
            let mut sums = vec![0.0_f32; batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                let unpacked = w_block.unpack();
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += crate::q8::Q4KBlock::dot_q8_scaled_preloaded(
                        &unpacked,
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        }
    });
}

pub fn matmul_q2_k_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q2KBlock],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    let q8_blocks_per_token = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        if batch_size <= MAX_STACK_BATCH_SUMS {
            let mut sums = [0.0_f32; MAX_STACK_BATCH_SUMS];
            let sums = &mut sums[..batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                let unpacked = w_block.unpack();
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += crate::q8::Q2KBlock::dot_q8_scaled_preloaded(
                        &unpacked,
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        } else {
            let mut sums = vec![0.0_f32; batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                let unpacked = w_block.unpack();
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += crate::q8::Q2KBlock::dot_q8_scaled_preloaded(
                        &unpacked,
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        }
    });
}

pub fn matmul_q8_k_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q8KBlock],
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    let q8_blocks_per_token = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        if batch_size <= MAX_STACK_BATCH_SUMS {
            let mut sums = [0.0_f32; MAX_STACK_BATCH_SUMS];
            let sums = &mut sums[..batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += w_block.dot_q8_scaled(
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                        selector,
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        } else {
            let mut sums = vec![0.0_f32; batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += w_block.dot_q8_scaled(
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                        selector,
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        }
    });
}

pub fn matmul_iq4_nl_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[IQ4NLBlock],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / 32;
    let q8_blocks_per_token = cols / 32;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        if batch_size <= MAX_STACK_BATCH_SUMS {
            let mut sums = [0.0_f32; MAX_STACK_BATCH_SUMS];
            let sums = &mut sums[..batch_size];
            for b in 0..blocks_per_row {
                let w_block = &w_row[b];
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                    *sum += w_block.dot_q8_scaled(x_block_vals, x_token_scales[b]);
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        } else {
            let mut sums = vec![0.0_f32; batch_size];
            for b in 0..blocks_per_row {
                let w_block = &w_row[b];
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                    *sum += w_block.dot_q8_scaled(x_block_vals, x_token_scales[b]);
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        }
    });
}

pub fn matmul_q3_k_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q3KBlock],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    let q8_blocks_per_token = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        if batch_size <= MAX_STACK_BATCH_SUMS {
            let mut sums = [0.0_f32; MAX_STACK_BATCH_SUMS];
            let sums = &mut sums[..batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                let unpacked = w_block.unpack();
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += crate::q8::Q3KBlock::dot_q8_scaled_preloaded(
                        &unpacked,
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        } else {
            let mut sums = vec![0.0_f32; batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                let unpacked = w_block.unpack();
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += crate::q8::Q3KBlock::dot_q8_scaled_preloaded(
                        &unpacked,
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        }
    });
}

pub fn matmul_q5_k_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q5KBlock],
    shape: BatchMatmulShape,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    let q8_blocks_per_token = cols / Q8_BLOCK_SIZE;
    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        if batch_size <= MAX_STACK_BATCH_SUMS {
            let mut sums = [0.0_f32; MAX_STACK_BATCH_SUMS];
            let sums = &mut sums[..batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                let unpacked = w_block.unpack();
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += crate::q8::Q5KBlock::dot_q8_scaled_preloaded(
                        &unpacked,
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        } else {
            let mut sums = vec![0.0_f32; batch_size];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                let unpacked = w_block.unpack();
                for (token_idx, sum) in sums.iter_mut().enumerate() {
                    let x_offset = token_idx * cols;
                    let scale_offset = token_idx * q8_blocks_per_token;
                    let x_token = &x_i8[x_offset..x_offset + cols];
                    let x_token_scales =
                        &x_scales[scale_offset..scale_offset + q8_blocks_per_token];
                    *sum += crate::q8::Q5KBlock::dot_q8_scaled_preloaded(
                        &unpacked,
                        &x_token[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_token_scales
                            [x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    );
                }
            }
            for (token_idx, &sum) in sums.iter().enumerate() {
                unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
            }
        }
    });
}

pub fn matmul_q4_0(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    _rows: usize,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
        matmul_q4_0_sdot_1x4(out, x_i8, x_scales, w, cols, selector);
        return;
    }

    let blocks_per_row = cols / 32;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for b in 0..blocks_per_row {
            let w_block = &w_row[b];
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            let dot_val = crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
            sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
        }
        *out_val = sum;
    });
}

fn matmul_q4_1(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_1Block],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / 32;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for b in 0..blocks_per_row {
            let w_block = &w_row[b];
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            let (weighted_sum, activation_sum) =
                crate::q8::dot_q4_1_q8_0_scalar(w_block, x_block_vals);
            sum += x_scales[b]
                * (w_block.scale_f32() * weighted_sum as f32
                    + w_block.min_f32() * activation_sum as f32);
        }
        *out_val = sum;
    });
}

fn matmul_q5_0(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[crate::q8::Q5_0Block],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for b in 0..blocks_per_row {
            let w_block = &w_row[b];
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            let dot_val = crate::q8::dot_q5_0_q8_0_scalar(w_block, x_block_vals);
            sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
        }
        *out_val = sum;
    });
}

fn matmul_q5_1(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[crate::q8::Q5_1Block],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for b in 0..blocks_per_row {
            let w_block = &w_row[b];
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            let (weighted_sum, activation_sum) =
                crate::q8::dot_q5_1_q8_0_scalar(w_block, x_block_vals);
            sum += x_scales[b]
                * (w_block.scale_f32() * weighted_sum as f32
                    + w_block.min_f32() * activation_sum as f32);
        }
        *out_val = sum;
    });
}

fn matmul_q4_0_sdot_1x4(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / 32;
    if should_parallelize_matmul(out.len()) {
        out.par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_sdot_1x4_chunk(
                    chunk_idx * 4,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    } else {
        out.chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_sdot_1x4_chunk(
                    chunk_idx * 4,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    }
}

fn compute_q4_0_sdot_1x4_chunk(
    row_base: usize,
    out_chunk: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    blocks_per_row: usize,
    selector: Q8DotKernelSelector,
) {
    if out_chunk.len() == 4 {
        let mut sums = [0.0_f32; 4];
        for (b, x_scale) in x_scales.iter().copied().enumerate().take(blocks_per_row) {
            #[cfg(target_arch = "aarch64")]
            {
                let next_b = b + 1;
                if next_b < blocks_per_row {
                    let next_w_base = row_base * blocks_per_row + next_b;
                    unsafe {
                        std::arch::asm!(
                            "prfm pldl1keep, [{ptr}]",
                            ptr = in(reg) w.as_ptr().add(next_w_base),
                            options(nostack, preserves_flags, readonly)
                        );
                    }
                }
            }
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            let w_base = row_base * blocks_per_row + b;
            let row0 = &w[w_base];
            let row1 = &w[w_base + blocks_per_row];
            let row2 = &w[w_base + 2 * blocks_per_row];
            let row3 = &w[w_base + 3 * blocks_per_row];
            let dots = dot_q4_0_q8_0_1x4_sdot([row0, row1, row2, row3], x_block_vals);
            sums[0] += row0.scale_f32() * x_scale * dots[0] as f32;
            sums[1] += row1.scale_f32() * x_scale * dots[1] as f32;
            sums[2] += row2.scale_f32() * x_scale * dots[2] as f32;
            sums[3] += row3.scale_f32() * x_scale * dots[3] as f32;
        }
        out_chunk.copy_from_slice(&sums);
        return;
    }

    for (lane, out_val) in out_chunk.iter_mut().enumerate() {
        let row = row_base + lane;
        let mut sum = 0.0_f32;
        let w_row = &w[row * blocks_per_row..(row + 1) * blocks_per_row];
        for b in 0..blocks_per_row {
            let w_block = &w_row[b];
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            let dot_val = crate::q8::dot_q4_0_q8_0_with_selector(w_block, x_block_vals, selector);
            sum += w_block.scale_f32() * x_scales[b] * dot_val as f32;
        }
        *out_val = sum;
    }
}

fn matmul_q4_0_swizzled_1x4(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &crate::model::Q4_0Swizzled1x4Matrix,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / 32;
    let w = &matrix.swizzled_1x4;
    debug_assert_eq!(w.len(), out.len() * blocks_per_row);
    debug_assert!(out.len().is_multiple_of(4));

    if let Some(aligned) = &matrix.page_aligned_1x4 {
        matmul_q4_0_page_aligned_1x4(out, x_i8, x_scales, aligned, cols, selector);
        return;
    }

    if should_parallelize_matmul(out.len()) {
        out.par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_sdot_1x4_swizzled_chunk(
                    chunk_idx,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    } else {
        out.chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_sdot_1x4_swizzled_chunk(
                    chunk_idx,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    }
}

fn compute_q4_0_sdot_1x4_swizzled_chunk(
    chunk_idx: usize,
    out_chunk: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q4_0Block],
    blocks_per_row: usize,
    selector: Q8DotKernelSelector,
) {
    debug_assert_eq!(out_chunk.len(), 4);

    let mut sums = [0.0_f32; 4];
    let chunk_base = chunk_idx * blocks_per_row * 4;
    for (b, x_scale) in x_scales.iter().copied().enumerate().take(blocks_per_row) {
        #[cfg(target_arch = "aarch64")]
        {
            let next_b = b + 1;
            if next_b < blocks_per_row {
                let next_w_base = chunk_base + next_b * 4;
                unsafe {
                    std::arch::asm!(
                        "prfm pldl1keep, [{ptr}]",
                        ptr = in(reg) w.as_ptr().add(next_w_base),
                        options(nostack, preserves_flags, readonly)
                    );
                }
            }
        }
        let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
        let w_base = chunk_base + b * 4;
        let row0 = &w[w_base];
        let row1 = &w[w_base + 1];
        let row2 = &w[w_base + 2];
        let row3 = &w[w_base + 3];
        let dots = if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
            dot_q4_0_q8_0_1x4_sdot([row0, row1, row2, row3], x_block_vals)
        } else {
            [
                crate::q8::dot_q4_0_q8_0_with_selector(row0, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row1, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row2, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row3, x_block_vals, selector),
            ]
        };
        sums[0] += row0.scale_f32() * x_scale * dots[0] as f32;
        sums[1] += row1.scale_f32() * x_scale * dots[1] as f32;
        sums[2] += row2.scale_f32() * x_scale * dots[2] as f32;
        sums[3] += row3.scale_f32() * x_scale * dots[3] as f32;
    }
    out_chunk.copy_from_slice(&sums);
}

fn dot_q4_0_q8_0_1x4_sdot(weights: [&Q4_0Block; 4], activation: &[i8; 32]) -> [i32; 4] {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: the 1x4 path is entered only after SDOT selector validation.
        unsafe { crate::q8::dot_q4_0_q8_0_1x4_sdot_aarch64(weights, activation) }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        [
            crate::q8::dot_q4_0_q8_0_scalar(weights[0], activation),
            crate::q8::dot_q4_0_q8_0_scalar(weights[1], activation),
            crate::q8::dot_q4_0_q8_0_scalar(weights[2], activation),
            crate::q8::dot_q4_0_q8_0_scalar(weights[3], activation),
        ]
    }
}

fn matmul_q4_0_page_aligned_1x4(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &PageAlignedQ4_0Swizzled1x4,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    debug_assert_eq!(matrix.blocks_per_row(), blocks_per_row);
    debug_assert_eq!(matrix.chunk_count(), out.len() / 4);
    debug_assert!(out.len().is_multiple_of(4));

    if should_parallelize_matmul(out.len()) {
        out.par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_page_aligned_1x4_chunk(
                    chunk_idx, out_chunk, x_i8, x_scales, matrix, selector,
                );
            });
    } else {
        out.chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q4_0_page_aligned_1x4_chunk(
                    chunk_idx, out_chunk, x_i8, x_scales, matrix, selector,
                );
            });
    }
}

fn compute_q4_0_page_aligned_1x4_chunk(
    chunk_idx: usize,
    out_chunk: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &PageAlignedQ4_0Swizzled1x4,
    selector: Q8DotKernelSelector,
) {
    debug_assert_eq!(out_chunk.len(), 4);

    let mut sums = [0.0_f32; 4];
    let chunk = matrix.chunk_ptr(chunk_idx);
    for (b, x_scale) in x_scales
        .iter()
        .copied()
        .enumerate()
        .take(matrix.blocks_per_row())
    {
        #[cfg(target_arch = "aarch64")]
        {
            let next_b = b + 1;
            if next_b < matrix.blocks_per_row() {
                let next_w_base = next_b * 4;
                unsafe {
                    std::arch::asm!(
                        "prfm pldl1keep, [{ptr}]",
                        ptr = in(reg) chunk.add(next_w_base),
                        options(nostack, preserves_flags, readonly)
                    );
                }
            }
        }
        let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
        let w_base = b * 4;
        let row0 = unsafe { &*chunk.add(w_base) };
        let row1 = unsafe { &*chunk.add(w_base + 1) };
        let row2 = unsafe { &*chunk.add(w_base + 2) };
        let row3 = unsafe { &*chunk.add(w_base + 3) };
        let dots = if selector.selected == Q8DotKernel::Sdot && q4_1x4_sdot_enabled() {
            dot_q4_0_q8_0_1x4_sdot([row0, row1, row2, row3], x_block_vals)
        } else {
            [
                crate::q8::dot_q4_0_q8_0_with_selector(row0, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row1, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row2, x_block_vals, selector),
                crate::q8::dot_q4_0_q8_0_with_selector(row3, x_block_vals, selector),
            ]
        };
        sums[0] += row0.scale_f32() * x_scale * dots[0] as f32;
        sums[1] += row1.scale_f32() * x_scale * dots[1] as f32;
        sums[2] += row2.scale_f32() * x_scale * dots[2] as f32;
        sums[3] += row3.scale_f32() * x_scale * dots[3] as f32;
    }

    out_chunk.copy_from_slice(&sums);
}

pub fn matmul_q6_k(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q6KBlock],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    #[cfg(target_arch = "aarch64")]
    if q6_k_sdot_enabled() && std::arch::is_aarch64_feature_detected!("dotprod") {
        for_each_matmul_row(out, |r, out_val| {
            let mut sum = 0.0_f32;
            let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
            for (block_idx, w_block) in w_row.iter().enumerate() {
                let x_block_start = block_idx * QK_K_BLOCK_SIZE;
                let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
                sum += unsafe {
                    w_block.dot_q8_scaled_sdot(
                        &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                        &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                    )
                };
            }
            *out_val = sum;
        });
        return;
    }

    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for (block_idx, w_block) in w_row.iter().enumerate() {
            let x_block_start = block_idx * QK_K_BLOCK_SIZE;
            let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
            sum += w_block.dot_q8_scaled(
                &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
            );
        }
        *out_val = sum;
    });
}

pub fn matmul_q4_k(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[crate::q8::Q4KBlock],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for (block_idx, w_block) in w_row.iter().enumerate() {
            let x_block_start = block_idx * QK_K_BLOCK_SIZE;
            let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
            let unpacked = w_block.unpack();
            sum += crate::q8::Q4KBlock::dot_q8_scaled_preloaded(
                &unpacked,
                &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
            );
        }
        *out_val = sum;
    });
}

pub fn matmul_q2_k(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q2KBlock],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for (block_idx, w_block) in w_row.iter().enumerate() {
            let x_block_start = block_idx * QK_K_BLOCK_SIZE;
            let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
            sum += w_block.dot_q8_scaled(
                &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
            );
        }
        *out_val = sum;
    });
}

pub fn matmul_q8_k(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q8KBlock],
    _rows: usize,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for (block_idx, w_block) in w_row.iter().enumerate() {
            let x_block_start = block_idx * QK_K_BLOCK_SIZE;
            let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
            sum += w_block.dot_q8_scaled(
                &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
                selector,
            );
        }
        *out_val = sum;
    });
}

pub fn matmul_iq4_nl(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[IQ4NLBlock],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / 32;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for b in 0..blocks_per_row {
            let w_block = &w_row[b];
            let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
            sum += w_block.dot_q8_scaled(x_block_vals, x_scales[b]);
        }
        *out_val = sum;
    });
}

pub fn matmul_q3_k(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q3KBlock],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for (block_idx, w_block) in w_row.iter().enumerate() {
            let x_block_start = block_idx * QK_K_BLOCK_SIZE;
            let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
            sum += w_block.dot_q8_scaled(
                &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
            );
        }
        *out_val = sum;
    });
}

pub fn matmul_q5_k(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q5KBlock],
    _rows: usize,
    cols: usize,
) {
    let blocks_per_row = cols / QK_K_BLOCK_SIZE;
    for_each_matmul_row(out, |r, out_val| {
        let mut sum = 0.0_f32;
        let w_row = &w[r * blocks_per_row..(r + 1) * blocks_per_row];
        for (block_idx, w_block) in w_row.iter().enumerate() {
            let x_block_start = block_idx * QK_K_BLOCK_SIZE;
            let x_scale_start = x_block_start / Q8_BLOCK_SIZE;
            let unpacked = w_block.unpack();
            sum += crate::q8::Q5KBlock::dot_q8_scaled_preloaded(
                &unpacked,
                &x_i8[x_block_start..x_block_start + QK_K_BLOCK_SIZE],
                &x_scales[x_scale_start..x_scale_start + (QK_K_BLOCK_SIZE / Q8_BLOCK_SIZE)],
            );
        }
        *out_val = sum;
    });
}

pub fn matmul_f32(out: &mut [f32], x: &[f32], w: &[f32], rows: usize, cols: usize) {
    // Each output row is an independent sequential dot product, so parallelizing
    // over rows is bit-identical to the scalar loop (no change in reduction order).
    // The tied-embedding LM head calls this over vocab_size (~128k) rows, which
    // was single-threaded scalar and dominated decode time.
    if should_parallelize_matmul(rows) {
        out.par_iter_mut()
            .with_min_len(matmul_min_rows())
            .enumerate()
            .for_each(|(r, dst)| {
                let w_row = &w[r * cols..(r + 1) * cols];
                let mut sum = 0.0_f32;
                for c in 0..cols {
                    sum += x[c] * w_row[c];
                }
                *dst = sum;
            });
    } else {
        for r in 0..rows {
            let mut sum = 0.0_f32;
            let w_row = &w[r * cols..(r + 1) * cols];
            for c in 0..cols {
                sum += x[c] * w_row[c];
            }
            out[r] = sum;
        }
    }
}

pub fn add_bias(values: &mut [f32], bias: &[f32]) {
    debug_assert_eq!(values.len(), bias.len());
    for (value, bias) in values.iter_mut().zip(bias) {
        *value += bias;
    }
}

pub fn silu(x: &mut [f32]) {
    for val in x.iter_mut() {
        let sigmoid = 1.0 / (1.0 + (-*val).exp());
        *val *= sigmoid;
    }
}

pub fn fused_silu_mul(gate_up: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert_eq!(gate_up.len(), gate.len());
    debug_assert_eq!(gate.len(), up.len());
    if gate_up.len() >= matmul_min_rows() {
        gate_up
            .par_iter_mut()
            .with_min_len(matmul_min_rows())
            .zip(gate.par_iter())
            .zip(up.par_iter())
            .for_each(|((dst, &g), &u)| {
                let sigmoid = 1.0 / (1.0 + (-g).exp());
                *dst = g * sigmoid * u;
            });
    } else {
        gate_up
            .iter_mut()
            .zip(gate)
            .zip(up)
            .for_each(|((dst, &g), &u)| {
                let sigmoid = 1.0 / (1.0 + (-g).exp());
                *dst = g * sigmoid * u;
            });
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertRoute {
    pub expert_idx: usize,
    pub weight: f32,
}

pub fn route_token_to_experts(
    hidden: &[f32],
    router: &[f32],
    expert_count: usize,
    expert_used_count: usize,
) -> Vec<ExpertRoute> {
    debug_assert!(expert_count > 0);
    debug_assert!(expert_used_count > 0);
    debug_assert!(expert_used_count <= expert_count);
    debug_assert_eq!(router.len(), expert_count * hidden.len());

    let mut logits = vec![0.0_f32; expert_count];
    for (expert_idx, logit) in logits.iter_mut().enumerate() {
        let row_start = expert_idx * hidden.len();
        *logit = router[row_start..row_start + hidden.len()]
            .iter()
            .zip(hidden)
            .map(|(&weight, &value)| weight * value)
            .sum();
    }

    let mut selected = Vec::with_capacity(expert_used_count);
    for expert_idx in 0..expert_count {
        let logit = logits[expert_idx];
        let insert_at = selected
            .iter()
            .position(|route: &ExpertRoute| logit > logits[route.expert_idx])
            .unwrap_or(selected.len());
        if insert_at < expert_used_count {
            selected.insert(
                insert_at,
                ExpertRoute {
                    expert_idx,
                    weight: logit,
                },
            );
            selected.truncate(expert_used_count);
        }
    }

    let max_logit = selected
        .iter()
        .map(|route| route.weight)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0_f32;
    for route in &mut selected {
        route.weight = (route.weight - max_logit).exp();
        exp_sum += route.weight;
    }
    if exp_sum != 0.0 {
        for route in &mut selected {
            route.weight /= exp_sum;
        }
    }

    selected
}

struct SingleTokenFfnBuffers<'a> {
    ffn_gate: &'a mut [f32],
    ffn_up: &'a mut [f32],
    ffn_gate_up: &'a mut [f32],
    x_i8: &'a mut [i8],
    x_scales: &'a mut [f32],
    expert_out: &'a mut [f32],
}

#[allow(clippy::too_many_arguments)]
fn run_moe_ffn_single_token(
    hidden_out: &mut [f32],
    residual: &[f32],
    norm_x: &[f32],
    router: &[f32],
    expert_used_count: usize,
    experts: &[MoeExpertWeights],
    config: &LlamaModelConfig,
    options: LlamaRuntimeOptions,
    buffers: SingleTokenFfnBuffers<'_>,
) {
    hidden_out.fill(0.0);
    let routes = route_token_to_experts(norm_x, router, experts.len(), expert_used_count);
    for route in routes {
        let expert = &experts[route.expert_idx];
        quantize_f32_to_q8_0(norm_x, buffers.x_i8, buffers.x_scales);
        matmul_quantized(
            buffers.ffn_gate,
            buffers.x_i8,
            buffers.x_scales,
            &expert.w1,
            config.feed_forward_length,
            config.embedding_length,
            options.q8_selector,
        );
        matmul_quantized(
            buffers.ffn_up,
            buffers.x_i8,
            buffers.x_scales,
            &expert.w3,
            config.feed_forward_length,
            config.embedding_length,
            options.q8_selector,
        );
        fused_silu_mul(buffers.ffn_gate_up, buffers.ffn_gate, buffers.ffn_up);

        quantize_f32_to_q8_0(buffers.ffn_gate_up, buffers.x_i8, buffers.x_scales);
        matmul_quantized(
            buffers.expert_out,
            buffers.x_i8,
            buffers.x_scales,
            &expert.w2,
            config.embedding_length,
            config.feed_forward_length,
            options.q8_selector,
        );
        for (accum, &value) in hidden_out.iter_mut().zip(buffers.expert_out.iter()) {
            *accum += route.weight * value;
        }
    }

    for (value, &residual) in hidden_out.iter_mut().zip(residual) {
        *value += residual;
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut max_val = f32::NEG_INFINITY;
    for &val in x.iter() {
        if val > max_val {
            max_val = val;
        }
    }

    let mut sum = 0.0;
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    let inv_sum = 1.0 / sum;
    for val in x.iter_mut() {
        *val *= inv_sum;
    }
}

fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON/ASIMD support.
        unsafe { dot_f32_neon(lhs, rhs) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_f32_scalar(lhs, rhs)
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dot_f32_scalar(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(&a, &b)| a * b).sum()
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dot_f32_neon(lhs: &[f32], rhs: &[f32]) -> f32 {
    use std::arch::aarch64::{vaddq_f32, vaddvq_f32, vdupq_n_f32, vld1q_f32, vmlaq_f32};

    let mut acc0 = unsafe { vdupq_n_f32(0.0) };
    let mut acc1 = unsafe { vdupq_n_f32(0.0) };
    let mut acc2 = unsafe { vdupq_n_f32(0.0) };
    let mut acc3 = unsafe { vdupq_n_f32(0.0) };
    let chunks = lhs.len() / 16;
    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 16;
        unsafe {
            acc0 = vmlaq_f32(
                acc0,
                vld1q_f32(lhs.as_ptr().add(offset)),
                vld1q_f32(rhs.as_ptr().add(offset)),
            );
            acc1 = vmlaq_f32(
                acc1,
                vld1q_f32(lhs.as_ptr().add(offset + 4)),
                vld1q_f32(rhs.as_ptr().add(offset + 4)),
            );
            acc2 = vmlaq_f32(
                acc2,
                vld1q_f32(lhs.as_ptr().add(offset + 8)),
                vld1q_f32(rhs.as_ptr().add(offset + 8)),
            );
            acc3 = vmlaq_f32(
                acc3,
                vld1q_f32(lhs.as_ptr().add(offset + 12)),
                vld1q_f32(rhs.as_ptr().add(offset + 12)),
            );
        }
    }

    let mut sum = unsafe { vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3))) };
    for idx in chunks * 16..lhs.len() {
        sum += lhs[idx] * rhs[idx];
    }
    sum
}

fn add_weighted_f32(out: &mut [f32], values: &[f32], weight: f32) {
    debug_assert_eq!(out.len(), values.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON/ASIMD support.
        unsafe {
            add_weighted_f32_neon(out, values, weight);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        add_weighted_f32_scalar(out, values, weight);
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn add_weighted_f32_scalar(out: &mut [f32], values: &[f32], weight: f32) {
    for (dst, &value) in out.iter_mut().zip(values) {
        *dst += weight * value;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn add_weighted_f32_neon(out: &mut [f32], values: &[f32], weight: f32) {
    use std::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmlaq_f32, vst1q_f32};

    let weight_vec = unsafe { vdupq_n_f32(weight) };
    let chunks = out.len() / 16;
    for chunk_idx in 0..chunks {
        let offset = chunk_idx * 16;
        for lane_offset in [0, 4, 8, 12] {
            unsafe {
                let dst_ptr = out.as_mut_ptr().add(offset + lane_offset);
                let dst = vld1q_f32(dst_ptr);
                let src = vld1q_f32(values.as_ptr().add(offset + lane_offset));
                vst1q_f32(dst_ptr, vmlaq_f32(dst, src, weight_vec));
            }
        }
    }

    for idx in chunks * 16..out.len() {
        out[idx] += weight * values[idx];
    }
}

pub fn forward_pass<'a>(
    token_id: usize,
    pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &'a mut LlamaWorkspace,
    options: LlamaRuntimeOptions,
) -> &'a [f32] {
    forward_pass_inner(
        token_id,
        pos,
        config,
        weights,
        cache,
        ws,
        LlamaRuntimeOptions {
            compute_logits: true,
            ..options
        },
    )
}

pub fn prefill_pass(
    token_id: usize,
    pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &mut LlamaWorkspace,
    options: LlamaRuntimeOptions,
) {
    let _ = forward_pass_inner(
        token_id,
        pos,
        config,
        weights,
        cache,
        ws,
        LlamaRuntimeOptions {
            compute_logits: false,
            ..options
        },
    );
}

pub fn prefill_pass_batch(
    token_ids: &[u32],
    start_pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &mut LlamaBatchWorkspace,
    options: LlamaRuntimeOptions,
) {
    let batch_size = token_ids.len();
    if batch_size == 0 {
        return;
    }
    debug_assert!(batch_size <= ws.max_batch);
    debug_assert!(start_pos + batch_size <= config.context_length);

    for (token_idx, &token_id) in token_ids.iter().enumerate() {
        let emb_start = token_id as usize * config.embedding_length;
        let hidden_start = token_idx * config.embedding_length;
        ws.hidden[hidden_start..hidden_start + config.embedding_length].copy_from_slice(
            &weights.token_embeddings[emb_start..emb_start + config.embedding_length],
        );
    }

    run_layer_range_batch(
        0,
        &weights.layers,
        batch_size,
        start_pos,
        config,
        cache,
        ws,
        options,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn run_layer_range_batch(
    layer_start: usize,
    layers: &[LlamaLayerWeights],
    batch_size: usize,
    start_pos: usize,
    config: &LlamaModelConfig,
    cache: &mut LlamaKvCache,
    ws: &mut LlamaBatchWorkspace,
    options: LlamaRuntimeOptions,
) {
    if batch_size == 0 {
        return;
    }
    debug_assert!(batch_size <= ws.max_batch);
    debug_assert!(start_pos + batch_size <= config.context_length);
    debug_assert!(layer_start + layers.len() <= config.block_count);

    for (local_layer_idx, layer) in layers.iter().enumerate() {
        let layer_started = Instant::now();
        let layer_idx = layer_start + local_layer_idx;
        let hidden_len = batch_size * config.embedding_length;
        let q_len = batch_size * config.attention_output_width;
        let attn_len = batch_size * config.attention_output_width;
        let kv_len = batch_size * config.kv_width;
        let ffn_len = batch_size * config.feed_forward_length;

        ws.residual[..hidden_len].copy_from_slice(&ws.hidden[..hidden_len]);
        let stage_started = Instant::now();
        rms_norm_batch(
            &mut ws.norm_x[..hidden_len],
            &ws.hidden[..hidden_len],
            &layer.attention_norm,
            config.rms_norm_epsilon,
            batch_size,
            config.embedding_length,
        );
        trace_record("batch.attn_norm", stage_started.elapsed());
        let stage_started = Instant::now();
        quantize_f32_to_q8_0_batch(
            &ws.norm_x[..hidden_len],
            &mut ws.x_i8[..batch_size * config.embedding_length],
            &mut ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            batch_size,
            config.embedding_length,
        );
        trace_record("batch.attn_quant", stage_started.elapsed());

        let attention_shape = BatchMatmulShape {
            batch_size,
            rows: config.attention_output_width,
            cols: config.embedding_length,
        };
        let stage_started = Instant::now();
        matmul_quantized_batch(
            &mut ws.q[..q_len],
            &ws.x_i8[..batch_size * config.embedding_length],
            &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            &layer.wq,
            attention_shape,
            options.q8_selector,
        );
        add_bias_batch(
            &mut ws.q[..q_len],
            &layer.wq_bias,
            batch_size,
            config.attention_output_width,
        );
        trace_record("batch.wq", stage_started.elapsed());

        let kv_shape = BatchMatmulShape {
            batch_size,
            rows: config.kv_width,
            cols: config.embedding_length,
        };
        let stage_started = Instant::now();
        matmul_quantized_batch(
            &mut ws.k[..kv_len],
            &ws.x_i8[..batch_size * config.embedding_length],
            &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            &layer.wk,
            kv_shape,
            options.q8_selector,
        );
        add_bias_batch(
            &mut ws.k[..kv_len],
            &layer.wk_bias,
            batch_size,
            config.kv_width,
        );
        trace_record("batch.wk", stage_started.elapsed());
        let stage_started = Instant::now();
        matmul_quantized_batch(
            &mut ws.v[..kv_len],
            &ws.x_i8[..batch_size * config.embedding_length],
            &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
            &layer.wav,
            kv_shape,
            options.q8_selector,
        );
        add_bias_batch(
            &mut ws.v[..kv_len],
            &layer.wav_bias,
            batch_size,
            config.kv_width,
        );
        trace_record("batch.wv", stage_started.elapsed());

        let stage_started = Instant::now();
        for token_idx in 0..batch_size {
            let pos = start_pos + token_idx;
            let q_start = token_idx * config.attention_output_width;
            let kv_start = token_idx * config.kv_width;
            apply_rope(
                &mut ws.q[q_start..q_start + config.attention_output_width],
                pos,
                config.attention_head_count,
                config.head_dim,
                config.rope_dimension_count,
                config.rope_freq_base,
                options.rope_scaling,
                config.rope_style,
            );
            apply_rope(
                &mut ws.k[kv_start..kv_start + config.kv_width],
                pos,
                config.attention_head_count_kv,
                config.head_dim,
                config.rope_dimension_count,
                config.rope_freq_base,
                options.rope_scaling,
                config.rope_style,
            );
            cache.store_kv(
                layer_idx,
                pos,
                &ws.k[kv_start..kv_start + config.kv_width],
                &ws.v[kv_start..kv_start + config.kv_width],
            );
        }
        trace_record("batch.rope_store_kv", stage_started.elapsed());

        let k_cache = cache.get_k_cache(layer_idx);
        let v_cache = cache.get_v_cache(layer_idx);
        let scale = 1.0 / (config.head_dim as f32).sqrt();
        ws.attn_output[..attn_len].fill(0.0);
        let parallel_heads = attention_head_parallel_enabled();

        let stage_started = Instant::now();
        for token_idx in 0..batch_size {
            let pos = start_pos + token_idx;
            let q_token_start = token_idx * config.attention_output_width;
            let out_token_start = token_idx * config.attention_output_width;
            let scores = if parallel_heads
                && ws.attn_scores.len()
                    >= batch_size * config.attention_head_count * config.context_length
            {
                let scores_start = token_idx * config.attention_head_count * config.context_length;
                let scores_end = scores_start + config.attention_head_count * config.context_length;
                &mut ws.attn_scores[scores_start..scores_end]
            } else {
                let scores_start = token_idx * config.context_length;
                &mut ws.attn_scores[scores_start..scores_start + config.context_length]
            };
            apply_attention_heads(
                &mut ws.attn_output
                    [out_token_start..out_token_start + config.attention_output_width],
                scores,
                AttentionInput {
                    q: &ws.q[q_token_start..q_token_start + config.attention_output_width],
                    k_cache,
                    v_cache,
                    pos,
                    head_count: config.attention_head_count,
                    kv_head_count: config.attention_head_count_kv,
                    head_dim: config.head_dim,
                    cache_kv_width: cache.kv_width,
                    context_length: config.context_length,
                    scale,
                },
                parallel_heads,
            );
        }
        trace_record("batch.attention", stage_started.elapsed());

        let stage_started = Instant::now();
        quantize_f32_to_q8_0_batch(
            &ws.attn_output[..attn_len],
            &mut ws.x_i8[..batch_size * config.attention_output_width],
            &mut ws.x_scales[..batch_size * (config.attention_output_width / Q8_BLOCK_SIZE)],
            batch_size,
            config.attention_output_width,
        );
        trace_record("batch.wo_quant", stage_started.elapsed());
        let stage_started = Instant::now();
        matmul_quantized_batch(
            &mut ws.hidden[..hidden_len],
            &ws.x_i8[..batch_size * config.attention_output_width],
            &ws.x_scales[..batch_size * (config.attention_output_width / Q8_BLOCK_SIZE)],
            &layer.wo,
            BatchMatmulShape {
                batch_size,
                rows: config.embedding_length,
                cols: config.attention_output_width,
            },
            options.q8_selector,
        );
        add_residual_batch(&mut ws.hidden[..hidden_len], &ws.residual[..hidden_len]);
        trace_record("batch.wo", stage_started.elapsed());

        ws.residual[..hidden_len].copy_from_slice(&ws.hidden[..hidden_len]);
        let stage_started = Instant::now();
        rms_norm_batch(
            &mut ws.norm_x[..hidden_len],
            &ws.hidden[..hidden_len],
            &layer.ffn_norm,
            config.rms_norm_epsilon,
            batch_size,
            config.embedding_length,
        );
        trace_record("batch.ffn_norm", stage_started.elapsed());
        match &layer.ffn {
            LlamaFfnWeights::Dense { w1, w3, w2 } => {
                let stage_started = Instant::now();
                quantize_f32_to_q8_0_batch(
                    &ws.norm_x[..hidden_len],
                    &mut ws.x_i8[..batch_size * config.embedding_length],
                    &mut ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
                    batch_size,
                    config.embedding_length,
                );
                trace_record("batch.ffn_quant", stage_started.elapsed());

                let ffn_shape = BatchMatmulShape {
                    batch_size,
                    rows: config.feed_forward_length,
                    cols: config.embedding_length,
                };
                let stage_started = Instant::now();
                matmul_quantized_batch(
                    &mut ws.ffn_gate[..ffn_len],
                    &ws.x_i8[..batch_size * config.embedding_length],
                    &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
                    w1,
                    ffn_shape,
                    options.q8_selector,
                );
                trace_record("batch.w1", stage_started.elapsed());
                let stage_started = Instant::now();
                matmul_quantized_batch(
                    &mut ws.ffn_up[..ffn_len],
                    &ws.x_i8[..batch_size * config.embedding_length],
                    &ws.x_scales[..batch_size * (config.embedding_length / Q8_BLOCK_SIZE)],
                    w3,
                    ffn_shape,
                    options.q8_selector,
                );
                trace_record("batch.w3", stage_started.elapsed());
                let stage_started = Instant::now();
                fused_silu_mul(
                    &mut ws.ffn_gate_up[..ffn_len],
                    &ws.ffn_gate[..ffn_len],
                    &ws.ffn_up[..ffn_len],
                );
                trace_record("batch.silu_mul", stage_started.elapsed());

                let stage_started = Instant::now();
                quantize_f32_to_q8_0_batch(
                    &ws.ffn_gate_up[..ffn_len],
                    &mut ws.x_i8[..batch_size * config.feed_forward_length],
                    &mut ws.x_scales[..batch_size * (config.feed_forward_length / Q8_BLOCK_SIZE)],
                    batch_size,
                    config.feed_forward_length,
                );
                trace_record("batch.down_quant", stage_started.elapsed());
                let down_shape = BatchMatmulShape {
                    batch_size,
                    rows: config.embedding_length,
                    cols: config.feed_forward_length,
                };
                let stage_started = Instant::now();
                matmul_quantized_batch(
                    &mut ws.hidden[..hidden_len],
                    &ws.x_i8[..batch_size * config.feed_forward_length],
                    &ws.x_scales[..batch_size * (config.feed_forward_length / Q8_BLOCK_SIZE)],
                    w2,
                    down_shape,
                    options.q8_selector,
                );
                add_residual_batch(&mut ws.hidden[..hidden_len], &ws.residual[..hidden_len]);
                trace_record("batch.w2", stage_started.elapsed());
            }
            LlamaFfnWeights::MoE {
                router,
                expert_used_count,
                experts,
            } => {
                let scale_len = config.feed_forward_length / Q8_BLOCK_SIZE + 1;
                for token_idx in 0..batch_size {
                    let hidden_start = token_idx * config.embedding_length;
                    let hidden_end = hidden_start + config.embedding_length;
                    let ffn_start = token_idx * config.feed_forward_length;
                    let ffn_end = ffn_start + config.feed_forward_length;
                    run_moe_ffn_single_token(
                        &mut ws.hidden[hidden_start..hidden_end],
                        &ws.residual[hidden_start..hidden_end],
                        &ws.norm_x[hidden_start..hidden_end],
                        router,
                        *expert_used_count,
                        experts,
                        config,
                        options,
                        SingleTokenFfnBuffers {
                            ffn_gate: &mut ws.ffn_gate[ffn_start..ffn_end],
                            ffn_up: &mut ws.ffn_up[ffn_start..ffn_end],
                            ffn_gate_up: &mut ws.ffn_gate_up[ffn_start..ffn_end],
                            x_i8: &mut ws.x_i8[..config.feed_forward_length],
                            x_scales: &mut ws.x_scales[..scale_len],
                            expert_out: &mut ws.attn_output[hidden_start..hidden_end],
                        },
                    );
                }
            }
        }
        trace_record("batch.layer_total", layer_started.elapsed());
    }
}

pub fn add_bias_batch(values: &mut [f32], bias: &Option<Vec<f32>>, batch_size: usize, dim: usize) {
    if let Some(bias) = bias {
        for token_idx in 0..batch_size {
            let start = token_idx * dim;
            add_bias(&mut values[start..start + dim], bias);
        }
    }
}

pub fn add_residual_batch(values: &mut [f32], residual: &[f32]) {
    debug_assert_eq!(values.len(), residual.len());
    for (value, residual) in values.iter_mut().zip(residual) {
        *value += residual;
    }
}

pub struct AttentionInput<'a> {
    pub q: &'a [f32],
    pub k_cache: KvCacheSlice<'a>,
    pub v_cache: KvCacheSlice<'a>,
    pub pos: usize,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub cache_kv_width: usize,
    pub context_length: usize,
    pub scale: f32,
}

pub fn apply_attention_heads(
    attn_output: &mut [f32],
    attn_scores: &mut [f32],
    input: AttentionInput<'_>,
    parallel_heads: bool,
) {
    let kv_mul = input.head_count / input.kv_head_count;
    let can_parallelize =
        parallel_heads && attn_scores.len() >= input.head_count * input.context_length;

    if can_parallelize {
        attn_output
            .par_chunks_exact_mut(input.head_dim)
            .zip(attn_scores.par_chunks_exact_mut(input.context_length))
            .enumerate()
            .for_each(|(h, (out_head, scores))| {
                apply_attention_head(out_head, scores, &input, h, kv_mul);
            });
    } else {
        for h in 0..input.head_count {
            let out_start = h * input.head_dim;
            apply_attention_head(
                &mut attn_output[out_start..out_start + input.head_dim],
                attn_scores,
                &input,
                h,
                kv_mul,
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]
/// # Safety
/// This function is unsafe because it performs raw pointer arithmetic and uses unsafe AArch64 intrinsics.
/// The caller must ensure that:
/// - `out_head` is valid for writes of size `head_dim`.
/// - `v_cache` is valid for reads up to `pos * kv_width + kv_h * head_dim + head_dim`.
/// - `scores` contains at least `pos + 1` elements.
pub unsafe fn accumulate_weighted_v_head_aarch64(
    out_head: &mut [f32],
    v_cache: &[f32],
    scores: &[f32],
    kv_width: usize,
    kv_h: usize,
    head_dim: usize,
    pos: usize,
) {
    use std::arch::aarch64::{vdupq_n_f32, vld1q_f32, vmlaq_f32, vst1q_f32};

    if head_dim % 16 == 0 {
        let chunks = head_dim / 16;
        for chunk_idx in 0..chunks {
            let offset = chunk_idx * 16;

            unsafe {
                // 1. Initialize register accumulators to 0
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let mut acc2 = vdupq_n_f32(0.0);
                let mut acc3 = vdupq_n_f32(0.0);

                // 2. Loop over sequence tokens - accumulating directly in CPU registers!
                for p in 0..=pos {
                    let scale_vec = vdupq_n_f32(scores[p]);
                    let v_ptr = v_cache
                        .as_ptr()
                        .add(p * kv_width + kv_h * head_dim + offset);

                    let v0 = vld1q_f32(v_ptr);
                    let v1 = vld1q_f32(v_ptr.add(4));
                    let v2 = vld1q_f32(v_ptr.add(8));
                    let v3 = vld1q_f32(v_ptr.add(12));

                    acc0 = vmlaq_f32(acc0, v0, scale_vec);
                    acc1 = vmlaq_f32(acc1, v1, scale_vec);
                    acc2 = vmlaq_f32(acc2, v2, scale_vec);
                    acc3 = vmlaq_f32(acc3, v3, scale_vec);
                }

                // 3. Store the accumulated values once at the end!
                let dst_ptr = out_head.as_mut_ptr().add(offset);
                vst1q_f32(dst_ptr, acc0);
                vst1q_f32(dst_ptr.add(4), acc1);
                vst1q_f32(dst_ptr.add(8), acc2);
                vst1q_f32(dst_ptr.add(12), acc3);
            }
        }
    } else {
        // Fallback for non-16-multiple head_dims
        for p in 0..=pos {
            let v_head =
                &v_cache[p * kv_width + kv_h * head_dim..p * kv_width + (kv_h + 1) * head_dim];
            add_weighted_f32(out_head, v_head, scores[p]);
        }
    }
}

fn apply_attention_head(
    out_head: &mut [f32],
    scores: &mut [f32],
    input: &AttentionInput<'_>,
    head_idx: usize,
    kv_mul: usize,
) {
    let kv_h = head_idx / kv_mul;
    let q_head = &input.q[head_idx * input.head_dim..(head_idx + 1) * input.head_dim];

    match (input.k_cache, input.v_cache) {
        (KvCacheSlice::F32(k_cache), KvCacheSlice::F32(v_cache)) => {
            for (p, score) in scores.iter_mut().enumerate().take(input.pos + 1) {
                let k_head = &k_cache[p * input.cache_kv_width + kv_h * input.head_dim
                    ..p * input.cache_kv_width + (kv_h + 1) * input.head_dim];
                *score = dot_f32(q_head, k_head) * input.scale;
            }
            softmax(&mut scores[0..=input.pos]);

            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    accumulate_weighted_v_head_aarch64(
                        out_head,
                        v_cache,
                        scores,
                        input.cache_kv_width,
                        kv_h,
                        input.head_dim,
                        input.pos,
                    );
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for (p, &score) in scores.iter().enumerate().take(input.pos + 1) {
                    let v_head = &v_cache[p * input.cache_kv_width + kv_h * input.head_dim
                        ..p * input.cache_kv_width + (kv_h + 1) * input.head_dim];
                    add_weighted_f32(out_head, v_head, score);
                }
            }
        }
        (KvCacheSlice::F16(k_cache), KvCacheSlice::F16(v_cache)) => {
            apply_attention_head_f16(out_head, scores, input, kv_h, q_head, k_cache, v_cache);
        }
        (KvCacheSlice::Q8_0(k_cache), KvCacheSlice::Q8_0(v_cache)) => {
            apply_attention_head_q8_0(out_head, scores, input, kv_h, q_head, k_cache, v_cache);
        }
        _ => unreachable!("KV cache key and value storage kinds must match"),
    }
}

/// Scalar quantization of a contiguous f32 row into `Q8_0Block`s.
///
/// Correctness over speed: each 32-element chunk gets its own `scale = max_abs / 127`
/// and elements are rounded to the nearest `i8`. `src.len()` must equal
/// `out.len() * Q8_BLOCK_SIZE`.
fn quantize_row_to_q8_0_blocks(src: &[f32], out: &mut [Q8_0Block]) {
    debug_assert_eq!(src.len(), out.len() * Q8_BLOCK_SIZE);
    for (chunk, block) in src.chunks_exact(Q8_BLOCK_SIZE).zip(out.iter_mut()) {
        let mut max_abs = 0.0f32;
        for &x in chunk {
            let a = x.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = max_abs / 127.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let mut values = [0i8; Q8_BLOCK_SIZE];
        for (dst, &x) in values.iter_mut().zip(chunk) {
            *dst = (x * inv).round().clamp(-127.0, 127.0) as i8;
        }
        *block = Q8_0Block::from_parts(crate::q8::fast_f32_to_f16(scale), values);
    }
}

fn apply_attention_head_q8_0(
    out_head: &mut [f32],
    scores: &mut [f32],
    input: &AttentionInput<'_>,
    kv_h: usize,
    q_head: &[f32],
    k_cache: &[Q8_0Block],
    v_cache: &[Q8_0Block],
) {
    assert!(
        input.head_dim.is_multiple_of(Q8_BLOCK_SIZE),
        "q8_0 KV cache attention requires head_dim to be a multiple of {Q8_BLOCK_SIZE} (got {})",
        input.head_dim
    );
    let blocks_per_head = input.head_dim / Q8_BLOCK_SIZE;
    let blocks_per_kv_width = input.cache_kv_width / Q8_BLOCK_SIZE;

    // Quantize q_head once (mirrors the store quantization) into stack/heap blocks.
    const MAX_STACK_HEAD_BLOCKS: usize = 256 / Q8_BLOCK_SIZE;
    let mut q_blocks_stack = [Q8_0Block::from_parts(0, [0i8; Q8_BLOCK_SIZE]); MAX_STACK_HEAD_BLOCKS];
    let mut q_blocks_heap;
    let q_blocks: &mut [Q8_0Block] = if blocks_per_head <= MAX_STACK_HEAD_BLOCKS {
        &mut q_blocks_stack[..blocks_per_head]
    } else {
        q_blocks_heap =
            vec![Q8_0Block::from_parts(0, [0i8; Q8_BLOCK_SIZE]); blocks_per_head];
        &mut q_blocks_heap[..]
    };
    quantize_row_to_q8_0_blocks(q_head, q_blocks);

    for (p, score) in scores.iter_mut().enumerate().take(input.pos + 1) {
        let base = p * blocks_per_kv_width + kv_h * blocks_per_head;
        let k_head_blocks = &k_cache[base..base + blocks_per_head];
        let mut acc = 0.0f32;
        for b in 0..blocks_per_head {
            acc += q_blocks[b].scaled_dot_f32(&k_head_blocks[b]);
        }
        *score = acc * input.scale;
    }
    softmax(&mut scores[0..=input.pos]);

    const MAX_STACK_HEAD_DIM: usize = 256;
    let mut v_head_stack = [0.0f32; MAX_STACK_HEAD_DIM];
    let mut v_head_heap;
    let v_head_f32: &mut [f32] = if input.head_dim <= MAX_STACK_HEAD_DIM {
        &mut v_head_stack[..input.head_dim]
    } else {
        v_head_heap = vec![0.0f32; input.head_dim];
        &mut v_head_heap[..]
    };

    for (p, &score) in scores.iter().enumerate().take(input.pos + 1) {
        let base = p * blocks_per_kv_width + kv_h * blocks_per_head;
        let v_head_blocks = &v_cache[base..base + blocks_per_head];
        for (b, block) in v_head_blocks.iter().enumerate() {
            let block_scale = block.scale_f32();
            let dst = &mut v_head_f32[b * Q8_BLOCK_SIZE..(b + 1) * Q8_BLOCK_SIZE];
            for (out, &q) in dst.iter_mut().zip(block.values().iter()) {
                *out = q as f32 * block_scale;
            }
        }
        add_weighted_f32(out_head, v_head_f32, score);
    }
}

fn apply_attention_head_f16(
    out_head: &mut [f32],
    scores: &mut [f32],
    input: &AttentionInput<'_>,
    kv_h: usize,
    q_head: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
) {
    const MAX_STACK_HEAD_DIM: usize = 256;
    if input.head_dim <= MAX_STACK_HEAD_DIM {
        let mut k_head_f32 = [0.0; MAX_STACK_HEAD_DIM];
        let mut v_head_f32 = [0.0; MAX_STACK_HEAD_DIM];
        apply_attention_head_f16_with_buffers(
            out_head,
            scores,
            input,
            kv_h,
            q_head,
            k_cache,
            v_cache,
            &mut k_head_f32[..input.head_dim],
            &mut v_head_f32[..input.head_dim],
        );
    } else {
        let mut k_head_f32 = vec![0.0; input.head_dim];
        let mut v_head_f32 = vec![0.0; input.head_dim];
        apply_attention_head_f16_with_buffers(
            out_head,
            scores,
            input,
            kv_h,
            q_head,
            k_cache,
            v_cache,
            &mut k_head_f32,
            &mut v_head_f32,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_attention_head_f16_with_buffers(
    out_head: &mut [f32],
    scores: &mut [f32],
    input: &AttentionInput<'_>,
    kv_h: usize,
    q_head: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    k_head_f32: &mut [f32],
    v_head_f32: &mut [f32],
) {
    for (p, score) in scores.iter_mut().enumerate().take(input.pos + 1) {
        let k_head = &k_cache[p * input.cache_kv_width + kv_h * input.head_dim
            ..p * input.cache_kv_width + (kv_h + 1) * input.head_dim];
        decode_f16_head(k_head, k_head_f32);
        *score = dot_f32(q_head, k_head_f32) * input.scale;
    }
    softmax(&mut scores[0..=input.pos]);

    for (p, &score) in scores.iter().enumerate().take(input.pos + 1) {
        let v_head = &v_cache[p * input.cache_kv_width + kv_h * input.head_dim
            ..p * input.cache_kv_width + (kv_h + 1) * input.head_dim];
        decode_f16_head(v_head, v_head_f32);
        add_weighted_f32(out_head, v_head_f32, score);
    }
}

fn decode_f16_head(src: &[u16], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for (out, &bits) in dst.iter_mut().zip(src) {
        *out = crate::q8::fast_f16_to_f32(bits);
    }
}

fn forward_pass_inner<'a>(
    token_id: usize,
    pos: usize,
    config: &LlamaModelConfig,
    weights: &LlamaWeights,
    cache: &mut LlamaKvCache,
    ws: &'a mut LlamaWorkspace,
    options: LlamaRuntimeOptions,
) -> &'a [f32] {
    embed_token(token_id, config, &weights.token_embeddings, ws);

    run_layer_range(0, &weights.layers, pos, config, cache, ws, options);

    if !options.compute_logits {
        return &ws.logits;
    }

    compute_logits_from_hidden(
        config,
        &weights.token_embeddings,
        &weights.output_norm,
        weights.output_projection.as_ref(),
        ws,
        options,
    )
}

pub fn embed_token(
    token_id: usize,
    config: &LlamaModelConfig,
    token_embeddings: &[f32],
    ws: &mut LlamaWorkspace,
) {
    let emb_start = token_id * config.embedding_length;
    ws.hidden
        .copy_from_slice(&token_embeddings[emb_start..emb_start + config.embedding_length]);
}

pub fn run_layer_range(
    layer_start: usize,
    layers: &[LlamaLayerWeights],
    pos: usize,
    config: &LlamaModelConfig,
    cache: &mut LlamaKvCache,
    ws: &mut LlamaWorkspace,
    options: LlamaRuntimeOptions,
) {
    debug_assert!(layer_start + layers.len() <= config.block_count);

    for (local_layer_idx, layer) in layers.iter().enumerate() {
        let layer_started = Instant::now();
        let layer_idx = layer_start + local_layer_idx;

        // --- Attention block ---
        // Save residual
        ws.residual.copy_from_slice(&ws.hidden);

        // RMSNorm
        let stage_started = Instant::now();
        rms_norm(
            &mut ws.norm_x,
            &ws.hidden,
            &layer.attention_norm,
            config.rms_norm_epsilon,
        );
        trace_record("decode.attn_norm", stage_started.elapsed());

        // Quantize normalized activations for Q8 matmul
        let stage_started = Instant::now();
        quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);
        trace_record("decode.attn_quant", stage_started.elapsed());

        // Q, K, V Projections
        let stage_started = Instant::now();
        matmul_quantized(
            &mut ws.q,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wq,
            config.attention_output_width,
            config.embedding_length,
            options.q8_selector,
        );
        if let Some(bias) = &layer.wq_bias {
            add_bias(&mut ws.q, bias);
        }
        trace_record("decode.wq", stage_started.elapsed());
        let stage_started = Instant::now();
        matmul_quantized(
            &mut ws.k,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wk,
            config.kv_width,
            config.embedding_length,
            options.q8_selector,
        );
        if let Some(bias) = &layer.wk_bias {
            add_bias(&mut ws.k, bias);
        }
        trace_record("decode.wk", stage_started.elapsed());
        let stage_started = Instant::now();
        matmul_quantized(
            &mut ws.v,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wav,
            config.kv_width,
            config.embedding_length,
            options.q8_selector,
        );
        if let Some(bias) = &layer.wav_bias {
            add_bias(&mut ws.v, bias);
        }
        trace_record("decode.wv", stage_started.elapsed());

        // Apply RoPE
        let stage_started = Instant::now();
        apply_rope(
            &mut ws.q,
            pos,
            config.attention_head_count,
            config.head_dim,
            config.rope_dimension_count,
            config.rope_freq_base,
            options.rope_scaling,
                config.rope_style,
        );

        apply_rope(
            &mut ws.k,
            pos,
            config.attention_head_count_kv,
            config.head_dim,
            config.rope_dimension_count,
            config.rope_freq_base,
            options.rope_scaling,
                config.rope_style,
        );
        trace_record("decode.rope", stage_started.elapsed());

        // Store to KV Cache
        let stage_started = Instant::now();
        cache.store_kv(layer_idx, pos, &ws.k, &ws.v);
        trace_record("decode.store_kv", stage_started.elapsed());

        // Retrieve K, V history
        let k_cache = cache.get_k_cache(layer_idx);
        let v_cache = cache.get_v_cache(layer_idx);

        // Score scale
        let scale = 1.0 / (config.head_dim as f32).sqrt();

        // Compute attention outputs per head
        ws.attn_output.fill(0.0);
        let stage_started = Instant::now();
        apply_attention_heads(
            &mut ws.attn_output,
            &mut ws.attn_scores,
            AttentionInput {
                q: &ws.q,
                k_cache,
                v_cache,
                pos,
                head_count: config.attention_head_count,
                kv_head_count: config.attention_head_count_kv,
                head_dim: config.head_dim,
                cache_kv_width: cache.kv_width,
                context_length: config.context_length,
                scale,
            },
            attention_head_parallel_enabled(),
        );
        trace_record("decode.attention", stage_started.elapsed());

        // Projection O (wo)
        let stage_started = Instant::now();
        quantize_f32_to_q8_0(&ws.attn_output, &mut ws.x_i8, &mut ws.x_scales);
        trace_record("decode.wo_quant", stage_started.elapsed());
        let stage_started = Instant::now();
        matmul_quantized(
            &mut ws.hidden,
            &ws.x_i8,
            &ws.x_scales,
            &layer.wo,
            config.embedding_length,
            config.attention_output_width,
            options.q8_selector,
        );

        // Residual addition
        for i in 0..config.embedding_length {
            ws.hidden[i] += ws.residual[i];
        }
        trace_record("decode.wo", stage_started.elapsed());

        // --- FFN block ---
        ws.residual.copy_from_slice(&ws.hidden);

        // RMSNorm
        let stage_started = Instant::now();
        rms_norm(
            &mut ws.norm_x,
            &ws.hidden,
            &layer.ffn_norm,
            config.rms_norm_epsilon,
        );
        trace_record("decode.ffn_norm", stage_started.elapsed());

        match &layer.ffn {
            LlamaFfnWeights::Dense { w1, w3, w2 } => {
                // Quantize
                let stage_started = Instant::now();
                quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);
                trace_record("decode.ffn_quant", stage_started.elapsed());

                // Gate (w1) & Up (w3) matmuls
                let stage_started = Instant::now();
                matmul_quantized(
                    &mut ws.ffn_gate,
                    &ws.x_i8,
                    &ws.x_scales,
                    w1,
                    config.feed_forward_length,
                    config.embedding_length,
                    options.q8_selector,
                );
                trace_record("decode.w1", stage_started.elapsed());
                let stage_started = Instant::now();
                matmul_quantized(
                    &mut ws.ffn_up,
                    &ws.x_i8,
                    &ws.x_scales,
                    w3,
                    config.feed_forward_length,
                    config.embedding_length,
                    options.q8_selector,
                );
                trace_record("decode.w3", stage_started.elapsed());

                // Fused SiLU activation on Gate and element-wise product with Up
                let stage_started = Instant::now();
                fused_silu_mul(&mut ws.ffn_gate_up, &ws.ffn_gate, &ws.ffn_up);
                trace_record("decode.silu_mul", stage_started.elapsed());

                // Down projection (w2)
                let stage_started = Instant::now();
                quantize_f32_to_q8_0(&ws.ffn_gate_up, &mut ws.x_i8, &mut ws.x_scales);
                trace_record("decode.down_quant", stage_started.elapsed());
                let stage_started = Instant::now();
                matmul_quantized(
                    &mut ws.hidden,
                    &ws.x_i8,
                    &ws.x_scales,
                    w2,
                    config.embedding_length,
                    config.feed_forward_length,
                    options.q8_selector,
                );

                // Residual addition
                for i in 0..config.embedding_length {
                    ws.hidden[i] += ws.residual[i];
                }
                trace_record("decode.w2", stage_started.elapsed());
            }
            LlamaFfnWeights::MoE {
                router,
                expert_used_count,
                experts,
            } => {
                run_moe_ffn_single_token(
                    &mut ws.hidden,
                    &ws.residual,
                    &ws.norm_x,
                    router,
                    *expert_used_count,
                    experts,
                    config,
                    options,
                    SingleTokenFfnBuffers {
                        ffn_gate: &mut ws.ffn_gate,
                        ffn_up: &mut ws.ffn_up,
                        ffn_gate_up: &mut ws.ffn_gate_up,
                        x_i8: &mut ws.x_i8,
                        x_scales: &mut ws.x_scales,
                        expert_out: &mut ws.attn_output,
                    },
                );
            }
        }
        trace_record("decode.layer_total", layer_started.elapsed());
    }
}

pub fn compute_logits_from_hidden<'a>(
    config: &LlamaModelConfig,
    token_embeddings: &[f32],
    output_norm: &[f32],
    output_projection: Option<&QuantizedMatrix>,
    ws: &'a mut LlamaWorkspace,
    options: LlamaRuntimeOptions,
) -> &'a [f32] {
    rms_norm(
        &mut ws.norm_x,
        &ws.hidden,
        output_norm,
        config.rms_norm_epsilon,
    );

    if let Some(out_proj) = output_projection {
        quantize_f32_to_q8_0(&ws.norm_x, &mut ws.x_i8, &mut ws.x_scales);
        matmul_quantized(
            &mut ws.logits,
            &ws.x_i8,
            &ws.x_scales,
            out_proj,
            config.vocab_size,
            config.embedding_length,
            options.q8_selector,
        );
    } else {
        // Tied embeddings (multiply norm_x by token_embeddings transposed)
        matmul_f32(
            &mut ws.logits,
            &ws.norm_x,
            token_embeddings,
            config.vocab_size,
            config.embedding_length,
        );
    }

    &ws.logits
}

pub fn sample_logits(logits: &[f32], temperature: f32) -> usize {
    if temperature <= 0.0 {
        // Greedy sampling: argmax
        let mut max_val = logits[0];
        let mut max_idx = 0;
        for (i, &val) in logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    } else {
        // Temperature sampling
        let mut scaled_probs = vec![0.0_f32; logits.len()];
        let mut max_logit = logits[0];
        for &val in logits.iter() {
            if val > max_logit {
                max_logit = val;
            }
        }

        let mut sum = 0.0;
        for (i, &val) in logits.iter().enumerate() {
            let p = ((val - max_logit) / temperature).exp();
            scaled_probs[i] = p;
            sum += p;
        }

        // Draw sample
        let mut r = rand_simple() * sum;
        for (i, &p) in scaled_probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        logits.len() - 1
    }
}

// Simple deterministic pseudo-random number generator for zero-dependency sampling
static mut RNG_STATE: u64 = 0x123456789abcdef0;
fn rand_simple() -> f32 {
    unsafe {
        RNG_STATE = RNG_STATE.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((RNG_STATE >> 32) as u32) as f32;
        x / (u32::MAX as f32)
    }
}

pub const Q8_1X4_SDOT_ENV: &str = "NANOCAMELID_Q8_1X4_SDOT";

fn q8_1x4_sdot_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(Q8_1X4_SDOT_ENV)
            .ok()
            .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "on" | "ON" | "yes"))
            .unwrap_or(true)
    })
}

fn matmul_q8_0_swizzled_1x4(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &crate::model::Q8_0Swizzled1x4Matrix,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / 32;
    let w = &matrix.swizzled_1x4;
    debug_assert_eq!(w.len(), out.len() * blocks_per_row);
    debug_assert!(out.len().is_multiple_of(4));

    if let Some(aligned) = &matrix.page_aligned_1x4 {
        matmul_q8_0_page_aligned_1x4(out, x_i8, x_scales, aligned, cols, selector);
        return;
    }

    if should_parallelize_matmul(out.len()) {
        out.par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q8_0_sdot_1x4_swizzled_chunk(
                    chunk_idx,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    } else {
        out.chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q8_0_sdot_1x4_swizzled_chunk(
                    chunk_idx,
                    out_chunk,
                    x_i8,
                    x_scales,
                    w,
                    blocks_per_row,
                    selector,
                );
            });
    }
}

fn compute_q8_0_sdot_1x4_swizzled_chunk(
    chunk_idx: usize,
    out_chunk: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q8_0Block],
    blocks_per_row: usize,
    selector: Q8DotKernelSelector,
) {
    debug_assert_eq!(out_chunk.len(), 4);

    let mut sums = [0.0_f32; 4];
    let chunk_base = chunk_idx * blocks_per_row * 4;
    for (b, x_scale) in x_scales.iter().copied().enumerate().take(blocks_per_row) {
        let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
        let w_base = chunk_base + b * 4;
        let row0 = &w[w_base];
        let row1 = &w[w_base + 1];
        let row2 = &w[w_base + 2];
        let row3 = &w[w_base + 3];
        let dots = if selector.selected == Q8DotKernel::Sdot && q8_1x4_sdot_enabled() {
            crate::q8::dot_q8_0_q8_0_1x4_sdot([row0, row1, row2, row3], x_block_vals)
        } else {
            [
                crate::q8::dot_i8_neon_32_selected(row0.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row1.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row2.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row3.values(), x_block_vals),
            ]
        };
        sums[0] += row0.scale_f32() * x_scale * dots[0] as f32;
        sums[1] += row1.scale_f32() * x_scale * dots[1] as f32;
        sums[2] += row2.scale_f32() * x_scale * dots[2] as f32;
        sums[3] += row3.scale_f32() * x_scale * dots[3] as f32;
    }
    out_chunk.copy_from_slice(&sums);
}

fn matmul_q8_0_page_aligned_1x4(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &PageAlignedQ8_0Swizzled1x4,
    cols: usize,
    selector: Q8DotKernelSelector,
) {
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    debug_assert_eq!(matrix.blocks_per_row(), blocks_per_row);
    debug_assert_eq!(matrix.chunk_count(), out.len() / 4);
    debug_assert!(out.len().is_multiple_of(4));

    if should_parallelize_matmul(out.len()) {
        out.par_chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q8_0_page_aligned_1x4_chunk(
                    chunk_idx, out_chunk, x_i8, x_scales, matrix, selector,
                );
            });
    } else {
        out.chunks_mut(4)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                compute_q8_0_page_aligned_1x4_chunk(
                    chunk_idx, out_chunk, x_i8, x_scales, matrix, selector,
                );
            });
    }
}

fn compute_q8_0_page_aligned_1x4_chunk(
    chunk_idx: usize,
    out_chunk: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &PageAlignedQ8_0Swizzled1x4,
    selector: Q8DotKernelSelector,
) {
    debug_assert_eq!(out_chunk.len(), 4);

    let mut sums = [0.0_f32; 4];
    let chunk = matrix.chunk_ptr(chunk_idx);
    for (b, x_scale) in x_scales
        .iter()
        .copied()
        .enumerate()
        .take(matrix.blocks_per_row())
    {
        let x_block_vals = unsafe { activation_block_ptr(x_i8, b) };
        let w_base = b * 4;
        let row0 = unsafe { &*chunk.add(w_base) };
        let row1 = unsafe { &*chunk.add(w_base + 1) };
        let row2 = unsafe { &*chunk.add(w_base + 2) };
        let row3 = unsafe { &*chunk.add(w_base + 3) };
        let dots = if selector.selected == Q8DotKernel::Sdot && q8_1x4_sdot_enabled() {
            crate::q8::dot_q8_0_q8_0_1x4_sdot([row0, row1, row2, row3], x_block_vals)
        } else {
            [
                crate::q8::dot_i8_neon_32_selected(row0.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row1.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row2.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row3.values(), x_block_vals),
            ]
        };
        sums[0] += row0.scale_f32() * x_scale * dots[0] as f32;
        sums[1] += row1.scale_f32() * x_scale * dots[1] as f32;
        sums[2] += row2.scale_f32() * x_scale * dots[2] as f32;
        sums[3] += row3.scale_f32() * x_scale * dots[3] as f32;
    }
    out_chunk.copy_from_slice(&sums);
}

fn matmul_q8_0_swizzled_1x4_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &crate::model::Q8_0Swizzled1x4Matrix,
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    let w = &matrix.swizzled_1x4;
    debug_assert_eq!(w.len(), rows * blocks_per_row);
    debug_assert!(rows.is_multiple_of(4));

    if let Some(aligned) = &matrix.page_aligned_1x4 {
        matmul_q8_0_page_aligned_1x4_batch(out, x_i8, x_scales, aligned, shape, selector);
        return;
    }

    if selector.selected == Q8DotKernel::Sdot && q8_1x4_sdot_enabled() {
        matmul_q8_0_swizzled_1x4_sdot_batch(out, x_i8, x_scales, w, shape, blocks_per_row);
        return;
    }

    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let chunk_idx = r / 4;
        let lane = r % 4;
        let chunk_base = chunk_idx * blocks_per_row * 4;
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let w_block = &w[chunk_base + b * 4 + lane];
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val = crate::q8::dot_i8_neon_32_selected(w_block.values(), x_block_vals);
                sum += w_block.scale_f32() * x_token_scales[b] * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

fn matmul_q8_0_swizzled_1x4_sdot_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    w: &[Q8_0Block],
    shape: BatchMatmulShape,
    blocks_per_row: usize,
) {
    let rows = shape.rows;
    let chunks = rows / 4;
    let out_addr = out.as_mut_ptr() as usize;
    let ctx = SwizzledQ8BatchChunkContext {
        out_addr,
        x_i8,
        x_scales,
        w,
        shape,
        blocks_per_row,
    };
    if should_parallelize_matmul(rows) {
        (0..chunks).into_par_iter().for_each(|chunk_idx| {
            compute_q8_0_swizzled_1x4_sdot_batch_chunk(chunk_idx, ctx);
        });
    } else {
        (0..chunks).for_each(|chunk_idx| {
            compute_q8_0_swizzled_1x4_sdot_batch_chunk(chunk_idx, ctx);
        });
    }
}

#[derive(Clone, Copy)]
struct SwizzledQ8BatchChunkContext<'a> {
    out_addr: usize,
    x_i8: &'a [i8],
    x_scales: &'a [f32],
    w: &'a [Q8_0Block],
    shape: BatchMatmulShape,
    blocks_per_row: usize,
}

fn compute_q8_0_swizzled_1x4_sdot_batch_chunk(
    chunk_idx: usize,
    ctx: SwizzledQ8BatchChunkContext<'_>,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = ctx.shape;
    let chunk_base = chunk_idx * ctx.blocks_per_row * 4;
    let row_base = chunk_idx * 4;
    let mut token_sums = vec![[0.0_f32; 4]; batch_size];
    for b in 0..ctx.blocks_per_row {
        let w_base = chunk_base + b * 4;
        let row0 = &ctx.w[w_base];
        let row1 = &ctx.w[w_base + 1];
        let row2 = &ctx.w[w_base + 2];
        let row3 = &ctx.w[w_base + 3];
        let row0_scale = row0.scale_f32();
        let row1_scale = row1.scale_f32();
        let row2_scale = row2.scale_f32();
        let row3_scale = row3.scale_f32();

        #[cfg(target_arch = "aarch64")]
        let unpacked = unsafe {
            use std::arch::aarch64::vld1q_s8;
            crate::q8::Q8_0Unpacked1x4Aarch64 {
                w0_low: vld1q_s8(row0.values().as_ptr()),
                w0_high: vld1q_s8(row0.values().as_ptr().add(16)),
                w1_low: vld1q_s8(row1.values().as_ptr()),
                w1_high: vld1q_s8(row1.values().as_ptr().add(16)),
                w2_low: vld1q_s8(row2.values().as_ptr()),
                w2_high: vld1q_s8(row2.values().as_ptr().add(16)),
                w3_low: vld1q_s8(row3.values().as_ptr()),
                w3_high: vld1q_s8(row3.values().as_ptr().add(16)),
            }
        };

        for (token_idx, sums) in token_sums.iter_mut().enumerate() {
            let x_offset = token_idx * cols;
            let x_scale = ctx.x_scales[token_idx * ctx.blocks_per_row + b];
            let x_token = &ctx.x_i8[x_offset..x_offset + cols];
            let x_block_vals = unsafe { activation_block_ptr(x_token, b) };

            #[cfg(target_arch = "aarch64")]
            let dots = unsafe {
                crate::q8::dot_q8_0_q8_0_1x4_sdot_preloaded_aarch64(unpacked, x_block_vals)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let dots = [
                crate::q8::dot_i8_neon_32_selected(row0.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row1.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row2.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row3.values(), x_block_vals),
            ];

            sums[0] += row0_scale * x_scale * dots[0] as f32;
            sums[1] += row1_scale * x_scale * dots[1] as f32;
            sums[2] += row2_scale * x_scale * dots[2] as f32;
            sums[3] += row3_scale * x_scale * dots[3] as f32;
        }
    }

    for (token_idx, sums) in token_sums.into_iter().enumerate() {
        for (lane, sum) in sums.into_iter().enumerate() {
            unsafe {
                write_batch_out(ctx.out_addr, token_idx, rows, row_base + lane, sum);
            }
        }
    }
}

fn matmul_q8_0_page_aligned_1x4_batch(
    out: &mut [f32],
    x_i8: &[i8],
    x_scales: &[f32],
    matrix: &PageAlignedQ8_0Swizzled1x4,
    shape: BatchMatmulShape,
    selector: Q8DotKernelSelector,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = shape;
    let blocks_per_row = cols / Q8_BLOCK_SIZE;
    debug_assert_eq!(matrix.blocks_per_row(), blocks_per_row);
    debug_assert_eq!(matrix.chunk_count(), rows / 4);

    if selector.selected == Q8DotKernel::Sdot && q8_1x4_sdot_enabled() {
        let chunks = rows / 4;
        let out_addr = out.as_mut_ptr() as usize;
        let ctx = PageAlignedQ8BatchChunkContext {
            out_addr,
            x_i8,
            x_scales,
            matrix,
            shape,
            blocks_per_row,
        };
        if should_parallelize_matmul(rows) {
            (0..chunks).into_par_iter().for_each(|chunk_idx| {
                compute_q8_0_page_aligned_1x4_sdot_batch_chunk(chunk_idx, ctx);
            });
        } else {
            (0..chunks).for_each(|chunk_idx| {
                compute_q8_0_page_aligned_1x4_sdot_batch_chunk(chunk_idx, ctx);
            });
        }
        return;
    }

    let out_addr = out.as_mut_ptr() as usize;
    for_each_batch_matmul_row(rows, |r| {
        let chunk_idx = r / 4;
        let lane = r % 4;
        let chunk = matrix.chunk_ptr(chunk_idx);
        for token_idx in 0..batch_size {
            let x_offset = token_idx * cols;
            let scale_offset = token_idx * blocks_per_row;
            let x_token = &x_i8[x_offset..x_offset + cols];
            let x_token_scales = &x_scales[scale_offset..scale_offset + blocks_per_row];
            let mut sum = 0.0_f32;
            for (b, x_scale) in x_token_scales
                .iter()
                .copied()
                .enumerate()
                .take(blocks_per_row)
            {
                let w_block = unsafe { &*chunk.add(b * 4 + lane) };
                let x_block_vals = unsafe { activation_block_ptr(x_token, b) };
                let dot_val = crate::q8::dot_i8_neon_32_selected(w_block.values(), x_block_vals);
                sum += w_block.scale_f32() * x_scale * dot_val as f32;
            }
            unsafe { write_batch_out(out_addr, token_idx, rows, r, sum) };
        }
    });
}

#[derive(Clone, Copy)]
struct PageAlignedQ8BatchChunkContext<'a> {
    out_addr: usize,
    x_i8: &'a [i8],
    x_scales: &'a [f32],
    matrix: &'a PageAlignedQ8_0Swizzled1x4,
    shape: BatchMatmulShape,
    blocks_per_row: usize,
}

fn compute_q8_0_page_aligned_1x4_sdot_batch_chunk(
    chunk_idx: usize,
    ctx: PageAlignedQ8BatchChunkContext<'_>,
) {
    let BatchMatmulShape {
        batch_size,
        rows,
        cols,
    } = ctx.shape;
    let chunk = ctx.matrix.chunk_ptr(chunk_idx);
    let row_base = chunk_idx * 4;
    let mut token_sums = vec![[0.0_f32; 4]; batch_size];
    for b in 0..ctx.blocks_per_row {
        let w_base = b * 4;
        let row0 = unsafe { &*chunk.add(w_base) };
        let row1 = unsafe { &*chunk.add(w_base + 1) };
        let row2 = unsafe { &*chunk.add(w_base + 2) };
        let row3 = unsafe { &*chunk.add(w_base + 3) };
        let row0_scale = row0.scale_f32();
        let row1_scale = row1.scale_f32();
        let row2_scale = row2.scale_f32();
        let row3_scale = row3.scale_f32();

        #[cfg(target_arch = "aarch64")]
        let unpacked = unsafe {
            use std::arch::aarch64::vld1q_s8;
            crate::q8::Q8_0Unpacked1x4Aarch64 {
                w0_low: vld1q_s8(row0.values().as_ptr()),
                w0_high: vld1q_s8(row0.values().as_ptr().add(16)),
                w1_low: vld1q_s8(row1.values().as_ptr()),
                w1_high: vld1q_s8(row1.values().as_ptr().add(16)),
                w2_low: vld1q_s8(row2.values().as_ptr()),
                w2_high: vld1q_s8(row2.values().as_ptr().add(16)),
                w3_low: vld1q_s8(row3.values().as_ptr()),
                w3_high: vld1q_s8(row3.values().as_ptr().add(16)),
            }
        };

        for (token_idx, sums) in token_sums.iter_mut().enumerate() {
            let x_offset = token_idx * cols;
            let x_scale = ctx.x_scales[token_idx * ctx.blocks_per_row + b];
            let x_token = &ctx.x_i8[x_offset..x_offset + cols];
            let x_block_vals = unsafe { activation_block_ptr(x_token, b) };

            #[cfg(target_arch = "aarch64")]
            let dots = unsafe {
                crate::q8::dot_q8_0_q8_0_1x4_sdot_preloaded_aarch64(unpacked, x_block_vals)
            };
            #[cfg(not(target_arch = "aarch64"))]
            let dots = [
                crate::q8::dot_i8_neon_32_selected(row0.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row1.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row2.values(), x_block_vals),
                crate::q8::dot_i8_neon_32_selected(row3.values(), x_block_vals),
            ];

            sums[0] += row0_scale * x_scale * dots[0] as f32;
            sums[1] += row1_scale * x_scale * dots[1] as f32;
            sums[2] += row2_scale * x_scale * dots[2] as f32;
            sums[3] += row3_scale * x_scale * dots[3] as f32;
        }
    }

    for (token_idx, sums) in token_sums.into_iter().enumerate() {
        for (lane, sum) in sums.into_iter().enumerate() {
            unsafe {
                write_batch_out(ctx.out_addr, token_idx, rows, row_base + lane, sum);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionInput, BatchMatmulShape, KvCacheSlice, RopeScaling, add_weighted_f32,
        add_weighted_f32_scalar, apply_attention_heads, apply_rope, dot_f32, dot_f32_scalar,
        embed_token, fused_silu_mul, matmul_iq4_nl, matmul_q2_k, matmul_q2_k_batch, matmul_q3_k,
        matmul_q3_k_batch, matmul_q4_0, matmul_q4_0_batch, matmul_q4_0_swizzled_1x4, matmul_q4_1,
        matmul_q4_1_batch, matmul_q4_k, matmul_q4_k_batch, matmul_q5_0, matmul_q5_0_batch,
        matmul_q5_1, matmul_q5_1_batch, matmul_q5_k, matmul_q5_k_batch, matmul_q6_k,
        matmul_q6_k_batch, matmul_q8_0, matmul_q8_0_batch, matmul_q8_0_swizzled_1x4, matmul_q8_k,
        quantize_f32_to_q8_0, quantize_f32_to_q8_0_batch, quantize_row_to_q8_0_blocks, rms_norm,
        rms_norm_batch, route_token_to_experts, run_layer_range, silu,
    };
    use crate::model::{
        LlamaFfnWeights, LlamaLayerWeights, LlamaModelConfig, LlamaWeights, QuantizedMatrix,
    };
    use crate::q8::{
        IQ4NLBlock, Q2KBlock, Q3KBlock, Q4_0Block, Q4_1Block, Q4KBlock, Q5_0Block, Q5_1Block,
        Q5KBlock, Q6KBlock, Q8_0Block, Q8_BLOCK_SIZE, Q8DotKernel, Q8DotKernelSelector, Q8KBlock,
        QK_K_BLOCK_SIZE, swizzle_q4_0_1x4,
    };

    fn selector(selected: Q8DotKernel) -> Q8DotKernelSelector {
        Q8DotKernelSelector {
            requested: Some(selected),
            selected,
            fallback_reason: None,
        }
    }

    fn q8_block(scale_bits: u16, seed: i16) -> Q8_0Block {
        let mut values = [0_i8; Q8_BLOCK_SIZE];
        for (idx, value) in values.iter_mut().enumerate() {
            *value = ((seed + idx as i16 * 7) % 63 - 31) as i8;
        }
        Q8_0Block::from_parts(scale_bits, values)
    }

    fn zero_q8_matrix(rows: usize, cols: usize) -> QuantizedMatrix {
        assert!(cols.is_multiple_of(Q8_BLOCK_SIZE));
        QuantizedMatrix::Q8_0(vec![
            Q8_0Block::from_parts(0, [0; Q8_BLOCK_SIZE]);
            rows * (cols / Q8_BLOCK_SIZE)
        ])
    }

    fn no_op_layer(config: &LlamaModelConfig) -> LlamaLayerWeights {
        LlamaLayerWeights {
            attention_norm: vec![1.0; config.embedding_length],
            wq: zero_q8_matrix(config.embedding_length, config.embedding_length),
            wk: zero_q8_matrix(config.kv_width, config.embedding_length),
            wav: zero_q8_matrix(config.kv_width, config.embedding_length),
            wq_bias: None,
            wk_bias: None,
            wav_bias: None,
            wo: zero_q8_matrix(config.embedding_length, config.embedding_length),
            ffn_norm: vec![1.0; config.embedding_length],
            ffn: LlamaFfnWeights::Dense {
                w1: zero_q8_matrix(config.feed_forward_length, config.embedding_length),
                w3: zero_q8_matrix(config.feed_forward_length, config.embedding_length),
                w2: zero_q8_matrix(config.embedding_length, config.feed_forward_length),
            },
        }
    }

    fn split_smoke_config() -> LlamaModelConfig {
        LlamaModelConfig {
            architecture: "llama".to_owned(),
            metadata_prefix: "llama".to_owned(),
            context_length: 4,
            embedding_length: 32,
            block_count: 2,
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

    fn split_smoke_weights(config: &LlamaModelConfig) -> LlamaWeights {
        let mut token_embeddings = Vec::with_capacity(config.vocab_size * config.embedding_length);
        for token_idx in 0..config.vocab_size {
            for dim_idx in 0..config.embedding_length {
                token_embeddings.push((token_idx as f32 + 1.0) * 0.01 + dim_idx as f32 * 0.001);
            }
        }

        LlamaWeights {
            token_embeddings,
            output_norm: vec![1.0; config.embedding_length],
            output_projection: None,
            layers: (0..config.block_count)
                .map(|_| no_op_layer(config))
                .collect(),
        }
    }

    fn q4_block(scale_bits: u16, seed: i16) -> Q4_0Block {
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        for (idx, value) in values.iter_mut().enumerate() {
            let low = ((seed + idx as i16 * 3).rem_euclid(16)) as u8;
            let high = ((seed + (idx as i16 + 16) * 3).rem_euclid(16)) as u8;
            *value = low | (high << 4);
        }
        Q4_0Block::from_parts(scale_bits, values)
    }

    fn q4_1_block(scale_bits: u16, min_bits: u16, seed: i16) -> Q4_1Block {
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        for (idx, value) in values.iter_mut().enumerate() {
            let low = ((seed + idx as i16 * 3).rem_euclid(16)) as u8;
            let high = ((seed + (idx as i16 + 16) * 5).rem_euclid(16)) as u8;
            *value = low | (high << 4);
        }
        Q4_1Block::from_parts(scale_bits, min_bits, values)
    }

    fn q5_0_block(scale_bits: u16, seed: i16) -> Q5_0Block {
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        for (idx, value) in values.iter_mut().enumerate() {
            let low = ((seed + idx as i16 * 3).rem_euclid(16)) as u8;
            let high = ((seed + (idx as i16 + 16) * 5).rem_euclid(16)) as u8;
            *value = low | (high << 4);
        }
        let high_bits = 0xA5A5_5A5A_u32.rotate_left(seed.rem_euclid(32) as u32);
        Q5_0Block::from_parts(scale_bits, high_bits, values)
    }

    fn q5_1_block(scale_bits: u16, min_bits: u16, seed: i16) -> Q5_1Block {
        let mut values = [0_u8; Q8_BLOCK_SIZE / 2];
        for (idx, value) in values.iter_mut().enumerate() {
            let low = ((seed + idx as i16 * 7).rem_euclid(16)) as u8;
            let high = ((seed + (idx as i16 + 16) * 3).rem_euclid(16)) as u8;
            *value = low | (high << 4);
        }
        let high_bits = 0x9696_6969_u32.rotate_left(seed.rem_euclid(32) as u32);
        Q5_1Block::from_parts(scale_bits, min_bits, high_bits, values)
    }

    fn k_scales(seed: i16) -> [u8; 12] {
        let mut scales = [0_u8; 12];
        for (idx, scale) in scales.iter_mut().enumerate() {
            *scale = ((seed + idx as i16 * 5).rem_euclid(64)) as u8;
        }
        scales
    }

    fn q2_k_block(scale_bits: u16, min_bits: u16, seed: i16) -> Q2KBlock {
        let mut scales = [0_u8; QK_K_BLOCK_SIZE / 16];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 4];
        for (idx, scale) in scales.iter_mut().enumerate() {
            let d = ((seed + idx as i16 * 3).rem_euclid(16)) as u8;
            let m = ((seed + idx as i16 * 5).rem_euclid(16)) as u8;
            *scale = d | (m << 4);
        }
        for (idx, value) in values.iter_mut().enumerate() {
            let q0 = ((seed + idx as i16 * 3).rem_euclid(4)) as u8;
            let q1 = ((seed + idx as i16 * 5).rem_euclid(4)) as u8;
            let q2 = ((seed + idx as i16 * 7).rem_euclid(4)) as u8;
            let q3 = ((seed + idx as i16 * 11).rem_euclid(4)) as u8;
            *value = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
        }
        Q2KBlock::from_parts(scales, values, scale_bits, min_bits)
    }

    fn q3_k_block(scale_bits: u16, seed: i16) -> Q3KBlock {
        let mut high_bits = [0_u8; QK_K_BLOCK_SIZE / 8];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 4];
        let mut scales = [0_u8; 12];
        for (idx, value) in high_bits.iter_mut().enumerate() {
            *value = ((seed + idx as i16 * 13).rem_euclid(256)) as u8;
        }
        for (idx, value) in values.iter_mut().enumerate() {
            let q0 = ((seed + idx as i16 * 3).rem_euclid(4)) as u8;
            let q1 = ((seed + idx as i16 * 5).rem_euclid(4)) as u8;
            let q2 = ((seed + idx as i16 * 7).rem_euclid(4)) as u8;
            let q3 = ((seed + idx as i16 * 11).rem_euclid(4)) as u8;
            *value = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
        }
        for (idx, scale) in scales.iter_mut().enumerate() {
            *scale = ((seed + idx as i16 * 5).rem_euclid(64)) as u8;
        }
        Q3KBlock::from_parts(high_bits, values, scales, scale_bits)
    }

    fn q4_k_block(scale_bits: u16, min_bits: u16, seed: i16) -> Q4KBlock {
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 2];
        for (idx, value) in values.iter_mut().enumerate() {
            let low = ((seed + idx as i16 * 3).rem_euclid(16)) as u8;
            let high = ((seed + idx as i16 * 7).rem_euclid(16)) as u8;
            *value = low | (high << 4);
        }
        Q4KBlock::from_parts(scale_bits, min_bits, k_scales(seed), values)
    }

    fn q5_k_block(scale_bits: u16, min_bits: u16, seed: i16) -> Q5KBlock {
        let mut high_bits = [0_u8; QK_K_BLOCK_SIZE / 8];
        let mut values = [0_u8; QK_K_BLOCK_SIZE / 2];
        for (idx, value) in high_bits.iter_mut().enumerate() {
            *value = ((seed + idx as i16 * 11).rem_euclid(256)) as u8;
        }
        for (idx, value) in values.iter_mut().enumerate() {
            let low = ((seed + idx as i16 * 3).rem_euclid(16)) as u8;
            let high = ((seed + idx as i16 * 5).rem_euclid(16)) as u8;
            *value = low | (high << 4);
        }
        Q5KBlock::from_parts(scale_bits, min_bits, k_scales(seed), high_bits, values)
    }

    fn q6_k_block(scale_bits: u16, seed: i16) -> Q6KBlock {
        let mut ql = [0_u8; 128];
        let mut qh = [0_u8; 64];
        let mut scales = [0_i8; 16];
        for (idx, value) in ql.iter_mut().enumerate() {
            *value = ((seed + idx as i16 * 5).rem_euclid(256)) as u8;
        }
        for (idx, value) in qh.iter_mut().enumerate() {
            *value = ((seed + idx as i16 * 7).rem_euclid(256)) as u8;
        }
        for (idx, scale) in scales.iter_mut().enumerate() {
            *scale = ((seed + idx as i16 * 3).rem_euclid(15) - 7) as i8;
        }
        Q6KBlock::from_parts(ql, qh, scales, scale_bits)
    }

    fn covered_matmul_rows(len: usize, worker_count: usize) -> Vec<usize> {
        let mut rows = Vec::new();
        for thread_idx in 0..=worker_count {
            let (start, end) = super::get_asymmetric_slice(len, thread_idx, worker_count);
            rows.extend(start..end);
        }
        rows
    }

    #[test]
    fn spin_pool_slices_cover_each_row_once_for_pi_and_fallback_counts() {
        for &(len, worker_count) in &[(129, 3), (256, 3), (129, 1), (257, 5)] {
            let rows = covered_matmul_rows(len, worker_count);
            assert_eq!(rows, (0..len).collect::<Vec<_>>());
        }
    }

    #[test]
    fn attention_vector_helpers_match_scalar_reference() {
        let lhs: Vec<f32> = (0..130).map(|idx| idx as f32 * 0.03125 - 2.0).collect();
        let rhs: Vec<f32> = (0..130).map(|idx| 1.0 - idx as f32 * 0.015625).collect();
        let candidate = dot_f32(&lhs, &rhs);
        let expected = dot_f32_scalar(&lhs, &rhs);
        assert!(
            (candidate - expected).abs() < 1e-4,
            "candidate {candidate} expected {expected}"
        );

        let mut candidate_out: Vec<f32> = (0..130).map(|idx| idx as f32 * 0.125).collect();
        let mut expected_out = candidate_out.clone();
        add_weighted_f32(&mut candidate_out, &rhs, 0.375);
        add_weighted_f32_scalar(&mut expected_out, &rhs, 0.375);

        for (idx, (&candidate, &expected)) in candidate_out.iter().zip(&expected_out).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn head_parallel_attention_matches_serial_attention() {
        let head_count = 4;
        let kv_head_count = 2;
        let head_dim = 8;
        let context_length = 5;
        let cache_kv_width = kv_head_count * head_dim;
        let pos = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q: Vec<f32> = (0..head_count * head_dim)
            .map(|idx| idx as f32 * 0.017 - 0.25)
            .collect();
        let k_cache: Vec<f32> = (0..context_length * cache_kv_width)
            .map(|idx| 0.5 - idx as f32 * 0.009)
            .collect();
        let v_cache: Vec<f32> = (0..context_length * cache_kv_width)
            .map(|idx| idx as f32 * 0.013 - 0.4)
            .collect();

        let mut serial_out = vec![0.0; head_count * head_dim];
        let mut serial_scores = vec![0.0; context_length];
        apply_attention_heads(
            &mut serial_out,
            &mut serial_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::F32(&k_cache),
                v_cache: KvCacheSlice::F32(&v_cache),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            false,
        );

        let mut parallel_out = vec![0.0; head_count * head_dim];
        let mut parallel_scores = vec![0.0; head_count * context_length];
        apply_attention_heads(
            &mut parallel_out,
            &mut parallel_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::F32(&k_cache),
                v_cache: KvCacheSlice::F32(&v_cache),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            true,
        );

        for (idx, (&candidate, &expected)) in parallel_out.iter().zip(&serial_out).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn split_layer_execution_matches_full_forward_for_single_token() {
        let config = split_smoke_config();
        let weights = split_smoke_weights(&config);
        let options = super::LlamaRuntimeOptions {
            q8_selector: selector(Q8DotKernel::Scalar),
            rope_scaling: RopeScaling::default(),
            compute_logits: true,
        };

        let mut full_cache =
            super::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
        let mut full_ws = super::LlamaWorkspace::new(&config);
        super::forward_pass(
            2,
            0,
            &config,
            &weights,
            &mut full_cache,
            &mut full_ws,
            options,
        );

        let mut split_cache =
            super::LlamaKvCache::new(config.block_count, config.context_length, config.kv_width);
        let mut split_ws = super::LlamaWorkspace::new(&config);
        embed_token(2, &config, &weights.token_embeddings, &mut split_ws);
        run_layer_range(
            0,
            &weights.layers[0..1],
            0,
            &config,
            &mut split_cache,
            &mut split_ws,
            options,
        );
        run_layer_range(
            1,
            &weights.layers[1..2],
            0,
            &config,
            &mut split_cache,
            &mut split_ws,
            options,
        );
        super::compute_logits_from_hidden(
            &config,
            &weights.token_embeddings,
            &weights.output_norm,
            weights.output_projection.as_ref(),
            &mut split_ws,
            options,
        );

        assert_eq!(split_ws.hidden, full_ws.hidden);
        assert_eq!(split_ws.logits, full_ws.logits);
    }

    #[test]
    fn fused_silu_mul_matches_two_pass_reference() {
        let gate: Vec<f32> = (0..257).map(|idx| idx as f32 * 0.025 - 3.0).collect();
        let up: Vec<f32> = (0..257).map(|idx| 2.0 - idx as f32 * 0.011).collect();
        let mut expected_gate = gate.clone();
        let mut expected = vec![0.0; gate.len()];
        silu(&mut expected_gate);
        for idx in 0..expected.len() {
            expected[idx] = expected_gate[idx] * up[idx];
        }

        let mut candidate = vec![0.0; gate.len()];
        fused_silu_mul(&mut candidate, &gate, &up);

        for (idx, (&candidate, &expected)) in candidate.iter().zip(&expected).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn route_token_to_experts_selects_top_k_and_normalizes_weights() {
        let hidden = [2.0, -1.0, 0.5];
        let router = [
            0.0, 1.0, 0.0, // -1.0
            1.0, 0.0, 0.0, // 2.0
            0.0, 0.0, 3.0, // 1.5
            -1.0, 0.0, 0.0, // -2.0
        ];

        let routes = route_token_to_experts(&hidden, &router, 4, 2);

        assert_eq!(routes[0].expert_idx, 1);
        assert_eq!(routes[1].expert_idx, 2);
        let weight_sum: f32 = routes.iter().map(|route| route.weight).sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
        assert!(routes[0].weight > routes[1].weight);
    }

    #[test]
    fn f16_attention_matches_explicitly_decoded_cache() {
        let head_count = 4;
        let kv_head_count = 2;
        let head_dim = 8;
        let context_length = 5;
        let cache_kv_width = kv_head_count * head_dim;
        let pos = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q: Vec<f32> = (0..head_count * head_dim)
            .map(|idx| idx as f32 * 0.017 - 0.25)
            .collect();
        let k_cache_f16: Vec<u16> = (0..context_length * cache_kv_width)
            .map(|idx| crate::q8::fast_f32_to_f16(0.5 - idx as f32 * 0.009))
            .collect();
        let v_cache_f16: Vec<u16> = (0..context_length * cache_kv_width)
            .map(|idx| crate::q8::fast_f32_to_f16(idx as f32 * 0.013 - 0.4))
            .collect();
        let k_cache_decoded: Vec<f32> = k_cache_f16
            .iter()
            .map(|&bits| crate::q8::fast_f16_to_f32(bits))
            .collect();
        let v_cache_decoded: Vec<f32> = v_cache_f16
            .iter()
            .map(|&bits| crate::q8::fast_f16_to_f32(bits))
            .collect();

        let mut expected = vec![0.0; head_count * head_dim];
        let mut expected_scores = vec![0.0; context_length];
        apply_attention_heads(
            &mut expected,
            &mut expected_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::F32(&k_cache_decoded),
                v_cache: KvCacheSlice::F32(&v_cache_decoded),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            false,
        );

        let mut candidate = vec![0.0; head_count * head_dim];
        let mut candidate_scores = vec![0.0; head_count * context_length];
        apply_attention_heads(
            &mut candidate,
            &mut candidate_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::F16(&k_cache_f16),
                v_cache: KvCacheSlice::F16(&v_cache_f16),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            true,
        );

        for (idx, (&candidate, &expected)) in candidate.iter().zip(&expected).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn q8_0_attention_matches_dequantized_cache() {
        // head_dim MUST be a multiple of Q8_BLOCK_SIZE (32) for the q8_0 arm.
        let head_count = 4;
        let kv_head_count = 2;
        let head_dim = 32;
        let context_length = 5;
        let cache_kv_width = 64; // kv_head_count * head_dim
        let pos = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..head_count * head_dim)
            .map(|idx| idx as f32 * 0.017 - 0.25)
            .collect();
        let k_cache_f32: Vec<f32> = (0..context_length * cache_kv_width)
            .map(|idx| 0.5 - idx as f32 * 0.009)
            .collect();
        let v_cache_f32: Vec<f32> = (0..context_length * cache_kv_width)
            .map(|idx| idx as f32 * 0.013 - 0.4)
            .collect();

        // Quantize the f32 K/V patterns into q8_0 blocks (same store quantization).
        let block_count = k_cache_f32.len() / Q8_BLOCK_SIZE;
        let mut k_cache_q8 =
            vec![Q8_0Block::from_parts(0, [0i8; Q8_BLOCK_SIZE]); block_count];
        let mut v_cache_q8 =
            vec![Q8_0Block::from_parts(0, [0i8; Q8_BLOCK_SIZE]); block_count];
        quantize_row_to_q8_0_blocks(&k_cache_f32, &mut k_cache_q8);
        quantize_row_to_q8_0_blocks(&v_cache_f32, &mut v_cache_q8);

        // Decode the blocks back to f32 so the reference sees the SAME quantized values.
        let mut k_cache_decoded = vec![0.0f32; k_cache_f32.len()];
        let mut v_cache_decoded = vec![0.0f32; v_cache_f32.len()];
        let dequantize_block = |block: &Q8_0Block, dst: &mut [f32]| {
            let s = block.scale_f32();
            for (out, &q) in dst.iter_mut().zip(block.values().iter()) {
                *out = q as f32 * s;
            }
        };
        for (block_idx, block) in k_cache_q8.iter().enumerate() {
            dequantize_block(
                block,
                &mut k_cache_decoded[block_idx * Q8_BLOCK_SIZE..(block_idx + 1) * Q8_BLOCK_SIZE],
            );
        }
        for (block_idx, block) in v_cache_q8.iter().enumerate() {
            dequantize_block(
                block,
                &mut v_cache_decoded[block_idx * Q8_BLOCK_SIZE..(block_idx + 1) * Q8_BLOCK_SIZE],
            );
        }

        let mut expected = vec![0.0; head_count * head_dim];
        let mut expected_scores = vec![0.0; context_length];
        apply_attention_heads(
            &mut expected,
            &mut expected_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::F32(&k_cache_decoded),
                v_cache: KvCacheSlice::F32(&v_cache_decoded),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            false,
        );

        let mut candidate = vec![0.0; head_count * head_dim];
        let mut candidate_scores = vec![0.0; head_count * context_length];
        apply_attention_heads(
            &mut candidate,
            &mut candidate_scores,
            AttentionInput {
                q: &q,
                k_cache: KvCacheSlice::Q8_0(&k_cache_q8),
                v_cache: KvCacheSlice::Q8_0(&v_cache_q8),
                pos,
                head_count,
                kv_head_count,
                head_dim,
                cache_kv_width,
                context_length,
                scale,
            },
            true,
        );

        // Loose bound: unlike the reference, the candidate additionally quantizes
        // q_head to q8_0, so scores differ by q8_0 precision (~1/127 relative).
        for (idx, (&candidate, &expected)) in candidate.iter().zip(&expected).enumerate() {
            assert!(
                (candidate - expected).abs() < 5e-2,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn matmul_q8_selected_kernels_match_scalar_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q8_block(0x3800, 1),
            q8_block(0x3c00, 2),
            q8_block(0x4000, 3),
            q8_block(0x4200, 4),
            q8_block(0x4400, 5),
            q8_block(0x4600, 6),
        ];
        let mut scalar = vec![0.0; rows];
        matmul_q8_0(
            &mut scalar,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        for kernel in [Q8DotKernel::Neon, Q8DotKernel::Sdot] {
            let mut candidate = vec![0.0; rows];
            matmul_q8_0(
                &mut candidate,
                &x_i8,
                &x_scales,
                &weights,
                rows,
                cols,
                selector(kernel),
            );
            assert_eq!(candidate, scalar, "{kernel:?} matmul diverged");
        }
    }

    #[test]
    fn matmul_q4_matches_dequantized_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q4_block(0x3800, 1),
            q4_block(0x3c00, 2),
            q4_block(0x4000, 3),
            q4_block(0x4200, 4),
            q4_block(0x4400, 5),
            q4_block(0x4600, 6),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q4_0(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let scale = block.scale_f32() * x_scales[b];
                let unpacked = block.unpack_values();
                for i in 0..Q8_BLOCK_SIZE {
                    sum += scale * unpacked[i] as f32 * x_i8[b * Q8_BLOCK_SIZE + i] as f32;
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-5,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q4_1_matches_dequantized_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q4_1_block(0x3800, 0x3000, 1),
            q4_1_block(0x3c00, 0x3400, 2),
            q4_1_block(0x4000, 0x3800, 3),
            q4_1_block(0x4200, 0x3a00, 4),
            q4_1_block(0x4400, 0x3c00, 5),
            q4_1_block(0x4600, 0x3e00, 6),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q4_1(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let scale = block.scale_f32();
                let min = block.min_f32();
                let unpacked = block.unpack_values();
                for i in 0..Q8_BLOCK_SIZE {
                    sum += (scale * unpacked[i] as f32 + min)
                        * x_i8[b * Q8_BLOCK_SIZE + i] as f32
                        * x_scales[b];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-5,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q5_0_matches_dequantized_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q5_0_block(0x3800, 1),
            q5_0_block(0x3c00, 2),
            q5_0_block(0x4000, 3),
            q5_0_block(0x4200, 4),
            q5_0_block(0x4400, 5),
            q5_0_block(0x4600, 6),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q5_0(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let scale = block.scale_f32() * x_scales[b];
                let unpacked = block.unpack_values();
                for i in 0..Q8_BLOCK_SIZE {
                    sum += scale * unpacked[i] as f32 * x_i8[b * Q8_BLOCK_SIZE + i] as f32;
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-5,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q5_1_matches_dequantized_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q5_1_block(0x3800, 0x3000, 1),
            q5_1_block(0x3c00, 0x3400, 2),
            q5_1_block(0x4000, 0x3800, 3),
            q5_1_block(0x4200, 0x3a00, 4),
            q5_1_block(0x4400, 0x3c00, 5),
            q5_1_block(0x4600, 0x3e00, 6),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q5_1(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / Q8_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let scale = block.scale_f32();
                let min = block.min_f32();
                let unpacked = block.unpack_values();
                for i in 0..Q8_BLOCK_SIZE {
                    sum += (scale * unpacked[i] as f32 + min)
                        * x_i8[b * Q8_BLOCK_SIZE + i] as f32
                        * x_scales[b];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-5,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q2_k_matches_dequantized_reference() {
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales: Vec<f32> = (0..cols / Q8_BLOCK_SIZE)
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights = [
            q2_k_block(0x3800, 0x3000, 1),
            q2_k_block(0x3c00, 0x3400, 2),
            q2_k_block(0x4000, 0x3800, 3),
            q2_k_block(0x4200, 0x3a00, 4),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q2_k(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / QK_K_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
                block.dequantize(&mut values);
                let x_block_start = b * QK_K_BLOCK_SIZE;
                for (i, value) in values.iter().enumerate() {
                    let x_idx = x_block_start + i;
                    sum += *value * x_i8[x_idx] as f32 * x_scales[x_idx / Q8_BLOCK_SIZE];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-3,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q8_k_matches_dequantized_reference() {
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 3) % 43) as i8 - 21).collect();
        let x_scales: Vec<f32> = (0..cols / Q8_BLOCK_SIZE)
            .map(|idx| 0.25 + idx as f32 * 0.05)
            .collect();

        let make_q8_k = |seed: i16| -> Q8KBlock {
            let mut bytes = [0_u8; 292];
            let d_bytes = 1.25_f32.to_le_bytes();
            bytes[0..4].copy_from_slice(&d_bytes);
            for i in 0..256 {
                bytes[4 + i] = ((seed + i as i16 * 3) % 23 - 11) as u8;
            }
            for i in 0..16 {
                let bsum = seed * 2 + i as i16;
                let bsum_bytes = bsum.to_le_bytes();
                bytes[260 + i * 2..260 + i * 2 + 2].copy_from_slice(&bsum_bytes);
            }
            Q8KBlock::from_bytes(&bytes)
        };

        let weights = [make_q8_k(1), make_q8_k(2), make_q8_k(3), make_q8_k(4)];

        let mut candidate = vec![0.0; rows];
        let sel = selector(Q8DotKernel::Scalar);
        matmul_q8_k(&mut candidate, &x_i8, &x_scales, &weights, rows, cols, sel);

        let blocks_per_row = cols / QK_K_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
                block.dequantize(&mut values);
                let x_block_start = b * QK_K_BLOCK_SIZE;
                for (i, value) in values.iter().enumerate() {
                    let x_idx = x_block_start + i;
                    sum += *value * x_i8[x_idx] as f32 * x_scales[x_idx / Q8_BLOCK_SIZE];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-2,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_iq4_nl_matches_dequantized_reference() {
        let rows = 2;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 7) % 31) as i8 - 15).collect();
        let x_scales: Vec<f32> = (0..cols / Q8_BLOCK_SIZE)
            .map(|idx| 0.5 + idx as f32 * 0.1)
            .collect();

        let make_iq4_nl = |seed: u16| -> IQ4NLBlock {
            let mut bytes = [0_u8; 18];
            let d_bits = crate::q8::fast_f32_to_f16(1.5 + seed as f32 * 0.25);
            let d_bytes = d_bits.to_le_bytes();
            bytes[0..2].copy_from_slice(&d_bytes);
            for i in 0..16 {
                let low = ((seed + i as u16) & 0x0F) as u8;
                let high = (((seed + i as u16 + 8) & 0x0F) as u8) << 4;
                bytes[2 + i] = low | high;
            }
            IQ4NLBlock::from_bytes(&bytes)
        };

        let weights = [
            make_iq4_nl(1),
            make_iq4_nl(2),
            make_iq4_nl(3),
            make_iq4_nl(4),
        ];

        let mut candidate = vec![0.0; rows];
        matmul_iq4_nl(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / 32;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let mut values = [0.0_f32; 32];
                block.dequantize(&mut values);
                let x_block_start = b * 32;
                for (i, value) in values.iter().enumerate() {
                    let x_idx = x_block_start + i;
                    sum += *value * x_i8[x_idx] as f32 * x_scales[x_idx / Q8_BLOCK_SIZE];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-1,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q3_k_matches_dequantized_reference() {
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales: Vec<f32> = (0..cols / Q8_BLOCK_SIZE)
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights = [
            q3_k_block(0x3800, 1),
            q3_k_block(0x3c00, 2),
            q3_k_block(0x4000, 3),
            q3_k_block(0x4200, 4),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q3_k(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / QK_K_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
                block.dequantize(&mut values);
                let x_block_start = b * QK_K_BLOCK_SIZE;
                for (i, value) in values.iter().enumerate() {
                    let x_idx = x_block_start + i;
                    sum += *value * x_i8[x_idx] as f32 * x_scales[x_idx / Q8_BLOCK_SIZE];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-4,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q4_k_matches_dequantized_reference() {
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales: Vec<f32> = (0..cols / Q8_BLOCK_SIZE)
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights = [
            q4_k_block(0x3800, 0x3000, 1),
            q4_k_block(0x3c00, 0x3400, 2),
            q4_k_block(0x4000, 0x3800, 3),
            q4_k_block(0x4200, 0x3a00, 4),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q4_k(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / QK_K_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
                block.dequantize(&mut values);
                let x_block_start = b * QK_K_BLOCK_SIZE;
                for (i, value) in values.iter().enumerate() {
                    let x_idx = x_block_start + i;
                    sum += *value * x_i8[x_idx] as f32 * x_scales[x_idx / Q8_BLOCK_SIZE];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-4,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q5_k_matches_dequantized_reference() {
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales: Vec<f32> = (0..cols / Q8_BLOCK_SIZE)
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights = [
            q5_k_block(0x3800, 0x3000, 1),
            q5_k_block(0x3c00, 0x3400, 2),
            q5_k_block(0x4000, 0x3800, 3),
            q5_k_block(0x4200, 0x3a00, 4),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q5_k(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / QK_K_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
                block.dequantize(&mut values);
                let x_block_start = b * QK_K_BLOCK_SIZE;
                for (i, value) in values.iter().enumerate() {
                    let x_idx = x_block_start + i;
                    sum += *value * x_i8[x_idx] as f32 * x_scales[x_idx / Q8_BLOCK_SIZE];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-4,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q6_k_matches_dequantized_reference() {
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales: Vec<f32> = (0..cols / Q8_BLOCK_SIZE)
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights = [
            q6_k_block(0x3800, 1),
            q6_k_block(0x3c00, 2),
            q6_k_block(0x4000, 3),
            q6_k_block(0x4200, 4),
        ];
        let mut candidate = vec![0.0; rows];
        matmul_q6_k(&mut candidate, &x_i8, &x_scales, &weights, rows, cols);

        let blocks_per_row = cols / QK_K_BLOCK_SIZE;
        let mut expected = vec![0.0; rows];
        for r in 0..rows {
            let mut sum = 0.0_f32;
            for b in 0..blocks_per_row {
                let block = &weights[r * blocks_per_row + b];
                let mut values = [0.0_f32; QK_K_BLOCK_SIZE];
                block.dequantize(&mut values);
                let x_block_start = b * QK_K_BLOCK_SIZE;
                for (i, value) in values.iter().enumerate() {
                    let x_idx = x_block_start + i;
                    sum += *value * x_i8[x_idx] as f32 * x_scales[x_idx / Q8_BLOCK_SIZE];
                }
            }
            expected[r] = sum;
        }

        for r in 0..rows {
            assert!(
                (candidate[r] - expected[r]).abs() < 1e-4,
                "row {r} candidate {} expected {}",
                candidate[r],
                expected[r]
            );
        }
    }

    #[test]
    fn matmul_q4_selected_kernels_match_scalar_reference() {
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols).map(|idx| ((idx * 5) % 61) as i8 - 30).collect();
        let x_scales = [0.25, 0.5];
        let weights = [
            q4_block(0x3800, 1),
            q4_block(0x3c00, 2),
            q4_block(0x4000, 3),
            q4_block(0x4200, 4),
            q4_block(0x4400, 5),
            q4_block(0x4600, 6),
        ];
        let mut scalar = vec![0.0; rows];
        matmul_q4_0(
            &mut scalar,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        for kernel in [Q8DotKernel::Neon, Q8DotKernel::Sdot] {
            let mut candidate = vec![0.0; rows];
            matmul_q4_0(
                &mut candidate,
                &x_i8,
                &x_scales,
                &weights,
                rows,
                cols,
                selector(kernel),
            );
            assert_eq!(candidate, scalar, "{kernel:?} Q4 matmul diverged");
        }
    }

    #[test]
    fn matmul_q4_swizzled_1x4_matches_row_major_reference() {
        let rows = 4;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols)
            .map(|idx| ((idx as i16 * 11).rem_euclid(83) - 41) as i8)
            .collect();
        let x_scales = [0.125, 0.375];
        let weights: Vec<Q4_0Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q4_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();
        let swizzled = swizzle_q4_0_1x4(&weights, rows, cols / Q8_BLOCK_SIZE);

        let mut expected = vec![0.0; rows];
        matmul_q4_0(
            &mut expected,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        let mut candidate = vec![0.0; rows];
        let matrix = crate::model::Q4_0Swizzled1x4Matrix {
            swizzled_1x4: swizzled.clone(),
            page_aligned_1x4: None,
            rows,
            cols,
        };
        matmul_q4_0_swizzled_1x4(
            &mut candidate,
            &x_i8,
            &x_scales,
            &matrix,
            cols,
            selector(Q8DotKernel::Sdot),
        );

        assert_eq!(candidate, expected);

        let mut aligned_candidate = vec![0.0; rows];
        let aligned_matrix = crate::model::Q4_0Swizzled1x4Matrix {
            page_aligned_1x4: Some(crate::model::PageAlignedQ4_0Swizzled1x4::from_swizzled(
                &swizzled,
                rows,
                cols / Q8_BLOCK_SIZE,
            )),
            swizzled_1x4: swizzled,
            rows,
            cols,
        };
        matmul_q4_0_swizzled_1x4(
            &mut aligned_candidate,
            &x_i8,
            &x_scales,
            &aligned_matrix,
            cols,
            selector(Q8DotKernel::Sdot),
        );

        assert_eq!(aligned_candidate, expected);
    }

    #[test]
    fn rms_norm_batch_matches_single_token_reference() {
        let batch_size = 3;
        let dim = 64;
        let input: Vec<f32> = (0..batch_size * dim)
            .map(|idx| idx as f32 * 0.013 - 1.75)
            .collect();
        let weight: Vec<f32> = (0..dim).map(|idx| 0.5 + idx as f32 * 0.007).collect();
        let mut candidate = vec![0.0; batch_size * dim];
        let mut expected = vec![0.0; batch_size * dim];

        rms_norm_batch(&mut candidate, &input, &weight, 1e-5, batch_size, dim);
        for token_idx in 0..batch_size {
            let start = token_idx * dim;
            rms_norm(
                &mut expected[start..start + dim],
                &input[start..start + dim],
                &weight,
                1e-5,
            );
        }

        for (idx, (&candidate, &expected)) in candidate.iter().zip(&expected).enumerate() {
            assert!(
                (candidate - expected).abs() < 1e-6,
                "idx {idx} candidate {candidate} expected {expected}"
            );
        }
    }

    #[test]
    fn quantize_batch_matches_single_token_reference() {
        let batch_size = 3;
        let dim = 64;
        let input: Vec<f32> = (0..batch_size * dim)
            .map(|idx| (idx as f32 * 0.03125).sin() * 3.0)
            .collect();
        let mut candidate_i8 = vec![0; batch_size * dim];
        let mut expected_i8 = vec![0; batch_size * dim];
        let mut candidate_scales = vec![0.0; batch_size * (dim / Q8_BLOCK_SIZE)];
        let mut expected_scales = vec![0.0; batch_size * (dim / Q8_BLOCK_SIZE)];

        quantize_f32_to_q8_0_batch(
            &input,
            &mut candidate_i8,
            &mut candidate_scales,
            batch_size,
            dim,
        );
        for token_idx in 0..batch_size {
            let value_start = token_idx * dim;
            let scale_start = token_idx * (dim / Q8_BLOCK_SIZE);
            quantize_f32_to_q8_0(
                &input[value_start..value_start + dim],
                &mut expected_i8[value_start..value_start + dim],
                &mut expected_scales[scale_start..scale_start + (dim / Q8_BLOCK_SIZE)],
            );
        }

        assert_eq!(candidate_i8, expected_i8);
        assert_eq!(candidate_scales, expected_scales);
    }

    #[test]
    fn matmul_q8_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 3;
        let cols = 64;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx * 5) % 61) as i8 - 30)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights: Vec<Q8_0Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q8_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q8_0_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
            selector(Q8DotKernel::Scalar),
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q8_0(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
                selector(Q8DotKernel::Scalar),
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q4_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 4;
        let cols = 64;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx as i16 * 11).rem_euclid(83) - 41) as i8)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.0625)
            .collect();
        let weights: Vec<Q4_0Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q4_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q4_0_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
            selector(Q8DotKernel::Scalar),
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q4_0(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
                selector(Q8DotKernel::Scalar),
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q4_1_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 4;
        let cols = 64;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx as i16 * 11).rem_euclid(83) - 41) as i8)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.0625)
            .collect();
        let weights: Vec<Q4_1Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q4_1_block(0x3800 + idx as u16, 0x3000 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q4_1_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q4_1(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q5_0_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 4;
        let cols = 64;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx as i16 * 11).rem_euclid(83) - 41) as i8)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.0625)
            .collect();
        let weights: Vec<Q5_0Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q5_0_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q5_0_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q5_0(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q5_1_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 4;
        let cols = 64;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx as i16 * 11).rem_euclid(83) - 41) as i8)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.0625)
            .collect();
        let weights: Vec<Q5_1Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q5_1_block(0x3800 + idx as u16, 0x3000 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q5_1_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q5_1(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q2_k_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx * 5) % 61) as i8 - 30)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights: Vec<Q2KBlock> = (0..rows * (cols / QK_K_BLOCK_SIZE))
            .map(|idx| q2_k_block(0x3800 + idx as u16, 0x3000 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q2_k_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q2_k(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q3_k_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx * 5) % 61) as i8 - 30)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights: Vec<Q3KBlock> = (0..rows * (cols / QK_K_BLOCK_SIZE))
            .map(|idx| q3_k_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q3_k_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q3_k(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q4_k_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx * 5) % 61) as i8 - 30)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights: Vec<Q4KBlock> = (0..rows * (cols / QK_K_BLOCK_SIZE))
            .map(|idx| q4_k_block(0x3800 + idx as u16, 0x3000 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q4_k_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q4_k(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q5_k_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx * 5) % 61) as i8 - 30)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights: Vec<Q5KBlock> = (0..rows * (cols / QK_K_BLOCK_SIZE))
            .map(|idx| q5_k_block(0x3800 + idx as u16, 0x3000 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q5_k_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q5_k(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn matmul_q6_k_batch_matches_single_token_reference() {
        let batch_size = 3;
        let rows = 2;
        let cols = QK_K_BLOCK_SIZE * 2;
        let x_i8: Vec<i8> = (0..batch_size * cols)
            .map(|idx| ((idx * 5) % 61) as i8 - 30)
            .collect();
        let x_scales: Vec<f32> = (0..batch_size * (cols / Q8_BLOCK_SIZE))
            .map(|idx| 0.125 + idx as f32 * 0.03125)
            .collect();
        let weights: Vec<Q6KBlock> = (0..rows * (cols / QK_K_BLOCK_SIZE))
            .map(|idx| q6_k_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();

        let mut candidate = vec![0.0; batch_size * rows];
        matmul_q6_k_batch(
            &mut candidate,
            &x_i8,
            &x_scales,
            &weights,
            BatchMatmulShape {
                batch_size,
                rows,
                cols,
            },
        );

        for token_idx in 0..batch_size {
            let mut expected = vec![0.0; rows];
            let value_start = token_idx * cols;
            let scale_start = token_idx * (cols / Q8_BLOCK_SIZE);
            matmul_q6_k(
                &mut expected,
                &x_i8[value_start..value_start + cols],
                &x_scales[scale_start..scale_start + (cols / Q8_BLOCK_SIZE)],
                &weights,
                rows,
                cols,
            );
            assert_eq!(
                &candidate[token_idx * rows..(token_idx + 1) * rows],
                expected.as_slice()
            );
        }
    }

    #[test]
    fn apply_rope_respects_partial_rope_dimension_count() {
        let mut data = vec![1.0, 0.0, 10.0, 20.0];
        apply_rope(&mut data, 1, 1, 4, 2, 10000.0, RopeScaling::default(), crate::model::RopeStyle::Norm);

        assert!((data[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((data[1] - 1.0_f32.sin()).abs() < 1e-6);
        assert_eq!(data[2], 10.0);
        assert_eq!(data[3], 20.0);
    }

    #[test]
    fn apply_rope_neox_rotates_split_half_pairs() {
        // head_dim = rope_dim = 4, head_count = 1, pos = 1, base = 10000.
        // NEOX pairs element i with i+rope_dim/2: (0,2) at theta0, (1,3) at theta1.
        let mut data = vec![1.0_f32, 2.0, 3.0, 4.0];
        apply_rope(
            &mut data,
            1,
            1,
            4,
            4,
            10000.0,
            RopeScaling::default(),
            crate::model::RopeStyle::Neox,
        );

        let theta0 = 10000_f32.powf(0.0); // i = 0
        let theta1 = 10000_f32.powf(-2.0 / 4.0); // i = 1
        let (s0, c0) = (1.0 * theta0).sin_cos();
        let (s1, c1) = (1.0 * theta1).sin_cos();
        // pair (0, 2)
        assert!((data[0] - (1.0 * c0 - 3.0 * s0)).abs() < 1e-5);
        assert!((data[2] - (1.0 * s0 + 3.0 * c0)).abs() < 1e-5);
        // pair (1, 3)
        assert!((data[1] - (2.0 * c1 - 4.0 * s1)).abs() < 1e-5);
        assert!((data[3] - (2.0 * s1 + 4.0 * c1)).abs() < 1e-5);
    }

    #[test]
    fn apply_rope_norm_and_neox_differ() {
        // Guards against a regression where NEOX silently behaves like NORM.
        let mut norm = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut neox = vec![1.0_f32, 2.0, 3.0, 4.0];
        apply_rope(
            &mut norm,
            3,
            1,
            4,
            4,
            10000.0,
            RopeScaling::default(),
            crate::model::RopeStyle::Norm,
        );
        apply_rope(
            &mut neox,
            3,
            1,
            4,
            4,
            10000.0,
            RopeScaling::default(),
            crate::model::RopeStyle::Neox,
        );
        assert_ne!(norm, neox, "NEOX rotation must differ from NORM");
    }

    #[test]
    fn matmul_q8_swizzled_1x4_matches_row_major_reference() {
        use crate::q8::swizzle_q8_0_1x4;
        let rows = 4;
        let cols = 64;
        let x_i8: Vec<i8> = (0..cols)
            .map(|idx| ((idx as i16 * 11).rem_euclid(83) - 41) as i8)
            .collect();
        let x_scales = [0.125, 0.375];
        let weights: Vec<Q8_0Block> = (0..rows * (cols / Q8_BLOCK_SIZE))
            .map(|idx| q8_block(0x3800 + idx as u16, idx as i16 + 1))
            .collect();
        let swizzled = swizzle_q8_0_1x4(&weights, rows, cols / Q8_BLOCK_SIZE);

        let mut expected = vec![0.0; rows];
        matmul_q8_0(
            &mut expected,
            &x_i8,
            &x_scales,
            &weights,
            rows,
            cols,
            selector(Q8DotKernel::Scalar),
        );

        let mut candidate = vec![0.0; rows];
        let matrix = crate::model::Q8_0Swizzled1x4Matrix {
            swizzled_1x4: swizzled.clone(),
            page_aligned_1x4: None,
            rows,
            cols,
        };
        matmul_q8_0_swizzled_1x4(
            &mut candidate,
            &x_i8,
            &x_scales,
            &matrix,
            cols,
            selector(Q8DotKernel::Sdot),
        );

        assert_eq!(candidate, expected);

        let mut aligned_candidate = vec![0.0; rows];
        let aligned_matrix = crate::model::Q8_0Swizzled1x4Matrix {
            page_aligned_1x4: Some(crate::model::PageAlignedQ8_0Swizzled1x4::from_swizzled(
                &swizzled,
                rows,
                cols / Q8_BLOCK_SIZE,
            )),
            swizzled_1x4: swizzled,
            rows,
            cols,
        };
        matmul_q8_0_swizzled_1x4(
            &mut aligned_candidate,
            &x_i8,
            &x_scales,
            &aligned_matrix,
            cols,
            selector(Q8DotKernel::Sdot),
        );

        assert_eq!(aligned_candidate, expected);
    }

    #[test]
    fn accumulate_weighted_v_head_aarch64_matches_reference() {
        let head_dim = 128;
        let pos = 50;
        let kv_width = head_dim;
        let kv_h = 0;

        let mut v_cache = vec![0.0_f32; (pos + 1) * kv_width];
        for (i, val) in v_cache.iter_mut().enumerate() {
            *val = ((i * 17 + 23) % 97) as f32 * 0.0625;
        }

        let mut scores = vec![0.0_f32; pos + 1];
        for (i, val) in scores.iter_mut().enumerate() {
            *val = ((i * 3 + 7) % 31) as f32 * 0.03125;
        }

        let mut expected = vec![0.0_f32; head_dim];
        for p in 0..=pos {
            let v_head =
                &v_cache[p * kv_width + kv_h * head_dim..p * kv_width + (kv_h + 1) * head_dim];
            add_weighted_f32(&mut expected, v_head, scores[p]);
        }

        let mut candidate = vec![0.0_f32; head_dim];
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                super::accumulate_weighted_v_head_aarch64(
                    &mut candidate,
                    &v_cache,
                    &scores,
                    kv_width,
                    kv_h,
                    head_dim,
                    pos,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            candidate.copy_from_slice(&expected);
        }

        for i in 0..head_dim {
            let diff = (candidate[i] - expected[i]).abs();
            assert!(
                diff < 1e-5,
                "Mismatch at index {i}: candidate={:?}, expected={:?}",
                candidate[i],
                expected[i]
            );
        }
    }
}
