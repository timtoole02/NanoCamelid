//! Per-node runtime setup: rayon thread-pool sizing + core pinning, and a thermal /
//! throttle watchdog. None of this affects numerics — it only governs how the existing
//! work is scheduled onto the 4 Cortex-A76 cores and surfaces thermal warnings.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Pi 5 (BCM2712) firmware begins throttling around 80–85 °C; warn a little early.
pub const DEFAULT_THERMAL_THRESHOLD_C: f32 = 80.0;
pub const DEFAULT_THERMAL_INTERVAL: Duration = Duration::from_secs(2);

/// Build the global rayon pool with one worker per core and pin each worker to a distinct
/// core, so the OS scheduler stops migrating the hot matmul threads between the A76 cores.
///
/// Overrides:
/// - `NANOCAMELID_THREADS=<n>` forces the worker count (default = detected core count).
/// - `NANOCAMELID_PIN=0` disables pinning (workers float).
///
/// Call once at startup. Pinning/thread-count never change results (matmul rows are
/// independent), so this is purely a scheduling optimization.
pub fn configure_compute_pool() -> Result<(), String> {
    let pin = std::env::var("NANOCAMELID_PIN")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true);

    let core_ids = core_affinity::get_core_ids();
    let detected = core_ids
        .as_ref()
        .map(|c| c.len())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(4);
    let threads = std::env::var("NANOCAMELID_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(detected);

    let core_ids_for_handler = core_ids.clone();
    let result = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |idx| {
            if pin {
                if let Some(ids) = &core_ids_for_handler {
                    if !ids.is_empty() {
                        let core = ids[idx % ids.len()];
                        core_affinity::set_for_current(core);
                    }
                }
            }
        })
        .build_global();

    match result {
        Ok(()) => {
            eprintln!(
                "[runtime] rayon pool: {threads} workers, pinning={}",
                if pin { "on" } else { "off" }
            );
            Ok(())
        }
        Err(e) => Err(format!("failed to build global rayon pool: {e}")),
    }
}

/// Read the SoC temperature in Celsius (Linux thermal sysfs). `None` off-Pi.
pub fn read_cpu_temp_celsius() -> Option<f32> {
    let raw = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp").ok()?;
    let milli: f32 = raw.trim().parse().ok()?;
    Some(milli / 1000.0)
}

/// Read the Raspberry Pi `throttled` bitmask via `vcgencmd get_throttled`. `None` if the
/// tool is unavailable. Low nibble = currently-active conditions (under-voltage / freq
/// capped / throttled / soft-temp-limit).
pub fn read_throttled_flags() -> Option<u32> {
    let out = std::process::Command::new("vcgencmd")
        .arg("get_throttled")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let hex = s.trim().strip_prefix("throttled=0x")?;
    u32::from_str_radix(hex.trim(), 16).ok()
}

/// Background watchdog logging a warning whenever temperature crosses `threshold_c` or the
/// firmware reports an active throttle condition. Pure observability.
pub struct ThermalMonitor {
    running: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ThermalMonitor {
    pub fn spawn(interval: Duration, threshold_c: f32) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();
        let handle = thread::Builder::new()
            .name("thermal-monitor".into())
            .spawn(move || {
                let step = Duration::from_millis(200);
                while r.load(Ordering::Relaxed) {
                    if let Some(temp) = read_cpu_temp_celsius() {
                        let flags = read_throttled_flags();
                        let throttling = flags.map(|f| f & 0xF != 0).unwrap_or(false);
                        if temp >= threshold_c || throttling {
                            let flag_str = flags
                                .map(|f| format!("0x{f:x}"))
                                .unwrap_or_else(|| "n/a".into());
                            let mut stderr = std::io::stderr();
                            let _ = writeln!(
                                stderr,
                                "[thermal] WARN cpu={temp:.1}C (threshold {threshold_c:.0}C) throttled={flag_str}"
                            );
                        }
                    }
                    // Sleep in small steps so stop() stays responsive.
                    let mut slept = Duration::ZERO;
                    while slept < interval && r.load(Ordering::Relaxed) {
                        thread::sleep(step);
                        slept += step;
                    }
                }
            })
            .ok();
        Self { running, handle }
    }

    pub fn stop(mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for ThermalMonitor {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}
