use std::env;
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

const HIDDEN_DIM: usize = 3584;
const BUFFER_BYTES: usize = HIDDEN_DIM * std::mem::size_of::<f32>();
const DEFAULT_PORT: &str = "5005";
const DEFAULT_ITERATIONS: usize = 1000;
const WARMUP_ITERATIONS: usize = 50;

fn main() -> io::Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("server") => {
            let port = args.next().unwrap_or_else(|| DEFAULT_PORT.to_string());
            run_server(&port)
        }
        Some("client") => {
            let Some(server_ip) = args.next() else {
                print_usage();
                return Ok(());
            };
            let port = args.next().unwrap_or_else(|| DEFAULT_PORT.to_string());
            let iterations = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_ITERATIONS);
            run_client(&server_ip, &port, iterations)
        }
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run --release --bin cluster_bench -- server [port]");
    println!("  cargo run --release --bin cluster_bench -- client <server_ip> [port] [iterations]");
}

fn run_server(port: &str) -> io::Result<()> {
    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("cluster_bench server listening on {addr}");

    for stream in listener.incoming() {
        let mut stream = stream?;
        stream.set_nodelay(true)?;
        println!("client connected from {}", stream.peer_addr()?);

        let mut buffer = vec![0_u8; BUFFER_BYTES];
        loop {
            match stream.read_exact(&mut buffer) {
                Ok(()) => stream.write_all(&buffer)?,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::UnexpectedEof
                            | io::ErrorKind::ConnectionReset
                            | io::ErrorKind::BrokenPipe
                    ) =>
                {
                    println!("client disconnected");
                    break;
                }
                Err(error) => return Err(error),
            }
        }
    }

    Ok(())
}

fn run_client(server_ip: &str, port: &str, iterations: usize) -> io::Result<()> {
    let addr = format!("{server_ip}:{port}");
    println!("connecting to {addr}");
    let mut stream = TcpStream::connect(&addr)?;
    stream.set_nodelay(true)?;
    println!("connected with TCP_NODELAY");

    let activations = vec![1.23_f32; HIDDEN_DIM];
    let bytes =
        unsafe { std::slice::from_raw_parts(activations.as_ptr().cast::<u8>(), BUFFER_BYTES) };
    let mut rx_buffer = vec![0_u8; BUFFER_BYTES];

    for _ in 0..WARMUP_ITERATIONS {
        stream.write_all(bytes)?;
        stream.read_exact(&mut rx_buffer)?;
    }

    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        stream.write_all(bytes)?;
        stream.read_exact(&mut rx_buffer)?;
        timings.push(start.elapsed());
    }

    print_metrics(timings);
    Ok(())
}

fn print_metrics(mut timings: Vec<Duration>) {
    timings.sort_unstable();
    let total: Duration = timings.iter().sum();
    let count = timings.len();
    let avg_ms = total.as_secs_f64() * 1000.0 / count as f64;
    let min_ms = timings[0].as_secs_f64() * 1000.0;
    let max_ms = timings[count - 1].as_secs_f64() * 1000.0;
    let p50_ms = timings[count / 2].as_secs_f64() * 1000.0;
    let p95_idx = ((count as f64 * 0.95) as usize).min(count - 1);
    let p95_ms = timings[p95_idx].as_secs_f64() * 1000.0;

    println!();
    println!("=== NanoCamelid Cluster Latency Spike ===");
    println!("vector_elements: {HIDDEN_DIM}");
    println!("payload_each_way_kb: {:.2}", BUFFER_BYTES as f64 / 1024.0);
    println!("iterations: {count}");
    println!("min_ms: {min_ms:.3}");
    println!("avg_ms: {avg_ms:.3}");
    println!("p50_ms: {p50_ms:.3}");
    println!("p95_ms: {p95_ms:.3}");
    println!("max_ms: {max_ms:.3}");
}
