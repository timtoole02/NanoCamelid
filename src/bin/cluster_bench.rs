use std::env;
use std::io::{self, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

const DEFAULT_VECTOR_ELEMENTS: usize = 3584;
const DEFAULT_PORT: &str = "5005";
const DEFAULT_ITERATIONS: usize = 1000;
const WARMUP_ITERATIONS: usize = 50;

fn main() -> io::Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("server") => {
            let port = args.next().unwrap_or_else(|| DEFAULT_PORT.to_string());
            let vector_elements =
                parse_positive_usize(args.next(), DEFAULT_VECTOR_ELEMENTS, "vector_elements")?;
            run_server(&port, vector_elements)
        }
        Some("client") => {
            let Some(server_ip) = args.next() else {
                print_usage();
                return Ok(());
            };
            let port = args.next().unwrap_or_else(|| DEFAULT_PORT.to_string());
            let iterations = parse_positive_usize(args.next(), DEFAULT_ITERATIONS, "iterations")?;
            let vector_elements =
                parse_positive_usize(args.next(), DEFAULT_VECTOR_ELEMENTS, "vector_elements")?;
            let warmup_iterations =
                parse_positive_usize(args.next(), WARMUP_ITERATIONS, "warmup_iterations")?;
            run_client(
                &server_ip,
                &port,
                iterations,
                vector_elements,
                warmup_iterations,
            )
        }
        _ => {
            print_usage();
            Ok(())
        }
    }
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run --release --bin cluster_bench -- server [port] [vector_elements]");
    println!(
        "  cargo run --release --bin cluster_bench -- client <server_ip> [port] [iterations] [vector_elements] [warmup_iterations]"
    );
}

fn run_server(port: &str, vector_elements: usize) -> io::Result<()> {
    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr)?;
    println!("cluster_bench server listening on {addr}");
    println!("vector_elements: {vector_elements}");
    println!(
        "payload_each_way_kb: {:.2}",
        payload_bytes_len(vector_elements) as f64 / 1024.0
    );

    for stream in listener.incoming() {
        let mut stream = stream?;
        stream.set_nodelay(true)?;
        println!("client connected from {}", stream.peer_addr()?);

        let mut buffer = vec![0_u8; payload_bytes_len(vector_elements)];
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

fn run_client(
    server_ip: &str,
    port: &str,
    iterations: usize,
    vector_elements: usize,
    warmup_iterations: usize,
) -> io::Result<()> {
    let addr = format!("{server_ip}:{port}");
    println!("connecting to {addr}");
    let mut stream = TcpStream::connect(&addr)?;
    stream.set_nodelay(true)?;
    println!("connected with TCP_NODELAY");

    let bytes = payload_bytes(vector_elements);
    let mut rx_buffer = vec![0_u8; bytes.len()];

    for _ in 0..warmup_iterations {
        stream.write_all(&bytes)?;
        stream.read_exact(&mut rx_buffer)?;
    }

    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        stream.write_all(&bytes)?;
        stream.read_exact(&mut rx_buffer)?;
        timings.push(start.elapsed());
    }

    print_metrics(timings, vector_elements, warmup_iterations);
    Ok(())
}

fn print_metrics(mut timings: Vec<Duration>, vector_elements: usize, warmup_iterations: usize) {
    debug_assert!(!timings.is_empty());
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
    println!("vector_elements: {vector_elements}");
    println!(
        "payload_each_way_kb: {:.2}",
        payload_bytes_len(vector_elements) as f64 / 1024.0
    );
    println!("warmup_iterations: {warmup_iterations}");
    println!("iterations: {count}");
    println!("min_ms: {min_ms:.3}");
    println!("avg_ms: {avg_ms:.3}");
    println!("p50_ms: {p50_ms:.3}");
    println!("p95_ms: {p95_ms:.3}");
    println!("max_ms: {max_ms:.3}");
    println!(
        "json: {{\"benchmark\":\"cluster-latency\",\"vector_elements\":{},\"payload_each_way_bytes\":{},\"warmup_iterations\":{},\"iterations\":{},\"min_ms\":{:.6},\"avg_ms\":{:.6},\"p50_ms\":{:.6},\"p95_ms\":{:.6},\"max_ms\":{:.6}}}",
        vector_elements,
        payload_bytes_len(vector_elements),
        warmup_iterations,
        count,
        min_ms,
        avg_ms,
        p50_ms,
        p95_ms,
        max_ms
    );
}

fn payload_bytes_len(vector_elements: usize) -> usize {
    vector_elements * std::mem::size_of::<f32>()
}

fn payload_bytes(vector_elements: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(payload_bytes_len(vector_elements));
    for _ in 0..vector_elements {
        bytes.extend_from_slice(&1.23_f32.to_le_bytes());
    }
    bytes
}

fn parse_positive_usize(
    value: Option<String>,
    default: usize,
    name: &'static str,
) -> io::Result<usize> {
    let Some(value) = value else {
        return Ok(default);
    };
    let parsed = value.parse::<usize>().map_err(|_| {
        io::Error::new(
            ErrorKind::InvalidInput,
            format!("{name} must be a positive integer, got {value:?}"),
        )
    })?;
    if parsed == 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            format!("{name} must be greater than zero"),
        ));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::{parse_positive_usize, payload_bytes, payload_bytes_len};

    #[test]
    fn payload_bytes_match_requested_f32_width() {
        let bytes = payload_bytes(3);
        assert_eq!(bytes.len(), 12);
        assert_eq!(payload_bytes_len(3), 12);
        assert_eq!(&bytes[0..4], &1.23_f32.to_le_bytes());
    }

    #[test]
    fn parser_rejects_zero_and_invalid_values() {
        assert!(parse_positive_usize(Some("0".to_string()), 7, "iterations").is_err());
        assert!(parse_positive_usize(Some("abc".to_string()), 7, "iterations").is_err());
        assert_eq!(
            parse_positive_usize(Some("11".to_string()), 7, "iterations").unwrap(),
            11
        );
        assert_eq!(parse_positive_usize(None, 7, "iterations").unwrap(), 7);
    }
}
