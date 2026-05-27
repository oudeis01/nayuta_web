// namedrop_tap — capture bert_install's ZMQ + OSC output to disk.
//
// Per web-version action plan §4. Independent process, never touches bert.c or
// graphics_consumer state. Two threads:
//
//   - zmq_thread: ZMQ SUB connecting to `--zmq-endpoint`. bert.c binds a PUB
//     socket; PUB drops frames when a slow subscriber lags past HWM. For each
//     received OpMsg frame, prepend a `(ts_ns_le:u64, len_le:u32)` header and
//     stream the raw payload into ops.bin.zst.
//
//   - osc_thread: UDP listener on `--osc-bind`. Each datagram is decoded with
//     `rosc` and written as one JSON line into events.jsonl.zst, preserving
//     path, OSC type string, and arg values. Lemma_ids fired by /bert/whisper
//     are collected into a HashSet for the manifest's prefetch list.
//
// Stop condition: whichever fires first of (a) `--duration-s` elapsed, (b)
// /bert/done event received, (c) SIGINT/SIGTERM. On stop, both compressed
// streams are flushed and a manifest.json is written next to them.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::net::UdpSocket;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::Parser;
use rosc::{OscPacket, OscType};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Parser, Debug, Clone)]
#[command(name = "namedrop_tap",
          about = "Capture bert_install ZMQ+OSC streams to zstd-compressed files")]
struct Args {
    /// Output directory. ops.bin.zst, events.jsonl.zst, manifest.json land here.
    #[arg(long)]
    out_dir: PathBuf,

    /// Short identifier — recorded in manifest, not used for filenames.
    #[arg(long)]
    tag: String,

    /// ZMQ PULL connect endpoint. bert_install binds tcp://*:5555 by default,
    /// so localhost works when tap and bert_install share a machine.
    #[arg(long, default_value = "tcp://127.0.0.1:5555")]
    zmq_endpoint: String,

    /// OSC UDP bind. Must match bert_install's `--osc-host` target.
    #[arg(long, default_value = "127.0.0.1:57120")]
    osc_bind: String,

    /// Hard time cap in seconds. 0 = no cap (rely on /bert/done or signal).
    #[arg(long, default_value_t = 0)]
    duration_s: u64,

    /// zstd compression level (1..=22). 3 is a good live-streaming default.
    #[arg(long, default_value_t = 3)]
    zstd_level: i32,

    /// Optional path to the corpus_demo sidecar JSON, copied into manifest.
    #[arg(long)]
    corpus_sidecar: Option<PathBuf>,

    /// Optional path to bert_install binary — its sha256 goes into the
    /// manifest so a future re-render can verify provenance.
    #[arg(long)]
    bert_install: Option<PathBuf>,
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn sha256_file(p: &Path) -> Option<String> {
    let mut f = File::open(p).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Some(hex::encode(hasher.finalize()))
}

// Shared counters between threads — small enough to keep behind atomics
// rather than introducing a channel just for stats.
struct Stats {
    n_ops_messages: AtomicU64,
    n_ops_bytes:    AtomicU64,
    n_osc_events:   AtomicU64,
    n_whisper:      AtomicU64,
    n_word_trigger: AtomicU64,
    n_op_flow:      AtomicU64,
    n_clock:        AtomicU64,
    saw_done:       AtomicBool,
}

impl Stats {
    fn new() -> Self {
        Self {
            n_ops_messages: AtomicU64::new(0),
            n_ops_bytes:    AtomicU64::new(0),
            n_osc_events:   AtomicU64::new(0),
            n_whisper:      AtomicU64::new(0),
            n_word_trigger: AtomicU64::new(0),
            n_op_flow:      AtomicU64::new(0),
            n_clock:        AtomicU64::new(0),
            saw_done:       AtomicBool::new(false),
        }
    }
}

fn run_zmq_thread(
    endpoint: String,
    out_path: PathBuf,
    zstd_level: i32,
    stop: Arc<AtomicBool>,
    stats: Arc<Stats>,
) -> thread::JoinHandle<std::io::Result<()>> {
    thread::Builder::new()
        .name("zmq".into())
        .spawn(move || {
            let ctx = zmq::Context::new();
            // bert.c binds a ZMQ_PUB socket; we subscribe to all topics.
            let sock = ctx.socket(zmq::SUB).expect("zmq SUB socket");
            sock.set_rcvtimeo(1000).expect("set RCVTIMEO");
            sock.set_rcvhwm(3000).expect("set RCVHWM");
            sock.connect(&endpoint).expect("zmq connect");
            sock.set_subscribe(b"").expect("zmq subscribe ''");

            let f = File::create(&out_path)?;
            let bw = BufWriter::with_capacity(1 << 20, f);
            let mut enc = zstd::stream::Encoder::new(bw, zstd_level)?;

            let mut buf = vec![0u8; 64 * 1024];
            while !stop.load(Ordering::Acquire) {
                // zmq_recv with timeout returns Err(EAGAIN) on timeout, which
                // is the expected path while we wait for the next batch.
                match sock.recv_into(&mut buf, 0) {
                    Ok(n) => {
                        let ts = now_ns().to_le_bytes();
                        let ln = (n as u32).to_le_bytes();
                        enc.write_all(&ts)?;
                        enc.write_all(&ln)?;
                        enc.write_all(&buf[..n])?;
                        stats.n_ops_messages.fetch_add(1, Ordering::Relaxed);
                        stats.n_ops_bytes.fetch_add(n as u64, Ordering::Relaxed);
                    }
                    Err(zmq::Error::EAGAIN) => continue,
                    Err(e) => {
                        eprintln!("[tap/zmq] recv error: {e}");
                        break;
                    }
                }
            }
            enc.finish()?.flush()?;
            eprintln!("[tap/zmq] flushed {} messages, {} bytes",
                      stats.n_ops_messages.load(Ordering::Relaxed),
                      stats.n_ops_bytes.load(Ordering::Relaxed));
            Ok(())
        })
        .expect("spawn zmq thread")
}

#[derive(Serialize)]
struct OscRecord<'a> {
    ts_ns: u64,
    path:  &'a str,
    types: String,
    args:  Vec<serde_json::Value>,
}

fn osc_arg_to_json(a: &OscType) -> (char, serde_json::Value) {
    // The OSC type tag should match what bert.c declares in its OSC_SEND
    // call sites (see bert.c §132–144). We render structural args verbatim
    // and pass binary blobs (used by /bert/whisper) through as base16 strings.
    use serde_json::json;
    match a {
        OscType::Int(i)    => ('i', json!(i)),
        OscType::Long(i)   => ('h', json!(i)),
        OscType::Float(f)  => ('f', json!(f)),
        OscType::Double(d) => ('d', json!(d)),
        OscType::String(s) => ('s', json!(s)),
        OscType::Blob(b)   => ('b', json!(hex::encode(b))),
        OscType::Bool(b)   => ('B', json!(b)),
        OscType::Nil       => ('N', serde_json::Value::Null),
        OscType::Inf       => ('I', json!("inf")),
        OscType::Char(c)   => ('c', json!(c.to_string())),
        OscType::Color(c)  => ('r', json!([c.red, c.green, c.blue, c.alpha])),
        OscType::Midi(_)   => ('m', json!("midi")),
        OscType::Time(t)   => ('t', json!([t.seconds, t.fractional])),
        OscType::Array(_)  => ('[', json!("array")),
    }
}

fn handle_packet(
    pkt: &OscPacket,
    ts_ns: u64,
    enc: &mut zstd::stream::Encoder<BufWriter<File>>,
    fired_lemmas: &Mutex<HashSet<i64>>,
    stats: &Stats,
) -> std::io::Result<()> {
    match pkt {
        OscPacket::Message(m) => {
            let mut types = String::with_capacity(m.args.len());
            let mut args = Vec::with_capacity(m.args.len());
            for a in &m.args {
                let (t, v) = osc_arg_to_json(a);
                types.push(t);
                args.push(v);
            }

            // Light per-path bookkeeping for the manifest.
            match m.addr.as_str() {
                "/bert/whisper" => {
                    stats.n_whisper.fetch_add(1, Ordering::Relaxed);
                    // /bert/whisper payload starts with lemma_id as int32.
                    if let Some(OscType::Int(lid)) = m.args.first() {
                        if let Ok(mut set) = fired_lemmas.lock() {
                            set.insert(*lid as i64);
                        }
                    }
                }
                "/bert/word_trigger" => {
                    stats.n_word_trigger.fetch_add(1, Ordering::Relaxed);
                    if let Some(OscType::Int(lid)) = m.args.first() {
                        if let Ok(mut set) = fired_lemmas.lock() {
                            set.insert(*lid as i64);
                        }
                    }
                }
                "/bert/op_flow" => { stats.n_op_flow.fetch_add(1, Ordering::Relaxed); }
                "/bert/clock"   => { stats.n_clock.fetch_add(1, Ordering::Relaxed); }
                "/bert/done"    => { stats.saw_done.store(true, Ordering::Release); }
                _ => {}
            }
            stats.n_osc_events.fetch_add(1, Ordering::Relaxed);

            let rec = OscRecord { ts_ns, path: &m.addr, types, args };
            let line = serde_json::to_string(&rec).expect("osc record serializes");
            enc.write_all(line.as_bytes())?;
            enc.write_all(b"\n")?;
        }
        OscPacket::Bundle(b) => {
            for inner in &b.content {
                handle_packet(inner, ts_ns, enc, fired_lemmas, stats)?;
            }
        }
    }
    Ok(())
}

fn run_osc_thread(
    bind: String,
    out_path: PathBuf,
    zstd_level: i32,
    stop: Arc<AtomicBool>,
    stats: Arc<Stats>,
    fired_lemmas: Arc<Mutex<HashSet<i64>>>,
) -> thread::JoinHandle<std::io::Result<()>> {
    thread::Builder::new()
        .name("osc".into())
        .spawn(move || {
            let sock = UdpSocket::bind(&bind).expect("UDP bind");
            sock.set_read_timeout(Some(Duration::from_secs(1)))
                .expect("set read timeout");

            let f = File::create(&out_path)?;
            let bw = BufWriter::with_capacity(1 << 20, f);
            let mut enc = zstd::stream::Encoder::new(bw, zstd_level)?;

            let mut buf = [0u8; 64 * 1024];
            while !stop.load(Ordering::Acquire) {
                match sock.recv_from(&mut buf) {
                    Ok((n, _src)) => {
                        let ts = now_ns();
                        match rosc::decoder::decode_udp(&buf[..n]) {
                            Ok((_, pkt)) => {
                                handle_packet(&pkt, ts, &mut enc,
                                              &fired_lemmas, &stats)?;
                            }
                            Err(e) => {
                                eprintln!("[tap/osc] decode error: {e:?}");
                            }
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock
                          || e.kind() == std::io::ErrorKind::TimedOut => continue,
                    Err(e) => {
                        eprintln!("[tap/osc] recv error: {e}");
                        break;
                    }
                }
            }
            enc.finish()?.flush()?;
            eprintln!("[tap/osc] flushed {} osc events ({} whisper, {} word, {} flow, {} clock)",
                      stats.n_osc_events.load(Ordering::Relaxed),
                      stats.n_whisper.load(Ordering::Relaxed),
                      stats.n_word_trigger.load(Ordering::Relaxed),
                      stats.n_op_flow.load(Ordering::Relaxed),
                      stats.n_clock.load(Ordering::Relaxed));
            Ok(())
        })
        .expect("spawn osc thread")
}

#[derive(Serialize)]
struct Manifest {
    tag: String,
    host: String,
    started_unix_ns: u64,
    ended_unix_ns: u64,
    duration_s: f64,
    duration_cap_s: u64,
    stopped_by: String,
    zmq_endpoint: String,
    osc_bind: String,
    zstd_level: i32,
    n_ops_messages: u64,
    n_ops_bytes_uncompressed: u64,
    n_osc_events: u64,
    n_whisper: u64,
    n_word_trigger: u64,
    n_op_flow: u64,
    n_clock: u64,
    saw_done: bool,
    fired_lemmas: Vec<i64>,
    bert_install_sha256: Option<String>,
    corpus_sidecar: Option<serde_json::Value>,
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| {
            std::process::Command::new("hostname")
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .map_err(|_| std::env::VarError::NotPresent)
        })
        .unwrap_or_else(|_| "unknown".to_string())
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out_dir)?;

    let ops_path    = args.out_dir.join("ops.bin.zst");
    let events_path = args.out_dir.join("events.jsonl.zst");
    let manifest_path = args.out_dir.join("manifest.json");

    let stop = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(Stats::new());
    let fired_lemmas = Arc::new(Mutex::new(HashSet::<i64>::new()));

    {
        let stop_c = stop.clone();
        ctrlc::set_handler(move || {
            eprintln!("[tap] signal — stopping");
            stop_c.store(true, Ordering::Release);
        }).expect("install signal handler");
    }

    let started_ns = now_ns();
    let started_inst = Instant::now();

    let zmq_h = run_zmq_thread(args.zmq_endpoint.clone(), ops_path.clone(),
                               args.zstd_level, stop.clone(), stats.clone());
    let osc_h = run_osc_thread(args.osc_bind.clone(), events_path.clone(),
                               args.zstd_level, stop.clone(), stats.clone(),
                               fired_lemmas.clone());

    eprintln!("[tap] started  tag={}  out={}  cap={}s  zmq={}  osc={}",
              args.tag, args.out_dir.display(), args.duration_s,
              args.zmq_endpoint, args.osc_bind);

    let mut stopped_by = String::from("signal");
    let report_every = Duration::from_secs(10);
    let mut last_report = Instant::now();

    loop {
        thread::sleep(Duration::from_millis(200));

        if stop.load(Ordering::Acquire) {
            stopped_by = "signal".into();
            break;
        }
        if stats.saw_done.load(Ordering::Acquire) {
            eprintln!("[tap] /bert/done received — stopping");
            stop.store(true, Ordering::Release);
            stopped_by = "bert_done".into();
            break;
        }
        if args.duration_s > 0 && started_inst.elapsed().as_secs() >= args.duration_s {
            eprintln!("[tap] duration cap reached — stopping");
            stop.store(true, Ordering::Release);
            stopped_by = "duration_cap".into();
            break;
        }

        if last_report.elapsed() >= report_every {
            last_report = Instant::now();
            eprintln!("[tap] +{:>5}s  ops_msgs={}  osc={} (whisper={} word={} flow={} clock={})",
                      started_inst.elapsed().as_secs(),
                      stats.n_ops_messages.load(Ordering::Relaxed),
                      stats.n_osc_events.load(Ordering::Relaxed),
                      stats.n_whisper.load(Ordering::Relaxed),
                      stats.n_word_trigger.load(Ordering::Relaxed),
                      stats.n_op_flow.load(Ordering::Relaxed),
                      stats.n_clock.load(Ordering::Relaxed));
        }
    }

    // Let the worker threads observe the stop flag and drain their buffers.
    zmq_h.join().expect("zmq join")?;
    osc_h.join().expect("osc join")?;

    let ended_ns = now_ns();
    let duration_s = (ended_ns - started_ns) as f64 / 1e9;

    let mut fired: Vec<i64> = fired_lemmas.lock().unwrap().iter().copied().collect();
    fired.sort_unstable();

    let corpus_sidecar = args.corpus_sidecar.as_deref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());

    let bert_install_sha256 = args.bert_install.as_deref().and_then(sha256_file);

    let manifest = Manifest {
        tag: args.tag.clone(),
        host: hostname(),
        started_unix_ns: started_ns,
        ended_unix_ns: ended_ns,
        duration_s,
        duration_cap_s: args.duration_s,
        stopped_by,
        zmq_endpoint: args.zmq_endpoint.clone(),
        osc_bind: args.osc_bind.clone(),
        zstd_level: args.zstd_level,
        n_ops_messages: stats.n_ops_messages.load(Ordering::Relaxed),
        n_ops_bytes_uncompressed: stats.n_ops_bytes.load(Ordering::Relaxed),
        n_osc_events: stats.n_osc_events.load(Ordering::Relaxed),
        n_whisper: stats.n_whisper.load(Ordering::Relaxed),
        n_word_trigger: stats.n_word_trigger.load(Ordering::Relaxed),
        n_op_flow: stats.n_op_flow.load(Ordering::Relaxed),
        n_clock: stats.n_clock.load(Ordering::Relaxed),
        saw_done: stats.saw_done.load(Ordering::Acquire),
        fired_lemmas: fired,
        bert_install_sha256,
        corpus_sidecar,
    };
    std::fs::write(&manifest_path,
                   serde_json::to_string_pretty(&manifest).unwrap())?;
    eprintln!("[tap] wrote {} ({:.1}s, {} ops_msgs, {} osc, {} fired_lemmas)",
              manifest_path.display(),
              duration_s,
              manifest.n_ops_messages,
              manifest.n_osc_events,
              manifest.fired_lemmas.len());

    Ok(())
}
