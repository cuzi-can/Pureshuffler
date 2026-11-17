//! Pure-Rust MP4 shuffler (no FFmpeg, no re-encoding, no external crates).
//!
//! The file is cut at keyframes into chunks of about the requested length, the
//! chunks are shuffled so that none stays in its original position, and a new
//! MP4 is written: the original compressed samples are copied byte-for-byte
//! into a new `mdat`, and the sample tables in `moov` are rebuilt to match.
//!
//! Supported input: ordinary (non-fragmented) MP4/MOV files with one video
//! track and optional audio tracks (H.264/HEVC + AAC and similar).

use std::collections::hash_map::RandomState;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, File};
use std::hash::{BuildHasher, Hasher};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

type BoxError = Box<dyn Error>;
type AppResult<T> = std::result::Result<T, BoxError>;
type FourCC = [u8; 4];

const CORRUPT: &str = "This file is damaged or uses an MP4 layout this tool does not support.";
const MAX_SAMPLES: usize = 50_000_000;
const MAX_MOOV_BYTES: u64 = 1 << 31;
const MAX_BOX_DEPTH: usize = 16;

const USAGE: &str = "\
Pure-Rust MP4 shuffler (no FFmpeg, no re-encoding)

Cuts an MP4 video at keyframes into chunks of about the requested length,
shuffles the chunks so that no chunk stays in its original position, and writes
a new MP4.

USAGE:
    pureshuffler [OPTIONS] [INPUT]

ARGS:
    INPUT    Video to shuffle (default: 0.mp4 in the current folder, or next
             to this program)

OPTIONS:
    -c, --chunk <SECONDS>   Wanted chunk length in seconds (default: 2).
                            Cuts can only land on keyframes, so real chunk
                            lengths follow the video's keyframe spacing.
    -o, --output <FILE>     Output file (default: 1.mp4 next to the input)
    -s, --seed <NUMBER>     Fixed random seed, for a repeatable shuffle
        --no-pause          Do not wait for Enter before exiting (Windows only)
    -h, --help              Show this help";

// ---------------------------------------------------------------------------
// Random numbers (std only)
// ---------------------------------------------------------------------------

/// SplitMix64: tiny, fast and statistically solid for shuffling.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..n` (`n` must be > 0), without modulo bias.
    fn below(&mut self, n: u64) -> u64 {
        let threshold = n.wrapping_neg() % n;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return value % n;
            }
        }
    }
}

/// Seed from the OS-randomised hasher keys mixed with the clock and process id.
fn random_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(nanos);
    hasher.write_u32(process::id());
    hasher.finish()
}

/// Shuffle with no item left in its original position (Sattolo's algorithm:
/// a single random cycle, so every element moves).
fn shuffle_segments<T>(mut items: Vec<T>, seed: Option<u64>) -> AppResult<Vec<T>> {
    if items.len() < 2 {
        return Err("Need at least two chunks to shuffle.".into());
    }
    let mut rng = SplitMix64::new(seed.unwrap_or_else(random_seed));
    for index in (1..items.len()).rev() {
        let other = rng.below(index as u64) as usize;
        items.swap(index, other);
    }
    Ok(items)
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Make a path absolute and lexically clean without touching the file system
/// (and without producing Windows `\\?\` verbatim paths).
fn absolutize(path: &Path) -> io::Result<PathBuf> {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };
    let mut cleaned = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                cleaned.pop();
            }
            other => cleaned.push(other.as_os_str()),
        }
    }
    Ok(cleaned)
}

/// True if both paths point at the same file (also through symlinks / case).
fn same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

fn exe_dir() -> Option<PathBuf> {
    env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
}

/// `0.mp4` in the current folder, else next to the program, else the former
/// (so the "Video not found" message names a sensible path).
fn default_input() -> PathBuf {
    let in_cwd = PathBuf::from("0.mp4");
    if in_cwd.is_file() {
        return in_cwd;
    }
    if let Some(dir) = exe_dir() {
        let beside_exe = dir.join("0.mp4");
        if beside_exe.is_file() {
            return beside_exe;
        }
    }
    in_cwd
}

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

fn slice_at(data: &[u8], pos: usize, len: usize) -> AppResult<&[u8]> {
    let end = pos.checked_add(len).ok_or(CORRUPT)?;
    Ok(data.get(pos..end).ok_or(CORRUPT)?)
}

fn u32_at(data: &[u8], pos: usize) -> AppResult<u32> {
    let b = slice_at(data, pos, 4)?;
    Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn u64_at(data: &[u8], pos: usize) -> AppResult<u64> {
    let b = slice_at(data, pos, 8)?;
    Ok(u64::from_be_bytes([
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
    ]))
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

// ---------------------------------------------------------------------------
// MP4 box tree (only the containers we need to look into are parsed)
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Body {
    Leaf(Vec<u8>),
    Container(Vec<Mp4Box>),
}

#[derive(Clone)]
struct Mp4Box {
    typ: FourCC,
    body: Body,
}

impl Mp4Box {
    fn leaf(typ: &FourCC, body: Vec<u8>) -> Self {
        Mp4Box {
            typ: *typ,
            body: Body::Leaf(body),
        }
    }
}

fn is_container(typ: &FourCC) -> bool {
    matches!(typ, b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl")
}

/// Parse a run of sibling boxes. Containers are parsed recursively, every other
/// box is kept as an opaque byte blob (so it can be copied back unchanged).
fn parse_boxes(data: &[u8], depth: usize) -> AppResult<Vec<Mp4Box>> {
    if depth > MAX_BOX_DEPTH {
        return Err(CORRUPT.into());
    }
    let mut boxes = Vec::new();
    let mut pos = 0usize;
    while data.len() - pos >= 8 {
        let size32 = u32_at(data, pos)? as u64;
        let typ: FourCC = [data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]];
        let remaining = (data.len() - pos) as u64;
        let (header_len, size) = match size32 {
            1 => (16u64, u64_at(data, pos + 8)?),
            0 => (8u64, remaining),
            n => (8u64, n),
        };
        if size < header_len || size > remaining {
            return Err(CORRUPT.into());
        }
        let body_bytes = &data[pos + header_len as usize..pos + size as usize];
        let body = if is_container(&typ) {
            Body::Container(parse_boxes(body_bytes, depth + 1)?)
        } else {
            Body::Leaf(body_bytes.to_vec())
        };
        boxes.push(Mp4Box { typ, body });
        pos += size as usize;
    }
    Ok(boxes)
}

fn write_box(out: &mut Vec<u8>, b: &Mp4Box) -> AppResult<()> {
    let start = out.len();
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&b.typ);
    match &b.body {
        Body::Leaf(bytes) => out.extend_from_slice(bytes),
        Body::Container(children) => {
            for child in children {
                write_box(out, child)?;
            }
        }
    }
    let size = u32::try_from(out.len() - start)
        .map_err(|_| BoxError::from("An MP4 header box became too large to write."))?;
    out[start..start + 4].copy_from_slice(&size.to_be_bytes());
    Ok(())
}

fn find_box<'a>(boxes: &'a [Mp4Box], typ: &FourCC) -> Option<&'a Mp4Box> {
    boxes.iter().find(|b| &b.typ == typ)
}

fn find_leaf<'a>(boxes: &'a [Mp4Box], typ: &FourCC) -> Option<&'a [u8]> {
    match find_box(boxes, typ) {
        Some(Mp4Box {
            body: Body::Leaf(bytes),
            ..
        }) => Some(bytes.as_slice()),
        _ => None,
    }
}

fn find_children<'a>(boxes: &'a [Mp4Box], typ: &FourCC) -> Option<&'a [Mp4Box]> {
    match find_box(boxes, typ) {
        Some(Mp4Box {
            body: Body::Container(kids),
            ..
        }) => Some(kids.as_slice()),
        _ => None,
    }
}

/// Replace the children of the first `stbl` found (searching containers).
fn replace_stbl(boxes: &mut [Mp4Box], replacement: &mut Option<Vec<Mp4Box>>) -> bool {
    for b in boxes.iter_mut() {
        if &b.typ == b"stbl" {
            return match replacement.take() {
                Some(children) => {
                    b.body = Body::Container(children);
                    true
                }
                None => false,
            };
        }
        if let Body::Container(children) = &mut b.body {
            if replace_stbl(children, replacement) {
                return true;
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tracks and sample tables
// ---------------------------------------------------------------------------

struct Sample {
    offset: u64,
    size: u32,
    dur: u32,
    cts: u32,
    sync: bool,
    desc: u32,
}

struct Track {
    /// Position of this `trak` among the children of `moov`.
    index: usize,
    timescale: u32,
    stsd: Vec<u8>,
    has_ctts: bool,
    ctts_header: [u8; 4],
    has_stss: bool,
    samples: Vec<Sample>,
    /// Decode time of every sample, in track timescale units.
    dts: Vec<u64>,
}

fn handler_of(trak: &[Mp4Box]) -> Option<FourCC> {
    let mdia = find_children(trak, b"mdia")?;
    let hdlr = find_leaf(mdia, b"hdlr")?;
    let bytes = hdlr.get(8..12)?;
    Some([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn parse_track(trak: &[Mp4Box], index: usize) -> AppResult<Track> {
    let mdia = find_children(trak, b"mdia").ok_or(CORRUPT)?;
    let mdhd = find_leaf(mdia, b"mdhd").ok_or(CORRUPT)?;
    let mdhd_version = *mdhd.first().ok_or(CORRUPT)?;
    let timescale = u32_at(mdhd, if mdhd_version == 1 { 20 } else { 12 })?;
    if timescale == 0 {
        return Err(CORRUPT.into());
    }
    let minf = find_children(mdia, b"minf").ok_or(CORRUPT)?;
    let stbl = find_children(minf, b"stbl").ok_or(CORRUPT)?;
    let stsd = find_leaf(stbl, b"stsd").ok_or(CORRUPT)?.to_vec();

    // Sample sizes.
    let stsz = find_leaf(stbl, b"stsz")
        .ok_or("This file has no standard sample size table (stz2 is not supported).")?;
    let fixed_size = u32_at(stsz, 4)?;
    let count = u32_at(stsz, 8)? as usize;
    if count > MAX_SAMPLES {
        return Err(CORRUPT.into());
    }
    let mut sizes: Vec<u32> = Vec::new();
    for i in 0..count {
        sizes.push(if fixed_size != 0 {
            fixed_size
        } else {
            u32_at(stsz, 12 + 4 * i)?
        });
    }

    // Sample durations.
    let stts = find_leaf(stbl, b"stts").ok_or(CORRUPT)?;
    let stts_entries = u32_at(stts, 4)? as usize;
    let mut durations: Vec<u32> = Vec::new();
    for i in 0..stts_entries {
        let run = u32_at(stts, 8 + 8 * i)? as usize;
        let delta = u32_at(stts, 12 + 8 * i)?;
        if durations.len() + run > count {
            return Err(CORRUPT.into());
        }
        durations.extend(std::iter::repeat(delta).take(run));
    }
    if durations.len() != count {
        return Err(CORRUPT.into());
    }

    // Composition offsets (optional; kept as raw 32-bit values so both signed
    // and unsigned versions of the box are preserved exactly).
    let mut ctts_values: Vec<u32> = Vec::new();
    let mut ctts_header = [0u8; 4];
    let has_ctts = match find_leaf(stbl, b"ctts") {
        Some(ctts) => {
            let header = slice_at(ctts, 0, 4)?;
            ctts_header = [header[0], header[1], header[2], header[3]];
            let entries = u32_at(ctts, 4)? as usize;
            for i in 0..entries {
                let run = u32_at(ctts, 8 + 8 * i)? as usize;
                let offset = u32_at(ctts, 12 + 8 * i)?;
                if ctts_values.len() + run > count {
                    return Err(CORRUPT.into());
                }
                ctts_values.extend(std::iter::repeat(offset).take(run));
            }
            if ctts_values.len() != count {
                return Err(CORRUPT.into());
            }
            true
        }
        None => false,
    };

    // Sync samples (keyframes). No stss box means every sample is a sync sample.
    let mut sync = vec![true; count];
    let has_stss = match find_leaf(stbl, b"stss") {
        Some(stss) => {
            sync = vec![false; count];
            let entries = u32_at(stss, 4)? as usize;
            for i in 0..entries {
                let number = u32_at(stss, 8 + 4 * i)? as usize;
                if number >= 1 && number <= count {
                    sync[number - 1] = true;
                }
            }
            true
        }
        None => false,
    };

    // Sample-to-chunk map and chunk offsets.
    let stsc = find_leaf(stbl, b"stsc").ok_or(CORRUPT)?;
    let stsc_entries = u32_at(stsc, 4)? as usize;
    let mut runs: Vec<(u32, u32, u32)> = Vec::new();
    for i in 0..stsc_entries {
        runs.push((
            u32_at(stsc, 8 + 12 * i)?,
            u32_at(stsc, 12 + 12 * i)?,
            u32_at(stsc, 16 + 12 * i)?,
        ));
    }

    let chunk_offsets: Vec<u64> = if let Some(stco) = find_leaf(stbl, b"stco") {
        let n = u32_at(stco, 4)? as usize;
        let mut offsets = Vec::new();
        for i in 0..n {
            offsets.push(u32_at(stco, 8 + 4 * i)? as u64);
        }
        offsets
    } else if let Some(co64) = find_leaf(stbl, b"co64") {
        let n = u32_at(co64, 4)? as usize;
        let mut offsets = Vec::new();
        for i in 0..n {
            offsets.push(u64_at(co64, 8 + 8 * i)?);
        }
        offsets
    } else {
        return Err(CORRUPT.into());
    };

    // Absolute file offset of every sample.
    let mut samples: Vec<Sample> = Vec::new();
    let mut run_index = 0usize;
    'chunks: for (chunk_index, &chunk_offset) in chunk_offsets.iter().enumerate() {
        let chunk_number = chunk_index as u64 + 1;
        while run_index + 1 < runs.len() && (runs[run_index + 1].0 as u64) <= chunk_number {
            run_index += 1;
        }
        let (_, per_chunk, desc) = *runs.get(run_index).ok_or(CORRUPT)?;
        let mut offset = chunk_offset;
        for _ in 0..per_chunk {
            let k = samples.len();
            if k >= count {
                break 'chunks;
            }
            samples.push(Sample {
                offset,
                size: sizes[k],
                dur: durations[k],
                cts: if has_ctts { ctts_values[k] } else { 0 },
                sync: sync[k],
                desc,
            });
            offset += sizes[k] as u64;
        }
    }
    if samples.len() != count {
        return Err(CORRUPT.into());
    }

    let mut dts = Vec::with_capacity(count);
    let mut clock = 0u64;
    for sample in &samples {
        dts.push(clock);
        clock += sample.dur as u64;
    }

    Ok(Track {
        index,
        timescale,
        stsd,
        has_ctts,
        ctts_header,
        has_stss,
        samples,
        dts,
    })
}

// ---------------------------------------------------------------------------
// Cutting into chunks
// ---------------------------------------------------------------------------

/// Sample indices (in the video track) where chunks 1, 2, ... begin. Each is
/// the first keyframe at or after the next multiple of the chunk length.
fn video_bounds(video: &Track, chunk: f64) -> Vec<usize> {
    let timescale = video.timescale as f64;
    let mut bounds = Vec::new();
    let mut target = chunk;
    for (i, sample) in video.samples.iter().enumerate().skip(1) {
        if !sample.sync {
            continue;
        }
        let t = video.dts[i] as f64 / timescale;
        if t >= target {
            bounds.push(i);
            target = ((t / chunk).floor() + 1.0) * chunk;
        }
    }
    bounds
}

/// Index of the sample of `track` closest in time to a video cut point.
fn cut_index(track: &Track, video_timescale: u32, video_dts: u64) -> usize {
    let vts = video_timescale as u128;
    let target = (video_dts as u128) * (track.timescale as u128);
    let idx = track.dts.partition_point(|&d| (d as u128) * vts < target);
    if idx == 0 || idx >= track.dts.len() {
        return idx;
    }
    let below = target - (track.dts[idx - 1] as u128) * vts;
    let above = (track.dts[idx] as u128) * vts - target;
    if below < above {
        idx - 1
    } else {
        idx
    }
}

struct Plan {
    /// `ranges[track][segment]` = half-open range of sample indices.
    ranges: Vec<Vec<(usize, usize)>>,
    /// Duration correction (track ticks) for the last sample of each segment,
    /// so audio chunks keep exactly the length of the video chunk they follow.
    fixes: Vec<Vec<i64>>,
    segments: usize,
}

/// `tracks[0]` must be the video track.
fn plan_segments(tracks: &[Track], chunk: f64) -> Plan {
    let video = &tracks[0];
    let bounds = video_bounds(video, chunk);
    let segments = bounds.len() + 1;

    let mut ranges: Vec<Vec<(usize, usize)>> = Vec::new();
    for (ti, track) in tracks.iter().enumerate() {
        let cuts: Vec<usize> = if ti == 0 {
            bounds.clone()
        } else {
            bounds
                .iter()
                .map(|&b| cut_index(track, video.timescale, video.dts[b]))
                .collect()
        };
        let mut track_ranges = Vec::with_capacity(segments);
        let mut start = 0usize;
        for seg in 0..segments {
            let end = if seg < cuts.len() {
                cuts[seg]
            } else {
                track.samples.len()
            };
            track_ranges.push((start, end.max(start)));
            start = end.max(start);
        }
        ranges.push(track_ranges);
    }

    let mut fixes: Vec<Vec<i64>> = Vec::new();
    for (ti, track) in tracks.iter().enumerate() {
        let mut fix = vec![0i64; segments];
        if ti > 0 {
            for seg in 0..segments - 1 {
                let (s, e) = ranges[ti][seg];
                if e <= s {
                    continue;
                }
                let (vs, ve) = ranges[0][seg];
                let video_ticks: u128 = video.samples[vs..ve].iter().map(|x| x.dur as u128).sum();
                let wanted = (video_ticks * track.timescale as u128 + video.timescale as u128 / 2)
                    / video.timescale as u128;
                let actual: u128 = track.samples[s..e].iter().map(|x| x.dur as u128).sum();
                fix[seg] = wanted as i64 - actual as i64;
            }
        }
        fixes.push(fix);
    }

    Plan {
        ranges,
        fixes,
        segments,
    }
}

// ---------------------------------------------------------------------------
// Rebuilding the sample tables
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Tables {
    durations: Vec<u32>,
    ctts: Vec<u32>,
    sizes: Vec<u32>,
    /// 1-based sample numbers of sync samples, in the new order.
    sync: Vec<u32>,
    /// (samples in chunk, sample description index) per new chunk.
    chunk_samples: Vec<(u32, u32)>,
    /// Offset of each new chunk relative to the start of the mdat payload.
    chunk_offsets: Vec<u64>,
}

/// Lay out the new file: for every segment in the new order, one chunk per
/// track. Returns the per-track tables and the total mdat payload size.
fn build_tables(tracks: &[Track], plan: &Plan, order: &[usize]) -> (Vec<Tables>, u64) {
    let mut tables: Vec<Tables> = tracks.iter().map(|_| Tables::default()).collect();
    let mut cursor: u64 = 0;

    for &seg in order {
        for (ti, track) in tracks.iter().enumerate() {
            let (s, e) = plan.ranges[ti][seg];
            let table = &mut tables[ti];

            let mut i = s;
            while i < e {
                let desc = track.samples[i].desc;
                let start = cursor;
                let mut j = i;
                while j < e && track.samples[j].desc == desc {
                    cursor += track.samples[j].size as u64;
                    j += 1;
                }
                table.chunk_samples.push(((j - i) as u32, desc));
                table.chunk_offsets.push(start);
                i = j;
            }

            let fix = plan.fixes[ti][seg];
            for k in s..e {
                let sample = &track.samples[k];
                let mut duration = sample.dur;
                if k + 1 == e && fix != 0 {
                    duration = (sample.dur as i64 + fix).clamp(1, u32::MAX as i64) as u32;
                }
                table.durations.push(duration);
                if track.has_ctts {
                    table.ctts.push(sample.cts);
                }
                table.sizes.push(sample.size);
                if sample.sync {
                    table.sync.push(table.sizes.len() as u32);
                }
            }
        }
    }
    (tables, cursor)
}

fn run_lengths(values: &[u32]) -> Vec<(u32, u32)> {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &value in values {
        if let Some(last) = runs.last_mut() {
            if last.1 == value {
                last.0 += 1;
                continue;
            }
        }
        runs.push((1, value));
    }
    runs
}

fn run_length_box(header: [u8; 4], values: &[u32]) -> Vec<u8> {
    let runs = run_lengths(values);
    let mut body = header.to_vec();
    push_u32(&mut body, runs.len() as u32);
    for &(count, value) in &runs {
        push_u32(&mut body, count);
        push_u32(&mut body, value);
    }
    body
}

fn stbl_children(
    track: &Track,
    tables: &Tables,
    base: u64,
    use_co64: bool,
) -> AppResult<Vec<Mp4Box>> {
    let mut kids = vec![Mp4Box::leaf(b"stsd", track.stsd.clone())];

    kids.push(Mp4Box::leaf(
        b"stts",
        run_length_box([0; 4], &tables.durations),
    ));
    if track.has_ctts {
        kids.push(Mp4Box::leaf(
            b"ctts",
            run_length_box(track.ctts_header, &tables.ctts),
        ));
    }
    if track.has_stss {
        let mut body = vec![0u8; 4];
        push_u32(&mut body, tables.sync.len() as u32);
        for &number in &tables.sync {
            push_u32(&mut body, number);
        }
        kids.push(Mp4Box::leaf(b"stss", body));
    }

    // stsc: one entry whenever (samples per chunk, description) changes.
    let mut entries: Vec<(u32, u32, u32)> = Vec::new();
    for (index, &(per_chunk, desc)) in tables.chunk_samples.iter().enumerate() {
        let changed = match entries.last() {
            Some(last) => last.1 != per_chunk || last.2 != desc,
            None => true,
        };
        if changed {
            entries.push((index as u32 + 1, per_chunk, desc));
        }
    }
    let mut stsc = vec![0u8; 4];
    push_u32(&mut stsc, entries.len() as u32);
    for &(first_chunk, per_chunk, desc) in &entries {
        push_u32(&mut stsc, first_chunk);
        push_u32(&mut stsc, per_chunk);
        push_u32(&mut stsc, desc);
    }
    kids.push(Mp4Box::leaf(b"stsc", stsc));

    let mut stsz = vec![0u8; 4];
    push_u32(&mut stsz, 0);
    push_u32(&mut stsz, tables.sizes.len() as u32);
    for &size in &tables.sizes {
        push_u32(&mut stsz, size);
    }
    kids.push(Mp4Box::leaf(b"stsz", stsz));

    let mut offsets = vec![0u8; 4];
    push_u32(&mut offsets, tables.chunk_offsets.len() as u32);
    if use_co64 {
        for &relative in &tables.chunk_offsets {
            offsets.extend_from_slice(&(base + relative).to_be_bytes());
        }
        kids.push(Mp4Box::leaf(b"co64", offsets));
    } else {
        for &relative in &tables.chunk_offsets {
            let absolute = u32::try_from(base + relative)
                .map_err(|_| BoxError::from("A chunk offset does not fit in 32 bits."))?;
            push_u32(&mut offsets, absolute);
        }
        kids.push(Mp4Box::leaf(b"stco", offsets));
    }
    Ok(kids)
}

/// Serialise a new `moov`: everything from the original is kept (movie header,
/// edit lists, codec setup, user data ...) except the per-sample tables of the
/// kept tracks, which are replaced, and tracks that were left out.
fn build_moov(
    template: &[Mp4Box],
    tracks: &[Track],
    tables: &[Tables],
    base: u64,
    use_co64: bool,
) -> AppResult<Vec<u8>> {
    let mut children: Vec<Mp4Box> = Vec::new();
    for (index, original) in template.iter().enumerate() {
        match &original.typ {
            b"trak" => {
                if let Some(ti) = tracks.iter().position(|t| t.index == index) {
                    let mut trak = original.clone();
                    let mut replacement =
                        Some(stbl_children(&tracks[ti], &tables[ti], base, use_co64)?);
                    let replaced = match &mut trak.body {
                        Body::Container(kids) => replace_stbl(kids, &mut replacement),
                        Body::Leaf(_) => false,
                    };
                    if !replaced {
                        return Err(CORRUPT.into());
                    }
                    children.push(trak);
                }
            }
            // The initial object descriptor lists track ids; drop it.
            b"iods" => {}
            _ => children.push(original.clone()),
        }
    }
    let moov = Mp4Box {
        typ: *b"moov",
        body: Body::Container(children),
    };
    let mut bytes = Vec::new();
    write_box(&mut bytes, &moov)?;
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Reading and writing files
// ---------------------------------------------------------------------------

/// Returns the `ftyp` payload (if any) and the children of `moov`.
fn read_top_level(file: &mut File, file_len: u64) -> AppResult<(Option<Vec<u8>>, Vec<Mp4Box>)> {
    let mut pos = 0u64;
    let mut ftyp: Option<Vec<u8>> = None;
    let mut moov: Option<Vec<Mp4Box>> = None;

    while pos + 8 <= file_len {
        file.seek(SeekFrom::Start(pos))?;
        let mut header = [0u8; 8];
        file.read_exact(&mut header)?;
        let size32 = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as u64;
        let typ: FourCC = [header[4], header[5], header[6], header[7]];
        let (header_len, size) = match size32 {
            0 => (8u64, file_len - pos),
            1 => {
                let mut extended = [0u8; 8];
                file.read_exact(&mut extended)?;
                (16u64, u64::from_be_bytes(extended))
            }
            n => (8u64, n),
        };
        if size < header_len {
            return Err(CORRUPT.into());
        }
        let fits = match pos.checked_add(size) {
            Some(end) => end <= file_len,
            None => false,
        };
        if !fits {
            if &typ == b"mdat" {
                break;
            }
            return Err("The file looks truncated (a header box runs past the end).".into());
        }

        match &typ {
            b"ftyp" => {
                let length = size - header_len;
                if length > (1 << 20) {
                    return Err(CORRUPT.into());
                }
                let mut payload = vec![0u8; length as usize];
                file.read_exact(&mut payload)?;
                ftyp = Some(payload);
            }
            b"moov" => {
                let length = size - header_len;
                if length > MAX_MOOV_BYTES {
                    return Err(CORRUPT.into());
                }
                let mut payload = vec![0u8; length as usize];
                file.read_exact(&mut payload)?;
                moov = Some(parse_boxes(&payload, 1)?);
            }
            b"moof" => {
                return Err("Fragmented MP4 files are not supported by this tool.".into());
            }
            _ => {}
        }
        pos += size;
    }

    match moov {
        Some(children) => Ok((ftyp, children)),
        None => Err("No movie header (moov) was found. Is this an MP4 file?".into()),
    }
}

fn default_ftyp_payload() -> Vec<u8> {
    let mut payload = b"isom".to_vec();
    push_u32(&mut payload, 512);
    for brand in [b"isom", b"iso2", b"avc1", b"mp41"] {
        payload.extend_from_slice(brand);
    }
    payload
}

/// Copy the sample data for `samples` (in order) from `input` to `out`,
/// merging samples that are adjacent in the input into single reads.
fn copy_samples<R: Read + Seek, W: Write>(
    input: &mut R,
    samples: &[Sample],
    out: &mut W,
    buffer: &mut [u8],
) -> io::Result<()> {
    let mut i = 0usize;
    while i < samples.len() {
        let start = samples[i].offset;
        let mut length = samples[i].size as u64;
        let mut j = i + 1;
        while j < samples.len() && samples[j].offset == start + length {
            length += samples[j].size as u64;
            j += 1;
        }
        input.seek(SeekFrom::Start(start))?;
        let mut remaining = length;
        while remaining > 0 {
            let take = remaining.min(buffer.len() as u64) as usize;
            input.read_exact(&mut buffer[..take])?;
            out.write_all(&buffer[..take])?;
            remaining -= take as u64;
        }
        i = j;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_output(
    part: &Path,
    ftyp: &[u8],
    moov: &[u8],
    large_mdat: bool,
    payload: u64,
    input: &mut File,
    tracks: &[Track],
    plan: &Plan,
    order: &[usize],
) -> AppResult<()> {
    let file = File::create(part)
        .map_err(|e| BoxError::from(format!("Could not create {}: {e}", part.display())))?;
    let mut out = BufWriter::with_capacity(1 << 20, file);

    out.write_all(ftyp)?;
    out.write_all(moov)?;
    if large_mdat {
        out.write_all(&1u32.to_be_bytes())?;
        out.write_all(b"mdat")?;
        out.write_all(&(payload + 16).to_be_bytes())?;
    } else {
        out.write_all(&((payload + 8) as u32).to_be_bytes())?;
        out.write_all(b"mdat")?;
    }

    let mut buffer = vec![0u8; 1 << 20];
    let mut written = 0u64;
    let mut last_decile = 0u64;
    for &seg in order {
        for (ti, track) in tracks.iter().enumerate() {
            let (s, e) = plan.ranges[ti][seg];
            let samples = &track.samples[s..e];
            copy_samples(input, samples, &mut out, &mut buffer).map_err(|err| {
                BoxError::from(format!(
                    "Could not copy the video data (is the file damaged?): {err}"
                ))
            })?;
            written += samples.iter().map(|x| x.size as u64).sum::<u64>();
        }
        let decile = if payload == 0 { 10 } else { written * 10 / payload };
        if decile > last_decile {
            last_decile = decile;
            println!("Writing: {}%", decile * 10);
        }
    }
    out.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The actual work
// ---------------------------------------------------------------------------

fn shuffle_video(
    input: &Path,
    chunk: f64,
    output: Option<&Path>,
    seed: Option<u64>,
) -> AppResult<PathBuf> {
    let input = absolutize(input)?;
    let output = match output {
        Some(path) => absolutize(path)?,
        None => input.with_file_name("1.mp4"),
    };

    if !input.is_file() {
        return Err(format!("Video not found: {}", input.display()).into());
    }
    if !chunk.is_finite() || chunk <= 0.0 {
        return Err("Chunk duration must be greater than zero.".into());
    }
    if same_file(&input, &output) {
        return Err("Input and output paths must be different.".into());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("{}", "=".repeat(60));
    println!("PURE RUST VIDEO SHUFFLER (no FFmpeg, no re-encoding)");
    println!("{}", "=".repeat(60));
    println!("Input:    {}", input.display());
    println!("Output:   {}", output.display());
    println!("Chunks:   about {chunk} seconds (cuts land on keyframes)\n");

    let mut file = File::open(&input)
        .map_err(|e| BoxError::from(format!("Could not open {}: {e}", input.display())))?;
    let file_len = file.metadata()?.len();
    let (ftyp_payload, moov_children) = read_top_level(&mut file, file_len)?;

    if find_box(&moov_children, b"mvex").is_some() {
        return Err("Fragmented MP4 files are not supported by this tool.".into());
    }

    // Keep the first video track and every audio track; leave anything else out.
    let mut video: Option<Track> = None;
    let mut audio: Vec<Track> = Vec::new();
    let mut skipped = 0usize;
    for (index, b) in moov_children.iter().enumerate() {
        if &b.typ != b"trak" {
            continue;
        }
        let kids = match &b.body {
            Body::Container(kids) => kids,
            Body::Leaf(_) => continue,
        };
        let is_video = match handler_of(kids) {
            Some(h) if &h == b"vide" => true,
            Some(h) if &h == b"soun" => false,
            _ => {
                skipped += 1;
                continue;
            }
        };
        let track = parse_track(kids, index)?;
        if track.samples.is_empty() {
            skipped += 1;
        } else if is_video {
            if video.is_none() {
                video = Some(track);
            } else {
                skipped += 1;
            }
        } else {
            audio.push(track);
        }
    }
    let video = video.ok_or("No video track was found in this file.")?;
    let mut tracks: Vec<Track> = vec![video];
    tracks.extend(audio);

    let keyframes = tracks[0].samples.iter().filter(|s| s.sync).count();
    let total_ticks = tracks[0].dts.last().copied().unwrap_or(0)
        + tracks[0].samples.last().map_or(0, |s| s.dur as u64);
    let seconds = total_ticks as f64 / tracks[0].timescale as f64;
    println!(
        "Video:    {} frames, {} keyframes, {:.2} seconds",
        tracks[0].samples.len(),
        keyframes,
        seconds
    );
    println!("Audio:    {} track(s)", tracks.len() - 1);
    if skipped > 0 {
        println!("Note:     {skipped} other track(s) (subtitles, timecode, extra video) left out");
    }

    let plan = plan_segments(&tracks, chunk);
    if plan.segments < 2 {
        return Err(format!(
            "Could not split this video into at least two chunks. Cuts can only land on \
             keyframes, and this video has {keyframes} keyframe(s) in {seconds:.1} seconds. \
             Try a smaller chunk duration, or use the FFmpeg version, which re-encodes and \
             can cut anywhere."
        )
        .into());
    }

    let order = shuffle_segments((0..plan.segments).collect::<Vec<usize>>(), seed)?;
    let order_text = order
        .iter()
        .map(|seg| format!("{seg:06}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("\nShuffling and joining {} chunks...", plan.segments);
    println!("Chunk order: {order_text}\n");

    let (tables, payload) = build_tables(&tracks, &plan, &order);

    let ftyp_payload = ftyp_payload.unwrap_or_else(default_ftyp_payload);
    let mut ftyp_box = Vec::new();
    push_u32(&mut ftyp_box, (8 + ftyp_payload.len()) as u32);
    ftyp_box.extend_from_slice(b"ftyp");
    ftyp_box.extend_from_slice(&ftyp_payload);

    // The moov goes in front of the mdat (fast start). Its size does not depend
    // on the offset values, so build it once to measure it, then for real.
    let large_mdat = payload + 8 > u32::MAX as u64;
    let mdat_header_len: u64 = if large_mdat { 16 } else { 8 };
    let probe = build_moov(&moov_children, &tracks, &tables, 0, false)?;
    let estimate = ftyp_box.len() as u64 + probe.len() as u64 + mdat_header_len + payload;
    let use_co64 = estimate > u32::MAX as u64 - (16 << 20);
    let moov_len = if use_co64 {
        build_moov(&moov_children, &tracks, &tables, 0, true)?.len()
    } else {
        probe.len()
    };
    let base = ftyp_box.len() as u64 + moov_len as u64 + mdat_header_len;
    let moov_bytes = build_moov(&moov_children, &tracks, &tables, base, use_co64)?;
    if moov_bytes.len() != moov_len {
        return Err("Internal error: the rebuilt header changed size.".into());
    }

    // Write to a temporary file next to the output, then move it into place,
    // so a failed run never leaves a half-written output file behind.
    let mut part_name = output.file_name().ok_or("Invalid output path.")?.to_os_string();
    part_name.push(".part");
    let part = output.with_file_name(part_name);

    if let Err(error) = write_output(
        &part,
        &ftyp_box,
        &moov_bytes,
        large_mdat,
        payload,
        &mut file,
        &tracks,
        &plan,
        &order,
    ) {
        let _ = fs::remove_file(&part);
        return Err(error);
    }
    drop(file);
    if let Err(error) = fs::rename(&part, &output) {
        let _ = fs::remove_file(&part);
        return Err(format!(
            "Could not move the finished file to {} (is it open in another program?): {error}",
            output.display()
        )
        .into());
    }

    println!("\nDone: {}", output.display());
    Ok(output)
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

struct Options {
    input: Option<PathBuf>,
    chunk: f64,
    output: Option<PathBuf>,
    seed: Option<u64>,
    no_pause: bool,
}

enum Cli {
    Run(Options),
    Help,
}

fn next_value(
    args: &mut impl Iterator<Item = OsString>,
    flag: &str,
) -> std::result::Result<OsString, String> {
    args.next()
        .ok_or_else(|| format!("Option {flag} needs a value."))
}

fn parse_args(args: impl Iterator<Item = OsString>) -> std::result::Result<Cli, String> {
    let mut args = args;
    let mut options = Options {
        input: None,
        chunk: 2.0,
        output: None,
        seed: None,
        no_pause: false,
    };

    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("-h") | Some("--help") => return Ok(Cli::Help),
            Some("--no-pause") => options.no_pause = true,
            Some("-c") | Some("--chunk") => {
                let value = next_value(&mut args, "--chunk")?;
                options.chunk = value
                    .to_str()
                    .and_then(|text| text.trim().parse::<f64>().ok())
                    .ok_or_else(|| {
                        format!("Invalid chunk duration: {}", value.to_string_lossy())
                    })?;
            }
            Some("-o") | Some("--output") => {
                options.output = Some(PathBuf::from(next_value(&mut args, "--output")?));
            }
            Some("-s") | Some("--seed") => {
                let value = next_value(&mut args, "--seed")?;
                options.seed = Some(
                    value
                        .to_str()
                        .and_then(|text| text.trim().parse::<u64>().ok())
                        .ok_or_else(|| format!("Invalid seed: {}", value.to_string_lossy()))?,
                );
            }
            Some(flag) if flag.starts_with('-') && flag.len() > 1 => {
                return Err(format!("Unknown option: {flag}"));
            }
            _ => {
                if options.input.is_some() {
                    return Err("Only one input video can be given.".to_string());
                }
                options.input = Some(PathBuf::from(arg.clone()));
            }
        }
    }
    Ok(Cli::Run(options))
}

fn pause() {
    print!("\nPress Enter to exit...");
    let _ = io::stdout().flush();
    let mut line = String::new();
    let _ = io::stdin().read_line(&mut line);
}

fn main() {
    let options = match parse_args(env::args_os().skip(1)) {
        Ok(Cli::Run(options)) => options,
        Ok(Cli::Help) => {
            println!("{USAGE}");
            return;
        }
        Err(message) => {
            eprintln!("ERROR: {message}\n\n{USAGE}");
            process::exit(2);
        }
    };

    // The original script always waited for Enter on Windows so a
    // double-clicked console window stayed open; keep that behaviour.
    let pause_on_exit = cfg!(windows) && !options.no_pause;
    let input = options.input.unwrap_or_else(default_input);

    let code = match shuffle_video(&input, options.chunk, options.output.as_deref(), options.seed)
    {
        Ok(_) => 0,
        Err(error) => {
            eprintln!("\nERROR: {error}");
            1
        }
    };

    if pause_on_exit {
        pause();
    }
    process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_box(typ: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 8 + body.len() as u32);
        bytes.extend_from_slice(typ);
        bytes.extend_from_slice(body);
        bytes
    }

    fn audio_track(timescale: u32, dts: Vec<u64>) -> Track {
        Track {
            index: 0,
            timescale,
            stsd: Vec::new(),
            has_ctts: false,
            ctts_header: [0; 4],
            has_stss: false,
            samples: Vec::new(),
            dts,
        }
    }

    #[test]
    fn no_item_stays_in_place() {
        for seed in 0..200u64 {
            let original: Vec<usize> = (0..8).collect();
            let shuffled = shuffle_segments(original.clone(), Some(seed)).unwrap();
            assert_eq!(shuffled.len(), original.len());
            assert!(shuffled.iter().zip(&original).all(|(a, b)| a != b));
        }
    }

    #[test]
    fn same_seed_gives_same_order() {
        let a = shuffle_segments((0..20).collect::<Vec<u32>>(), Some(42)).unwrap();
        let b = shuffle_segments((0..20).collect::<Vec<u32>>(), Some(42)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_fewer_than_two_items() {
        assert!(shuffle_segments(vec![1], None).is_err());
        assert!(shuffle_segments(Vec::<i32>::new(), None).is_err());
    }

    #[test]
    fn box_tree_round_trips() {
        let inner = raw_box(b"free", &[1, 2, 3]);
        let trak = raw_box(b"trak", &inner);
        let boxes = parse_boxes(&trak, 0).unwrap();
        let mut rewritten = Vec::new();
        for b in &boxes {
            write_box(&mut rewritten, b).unwrap();
        }
        assert_eq!(rewritten, trak);
    }

    #[test]
    fn truncated_box_is_rejected() {
        let mut data = raw_box(b"free", &[1, 2, 3, 4]);
        data.truncate(data.len() - 1);
        assert!(parse_boxes(&data, 0).is_err());
    }

    #[test]
    fn run_lengths_merge_equal_neighbours() {
        assert_eq!(
            run_lengths(&[1, 1, 2, 2, 2, 3]),
            vec![(2, 1), (3, 2), (1, 3)]
        );
        assert!(run_lengths(&[]).is_empty());
    }

    #[test]
    fn cut_index_picks_the_nearest_sample() {
        let track = audio_track(1000, vec![0, 100, 200, 300]);
        assert_eq!(cut_index(&track, 1000, 140), 1);
        assert_eq!(cut_index(&track, 1000, 160), 2);
        assert_eq!(cut_index(&track, 1000, 0), 0);
        assert_eq!(cut_index(&track, 1000, 10_000), 4);
    }
}
