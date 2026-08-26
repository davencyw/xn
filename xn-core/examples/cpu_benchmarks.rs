use clap::{Parser, Subcommand};
use rayon::prelude::*;
use xn::Result;
use xn::quantized::GgmlType;
use xn::quantized::k_quants::{BlockQ8_0, QK8_0};

type Tensor = xn::Tensor<f32, xn::CpuDevice>;

trait Benchmark {
    type PreProcessData;
    type RunResult;

    fn preprocess() -> Result<Self::PreProcessData>;
    fn run_one(_: &Self::PreProcessData) -> Result<Self::RunResult>;

    const ITERS: usize;
}

struct MatMul;
impl Benchmark for MatMul {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = Tensor::zeros((125, 4096), &xn::CPU)?;
        let rhs = Tensor::zeros((4096, 1024), &xn::CPU)?;
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.matmul(&d.1)
    }

    const ITERS: usize = 5;
}

// Shared dimensions for the q8_0 matmul benchmarks. `K` must be a multiple of
// QK8_0 (32) since q8_0 packs 32 elements per block.
const QM: usize = 125;
const QK: usize = 4096;
const QN: usize = 1024;

// Number of distinct weight matrices to rotate through. With QN=1024,
// QK=4096, q8_0, each weight is ~4.25 MiB; 24 of them (~102 MiB) is well
// past typical L3 caches so each iteration pays the cost of streaming the
// weight from RAM, matching real LLM inference behaviour.
const Q_WEIGHTS: usize = 24;

// Pre-quantized q8_0 lhs (m × k_blocks) and `Q_WEIGHTS` q8_0 rhs matrices
// (n × k_blocks each). Both `QMatMul` and `QMatMulSgemm` reuse this so the
// f32→q8_0 conversion is not timed. The atomic counter cycles through the
// weight matrices to defeat L3 caching.
struct QData {
    lhs: Vec<BlockQ8_0>,
    rhs: Vec<Vec<BlockQ8_0>>,
    counter: std::sync::atomic::AtomicUsize,
}

fn q_preprocess() -> Result<QData> {
    let k_blocks = QK / QK8_0;
    let lhs = vec![BlockQ8_0::zeros(); QM * k_blocks];
    let rhs = (0..Q_WEIGHTS).map(|_| vec![BlockQ8_0::zeros(); QN * k_blocks]).collect();
    Ok(QData { lhs, rhs, counter: std::sync::atomic::AtomicUsize::new(0) })
}

fn q_pick_rhs(d: &QData) -> &[BlockQ8_0] {
    let idx = d.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % d.rhs.len();
    &d.rhs[idx]
}

// Existing per-row mul-vec path: for each output row, parallelise over
// columns and call `vec_dot` (matches the inner loop of `k_quants::matmul`,
// minus the lhs quantization step that we pre-do in `q_preprocess`).
struct QMatMul;
impl Benchmark for QMatMul {
    type PreProcessData = QData;
    type RunResult = Vec<f32>;
    fn preprocess() -> Result<Self::PreProcessData> {
        q_preprocess()
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        let k_blocks = QK / QK8_0;
        let rhs = q_pick_rhs(d);
        let mut dst = vec![0f32; QM * QN];
        for row_idx in 0..QM {
            let lhs_row = &d.lhs[row_idx * k_blocks..(row_idx + 1) * k_blocks];
            let dst_row = &mut dst[row_idx * QN..(row_idx + 1) * QN];
            let result: Result<Vec<_>> = dst_row
                .into_par_iter()
                .enumerate()
                .with_min_len(128)
                .with_max_len(512)
                .map(|(col_idx, dst)| {
                    let rhs_col = &rhs[col_idx * k_blocks..(col_idx + 1) * k_blocks];
                    BlockQ8_0::vec_dot(QK, rhs_col, lhs_row).map(|value| *dst = value)
                })
                .collect();
            result?;
        }
        Ok(dst)
    }

    const ITERS: usize = Q_WEIGHTS;
}

// New blocked sgemm path: q8_0 × q8_0 → f32. Dispatches at compile time to
// `neon::sgemm_q8_0_q8_0` on aarch64 and `avx::sgemm_q8_0_q8_0` on x86. Both
// kernels are single-threaded — the existing matmul uses rayon over output
// columns, so expect the gap to shrink on multi-core machines.
#[cfg(any(target_feature = "neon", target_feature = "avx"))]
struct QMatMulSgemm;
#[cfg(any(target_feature = "neon", target_feature = "avx"))]
impl Benchmark for QMatMulSgemm {
    type PreProcessData = QData;
    type RunResult = Vec<f32>;
    fn preprocess() -> Result<Self::PreProcessData> {
        q_preprocess()
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        let k_blocks = QK / QK8_0;
        let rhs = q_pick_rhs(d);
        // sgemm output is column-major with stride `ldc`; here we use ldc = QM
        // so the buffer is tightly packed.
        let mut dst = vec![0f32; QM * QN];
        #[cfg(target_feature = "neon")]
        xn::quantized::neon::sgemm_q8_0_q8_0(
            QM, QN, k_blocks, &d.lhs, k_blocks, rhs, k_blocks, &mut dst, QM, 0, 1,
        )?;
        #[cfg(all(target_feature = "avx", not(target_feature = "neon")))]
        xn::quantized::avx::sgemm_q8_0_q8_0(
            QM, QN, k_blocks, &d.lhs, k_blocks, rhs, k_blocks, &mut dst, QM, 0, 1,
        )?;
        Ok(dst)
    }

    const ITERS: usize = Q_WEIGHTS;
}

// ---------------------------------------------------------------------------
// qmatmul-sgemm shape sweep
// ---------------------------------------------------------------------------

// (k, m, n) triples. m = 1 is the single-token decode case, where the sgemm
// kernel degenerates to a mat-vec and the fixed per-call overheads (rayon
// fan-out, tile setup) are a large share of the runtime.
const SGEMM_SHAPES: &[(usize, usize, usize)] = &[(3072, 1, 768), (768, 1, 3072), (768, 1, 768)];

// Rotate through enough distinct weight matrices to cover this many bytes, so
// that each iteration streams its weights from RAM instead of replaying them
// out of L2/SLC. These shapes are small (0.6-2.4 MiB of q8_0 weights), so
// without the rotation the benchmark would measure a cache-resident best case
// that never happens during real decoding.
const SGEMM_WEIGHT_BYTES: usize = 128 << 20;
const SGEMM_MIN_WEIGHTS: usize = 8;
const SGEMM_MAX_WEIGHTS: usize = 96;

// Pre-quantized operands for one (k, m, n) shape: a single q8_0 activation
// matrix (m x k_blocks) and a rotating pool of q8_0 weights (n x k_blocks
// each, row-major over n, i.e. already transposed as `rhs_t`).
struct ShapeData {
    m: usize,
    n: usize,
    k: usize,
    k_blocks: usize,
    lhs: Vec<BlockQ8_0>,
    rhs: Vec<Vec<BlockQ8_0>>,
    counter: std::sync::atomic::AtomicUsize,
}

impl ShapeData {
    fn pick_rhs(&self) -> &[BlockQ8_0] {
        let idx = self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.rhs.len();
        &self.rhs[idx]
    }

    // Bytes touched per call: weights (streamed) + activations + output.
    fn bytes_per_call(&self) -> usize {
        let block = std::mem::size_of::<BlockQ8_0>();
        (self.n + self.m) * self.k_blocks * block + self.m * self.n * 4
    }

    fn flops_per_call(&self) -> usize {
        2 * self.m * self.n * self.k
    }
}

// Deterministic pseudo-random values in [-1, 1), quantized to q8_0. Zeroed
// blocks would give every block a scale of 0, which is not representative of
// real weights (and would hide any denormal/branchy behaviour in the scale
// handling).
fn rand_q8_0(n_blocks: usize, seed: &mut u64) -> Result<Vec<BlockQ8_0>> {
    let values: Vec<f32> = (0..n_blocks * QK8_0)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*seed >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect();
    let mut blocks = vec![BlockQ8_0::zeros(); n_blocks];
    BlockQ8_0::from_float(&values, &mut blocks)?;
    Ok(blocks)
}

fn shape_data(k: usize, m: usize, n: usize) -> Result<ShapeData> {
    if !k.is_multiple_of(QK8_0) {
        xn::bail!("k {k} is not a multiple of {QK8_0}")
    }
    let k_blocks = k / QK8_0;
    let weight_bytes = n * k_blocks * std::mem::size_of::<BlockQ8_0>();
    let n_weights =
        (SGEMM_WEIGHT_BYTES / weight_bytes.max(1)).clamp(SGEMM_MIN_WEIGHTS, SGEMM_MAX_WEIGHTS);
    let mut seed = 0x5eed_1234_9876_4321u64 ^ ((k as u64) << 32) ^ ((n as u64) << 8) ^ m as u64;
    let lhs = rand_q8_0(m * k_blocks, &mut seed)?;
    let rhs = (0..n_weights).map(|_| rand_q8_0(n * k_blocks, &mut seed)).collect::<Result<_>>()?;
    Ok(ShapeData { m, n, k, k_blocks, lhs, rhs, counter: std::sync::atomic::AtomicUsize::new(0) })
}

// The dispatch used in production: `BlockQ8_0::matmul` routes k % 32 == 0
// through `matmul_q8_0_sgemm`, which fans the sgemm tiles out over rayon.
fn sgemm_mt(d: &ShapeData, rhs: &[BlockQ8_0], dst: &mut [f32]) -> Result<()> {
    BlockQ8_0::matmul(&d.lhs, rhs, dst, d.m, d.n, d.k)
}

// Single-threaded sgemm, to separate kernel cost from rayon fan-out cost.
// Roles are swapped (weights as A, activations as B, `(m', n') = (n, m)`,
// ldc = n) so the column-major sgemm output lands as row-major `dst[i*n+j]`,
// matching what `matmul_q8_0_sgemm` does.
#[cfg(any(target_feature = "neon", target_feature = "avx", target_feature = "simd128"))]
fn sgemm_st(d: &ShapeData, rhs: &[BlockQ8_0], dst: &mut [f32]) -> Result<()> {
    #[cfg(target_feature = "avx")]
    return xn::quantized::avx::sgemm_q8_0_q8_0(
        d.n, d.m, d.k_blocks, rhs, d.k_blocks, &d.lhs, d.k_blocks, dst, d.n, 0, 1,
    );
    #[cfg(all(target_feature = "neon", not(target_feature = "avx")))]
    return xn::quantized::neon::sgemm_q8_0_q8_0(
        d.n, d.m, d.k_blocks, rhs, d.k_blocks, &d.lhs, d.k_blocks, dst, d.n, 0, 1,
    );
    #[cfg(all(
        target_feature = "simd128",
        not(target_feature = "avx"),
        not(target_feature = "neon")
    ))]
    return xn::quantized::simd128::sgemm_q8_0_q8_0(
        d.n, d.m, d.k_blocks, rhs, d.k_blocks, &d.lhs, d.k_blocks, dst, d.n, 0, 1,
    );
}

// Baseline: the pre-sgemm path, one `vec_dot` per output column with rayon
// over the columns (mirrors `k_quants::matmul_by_row`).
fn by_row(d: &ShapeData, rhs: &[BlockQ8_0], dst: &mut [f32]) -> Result<()> {
    for row_idx in 0..d.m {
        let lhs_row = &d.lhs[row_idx * d.k_blocks..(row_idx + 1) * d.k_blocks];
        let dst_row = &mut dst[row_idx * d.n..(row_idx + 1) * d.n];
        let result: Result<Vec<_>> = dst_row
            .into_par_iter()
            .enumerate()
            .with_min_len(128)
            .with_max_len(512)
            .map(|(col_idx, dst)| {
                let rhs_col = &rhs[col_idx * d.k_blocks..(col_idx + 1) * d.k_blocks];
                BlockQ8_0::vec_dot(d.k, rhs_col, lhs_row).map(|value| *dst = value)
            })
            .collect();
        result?;
    }
    Ok(())
}

type ShapeVariant = (&'static str, fn(&ShapeData, &[BlockQ8_0], &mut [f32]) -> Result<()>);

const SGEMM_VARIANTS: &[ShapeVariant] = &[
    ("sgemm", sgemm_mt),
    #[cfg(any(target_feature = "neon", target_feature = "avx", target_feature = "simd128"))]
    ("sgemm-1t", sgemm_st),
    ("by-row", by_row),
];

// All variants compute the same product, so a disagreement means the
// benchmark is timing something wrong. q8_0 inputs are identical across
// variants, only the summation order differs, hence the loose tolerance.
fn check_variants(d: &ShapeData) -> Result<()> {
    let rhs = &d.rhs[0];
    let mut reference = vec![0f32; d.m * d.n];
    by_row(d, rhs, &mut reference)?;
    let scale = reference.iter().fold(0f32, |acc, v| acc.max(v.abs())).max(1e-6);
    for (name, f) in SGEMM_VARIANTS.iter() {
        let mut dst = vec![0f32; d.m * d.n];
        f(d, rhs, &mut dst)?;
        let max_diff =
            dst.iter().zip(reference.iter()).fold(0f32, |acc, (a, b)| acc.max((a - b).abs()));
        if max_diff > 1e-3 * scale {
            xn::bail!(
                "variant {name} disagrees with the by-row reference: max diff {max_diff:e} \
                 (scale {scale:e})"
            )
        }
    }
    Ok(())
}

fn run_shape(k: usize, m: usize, n: usize, iters: usize) -> Result<()> {
    use std::hint::black_box;

    let d = shape_data(k, m, n)?;
    check_variants(&d)?;
    let weight_mib = (d.n * d.k_blocks * std::mem::size_of::<BlockQ8_0>()) as f64 / (1024. * 1024.);
    println!(
        "q8_0 k={k} m={m} n={n} | weights {:.2} MiB x {} | {} rayon threads | {iters} iters",
        weight_mib,
        d.rhs.len(),
        rayon::current_num_threads(),
    );
    for (name, f) in SGEMM_VARIANTS.iter() {
        let mut dst = vec![0f32; d.m * d.n];
        // Warm up (and let rayon spin its workers up) before timing.
        for _ in 0..(iters / 10).clamp(2, 50) {
            f(&d, black_box(d.pick_rhs()), &mut dst)?;
            black_box(&dst);
        }
        let mut best = std::time::Duration::MAX;
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let iter_start = std::time::Instant::now();
            f(&d, black_box(d.pick_rhs()), &mut dst)?;
            black_box(&dst);
            best = best.min(iter_start.elapsed());
        }
        let mean = start.elapsed() / iters as u32;
        let secs = mean.as_secs_f64();
        println!(
            "  {name:<9} {:>10.3?} mean {:>10.3?} best {:>8.1} GFLOP/s {:>8.1} GiB/s",
            mean,
            best,
            d.flops_per_call() as f64 / secs / 1e9,
            d.bytes_per_call() as f64 / secs / (1024. * 1024. * 1024.),
        );
    }
    Ok(())
}

// `--shape k,m,n`, e.g. `--shape 3072,1,768`.
fn parse_shape(s: &str) -> Result<(usize, usize, usize)> {
    let parts: Vec<&str> = s.split(',').collect();
    let [k, m, n] = parts.as_slice() else {
        xn::bail!("expected a shape of the form k,m,n, got {s:?}")
    };
    let parse = |v: &str, name: &str| match v.trim().parse::<usize>() {
        Ok(v) => Ok(v),
        Err(_) => xn::bail!("cannot parse {name}={v:?} in shape {s:?}"),
    };
    Ok((parse(k, "k")?, parse(m, "m")?, parse(n, "n")?))
}

struct MatVec;
impl Benchmark for MatVec {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = Tensor::zeros((1024 * 4, 1024 * 4), &xn::CPU)?;
        let rhs = Tensor::zeros((1024 * 4, 1), &xn::CPU)?;
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.matmul(&d.1)
    }

    const ITERS: usize = 100;
}

fn run<B: Benchmark>(iters: Option<usize>) -> Result<()> {
    use std::hint::black_box;

    let iters = iters.unwrap_or(B::ITERS);
    let d = B::preprocess()?;
    let start = std::time::Instant::now();
    for _iter in 0..iters {
        let _res = black_box(B::run_one(black_box(&d))?);
    }
    println!("{:?}", start.elapsed() / iters as u32);
    Ok(())
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Matmul,
    Matvec,
    Qmatmul,
    #[cfg(any(target_feature = "neon", target_feature = "avx"))]
    QmatmulSgemm,
    /// q8_0 matmul across a set of (k, m, n) shapes, comparing the rayon sgemm
    /// path, the single-threaded sgemm kernel and the per-row vec_dot path.
    QmatmulSgemmShapes {
        /// Shapes to benchmark, as `k,m,n`. Repeatable; defaults to the
        /// single-token decode shapes in `SGEMM_SHAPES`.
        #[arg(long = "shape")]
        shapes: Vec<String>,
    },
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The benchmark to be run.
    #[command(subcommand)]
    task: Task,

    #[arg(long, global = true)]
    iters: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.task {
        Task::Matmul => {
            for _ in 0..20 {
                run::<MatMul>(args.iters)?
            }
        }
        Task::Matvec => {
            for _ in 0..20 {
                run::<MatVec>(args.iters)?
            }
        }
        Task::Qmatmul => {
            for _ in 0..20 {
                run::<QMatMul>(args.iters)?
            }
        }
        #[cfg(any(target_feature = "neon", target_feature = "avx"))]
        Task::QmatmulSgemm => {
            for _ in 0..20 {
                run::<QMatMulSgemm>(args.iters)?
            }
        }
        Task::QmatmulSgemmShapes { shapes } => {
            let shapes = if shapes.is_empty() {
                SGEMM_SHAPES.to_vec()
            } else {
                shapes.iter().map(|s| parse_shape(s)).collect::<Result<Vec<_>>>()?
            };
            let iters = args.iters.unwrap_or(2000);
            for (k, m, n) in shapes {
                run_shape(k, m, n, iters)?
            }
        }
    }
    Ok(())
}
