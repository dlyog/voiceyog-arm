# kheledi — can Arm KleidiAI make this model faster?

**No. And finding out produced something more useful than a yes would have.**

This directory is self-contained. It has its own virtualenvs, its own frozen
inputs, and its own evidence. It imports nothing from the rest of the
repository, and `rm -rf kheledi/` leaves the project exactly as it was.

```bash
bash setup.sh 1.20.1                      # isolated venv, the pinned runtime
./.venv-1.20.1/bin/python3 1_platform.py  # what this Arm part can execute
./.venv-1.20.1/bin/python3 2_bench.py --pin
./.venv-1.20.1/bin/python3 3_compare.py   # every machine, side by side
```

---

## The answer

KleidiAI is not reachable on either machine this project ships for, and the
reason is one line of ONNX Runtime. Every KleidiAI kernel — fp32 GEMM, fp16,
bf16, and convolution — is installed only inside this guard in
`onnxruntime/core/mlas/lib/platform.cpp`:

```cpp
if (MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME() ||
    MLAS_CPUIDINFO::GetCPUIDInfo().HasArm_SME2())
```

Neither target has SME:

| | DGX Spark GB10 | Apple M1 Max |
|---|---|---|
| CPU | Cortex-X925 + Cortex-A725 | Apple M1 Max |
| SME / SME2 | **no / no** | **no / no** |
| I8MM · BF16 · SVE2 | yes · yes · yes | no · no · no |
| dotprod | yes | yes |
| KleidiAI reachable | **no** | **no** |

Reading a gate is not the same as testing one, so KleidiAI was installed from
Arm's repository on both machines and its kernels called directly, with ONNX
Runtime out of the picture:

| kernel | DGX Spark | M1 Max |
|---|---|---|
| `..._neon_dotprod` (INT4) | returns normally | returns normally |
| `..._sme_mopa` (FP32) | **SIGILL** | **SIGILL** |

Same library, same compiler, same binary — so the CPU is the only variable. It
dies inside `kai_get_m_step_...`, which reads the streaming vector length,
before touching a single matrix element. **The gate in ONNX Runtime is not a
limitation. It is what stops the server crashing.**

It would also have been aimed at the wrong operator. Profiling this model puts
about **70%** of kernel time in convolution (68.6-71.1% across four pinned
runs) and under **1%** in matmul (0.78-0.85%), and its 26
MatMuls are activation×activation, so there is no constant weight matrix to
prepack even in principle.

---

## What the benchmark found instead

DGX Spark GB10, 9 threads pinned to the performance cores, 1,200 timed
inferences each, identical frozen inputs, machine idle and verified idle.

| metric | ORT 1.20.1 | ORT 1.28.0 | change |
|---|---:|---:|---:|
| p50 latency | 59.204 ms | **56.141 ms** | −5.2% |
| p95 latency | 66.688 ms | **63.115 ms** | −5.4% |
| p99 latency | 68.182 ms | 63.878 ms | −6.3% |
| throughput | 17.102 inf/s | **17.965 inf/s** | +5.0% |
| RTF | 0.029997 | 0.028555 | −4.8% |
| peak RSS | 200.1 MB | 208.6 MB | +4.2% |
| KleidiAI kernels linked | 0 | 13 | — |

The 13 KleidiAI kernels in the 1.28.0 build contribute **nothing** — they are
never dispatched. The gain is eight releases of general ONNX Runtime
improvement, and it costs 8 MB of resident memory.

### The finding that matters more than the 5%

**ONNX Runtime 1.28.0 is bimodal when it is not pinned, and 1.20.1 is not.**

| | unpinned | pinned to P-cores | pinned to E-cores |
|---|---:|---:|---:|
| ORT 1.20.1 | 60.6 ms | 60.8 ms | 142.9 ms |
| ORT 1.28.0 | **120.4 ms** | **59.6 ms** | 136.7 ms |

Across every run taken, 1.20.1 unpinned stayed between 59 and 61 ms. 1.28.0
unpinned was usually 58–60 ms but landed at ~120 ms on several occasions — a
**2× regression on an idle machine with nothing else running**. Pinned to the
performance cores it is stable at 59.6 ms.

The slow mode is thread placement: the scheduler puts some of the intra-op
pool on Cortex-A725 efficiency cores, and because the intra-op join is a
barrier, every operator then waits on the slowest thread. On this asymmetric
part that is worth 2×.

**So the upgrade is not free.** Taken unpinned it is a coin flip between 5%
faster and 2× slower. Taken with the performance cores pinned it is a
reliable 5%. That is the recommendation: **upgrade only together with
pinning.**

I did not fully isolate what tips 1.28.0 into the slow mode — it appeared
after deep idle and after heavy I/O, and not during back-to-back runs. The
mitigation is measured and reliable; the trigger is not characterised, and I
am not claiming otherwise.

---

## Files

```
1_platform.py    Arm topology and ISA, identically on Linux and macOS
2_bench.py       the benchmark; --pin applies the finding above
3_compare.py     every bench_*.json in one table
setup.sh         an isolated venv per runtime version
inputs/          the frozen fixture, sha256 771d0623096affc7…
evidence/        results, one JSON per run
```

### Why the inputs are frozen rather than generated

The DGX Spark has espeak-ng 1.51 and the Mac has 1.52, and the two disagree on
liaison and function-word reduction. Phonemizing locally on each machine would
have given the two platforms different phoneme sequences and silently
invalidated every comparison. So espeak runs once, and both machines load the
same bytes.

`noise_scale` and `noise_w` are zero throughout. The graph has two
`RandomNormalLike` nodes, and through the stochastic duration predictor they
change how many samples come out — leaving them on would put the model's own
variance directly on top of the effect being measured.

### The validity flag

Every result records the load average and marks itself usable or not. This is
not decoration: the first Mac run was taken while `rustc` was using two cores,
and reported 177 ms p50 with a p99 of 428 ms — numbers that describe a busy
laptop, not the model.

Its known limitation is that it cannot tell foreign load from this
benchmark's own: running two benchmarks back to back trips it, because a
9-thread run at 900% CPU is still in the 1-minute average when the next one
starts. Cool down between runs.

---

## Apple Silicon

The code runs correctly on macOS — same script, correct topology (8 performance
+ 2 efficiency cores, 8 threads derived), full 1,200 inferences. **No usable
timing was captured**, because the machine was compiling Rust throughout and
the run is flagged `CONTENDED`. The DGX Spark is the reporting target here; the
Mac was checked for portability only.

One portability note found on the way: `onnxruntime==1.20.1` has no wheel for
Python 3.13 or newer, so on a machine whose `python3` is 3.14 the pinned
version cannot be installed at all and pip reports the oldest available as
1.24.1. `setup.sh` prefers Python 3.12 for that reason.

---

## Verdict

| | |
|---|---|
| Can this model use KleidiAI? | **No** — no SME on either target; SIGILL when forced |
| Would it help if it could? | **No** — it targets under 1% of this model's CPU time |
| Is it in the shipped package? | **No** — the packages pin ORT 1.20.1, which has 0 KleidiAI symbols |
| What should change instead? | Pin to performance cores. Then, optionally, ORT 1.28.0 for 5% |
| What is the real remaining lever? | INT8 on the convolutions — about 70% of the time is there |
