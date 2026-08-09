# Evidence

Every number in this submission, in the file that produced it.

```bash
python3 verify_claims.py          # 33 claims, no dependencies, no model, no network
python3 verify_claims.py --show   # each one with its measured value and source
```

Exit 0 means every figure in `README.md` and `SUBMISSION.md` is backed by a
measurement here. Exit 1 names the ones that are not.

| file | what produced it |
|---|---|
| `benchmark_of_record_dgx_spark.json` | `5_BenchmarkOnDGX/benchmark.py` — four engines, each in its own subprocess, on an idle GPU |
| `thread_sweep_dgx_spark.json` | `2_arm_optimization/2_thread_sweep.py` on GB10 |
| `thread_sweep_m1_max.json` | the same script on an Apple M1 Max |
| `int8_negative_result_dgx_spark.json` | `3_int8_negative_result.py` on GB10 |
| `int8_negative_result_m1_max.json` | the same script on M1 Max |
| `core_topology_dgx_spark.json` | `1_core_topology.py --json`, read from `MIDR_EL1` and `cpu_capacity` |
| `core_topology_m1_max.json` | the same script, read from `hw.perflevel*` |
| `fresh_install_dgx_spark.txt` | a clean unzip and offline install, verbatim |
| `fresh_install_apple_silicon.txt` | the same, on the Mac |
| `kleidiai_symbol_scan.json` | symbol counts for KleidiAI in the shipped runtime and in a newer one |

All measurements are on **kokoro-heart-new v3**, `sha256 63ec62a3…`, which is
the model inside both downloadable packages. The benchmark records its own
checkpoint path and the GPU utilisation at start, so you can tell whether the
run was clean without taking my word for it.

## Why this directory exists

An earlier version of this project's documentation claimed *"3× faster than
Kokoro-GPU"*. Re-measured with both engines in one run on an idle GPU, it was
**1.77×**. The original figure had been taken while the GPU was busy, which
made the baseline look slow.

Nobody lied. A number was carried by hand out of its context and stopped being
true — which is the normal way documentation goes wrong, and it is invisible
until someone checks. So no figure here is typed by a person: a script writes
it into JSON, the writeup quotes it, and `verify_claims.py` fails the moment
the two disagree.

The same discipline is why this submission reports that INT8 does not work here
and that the cooperative CPU→GPU path is **0.72× Kokoro-GPU — slower**. Those
results are as measured as the good ones.

## Methodology notes worth knowing before you read the numbers

**Each engine runs in its own subprocess.** Peak RSS is meaningless if PyTorch,
ONNX Runtime and Kokoro are all resident in one process — whichever loaded
first would be charged for the others.

**The GPU must be idle, and the script refuses if it is not.** Measured
directly: with a training job resident, Kokoro-GPU reported RTF 0.03787 against
0.01438 idle — a 2.6× inflation. Comparing a contended baseline against an idle
candidate is how a benchmark lies.

**`ru_maxrss` is kilobytes on Linux and bytes on macOS.** Resolved from
`sys.platform` rather than assumed; getting it wrong is a silent 1024× error.

**Sweep results are medians, not means.** One scheduler hiccup skews a mean of
twenty samples and would make a rerun disagree with this one for no reason.

**The topology's thread recommendation is recorded before the sweep runs**
(`topology_recommendation` in each sweep file), so "the prediction matched the
optimum" cannot be back-fitted.

## The figure this directory does not cover

WER/CER against the teacher. Re-running it needs the Kokoro teacher and a
speech-recognition model, so it is not reproducible from these packages alone
and `verify_claims.py` does not check it. It is flagged as such wherever it
appears rather than sheltering under "33 claims verified".
