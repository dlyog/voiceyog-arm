# The optimization, as things you can run

Four items. Each is standalone, each prints a result for **your** hardware, and
each writes JSON that [`../3_evidence/verify_claims.py`](../3_evidence/verify_claims.py)
checks the writeup against.

| script | needs | what it answers |
|---|---|---|
| `1_core_topology.py` | Python 3 only | What are my cores, really? |
| `2_thread_sweep.py` | an installed bundle | How many threads should ONNX Runtime get, and what does the wrong answer cost? |
| `3_int8_negative_result.py` | + `pip install onnx` | Should I quantize? |
| `4_cooperative_pipeline/` | DGX Spark, torch + CUDA | Can the Arm CPU and the GPU split one utterance? |
| `5_profile_with_performix.py` | Arm Performix CLI | Where does the Arm CPU time actually go? |

```bash
B=../1_packages/voiceyog-local-tts-kokoro-heart-new-apple-silicon-1.0.0

python3 1_core_topology.py
$B/.venv/bin/python3 2_thread_sweep.py --bundle $B --json sweep.json
$B/.venv/bin/pip install onnx
$B/.venv/bin/python3 3_int8_negative_result.py --bundle $B
```

Run anything that loads the model with the **bundle's** interpreter — that is
where onnxruntime lives. `1_core_topology.py` reads `/sys` and `sysctl` and
needs nothing at all.

---

## 1 — Read the silicon, do not guess it

Both targets are asymmetric, and the asymmetry is the whole story:

```
DGX Spark GB10   10x Cortex-X925 @ 3.90 GHz  +  10x Cortex-A725 @ 2.81 GHz
Apple M1 Max      8 performance cores        +   2 efficiency cores
```

On Linux this comes from `MIDR_EL1` — an Arm architectural register, read per
core, whose bits [15:4] name the core design — and from `cpu_capacity`, the
scheduler's own relative-performance number for big.LITTLE-style layouts. On
macOS, from `hw.perflevel0.logicalcpu`. Not from a core count, and not from a
model-name lookup table.

**The finding that changed the code:** on GB10 the performance cores are not
contiguous. They are `5-9,15-19`, interleaved across two clusters, so pinning
to a *range* would put half the threads on the wrong cluster while looking like
an optimization.

An earlier version of this code counted cores per MIDR and called the largest
group "performance". With exactly ten of each, `max()` picked arbitrarily and
was right only by luck. Capacity decides, not population.

---

## 2 — The thread sweep, which is the actual optimization

A standard technique — set `intra_op_num_threads` — with a hardware-specific
value derived at runtime. That combination is the point: **one ONNX graph, one
binary, retuned per Arm part with no recompilation and no separate build.**

Measured on the shipped model over the 20 held-out eval sentences, median:

| threads | DGX Spark GB10 | Apple M1 Max |
|---|---|---|
| ONNX Runtime's default | 136.02 ms | 154.33 ms |
| 1 | 350.01 ms | 524.20 ms |
| 8 | 84.97 ms | **128.47 ms** ← best |
| 9 | **82.49 ms** ← best | 228.57 ms |
| 10 | 84.47 ms | 283.56 ms |
| all cores (20 / 10) | 114.37 ms | 283.56 ms |

Read the last row against the best one. **Handing ONNX Runtime every core is
1.39× slower on DGX Spark and 2.21× slower on an M1 Max** than handing it the
right subset.

Why an operator gets slower when you add cores to it: ONNX Runtime splits the
operator across intra-op threads and joins. The join is a barrier, so the
operator finishes when its slowest thread finishes. A thread scheduled onto a
core 28% slower holds up every other thread in that operator, on every
operator, for the whole graph.

**On both machines the topology's prediction was the measured optimum** — 9 on
GB10, 8 on M1 Max — and the prediction is recorded in the JSON
(`topology_recommendation`) before the sweep runs, so it cannot be back-fitted.

### How the threads actually land

I choose the count; I do not place the threads. ONNX Runtime sets no affinity
here and Linux does the placing. The reason the count matters is arithmetic:
there are 10 performance cores, so asking for 20 runnable threads puts **at
least 10 of them on efficiency cores by pigeonhole**, no matter how good the
scheduler is. Ask for 9 and none is forced there. You are not steering
placement — you are removing the guarantee of bad placement.

One below the performance-core count, rather than exactly it, leaves a fast
core for the OS and for the `espeak-ng` phonemizer subprocess, which runs per
utterance and would otherwise preempt an inference thread.

### This was not a theoretical exercise

Building this submission is how I found that the shipped DGX package ran at
**20 threads**: `tuned_threads()` returned `os.cpu_count()` on Linux, under a
docstring asserting "on DGX OS every core is the same". That is false, and it
cost 1.39×. Fixed, and verified on the box:

```
before   "threads": 20    RTF 0.0528
after    "threads":  9    RTF 0.0339
```

An optimization is only real once something measures it.

---

## 3 — The one I did not ship

INT8 dynamic quantization is the reflex answer for Arm, and on this graph it
does not merely lose — **the quantized model will not load at all**, on either
target, at the same node:

```
float32   71.8M    80.29 ms      (DGX Spark, 9 threads)
int8      21.5M    does not run

NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
                  node '/enc_p/encoder/attn_layers.0/conv_q/Conv_quant'

root cause, read off the quantized graph:
    183x  ConvInteger
    162x  DynamicQuantizeLinear
```

`quantize_dynamic` rewrote all 183 convolutions into `ConvInteger`, and ONNX
Runtime's CPU provider has no `ConvInteger` kernel for this graph on aarch64.

This is not "INT8 is bad on Arm". It is: INT8 is a **trade**, its payoff is a
property of the model and the target rather than of the technique, and a
one-line change that makes the file 3.3× smaller produced an artifact that runs
nowhere. Shipping it on the strength of the file size — without loading it once
— is the failure this script exists to make visible.

Which is also why the win here was in scheduling and not in numeric format.

---

## 4 — The Arm CPU and the GPU sharing one utterance

`4_cooperative_pipeline/` is the DGX Spark path: the Arm CPU runs the text
encoder, duration predictor and normalising flow through ONNX Runtime, and the
Blackwell GPU runs the vocoder. Reference code and measured results are in that
directory — it needs torch and CUDA, so it is kept out of the CPU-only bundle
rather than adding a 2 GB dependency to every Mac install for a path that never
runs there.

Measured on an idle GPU, 20 sentences:

| stage | time | share |
|---|---|---|
| Arm CPU — encoder, duration predictor, flow | 14.21 ms | **57.4%** |
| handoff — unified memory | **0.1215 ms** | 0.49% |
| Blackwell GPU — vocoder | 10.43 ms | 40.3% |

The Arm CPU does the majority of the work on every utterance, and the handoff
across NVLink-C2C costs a tenth of a millisecond. Neither processor can produce
audio alone — a cooperative split, not request routing.

**Stated plainly: this path is 0.72× Kokoro-GPU — slower.** Running an entire
model on CUDA beats splitting one. What the split buys is meaningful Arm CPU
participation at GPU-class latency in 1.8× less memory. I report it that way
because the alternative is the "3× faster" number that turned out to be 1.77×.

---

## 5 — Where the Arm CPU time actually goes

Tuning ONNX Runtime's thread count is only worth doing if ONNX Runtime is where
the cycles are. That is a claim about where time goes, so it is measured with
Arm's own profiler rather than inferred from wall-clock timings.

[Arm Performix](https://developer.arm.com/documentation/109842/latest/),
`code_hotspots` recipe, 41,593 samples of pure Arm-CPU synthesis on a DGX Spark
GB10:

| share | image |
|---|---|
| **94.0%** | ONNX Runtime — fused CPU kernels |
| 3.1% | OpenBLAS |
| 2.0% | libc |
| 0.3% | espeak-ng |
| 0.2% | Python |

**Almost nothing is interpreter overhead**, which is the result that makes the
thread sweep meaningful: the knob is attached to 94% of the workload rather
than to a wrapper around it.

```bash
python3 5_profile_with_performix.py            # profile and report
python3 5_profile_with_performix.py --list     # past runs
python3 5_profile_with_performix.py --run-id <id>
```

**Honest limitation:** ONNX Runtime ships stripped, so samples inside it
resolve to `<Unknown code in onnxruntime...>` rather than to individual
kernels. Image-level attribution is still meaningful — it separates fused
kernels from Python, libc and the phonemizer — but per-kernel names would need
a symbolised build.

**A caution from doing this.** An earlier run of this profile attributed 93% of
the time to scipy's OpenBLAS, which would have contradicted everything above.
It had profiled a different workload. The run recorded here samples the shipped
v3 model through the same CPU engine the packages use, and its run id is in the
JSON so it can be re-exported and checked.
