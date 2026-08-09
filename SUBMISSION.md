# VoiceYog

### One voice, one language, running on Arm

**Tarun Kumar Chawdhury** · DLYog Lab Research Services LLC · Apache 2.0
NVIDIA DGX Spark (GB10) + Apple Silicon · 2026

A 326 MB multi-voice text-to-speech model, distilled to **68.5 MB** for a
single voice, then tuned for the asymmetric Arm CPUs it actually runs on.

**Two downloads. One command each. Everything else here is evidence for what
those two downloads do.**

---

## 1. The problem this is actually for

Every general TTS model carries many voices and several languages. Almost
nobody uses them that way. A person who has **banked their own voice** — before
illness takes it, or simply because it is theirs — needs exactly one voice,
forever, on hardware they own, offline, with no subscription and no company
that has to still exist in ten years.

That is not "the same product, smaller". It changes what the model must
contain:

| Kokoro-82M component | params | needed for one voice? |
|---|---|---|
| `decoder.decode` (style conditioning) | 27.94 M | **no — multi-voice machinery** |
| `predictor` (prosody) | 16.20 M | no |
| `decoder.encode` | 5.66 M | no |
| `bert` + `text_encoder` | 11.90 M | ours is smaller |
| `decoder.generator` (vocoder) | 19.69 M | yes — ours is 3.76 M |

The single largest component exists only to condition on *which* voice you
asked for. For this user it is dead weight — not capability being sacrificed.
**That is what makes single-voice distillation a legitimate optimization target
rather than a lossy shortcut**, and it is why the rest of this document is
about making one voice run well on a CPU instead of making many voices run at
all.

---

## 2. What was optimized

### 2.1 The model — 326 MB to 68.5 MB

Distilled into a VITS/HiFi-GAN student trained on 3.0 h / 4,104 clips of
teacher audio. **4.8× smaller**, and the vocoder is 3.76 M parameters against
the teacher's 19.69 M.

### 2.2 The runtime — reading the silicon instead of counting cores

Both targets are **asymmetric** Arm parts:

```
DGX Spark GB10   10x Cortex-X925 @ 3.90 GHz  +  10x Cortex-A725 @ 2.81 GHz
Apple M1 Max      8 performance cores        +   2 efficiency cores
```

ONNX Runtime parallelises an operator across intra-op threads and joins them.
That join is a **barrier** — the operator finishes when its slowest thread
does. One thread on a core 28% slower holds up every other thread, on every
operator, for the whole graph. So on these parts, adding cores subtracts
latency:

| | best threads | vs every core | vs ONNX Runtime's default |
|---|---|---|---|
| DGX Spark GB10 (20 cores) | 9 | **1.39× faster** | **1.65× faster** |
| Apple M1 Max (10 cores) | 8 | **2.21× faster** | 1.20× faster |

The count is derived at runtime from `MIDR_EL1` and `cpu_capacity` on Linux and
`hw.perflevel0` on macOS — never hard-coded, never inferred from a core count.
**One ONNX graph, one binary, retuned per Arm part with no recompilation.**

On both machines the topology's prediction *was* the measured optimum — 9 on
GB10, 8 on M1 Max — and the prediction is written into the sweep JSON before
the sweep runs, so it cannot be back-fitted.

The technique is as standard as it gets: set `intra_op_num_threads`. What makes
it an **Arm** optimization is that the correct value is a property of the core
topology and can only be read off the hardware.

> **What a core count would have missed.** On GB10 the performance cores are
> **not contiguous** — they are `5-9,15-19`, interleaved across two clusters.
> `taskset -c 5-14` reads exactly like "pin to the fast half" and lands five of
> ten threads on efficiency cores.
>
> An earlier version of this code counted cores per MIDR and called the largest
> group "performance". With exactly ten of each, that was right only by luck.

**How the threads actually land.** I choose the count; I do not place the
threads — ONNX Runtime sets no affinity here and Linux does the placing. The
reason the count matters is arithmetic: there are 10 performance cores, so
asking for 20 runnable threads puts **at least 10 on efficiency cores by
pigeonhole**, however good the scheduler is. Ask for 9 and none is forced
there. You are not steering placement; you are removing the guarantee of bad
placement.

### 2.3 The pipeline — the Arm CPU doing the majority of the work

On DGX Spark the Arm CPU runs the text encoder, duration predictor and
normalising flow; the Blackwell GPU runs the vocoder; the latent crosses
between them through unified memory:

| stage | time | share |
|---|---|---|
| Arm CPU — encoder, duration predictor, flow | 14.21 ms | **57.4%** |
| handoff — unified memory (NVLink-C2C) | **0.1215 ms** | 0.49% |
| Blackwell GPU — vocoder | 10.43 ms | 40.3% |

Every utterance uses both processors; neither can produce audio alone. This is
a cooperative split, not request routing. The handoff costs a tenth of a
millisecond — the fact that makes the design viable on this machine and
pointless on a PCIe-separated GPU.

---

## 3. The numbers

DGX Spark GB10, idle GPU, 20 held-out sentences, each engine in its own
process. Source: [`3_evidence/benchmark_of_record_dgx_spark.json`](3_evidence/benchmark_of_record_dgx_spark.json)

| engine | RTF | latency | peak RSS | model |
|---|---|---|---|---|
| **ours — Arm CPU only** | **0.03888** | 82.9 ms | **356 MB** | 68.5 MB |
| ours — Arm CPU → GPU | 0.01957 | 42.2 ms | 1898 MB | 68.5 MB |
| Kokoro-82M — GPU | 0.01418 | 40.1 ms | 3464 MB | 326 MB |
| Kokoro-82M — Arm CPU | 0.33447 | 947.5 ms | 2661 MB | 326 MB |

**On the Arm CPU alone: 8.6× faster than the teacher, in 7.5× less memory,
from a model 4.8× smaller.** That is the row both packages ship, and it needs
no GPU at all.

**On DGX Spark specifically: the GPU builds the model once; the Arm CPU runs it
forever.** The Arm side is not a host babysitting CUDA — it is the deployment
target.

**And the row that goes the wrong way: the cooperative path is 0.72×
Kokoro-GPU — slower.** Running an entire model on CUDA beats splitting one
across a device boundary. What the split buys is real Arm CPU participation at
GPU-class latency (42.2 ms against 40.1 ms) in **1.8× less memory**.

> **Why that is phrased so carefully.** An earlier version of this project's
> documentation claimed "3× faster than Kokoro-GPU". Re-measured with both
> engines in one run on an idle GPU, it was **1.77×** — the original figure had
> been taken while the GPU was busy, which made the baseline look slow. Nobody
> lied; a number was carried by hand out of its context and stopped being true.
>
> Every figure in this document is now emitted by a script into JSON, and
> `verify_claims.py` exits non-zero if the writeup and the JSON disagree.

---

## 4. The optimization I did not ship

INT8 dynamic quantization is the reflex answer for Arm. On this graph it does
not merely lose — **the quantized model will not load at all**, on both
targets, at the same node:

```
float32   71.8M    80.29 ms      (DGX Spark, 9 threads)
int8      21.5M    does not run

NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
                  node '/enc_p/encoder/attn_layers.0/conv_q/Conv_quant'

root cause, read off the quantized graph:
    183x  ConvInteger        162x  DynamicQuantizeLinear
```

`quantize_dynamic` rewrote all 183 convolutions into `ConvInteger`, for which
ONNX Runtime's CPU provider has no aarch64 kernel.

The useful conclusion is not "INT8 is bad on Arm". It is that quantization is a
**trade** whose payoff belongs to the model and the target rather than to the
technique — and that a one-line change making the file 3.3× smaller produced an
artifact that runs nowhere. Shipping it on the strength of the file size,
without loading it once, is the failure this submission is built to catch. It
is also why the win here turned out to be in scheduling rather than in numeric
format.

Reproduce it: `2_arm_optimization/3_int8_negative_result.py`.

---

## 5. Reproducibility

The first thing to run needs nothing at all:

```bash
python3 3_evidence/verify_claims.py

  All 33 claims are backed by the measurements in this directory.
```

No dependencies, no model, no network. It checks every figure in this document
against the JSON the measurement scripts wrote, and fails loudly if they
disagree.

Then, on your own silicon:

```bash
python3 2_arm_optimization/1_core_topology.py
$B/.venv/bin/python3 2_arm_optimization/2_thread_sweep.py --json sweep.json
```

The sweep takes about a minute and prints the tuning table for the machine in
front of you. It will contradict me if I am wrong.

Both packages were installed from a clean extraction on real hardware before
submission — offline, from bundled wheels — and the transcripts ship in
`3_evidence/`:

```
fresh_install_apple_silicon.txt   102 checksums · 32 wheels, no network · 8 threads
fresh_install_dgx_spark.txt       102 checksums · 32 wheels, no network · 9 threads
```

### What building this submission found

Preparing the packages is how I discovered the shipped DGX build was running
at **20 threads**: `tuned_threads()` returned `os.cpu_count()` on Linux, under
a docstring asserting "on DGX OS every core is the same". That is false, and it
cost 1.39×. Fixed, and verified on the box:

```
before   "threads": 20    RTF 0.0528
after    "threads":  9    RTF 0.0339
```

An optimization is only real once something measures it.

---

## 6. What another developer can take

The reusable artifact is not the app. It is four scripts, each standalone, each
printing a result for the machine in front of you, each writing JSON the
claim-checker reads.

| script | needs | answers |
|---|---|---|
| `1_core_topology.py` | Python 3 only | What are my cores, really? |
| `2_thread_sweep.py` | an installed bundle | How many threads, and what does the wrong answer cost? |
| `3_int8_negative_result.py` | + `onnx` | Should I quantize *here*? |
| `4_cooperative_pipeline/` | DGX Spark, torch + CUDA | Can the CPU and GPU split one utterance? |

`1_core_topology.py` is the one I would most want someone to take. "How many threads
should ONNX Runtime get" comes up for **every** ONNX workload on **every**
asymmetric Arm part, and the honest answer is always read off `MIDR_EL1` and
`cpu_capacity` rather than from `nproc`.

---

## 7. Limitations

- **Quality is below the teacher's.** 3 h of distilled audio and a 3.76 M
  vocoder against Kokoro's 19.69 M. The training audio came *from* Kokoro, so
  Kokoro is a hard ceiling by construction. *(The WER/CER figures for this are
  the one set `verify_claims.py` does **not** check — re-running them needs the
  teacher and a speech-recognition model, so they are not reproducible from
  these packages alone.)*
- **Chunked pipelining is not built.** The Arm CPU prefix is 57.4% of the
  pipeline and runs strictly before the GPU stage, so each processor idles
  while the other works. Overlapping `prefix(N+1)` with `decode(N)` is the
  largest remaining win. Designed, not built, not claimed.
- **KleidiAI is absent from the runtime I pin, and present in a newer one.**
  The packages ship onnxruntime 1.20.1, which contains **0** KleidiAI symbols
  on either target. onnxruntime 1.28.0 on the same DGX Spark contains **11**
  `kai_run_matmul_*` symbols. Arm's optimized matmul kernels would accelerate
  exactly the CPU prefix that is now the bottleneck, so upgrading the pinned
  runtime is a concrete measurable next step rather than a wish — I have not
  measured it, so I am not claiming it.
- **Thread placement is inferred, not observed.** The topology and the latency
  curve are measured; which physical core each thread ran on is not. The
  pigeonhole argument in §2.2 holds regardless, but the causal story is
  reasoning rather than measurement.
- **English only, one voice, by design.** Not validated beyond ~6.25 s per
  chunk, the longest clip in the training data; longer text is chunked by
  sentence automatically.

---

## Attribution

Copyright © 2026 Tarun Kumar Chawdhury, DLYog Lab Research Services LLC.
Apache License 2.0.

Distilled from **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)**
(Apache-2.0), which is also the comparison baseline — this work is not
affiliated with or endorsed by it. Architecture adapted from
**[VITS](https://github.com/jaywalnut310/vits)** and
**[HiFi-GAN](https://github.com/jik876/hifi-gan)** (both MIT).
**[CMU ARCTIC](http://www.festvox.org/cmu_arctic/)** prompt text only — none of
their recorded audio is used. **espeak-ng** (GPL-3.0) is invoked as a separate
process, never linked; the shipped runtime is permissive throughout.
