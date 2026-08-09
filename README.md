# VoiceYog — one voice, one language, running on Arm

**Tarun Kumar Chawdhury** · DLYog Lab Research Services LLC · Apache 2.0

A 326 MB multi-voice text-to-speech model, distilled to **68.5 MB** for a single
voice, then tuned for the asymmetric Arm CPUs it actually runs on — the Grace
CPU on **NVIDIA DGX Spark (GB10)** and **Apple Silicon**.

**No GPU. No CUDA. No PyTorch at inference. No network.** Each package carries
its own Python wheels and verifies 102 checksums before it does anything.

---

## Three steps, from a clean machine to a voice

```bash
brew install espeak-ng                 # macOS      (sudo apt-get install espeak-ng on Linux)
git clone https://github.com/dlyog/voiceyog-arm.git && cd voiceyog-arm
bash manage.sh install
```

That's it. `install` detects whether you are on Apple Silicon or a DGX Spark,
downloads the right package, verifies its sha256, unpacks it, runs the bundle's
own installer (102 more checksums, 32 vendored wheels, **pip never touches the
network**), starts the server in the background and prints a URL:

```
  serving            kokoro-heart-new:v3  8 threads  68.5 MB
  open  http://127.0.0.1:8823
```

Open it and type something.

```
bash manage.sh install | start | stop | status | log [N] | demo [text] | uninstall
```

Prerequisites are checked **before** the 170 MB download, not after — a missing
phonemizer should cost you two seconds, not two minutes. `espeak-ng` runs as a
**separate process**, never linked, which keeps its GPL terms out of the
shipped runtime.

Prefer to do it by hand? The package is self-contained and needs none of the
above:

```bash
bash 1_packages/download.sh          # detects your machine, verifies sha256
unzip voiceyog-kokoro-heart-new-v3-*.zip
cd voiceyog-local-tts-kokoro-heart-new-*
bash install.sh && bash start.sh
```

The two packages are **[release assets](https://github.com/dlyog/voiceyog-arm/releases/latest)**,
not files in this repository — both exceed GitHub's 100 MB per-file limit, and
Git LFS bills bandwidth per download. Checksums are committed in
[`1_packages/SHA256SUMS`](1_packages/SHA256SUMS).

| target | hardware | size |
|---|---|---|
| `apple-silicon` | macOS 13+ on M-series | 171 MB |
| `dgx-spark` | aarch64 Linux, DGX Spark GB10 | 154 MB |

**Architecture at a glance:** [`ARCHITECTURE.html`](ARCHITECTURE.html) — the
pipeline, the split point, and what runs where.

---

## Why one voice

Kokoro-82M serves eleven voices and several languages. Almost nobody needs
that. A person who has banked their own voice — before illness takes it, or
simply because it is theirs — needs exactly **one** voice, forever, on hardware
they own, with no network and no subscription.

That is not a smaller version of the same product. It changes what the model
has to contain:

| Kokoro-82M component | params | needed for one voice? |
|---|---|---|
| `decoder.decode` (style conditioning) | 27.94 M | **no — multi-voice machinery** |
| `predictor` (prosody) | 16.20 M | no |
| `decoder.encode` | 5.66 M | no |
| `bert` + `text_encoder` | 11.90 M | ours is smaller |
| `decoder.generator` (vocoder) | 19.69 M | yes — ours is 3.76 M |

The single largest component exists only to condition on *which* voice you
asked for. For this user it is dead weight, not a feature being sacrificed.

---

## The optimization

Both targets are **asymmetric** Arm parts, and that asymmetry is the whole
thing:

```
DGX Spark GB10   10x Cortex-X925 @ 3.90 GHz  +  10x Cortex-A725 @ 2.81 GHz
Apple M1 Max      8 performance cores        +   2 efficiency cores
```

ONNX Runtime parallelises an operator across intra-op threads and joins them.
That join is a barrier — the operator finishes when its **slowest** thread
does. One thread on a core 28% slower holds up every other thread, on every
operator, for the whole graph. So on these parts, **adding cores subtracts
latency**:

| | best threads | vs handing it every core | vs ONNX Runtime's default |
|---|---|---|---|
| DGX Spark GB10 (20 cores) | 9 | **1.39× faster** | **1.65× faster** |
| Apple M1 Max (10 cores) | 8 | **2.21× faster** | 1.20× faster |

The count is derived at runtime from `MIDR_EL1` and `cpu_capacity` on Linux and
`hw.perflevel0` on macOS — never hard-coded, never guessed from a core count.
**One ONNX graph, one binary, retuned per Arm part with no recompilation.**

On both machines the topology's prediction *was* the measured optimum: 9 on
GB10, 8 on M1 Max.

> **The detail a core count would have missed.** On GB10 the performance cores
> are **not contiguous** — they are `5-9,15-19`, interleaved across two
> clusters. `taskset -c 5-14` reads exactly like "pin to the fast half" and
> lands five of ten threads on efficiency cores.

---

## Results

DGX Spark GB10, idle GPU, 20 held-out sentences, each engine in its own
process. Raw output: [`3_evidence/benchmark_of_record_dgx_spark.json`](3_evidence/benchmark_of_record_dgx_spark.json)

| engine | RTF | latency | peak RSS | model |
|---|---|---|---|---|
| **ours — Arm CPU only** | **0.03888** | 82.9 ms | **356 MB** | 68.5 MB |
| ours — Arm CPU → GPU | 0.01957 | 42.2 ms | 1898 MB | 68.5 MB |
| Kokoro-82M — GPU | 0.01418 | 40.1 ms | 3464 MB | 326 MB |
| Kokoro-82M — Arm CPU | 0.33447 | 947.5 ms | 2661 MB | 326 MB |

**On the Arm CPU alone: 8.6× faster than the teacher, in 7.5× less memory,
from a model 4.8× smaller.** That is the row both packages ship.

**And the one that goes the wrong way: the cooperative path is 0.72×
Kokoro-GPU — slower.** Running an entire model on CUDA beats splitting one
across a device boundary. What the split buys is real Arm CPU participation at
GPU-class latency in 1.8× less memory:

| stage | time | share |
|---|---|---|
| Arm CPU — encoder, duration predictor, flow | 14.21 ms | **57.4%** |
| handoff — unified memory (NVLink-C2C) | **0.1215 ms** | 0.49% |
| Blackwell GPU — vocoder | 10.43 ms | 40.3% |

Every utterance uses both processors; neither can produce audio alone.

---

## Verify it yourself

```bash
python3 3_evidence/verify_claims.py
```

No dependencies, no model, no network. It checks **all 33 figures** in this
README and `SUBMISSION.md` against the JSON the measurement scripts wrote, and
exits non-zero if any disagree.

Then, on your own silicon:

```bash
python3 2_arm_optimization/1_core_topology.py           # your cores, from the registers
B=1_packages/voiceyog-local-tts-kokoro-heart-new-*
$B/.venv/bin/python3 2_arm_optimization/2_thread_sweep.py --json sweep.json
```

The sweep takes about a minute and prints the tuning table for **your** machine.
It will contradict us if we are wrong.

---

## What is here

```
manage.sh            install | start | stop | status | log | demo | uninstall
1_packages/          download.sh + SHA256SUMS   (zips are release assets)
2_arm_optimization/  the optimization, as scripts you can run
3_evidence/          every number, in the file that produced it
4_voice_pipeline/    how a voice becomes an Arm-native model
SUBMISSION.md        the full narrative (SUBMISSION.html renders identically)
```

The reusable part is `2_arm_optimization/`. `1_core_topology.py` answers "how
many threads should ONNX Runtime get on this machine" for **any** ONNX workload
on **any** asymmetric Arm part — not just this one, and not just TTS.

---

## The optimization we did not ship

INT8 dynamic quantization is the reflex answer for Arm. Here the quantized
model **will not load at all**, on both targets, at the same node:

```
float32   71.8M    80.29 ms      (DGX Spark, 9 threads)
int8      21.5M    does not run

NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
                  node '/enc_p/encoder/attn_layers.0/conv_q/Conv_quant'

    183x  ConvInteger        162x  DynamicQuantizeLinear
```

`quantize_dynamic` rewrote all 183 convolutions into `ConvInteger`, for which
ONNX Runtime's CPU provider has no aarch64 kernel. A one-line change made the
file 3.3× smaller and produced an artifact that runs nowhere.

Reproduce it: `2_arm_optimization/3_int8_negative_result.py`.

---

## Limitations

- **Quality is below the teacher's.** 3 h of distilled audio and a 3.76 M
  vocoder against Kokoro's 19.69 M. The training audio came *from* Kokoro, so
  Kokoro is a hard ceiling by construction.
- **Chunked pipelining is not built.** The Arm CPU prefix is 57.4% of the
  pipeline and runs strictly before the GPU stage, so each processor idles
  while the other works. Overlapping `prefix(N+1)` with `decode(N)` is the
  largest remaining win. Designed, not built, not claimed.
- **KleidiAI is absent** from the ONNX Runtime build used here (0 symbols).
  Arm's optimized matmul kernels would accelerate exactly the CPU prefix that
  is now the bottleneck.
- **English only, one voice, by design.** Not validated beyond ~6.25 s per
  chunk; longer text is chunked by sentence automatically.

---

## Licence and attribution

Copyright © 2026 Tarun Kumar Chawdhury, DLYog Lab Research Services LLC.
Apache License 2.0 — see [`LICENSE`](LICENSE).

The shipped runtime is permissive throughout: onnxruntime (MIT), numpy (BSD),
FastAPI/uvicorn/pydantic (MIT/BSD), cryptography (Apache-2.0).

- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** (Apache-2.0) —
  the teacher for the training audio and the comparison baseline. This work is
  not affiliated with or endorsed by it.
- Architecture adapted from **[VITS](https://github.com/jaywalnut310/vits)** and
  **[HiFi-GAN](https://github.com/jik876/hifi-gan)** (both MIT).
- **[CMU ARCTIC](http://www.festvox.org/cmu_arctic/)** — prompt text only; none
  of their recorded audio is used.
- **espeak-ng** (GPL-3.0) is invoked as a separate process, never linked. The
  Python bindings that *would* link it are deliberately not used.
