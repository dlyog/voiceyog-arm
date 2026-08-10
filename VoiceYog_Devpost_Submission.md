# VoiceYog: Arm-Optimized TTS

## Inspiration

The original motivation was personal voice ownership.

People can lose their voice because of ALS, cancer, stroke, age, or other medical conditions. Voice cloning can help, but many solutions still depend on a cloud API, an account, a subscription, and somebody else's infrastructure.

I wanted to explore a different model: train a voice once, then let the owner run it locally for years.

While working on that, I noticed another practical problem. TTS models often support many voices, but a user may only care about one. Kokoro is compact for a modern TTS model, but it still carries capability that a single-voice deployment does not need.

That led to a simple question:

**How small and how fast can one useful voice become on an Arm CPU somebody already owns?**

---

## What it does

VoiceYog is a single-voice TTS model designed for local Arm CPU inference.

After installation, speech generation runs without CUDA, without PyTorch, and without a network connection.

```bash
brew install espeak-ng
git clone https://github.com/dlyog/voiceyog-arm.git
cd voiceyog-arm
bash manage.sh install
```

`espeak-ng` converts text into phonemes. The installer then detects whether it is running on Apple Silicon or DGX Spark, downloads the matching package, verifies it, installs the bundled Python wheels, and starts the local inference server.

The interesting part is what happens after that.

### CPU benchmark

Measured on the DGX Spark GB10 with an idle GPU, using 20 held-out sentences and a separate process for each engine:

| Engine | RTF | Latency | Peak RSS | Model size |
|---|---:|---:|---:|---:|
| **VoiceYog - Arm CPU** | **0.03888** | **82.9 ms** | **356 MB** | **68.5 MB** |
| VoiceYog - CPU to GPU | 0.01957 | 42.2 ms | 1898 MB | 68.5 MB |
| Kokoro-82M - GPU | 0.01418 | 40.1 ms | 3464 MB | 326 MB |
| Kokoro-82M - Arm CPU | 0.33447 | 947.5 ms | 2661 MB | 326 MB |

The comparison I care about most is **VoiceYog CPU versus Kokoro CPU on the same Arm system**:

- **8.6x faster**
- **7.5x less runtime memory**
- **4.8x smaller model on disk**

A GPU is still faster once Kokoro is fully loaded. I am not trying to hide that. The point is that this workload no longer *needs* a GPU.

Cold start also changes the experience. Kokoro GPU took 5.42 seconds from process start to first audio in my test, while VoiceYog CPU took 0.94 seconds.

### Private voice deployment

The downloadable model does **not** contain my personal cloned voice. I distribute a single Kokoro-derived voice for the demo package.

The repository also includes the pipeline I used so people can build their own model.

A user can provide a voice sample, generate a focused training corpus with a larger local TTS model, train the smaller student once on a GPU, export it, and then run that voice locally on an Arm CPU.

That means the long-term inference path does not need a cloud voice-cloning API and the user's voice does not have to be sent to a remote service every time they want speech.

For me, that privacy story is just as important as the speed.

### Provenance

Every generated clip also carries a signed provenance record containing information such as the model, checksum, runtime, and platform.

The goal is simple: generated voice should be fast and private, but it should also be traceable.

---

## Architecture

The whole project is one pipeline. A sample of audio goes in at the top, and a
voice that runs on an Arm CPU comes out at the bottom. Every box is a script in
this repository, and every one of them was run on the DGX Spark before it was
written about here.

![VoiceYog architecture: sample audio and a built corpus feed 5_generate_dataset.py, then 6_validate_dataset.py, 7_train.py on the GB10 GPU, 8_export_onnx.py, 1_core_topology.py, manage.sh install, and finally CPU-only inference at 82.9 ms per sentence in 356 MB](https://raw.githubusercontent.com/dlyog/voiceyog-arm/main/.github/architecture.png)

| Step | Script | What comes out |
|---|---|---|
| Build the prompt corpus | `1_word_bank.py`, `2_cmu_arctic_sentences.py`, `3_llm_sentences.py` → `4_build_corpus.py` | 1,194 sentences, offline, no network |
| Generate data from a sample | `5_generate_dataset.py` | A teacher speaks the corpus. Kokoro for an AI voice, or a local clone service conditioned on one recording of you. This is the only step that changes between the two |
| Validate before training | `6_validate_dataset.py` | 4,104 clips, 24 kHz, 3.057 hours. Catches wrong sample rates, stereo files and bad delimiters before a training run wastes hours on them |
| Train the student | `7_train.py` | 25.74 M generator with a 3.76 M vocoder, about 125 s per epoch on the GB10 |
| Export for Arm | `8_export_onnx.py` | A 68.5 MB ONNX graph that runs on any modern Arm CPU |
| Tune for the Arm CPU | `2_arm_optimization/1_core_topology.py` | The thread count, read from the hardware at runtime: 9 on DGX Spark, 8 on M1 Max |
| Install and serve | `manage.sh install` | Detects the platform, installs the matching package from bundled wheels, starts the local server |

**The GPU appears exactly once, in the training step.** Everything below the
export runs on the CPU, which is the whole point of the project.

### What you get at the end

Against the same Kokoro model on the same DGX Spark Arm CPU:

- **8.6x faster** and **7.5x less memory**
- 68.5 MB on disk instead of 326 MB

And against Kokoro running on the GB10 GPU:

- **9.7x less memory** while running, 356 MB against 3464 MB
- **5.8x faster to the first word** from a cold start, 0.94 s against 5.42 s
- but the GPU is still **2.7x faster per sentence** once it is loaded, and I
  would rather say so than have somebody find it

---

## How I built it

The project came together in four main steps.

### 1. Build a smaller single-voice model

The student is a smaller VITS/HiFi-GAN model trained from scratch on roughly three hours of Kokoro-generated speech.

Kokoro supports multiple voices. VoiceYog does not need that flexibility at inference time, so the student architecture can be much smaller.

That is how the deployed model comes down from 326 MB to 68.5 MB.

The expensive part happens on the DGX Spark GB10 GPU during data generation and training. Once the model is trained and exported, the GPU is no longer required for normal inference.

### 2. Profile before optimizing

I did not want to guess where the CPU time was going, so I profiled the shipped workload with **Arm Performix**.

Across 41,593 samples:

| CPU time | Component |
|---:|---|
| **94.0%** | ONNX Runtime |
| 3.1% | OpenBLAS |
| 2.0% | libc |
| 0.3% | espeak-ng |
| 0.2% | Python |

That was useful because it showed that almost all of the work was in ONNX Runtime, not Python glue code.

So that is where I focused the optimization.

### 3. Make threading aware of the Arm CPU

Both of my targets are asymmetric Arm systems. They mix faster and slower cores.

That matters for latency. If an ONNX operator runs across several threads, the operation cannot finish until the slowest participating thread is done. Using more cores can therefore make inference slower when some of those cores are significantly slower.

Instead of hard-coding a thread count, VoiceYog reads the hardware topology at runtime.

On Linux it uses information including `MIDR_EL1` and `cpu_capacity`. On macOS it uses the performance-level information exposed by the system.

The measured result:

| Platform | Best threads | Versus all cores | Versus ORT default |
|---|---:|---:|---:|
| DGX Spark GB10 | **9** | **1.39x faster** | **1.65x faster** |
| Apple M1 Max | **8** | **2.21x faster** | **1.20x faster** |

This is why the demo shows different thread counts on the two machines. The runtime is not just asking for every available CPU core. It is adapting to the Arm hardware it actually finds.

### 4. Make the claims reproducible

While preparing the submission, I caught several numbers that had become stale after code or model changes.

So I stopped trusting numbers copied by hand.

`verify_claims.py` checks the figures in the project against the JSON produced by the measurement scripts and exits with an error when they disagree.

That became one of the most useful pieces of the project because optimization claims are easy to get wrong when hardware, model versions, and runtime settings keep changing.

---

## Challenges I ran into

### A benchmark I had to correct

An earlier version of my notes said VoiceYog was "3x faster than Kokoro GPU."

When I re-ran both engines fairly on an idle system, that claim did not hold. The old baseline had been measured while the GPU was busy.

I removed the claim and built the checker so that kind of drift is harder to publish again.

### My own package was using the wrong thread count

At one point the DGX build was using all 20 CPU threads even though the documentation said the runtime was tuned.

The reason was embarrassing but useful: an old assumption in the Linux code treated all DGX CPU cores as equivalent.

They are not.

Fixing the topology detection improved the measured RTF from 0.0528 to 0.0339 in that test.

### The fast cores were not where I expected

On the GB10, the performance cores are not simply the first half or second half of the CPU IDs. They are spread across the topology.

That broke an early attempt to pin threads using a simple CPU range and convinced me that the runtime should read the hardware instead of guessing from core numbers.

### INT8 was smaller, but unusable

Dynamic INT8 quantization made the model file about 3.3x smaller.

It also rewrote convolutions into `ConvInteger`, which the ONNX Runtime CPU provider did not have an AArch64 kernel for in this workload.

So the model would not load on either target.

I kept that result in the project because "smaller file" is not an optimization if the artifact cannot run.

### The pipeline had to become real, not reference code

Some of the original training scripts were effectively reference copies and depended on paths from my earlier project.

When I tried to run the pipeline cleanly from this repository, those assumptions broke.

I fixed the paths, separated teacher outputs, and ran the complete flow on the GB10 before documenting it.

That was a useful reminder that a pipeline is only part of the submission once another person can actually execute it.

---

## Accomplishments I am proud of

The biggest win is not just the 68.5 MB model. It is that the whole deployment behaves like an Arm application instead of a GPU application that happens to have a CPU fallback.

The same repository can install on DGX Spark or Apple Silicon and choose a hardware-appropriate thread configuration automatically.

The core-topology utility is also reusable beyond TTS. The same idea applies to other ONNX workloads running on asymmetric Arm CPUs.

I am also happy that the demo is produced with the project's own models. Most of the narration uses my voice cloned locally with Qwen3-TTS on the DGX Spark, and the small VoiceYog model speaks for itself during the inference scenes.

And finally, I am proud that the failed experiments are still visible. The INT8 result failed. The cooperative CPU-to-GPU path did not beat pure Kokoro GPU inference. Those results helped narrow the project toward the part that actually worked well: **small, local Arm CPU inference**.

---

## What I learned

The biggest lesson was that **more CPU cores do not automatically mean more performance**.

On an asymmetric Arm processor, a latency-sensitive workload can end up waiting for its slowest thread. On the M1 Max, the difference between the right thread count and using every core was 2.21x.

I also learned to read the hardware instead of inferring it from names or CPU numbering. The topology information was more reliable than every heuristic I tried.

Another lesson was that specialization can be more useful than generic compression. I did not make all of Kokoro smaller while preserving every feature. I built a model around the narrower job I actually wanted: one voice, local inference, on Arm.

And finally, benchmark documentation needs to be treated like code. If the measurements change, the write-up has to change with them.

---

## What's next

### Better CPU pipelining

There is still work left in the cooperative CPU/GPU path. Today the stages run mostly one after another, so there is room to overlap work instead of leaving one processor idle while the other is active.

I have not implemented that yet, so I am not claiming a result.

### Newer ONNX Runtime and KleidiAI

The current package pins ONNX Runtime 1.20.1. A newer runtime on the same DGX Spark contains Arm KleidiAI kernels that are not present in the pinned version.

That makes upgrading and re-measuring the CPU path an obvious next experiment.

### Observe real thread placement

I currently measure the topology and the latency result, but I do not yet record exactly which physical core every inference thread lands on.

Adding that measurement would make the threading analysis stronger.

### Make bring-your-own-voice easier

The eight training pipeline scripts now ship and run end to end, covering corpus generation, teacher generation, validation, training, and ONNX export.

The next step is making that process simple enough for somebody who cares about preserving their own voice, not somebody who wants to debug a TTS training pipeline.

### More Arm targets

The inference path is ONNX-based, so Android, iOS, and Windows on Arm are natural next targets.

The goal is to keep the same model and the same idea: detect the hardware, tune locally, and run the voice where the user already is.

---

## Built with

`onnxruntime` · `numpy` · `FastAPI` · `uvicorn` · `pydantic` · `cryptography` · `espeak-ng` · `PyTorch` (training only) · **Arm Performix** · **Qwen3-TTS** · `Playwright` · `ffmpeg` · **Kokoro-82M** · **VITS** · **HiFi-GAN** · CMU ARCTIC · NVIDIA DGX Spark GB10 · Apple Silicon

---

## Thanks

Kokoro-82M is both the teacher behind this experiment and the baseline I used to measure it.

VoiceYog is not an attempt to replace everything Kokoro can do. The project asks a narrower question: if I only need one voice, how much of the deployment cost can I remove, and how well can I make what remains run on Arm?

For this project, the answer was: enough to make CPU-only TTS genuinely practical for my use case.

---

## Summary

Kokoro-82M is one of the best lightweight text-to-speech models I have used. It is already very fast on a GPU, but on an Arm CPU it can take close to a second to generate a sentence.

That felt wasteful for my use case. Most of the time I need one language and one voice, not a full multi-voice TTS system.

So I built VoiceYog: a smaller, single-voice TTS model and an Arm-aware inference runtime around it.

On the same DGX Spark Arm CPU, VoiceYog is **8.6x faster than Kokoro CPU inference while using 7.5x less memory**. The model is 68.5 MB on disk and uses about 356 MB while running. Inference needs no GPU, no cloud API, and no internet connection.

**Demo:** https://www.youtube.com/watch?v=L4THa8PWQi4  
**Repository:** https://github.com/dlyog/voiceyog-arm

The DGX Spark was my main development box. I generated the training data there, trained the student model on its GB10 Blackwell GPU, built the inference path there, and used it for the benchmark, profiling, and thread-tuning work.

From the same project I prepared inference packages for two Arm targets: **DGX Spark and Apple Silicon**.

That split became the main idea behind the project:

**Use the GPU for the expensive work you do once. Run the voice on the Arm CPU for the work you do again and again.**
