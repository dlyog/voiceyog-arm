# Arm CPU → GPU, one utterance, split between them

Reference code for the DGX Spark path. This is the part that needs torch and
CUDA, which is why it is here rather than inside the CPU-only bundle — adding
a 2 GB dependency to every Mac install to support a path that never runs there
would be the wrong trade.

| file | what it is |
|---|---|
| `hybrid.py` | the cooperative engine: ONNX prefix on the Arm CPU, PyTorch vocoder on the GPU |
| `export_split.py` | splits the trained model into the CPU prefix and the GPU decoder |
| `benchmark_of_record.py` | the measurement; refuses to run on a busy GPU |

Results: [`../../3_evidence/benchmark_of_record_dgx_spark.json`](../../3_evidence/benchmark_of_record_dgx_spark.json)

---

## The split

```
text ─► Arm CPU (ONNX Runtime)              ─► latent ─► Blackwell GPU (PyTorch) ─► audio
        text encoder, duration predictor,      unified     HiFi-GAN vocoder
        normalising flow                       memory      3.76 M params
```

Every utterance uses both processors. Neither can produce audio alone — this
is a cooperative split, not a router that sends some requests to the CPU and
others to the GPU.

| stage | time | share |
|---|---|---|
| Arm CPU — encoder, duration predictor, flow | 14.21 ms | **57.4%** |
| handoff — unified memory | **0.1215 ms** | 0.49% |
| Blackwell GPU — vocoder | 10.43 ms | 40.3% |

---

## Why the split is worth doing on this hardware specifically

On DGX Spark the Grace CPU and the Blackwell GPU share physical memory over
NVLink-C2C, so handing a tensor across costs **0.1215 ms — 0.49% of the
pipeline**. On a PCIe-separated GPU that copy would dominate and the whole
design would be pointless. The handoff is timed as its own stage precisely so
that claim is checkable rather than asserted.

## Why the CPU half runs ONNX and not PyTorch

Eager PyTorch pays per-op dispatch overhead that ONNX Runtime's fused graph
avoids — roughly 6× on this architecture in the parent project's measurements.
Enough that an eager-PyTorch prefix made the whole split **lose** to running
everything on the GPU. The runtime mattered more than the device boundary,
which was the non-obvious part.

## Why the GPU half runs PyTorch and not ONNX

ONNX Runtime in this environment has no CUDA provider. Checked, not assumed:
`get_available_providers()` returns `['AzureExecutionProvider', 'CPUExecutionProvider']`.

---

## The honest result

| engine | RTF | latency | peak RSS |
|---|---|---|---|
| ours — Arm CPU only | 0.03888 | 82.9 ms | **356 MB** |
| ours — Arm CPU → GPU | 0.01957 | 42.2 ms | 1898 MB |
| Kokoro-82M — GPU | **0.01418** | 40.1 ms | 3464 MB |
| Kokoro-82M — Arm CPU | 0.33447 | 947.5 ms | 2661 MB |

**The cooperative path is 0.72× Kokoro-GPU. It is slower.** Running an entire
model on CUDA beats splitting one across a device boundary, and we are not
going to bury that.

What the split does buy, measured on the same box in the same run:

- meaningful Arm CPU participation — 57.4% of every utterance — at GPU-class
  latency (42.2 ms against 40.1 ms)
- **1.8× less memory** than the GPU baseline needs
- and the CPU-only path, which needs no GPU at all, is **8.6× faster than the
  same baseline on the same Arm CPU in 7.5× less memory**

That last row is the one that ships. The cooperative path is the interesting
one.

---

## Known limitation, stated because it is the largest remaining win

The Arm CPU prefix is 57.4% of the pipeline and runs strictly **before** the
GPU stage, so each processor idles while the other works. Overlapping
`prefix(N+1)` with `decode(N)` across sentence chunks would hide most of one
behind the other. It is designed and not built; claiming it as a result would
be exactly the kind of number this submission is organised to avoid.
