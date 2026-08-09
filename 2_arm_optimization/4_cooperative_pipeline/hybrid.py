"""
Arm CPU + GPU cooperative inference.

The Arm CPU runs the text encoder, duration predictor and normalising flow
through ONNX Runtime; the Blackwell GPU runs the vocoder in PyTorch. The
intermediate latent crosses between them.

This is a cooperative split, not request routing. Every single utterance uses
both processors, and neither can produce audio alone.

Why the split is worth doing on this hardware
---------------------------------------------
On DGX Spark the Grace CPU and Blackwell GPU share physical memory over
NVLink-C2C, so handing a tensor across costs almost nothing. On a
PCIe-separated GPU the copy would dominate and this design would be pointless.
`benchmark_hybrid.py` measures the handoff separately so that claim is
checkable rather than asserted.

Why the CPU half runs ONNX rather than PyTorch
----------------------------------------------
Eager PyTorch pays per-op dispatch overhead that ONNX Runtime's fused graph
avoids. The earlier incarnation of this project measured roughly 6x on the
same architecture -- enough that an eager-PyTorch prefix made the whole split
LOSE to running everything on the GPU. The runtime mattered more than the
device boundary, which was the non-obvious part.

Why the GPU half is PyTorch rather than ONNX
--------------------------------------------
ONNX Runtime in this environment has no CUDA provider -- checked, not
assumed: get_available_providers() returns
['AzureExecutionProvider', 'CPUExecutionProvider'].

Thread count on the CPU side comes from tts.engine, which reads Linux
cpu_capacity to find the performance cores. Using all 20 cores here is 1.7x
SLOWER than using 9, because a batch finishes when its slowest thread does
and the efficiency cluster drags.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import onnxruntime as ort
import torch

from .export_onnx import build_from_checkpoint  # noqa: E402
from .engine import Phonemizer, _tuned_thread_count, split_sentences, write_wav  # noqa: E402


class HybridEngine:
    """CPU prefix (ONNX) -> GPU decoder (PyTorch)."""

    def __init__(self, decoder: str | Path, prefix_onnx: str | Path,
                 device: str = "cuda", num_threads: int | None = None):
        """`decoder` is an exported decoder file (models/af_heart_decoder.pt).

        A full training checkpoint is still accepted, because that is what this
        used to take -- but it is a 834 MB file of which only `net_g.dec` is
        ever read, so tts/export_decoder.py exists to strip it down. The
        exported form is the one distributed.
        """
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available; use device='cpu' to compare")
        self.device = torch.device(device)

        # Try the exported decoder first: it is the small, distributed form, so
        # the common path never pays to read an 834 MB checkpoint just to find
        # out which kind of file it was handed.
        from .export_decoder import load_decoder
        try:
            self.dec, payload = load_decoder(Path(decoder), device=device)
            self.epoch = payload["epoch"]
            self.sample_rate = payload["sample_rate"]
            self.phoneme_id_map = payload["phoneme_id_map"]
            inf = payload.get("inference", {})
            espeak_voice = payload.get("espeak_voice", "en-us")
        except Exception:
            # Legacy path: a full training checkpoint. Only net_g.dec is used.
            net_g, ckpt, voice_cfg = build_from_checkpoint(decoder, device="cpu")
            # Only the decoder goes to the GPU. Keeping the rest on CPU is the
            # point of the split, and it also keeps GPU memory small.
            self.dec = net_g.dec.to(self.device).eval()
            self.epoch = ckpt["epoch"]
            self.sample_rate = voice_cfg["audio"]["sample_rate"]
            self.phoneme_id_map = voice_cfg["phoneme_id_map"]
            inf = voice_cfg.get("inference", {})
            espeak_voice = voice_cfg.get("espeak", {}).get("voice", "en-us")
        self.scales = np.array([
            float(inf.get("noise_scale", 0.667)),
            float(inf.get("length_scale", 1.0)),
            float(inf.get("noise_w", 0.8)),
        ], dtype=np.float32)

        so = ort.SessionOptions()
        self.num_threads = num_threads or _tuned_thread_count()
        if self.num_threads:
            so.intra_op_num_threads = self.num_threads
        self.prefix = ort.InferenceSession(str(prefix_onnx), so,
                                           providers=["CPUExecutionProvider"])
        self.phonemizer = Phonemizer(espeak_voice)

        # Per-stage timing, so the CPU/GPU/handoff breakdown is measurable
        # rather than inferred from the total.
        self.last_timing: dict[str, float] = {}

    def _ids(self, phonemes: str) -> list[int]:
        import unicodedata
        pad = self.phoneme_id_map["_"]
        ids = list(self.phoneme_id_map["^"]) + list(pad)
        for ch in unicodedata.normalize("NFD", phonemes):
            m = self.phoneme_id_map.get(ch)
            if m is None:
                continue
            ids.extend(m)
            ids.extend(pad)
        ids.extend(self.phoneme_id_map["$"])
        return ids

    @torch.no_grad()
    def _synth_one(self, phonemes: str) -> np.ndarray:
        ids = self._ids(phonemes)
        arr = np.expand_dims(np.array(ids, dtype=np.int64), 0)

        t0 = time.perf_counter()
        z = self.prefix.run(None, {
            "input": arr,
            "input_lengths": np.array([arr.shape[1]], dtype=np.int64),
            "scales": self.scales,
        })[0]
        t_cpu = time.perf_counter() - t0

        # Handoff timing WITHOUT a synchronize inside the timed region.
        # An earlier version wrapped this in torch.cuda.synchronize() to time
        # it "accurately" and reported 0.18 ms -- almost all of which was the
        # synchronize itself. Measured in isolation the copy is 0.013 ms for
        # ~98 KB, i.e. 0.05% of the pipeline. On GB10 the CPU and GPU share
        # one 121.6 GiB pool (CUDA total == MemTotal, and nvidia-smi reports
        # [N/A] for VRAM because there is none), so this is a coherent-memory
        # transfer rather than a bus copy. Pinned and pre-allocated variants
        # measured 0.011-0.012 ms: nothing left to win.
        t1 = time.perf_counter()
        z_gpu = torch.from_numpy(z).to(self.device, non_blocking=True)
        t_handoff = time.perf_counter() - t1

        # The decoder still needs a synchronize to be timed honestly, since
        # CUDA launches are async and would otherwise appear free.
        t2 = time.perf_counter()
        audio = self.dec(z_gpu)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_gpu = time.perf_counter() - t2

        out = audio.squeeze().float().cpu().numpy()
        self.last_timing = {
            "cpu_prefix_ms": t_cpu * 1000,
            "handoff_ms": t_handoff * 1000,
            "gpu_decoder_ms": t_gpu * 1000,
            "handoff_bytes": z.nbytes,
        }
        return out

    def stream(self, text: str) -> Iterator[np.ndarray]:
        for ph in self.phonemizer.phonemize(text):
            ids = self._ids(ph)
            if len(ids) <= 3:
                continue
            a = self._synth_one(ph)
            if a.size:
                yield a.astype(np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        chunks = list(self.stream(text))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)

    def synthesize_to_wav(self, text: str, out_path: str | Path) -> float:
        a = self.synthesize(text)
        write_wav(out_path, a, self.sample_rate)
        return a.shape[-1] / self.sample_rate


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default="hybrid_out.wav")
    ap.add_argument("--text", default="The weather changed suddenly this afternoon.")
    args = ap.parse_args()

    eng = HybridEngine(args.checkpoint, args.prefix)
    print(f"epoch {eng.epoch}, CPU threads {eng.num_threads}, device {eng.device}")
    t0 = time.perf_counter()
    dur = eng.synthesize_to_wav(args.text, args.out)
    el = time.perf_counter() - t0
    print(f"{dur:.2f}s audio in {el*1000:.0f}ms  RTF {el/dur:.4f}")
    print("last sentence stage breakdown:", {k: round(v, 3) for k, v in eng.last_timing.items()})
