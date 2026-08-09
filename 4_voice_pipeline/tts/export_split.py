"""
Export the CPU half of the model for the Arm-CPU / GPU cooperative pipeline.

VITS splits cleanly at one line of `SynthesizerTrn.infer`:

    x, m_p, logs_p, x_mask = self.enc_p(...)      <-- text encoder
    logw = self.dp(...)                            <-- duration predictor
    ... attention / path expansion ...
    z = self.flow(z_p, y_mask, reverse=True)       <-- normalising flow
    o = self.dec((z * y_mask)[...])                <-- vocoder  <<< SPLIT HERE

Everything above the split is small, sequential and latency-bound: attention
over ~80 phonemes, a few thousand flow ops. That is Arm CPU work, and running
it under ONNX Runtime's fused graph is far faster than eager PyTorch -- the
earlier incarnation of this project measured ~6x on the same architecture,
which was the difference between the split winning and losing.

Everything below is dense convolution over tens of thousands of samples.
That is accelerator work.

This exports the prefix to ONNX. The decoder stays in PyTorch and runs on
CUDA, because ONNX Runtime in this environment ships no CUDA provider
(`get_available_providers()` -> ['AzureExecutionProvider',
'CPUExecutionProvider']). Checked, not assumed.

    python3 export_split.py --checkpoint <ckpt> --output-file exported/prefix.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .export_onnx import build_from_checkpoint  # noqa: E402
from .vits import commons  # noqa: E402

OPSET = 15


class EncoderPrefix(torch.nn.Module):
    """enc_p + duration predictor + flow, stopping before the vocoder.

    Returns the exact tensor the decoder consumes -- `z * y_mask` -- so the
    GPU side is a single `dec(z)` call with nothing to reconstruct.
    """

    def __init__(self, net_g):
        super().__init__()
        self.net_g = net_g

    def forward(self, input, input_lengths, scales):
        noise_scale, length_scale, noise_scale_w = scales[0], scales[1], scales[2]
        g = self.net_g
        x, m_p, logs_p, x_mask = g.enc_p(input, input_lengths)

        if g.use_sdp:
            logw = g.dp(x, x_mask, g=None, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = g.dp(x, x_mask, g=None)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = g.flow(z_p, y_mask, g=None, reverse=True)
        return z * y_mask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-file", required=True, type=Path)
    args = ap.parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    net_g, ckpt, voice_cfg = build_from_checkpoint(args.checkpoint)
    print(f"checkpoint: epoch {ckpt['epoch']}, step {ckpt['global_step']}")

    prefix = EncoderPrefix(net_g).eval()
    dummy_input = torch.randint(1, 40, (1, 60), dtype=torch.long)
    dummy_lengths = torch.LongTensor([dummy_input.size(1)])
    dummy_scales = torch.FloatTensor([0.667, 1.0, 0.8])

    with torch.no_grad():
        torch.onnx.export(
            prefix,
            (dummy_input, dummy_lengths, dummy_scales),
            str(args.output_file),
            input_names=["input", "input_lengths", "scales"],
            output_names=["z"],
            dynamic_axes={
                "input": {0: "batch", 1: "phonemes"},
                "input_lengths": {0: "batch"},
                "z": {0: "batch", 2: "frames"},
            },
            opset_version=OPSET,
            # The flow's rational-quadratic spline branches on data, which the
            # dynamo exporter cannot trace. Same constraint as the full-model
            # export.
            dynamo=False,
            do_constant_folding=True,
        )

    size_mb = args.output_file.stat().st_size / 1048576
    print(f"exported prefix: {args.output_file}  ({size_mb:.1f} MB)")

    # Verify the graph loads and emits a latent of the right width. The
    # decoder expects exactly inter_channels rows; anything else means the
    # split was taken at the wrong point and would fail only at runtime.
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.output_file), providers=["CPUExecutionProvider"])
    z = sess.run(None, {
        "input": dummy_input.numpy(),
        "input_lengths": dummy_lengths.numpy(),
        "scales": dummy_scales.numpy(),
    })[0]
    expected = net_g.dec.conv_pre.in_channels
    print(f"smoke test: z shape {z.shape}  (decoder expects {expected} channels)")
    if z.shape[1] != expected:
        print(f"ERROR: latent width {z.shape[1]} != decoder input {expected}")
        return 1
    if not np.isfinite(z).all():
        print("ERROR: latent contains NaN/Inf")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
