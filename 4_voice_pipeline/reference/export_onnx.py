"""
Export a GPL-free training checkpoint to ONNX.

The exported graph deliberately matches piper's interface exactly:

    inputs   input          int64  [1, T]   phoneme ids
             input_lengths  int64  [1]
             scales         fp32   [3]      noise_scale, length_scale, noise_w
    output   output         fp32   [1, 1, S]

That is not cosmetic. Everything downstream in this project already speaks
that interface -- tts/engine.py, both web apps, the encoder-prefix split for the
CPU/accelerator pipeline, and the benchmark harness. Matching it means a model
trained here is a drop-in replacement for the shipped one, with no code
changes anywhere else.

The model's own `infer()` takes the three scales as separate Python floats, so
a thin wrapper unpacks them from the tensor. Without the wrapper the scales
would be traced as constants and the exported model would ignore whatever the
caller passes -- it would still run, still produce audio, and silently ignore
your inference settings.

    python3 export_onnx.py --checkpoint output_mit_3hr/checkpoints/last.pt \
        --output-file exported/af_heart_v3.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

from .vits.models import SynthesizerTrn  # noqa: E402

OPSET = 15


def _resolve_spec(spec_path: str) -> Path:
    """Find the phoneme spec a checkpoint was trained against.

    Checkpoints embed this path ABSOLUTELY, as it stood when training ran. The
    released weights later moved out of the checkout and into the app home, so
    every checkpoint written before that move names a file that is no longer
    there -- and an 834 MB checkpoint cannot be regenerated in under a day.

    The spec is identical for every clone, so locating it by name in the
    current weights directory is safe. The alternative is that a directory
    rename permanently bricks every checkpoint written before it.
    """
    p = Path(spec_path)
    if p.is_file():
        return p
    home = Path(os.environ.get("VOICEYOG_HOME", str(Path.home() / ".voiceyog")))
    repo = Path(__file__).resolve().parents[2]
    for cand in (home / "models" / "released" / p.name, repo / "models" / p.name):
        if cand.is_file():
            print(f"  spec {p} not found; using {cand}", flush=True)
            return cand
    raise FileNotFoundError(
        f"phoneme spec {spec_path!r} is named by the checkpoint but does not "
        f"exist, and no {p.name} was found in {home / 'models' / 'released'} "
        f"or {repo / 'models'}. Run download_models.sh to fetch the released "
        f"weights.")


class InferenceWrapper(torch.nn.Module):
    """Adapts SynthesizerTrn.infer to piper's tensor-in/tensor-out signature.

    Emits ONLY the audio. It is tempting to also expose `z` here and drop
    encoder_prefix.onnx, since that file is a subgraph of this one and
    duplicates ~54 MB of the same weights. That was built and MEASURED, and it
    does not work: ONNX Runtime fixes its execution plan when the session is
    created, not per call, so asking a two-output model for `z` alone still
    runs the whole graph --

        standalone encoder_prefix.onnx -> z      12.17 ms
        merged two-output model        -> z     146.59 ms
        merged two-output model        -> audio 146.59 ms

    -- which would run the vocoder on the CPU as well as the GPU and destroy
    the point of the split. The separate prefix export is what makes the
    cooperative path fast, and the duplicated weights are its price.
    (The merged `z` was verified bit-identical to the prefix's, so the idea was
    sound; only ORT's execution model rules it out.)
    """

    def __init__(self, net_g: SynthesizerTrn):
        super().__init__()
        self.net_g = net_g

    def forward(self, input, input_lengths, scales):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = self.net_g.infer(
            input,
            input_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
        )[0]
        return audio


def build_from_checkpoint(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m, d, t = cfg["model"], cfg["data"], cfg["train"]

    # `phoneme_config` since the rename; `voice_config` before it. Checkpoints
    # embed the config they were trained with, so old ones carry the old key and
    # must keep loading -- an 834 MB checkpoint is not reproducible on demand.
    spec_path = d.get("phoneme_config") or d["voice_config"]
    voice_cfg = json.loads(_resolve_spec(spec_path).read_text())
    n_symbols = max(max(v) for v in voice_cfg["phoneme_id_map"].values()) + 1

    net_g = SynthesizerTrn(
        n_symbols,
        d["filter_length"] // 2 + 1,
        t["segment_size"] // d["hop_length"],
        inter_channels=m["inter_channels"],
        hidden_channels=m["hidden_channels"],
        filter_channels=m["filter_channels"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        kernel_size=m["kernel_size"],
        p_dropout=m["p_dropout"],
        resblock=m["resblock"],
        resblock_kernel_sizes=m["resblock_kernel_sizes"],
        resblock_dilation_sizes=m["resblock_dilation_sizes"],
        upsample_rates=m["upsample_rates"],
        upsample_initial_channel=m["upsample_initial_channel"],
        upsample_kernel_sizes=m["upsample_kernel_sizes"],
        n_speakers=m.get("n_speakers", 0),
        gin_channels=m.get("gin_channels", 0),
        use_sdp=m.get("use_sdp", True),
    ).to(device)

    net_g.load_state_dict(ckpt["net_g"])
    net_g.eval()

    # weight_norm is a training-time reparameterisation. Folding it away
    # before export removes a pile of ops from the graph and is what piper
    # does too, so the exported sizes stay comparable.
    net_g.dec.remove_weight_norm()

    return net_g, ckpt, voice_cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-file", required=True, type=Path)
    ap.add_argument("--voice-config", type=Path, default=None,
                    help="defaults to the one recorded in the checkpoint")
    args = ap.parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    net_g, ckpt, voice_cfg = build_from_checkpoint(args.checkpoint)
    print(f"checkpoint: epoch {ckpt['epoch']}, step {ckpt['global_step']}, "
          f"train_mel {ckpt['train_mel']:.4f}")

    model = InferenceWrapper(net_g)
    model.eval()

    # A realistic dummy: 50 phoneme ids. Dynamic axes make the real length
    # free, but tracing with length 1 would fold away the sequence logic.
    dummy_input = torch.randint(1, 40, (1, 50), dtype=torch.long)
    dummy_lengths = torch.LongTensor([dummy_input.size(1)])
    dummy_scales = torch.FloatTensor([0.667, 1.0, 0.8])

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input, dummy_lengths, dummy_scales),
            str(args.output_file),
            input_names=["input", "input_lengths", "scales"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 1: "phonemes"},
                "input_lengths": {0: "batch"},
                "output": {0: "batch", 1: "channel", 2: "samples"},
            },
            opset_version=OPSET,
            # The flow's rational-quadratic spline has data-dependent
            # branching the dynamo exporter cannot trace. Same reason piper
            # pins the legacy exporter here.
            dynamo=False,
            do_constant_folding=True,
        )

    # piper's convention: the .onnx must sit beside a matching .onnx.json with
    # the same base name, or PiperVoice-style loaders cannot find the phoneme
    # map or sample rate.
    _d = ckpt["config"]["data"]
    src_cfg = args.voice_config or _resolve_spec(
        _d.get("phoneme_config") or _d["voice_config"])
    dst_cfg = Path(f"{args.output_file}.json")
    shutil.copy(src_cfg, dst_cfg)

    size_mb = args.output_file.stat().st_size / 1048576
    print(f"exported: {args.output_file}  ({size_mb:.1f} MB)")
    print(f"config:   {dst_cfg}")

    # Prove the graph loads and produces audio of a sane length, rather than
    # trusting that a successful export means a working model.
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.output_file),
                                    providers=["CPUExecutionProvider"])
        out = sess.run(None, {
            "input": dummy_input.numpy(),
            "input_lengths": dummy_lengths.numpy(),
            "scales": dummy_scales.numpy(),
        })[0]
        secs = out.shape[-1] / voice_cfg["audio"]["sample_rate"]
        print(f"smoke test: {out.shape} -> {secs:.2f}s of audio at "
              f"{voice_cfg['audio']['sample_rate']} Hz")
        if not np.isfinite(out).all():
            print("WARNING: output contains NaN/Inf")
            return 1
    except Exception as exc:
        print(f"WARNING: exported, but the smoke test failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
