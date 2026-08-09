"""
GPL-free VITS training.

Replaces `python3 -m piper.train fit`. Every import here is MIT, BSD or
Apache-2.0:

  tts/vits/*            adapted from jaywalnut310/vits (MIT) and
                        jik876/hifi-gan (MIT) -- see tts/vits/LICENSE-*
  tts/vits/monotonic_align.py   rewritten to drop the Cython build step,
                        verified identical to the compiled reference by
                        4_VerifyCorrectness/monotonic_align.py
  tts/vits/discriminators.py    MRD, written here on torch primitives
  tts.engine.Phonemizer    espeak-ng as a SEPARATE PROCESS (no linking)
  torch, numpy, scipy   BSD

Checkpoints are written in a format the ONNX exporter understands, and the
phoneme inventory matches piper's, so a model trained here is interchangeable
with the shipped one.

    python3 train.py --config config.json
    python3 train.py --config config.json --resume auto
"""
from __future__ import annotations

import sys
from pathlib import Path

# repo root on the path so `tts` imports as a package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "3_ExportAndInferenceEngine"))  # the tts package lives there

import argparse
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tts.vits import commons
from tts.vits.data import LengthBucketSampler, TextAudioCollate, TextAudioDataset
from tts.vits.discriminators import MultiResolutionDiscriminator
from tts.vits.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from tts.vits.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from tts.vits.models import MultiPeriodDiscriminator, SynthesizerTrn
from tts.vits.validate import validate_wer


def build_models(cfg, n_symbols, device):
    m = cfg["model"]
    net_g = SynthesizerTrn(
        n_symbols,
        cfg["data"]["filter_length"] // 2 + 1,
        cfg["train"]["segment_size"] // cfg["data"]["hop_length"],
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

    net_d = MultiPeriodDiscriminator(m.get("use_spectral_norm", False)).to(device)
    net_mrd = (
        MultiResolutionDiscriminator().to(device) if m.get("use_mrd", True) else None
    )
    return net_g, net_d, net_mrd


def param_report(net_g, net_d, net_mrd):
    """The startup oracle.

    A config that reads correctly can still build the wrong architecture --
    `resblock` is compared as a string, so an int silently selects the weaker
    block. The parameter count is the fastest way to know what was actually
    built, and it prints before any training happens. Read it before walking
    away.
    """
    dec = sum(p.numel() for p in net_g.dec.parameters())
    tot_g = sum(p.numel() for p in net_g.parameters())
    tot_d = sum(p.numel() for p in net_d.parameters())
    tot_m = sum(p.numel() for p in net_mrd.parameters()) if net_mrd else 0
    block = type(net_g.dec.resblocks[0]).__name__
    print("=" * 62)
    print(f"  generator          {tot_g/1e6:8.2f} M")
    print(f"    of which vocoder {dec/1e6:8.2f} M   ({block})")
    print(f"  MPD                {tot_d/1e6:8.2f} M")
    print(f"  MRD                {tot_m/1e6:8.2f} M" if net_mrd else "  MRD                  disabled")
    print(f"  TOTAL trainable    {(tot_g+tot_d+tot_m)/1e6:8.2f} M")
    print("=" * 62, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--resume", default=None,
                    help="'auto' for newest checkpoint, or a path")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())

    # Paths in config.json are RELATIVE TO THE REPOSITORY ROOT and resolved
    # here. They used to be absolute paths into the author's own workspace,
    # so training could not start from a fresh clone at all.
    repo = Path(__file__).resolve().parents[1]
    for section, keys in (("data", ("csv_path", "audio_dir", "phoneme_config",
                                    "voice_config", "cache_dir")),
                          ("train", ("output_dir", "val_sentences"))):
        for k in keys:
            v = cfg.get(section, {}).get(k)
            if isinstance(v, str) and v and not Path(v).is_absolute():
                cfg[section][k] = str(repo / v)

    # `phoneme_config` was called `voice_config`, which misdescribed it: the file
    # holds the espeak phoneme set, id map, sample rate and hop length, and is
    # used unchanged for EVERY voice this pipeline trains. Nothing in it selects
    # a voice. The old key is still accepted so configs and checkpoints written
    # before the rename keep working.
    if "phoneme_config" not in cfg.get("data", {}) and "voice_config" in cfg.get("data", {}):
        cfg["data"]["phoneme_config"] = cfg["data"]["voice_config"]

    out_dir = Path(cfg["train"]["output_dir"])
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no CUDA device. Training on CPU is not viable at this scale.")
    torch.manual_seed(cfg["train"].get("seed", 1234))

    # -- data ----------------------------------------------------------------
    d = cfg["data"]
    ds = TextAudioDataset(
        csv_path=d["csv_path"], audio_dir=d["audio_dir"],
        config_path=d["phoneme_config"], cache_dir=d["cache_dir"],
        sample_rate=d["sample_rate"], filter_length=d["filter_length"],
        hop_length=d["hop_length"], win_length=d["win_length"],
        espeak_voice=d.get("espeak_voice", "en-us"),
    )
    print(f"dataset: {len(ds)} clips from {d['csv_path']}", flush=True)

    sampler = LengthBucketSampler(ds.audio_lengths(), cfg["train"]["batch_size"])
    loader = DataLoader(
        ds, batch_sampler=sampler, collate_fn=TextAudioCollate(),
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True, persistent_workers=True,
    )
    print(f"{len(sampler)} batches/epoch at batch_size={cfg['train']['batch_size']}",
          flush=True)

    # Held-out sentences for the in-loop WER check. Same fixed set used for
    # every other model in this project, so the numbers are comparable.
    t = cfg["train"]
    val_sentences = []
    vs_path = t.get("val_sentences")
    if vs_path and Path(vs_path).exists():
        val_sentences = [l.strip() for l in Path(vs_path).read_text().splitlines() if l.strip()]
        print(f"validation: {len(val_sentences)} held-out sentences, "
              f"WER every {t.get('val_every_epochs', 10)} epochs", flush=True)
    else:
        print("validation: no val_sentences configured -- WER will not be tracked", flush=True)

    n_symbols = max(max(v) for v in ds.phoneme_id_map.values()) + 1
    net_g, net_d, net_mrd = build_models(cfg, n_symbols, device)
    param_report(net_g, net_d, net_mrd)

    opt_g = torch.optim.AdamW(net_g.parameters(), t["learning_rate"],
                              betas=t["betas"], eps=t["eps"])
    d_params = list(net_d.parameters()) + (list(net_mrd.parameters()) if net_mrd else [])
    opt_d = torch.optim.AdamW(d_params, t["learning_rate_d"],
                              betas=t["betas_d"], eps=t["eps"])

    start_epoch, global_step = 0, 0
    ckpt_path = _resolve_resume(args.resume, out_dir)
    if ckpt_path:
        print(f"RESUMING from {ckpt_path}", flush=True)
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        net_g.load_state_dict(st["net_g"])
        net_d.load_state_dict(st["net_d"])
        if net_mrd and st.get("net_mrd"):
            net_mrd.load_state_dict(st["net_mrd"])
        opt_g.load_state_dict(st["opt_g"])
        opt_d.load_state_dict(st["opt_d"])
        start_epoch, global_step = st["epoch"] + 1, st["global_step"]

    sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, t["lr_decay"], last_epoch=start_epoch - 1)
    sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, t["lr_decay"], last_epoch=start_epoch - 1)

    # bf16 rather than fp16: same exponent range as fp32, so GAN training does
    # not need a gradient scaler and cannot silently overflow.
    amp_dtype = torch.bfloat16 if t.get("bf16", True) else torch.float32
    use_amp = device.type == "cuda" and amp_dtype == torch.bfloat16

    print(f"starting at epoch {start_epoch}, step {global_step}", flush=True)
    for epoch in range(start_epoch, t["epochs"]):
        sampler.set_epoch(epoch)
        net_g.train(); net_d.train()
        if net_mrd:
            net_mrd.train()
        t0 = time.time()
        run = {"mel": 0.0, "kl": 0.0, "gen": 0.0, "disc": 0.0, "n": 0}

        for batch in loader:
            x, x_len, spec, spec_len, y, y_len = [b.to(device, non_blocking=True) for b in batch]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                (y_hat, l_length, _attn, ids_slice, _x_mask, z_mask,
                 (_z, z_p, m_p, logs_p, _m_q, logs_q)) = net_g(x, x_len, spec, spec_len)

                mel = spec_to_mel_torch(
                    spec.float(), d["filter_length"], d["n_mel_channels"],
                    d["sample_rate"], d["mel_fmin"], d["mel_fmax"])
                y_mel = commons.slice_segments(
                    mel, ids_slice, t["segment_size"] // d["hop_length"])
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1), d["filter_length"], d["n_mel_channels"],
                    d["sample_rate"], d["hop_length"], d["win_length"],
                    d["mel_fmin"], d["mel_fmax"])
                y_sliced = commons.slice_segments(y, ids_slice * d["hop_length"], t["segment_size"])

            # -- discriminators --------------------------------------------
            # The discriminator forwards must run INSIDE autocast too. Outside
            # it, y_hat arrives as bfloat16 while the discriminator weights are
            # still fp32, and conv1d refuses to mix them. Losses are computed
            # in fp32 (autocast handles the cast back) and backward runs
            # outside -- bf16 has fp32's exponent range, so no GradScaler.
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                y_dr, y_dg, _, _ = net_d(y_sliced, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_dr, y_dg)
                if net_mrd:
                    r_dr, r_dg, _, _ = net_mrd(y_sliced, y_hat.detach())
                    loss_disc = loss_disc + discriminator_loss(r_dr, r_dg)[0]
            opt_d.zero_grad(set_to_none=True)
            loss_disc.float().backward()
            commons.clip_grad_value_(d_params, None)
            opt_d.step()

            # -- generator --------------------------------------------------
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                y_dr, y_dg, fmap_r, fmap_g = net_d(y_sliced, y_hat)
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, _ = generator_loss(y_dg)
                if net_mrd:
                    r_dr, r_dg, r_fmap_r, r_fmap_g = net_mrd(y_sliced, y_hat)
                    loss_fm = loss_fm + feature_loss(r_fmap_r, r_fmap_g)
                    loss_gen = loss_gen + generator_loss(r_dg)[0]
            # Reconstruction and KL terms in fp32: they are the ones that
            # actually determine convergence, and they are cheap.
            loss_mel = F.l1_loss(y_mel.float(), y_hat_mel.float()) * t["c_mel"]
            loss_dur = torch.sum(l_length.float())
            loss_kl = kl_loss(z_p.float(), logs_q.float(), m_p.float(),
                              logs_p.float(), z_mask.float()) * t["c_kl"]
            loss_g = loss_gen.float() + loss_fm.float() + loss_mel + loss_dur + loss_kl

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            commons.clip_grad_value_(net_g.parameters(), None)
            opt_g.step()

            run["mel"] += loss_mel.item(); run["kl"] += loss_kl.item()
            run["gen"] += loss_gen.item(); run["disc"] += loss_disc.item()
            run["n"] += 1
            global_step += 1

            if global_step % t.get("log_every", 50) == 0:
                n = max(run["n"], 1)
                print(f"epoch {epoch} step {global_step}  "
                      f"mel {run['mel']/n:.4f}  kl {run['kl']/n:.4f}  "
                      f"gen {run['gen']/n:.4f}  disc {run['disc']/n:.4f}  "
                      f"{run['n']/(time.time()-t0):.2f} it/s", flush=True)

        sch_g.step(); sch_d.step()
        n = max(run["n"], 1)
        print(f"[epoch {epoch}] mel={run['mel']/n:.4f} kl={run['kl']/n:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

        wer = None
        if val_sentences and (epoch + 1) % t.get("val_every_epochs", 10) == 0:
            v = validate_wer(net_g, ds.phonemizer, ds.phoneme_id_map, val_sentences,
                             d["sample_rate"], device,
                             url=t.get("whisper_url", ""))
            if v["mean_wer"] is not None:
                wer = v["mean_wer"]
                print(f"[epoch {epoch}] VAL wer={wer:.4f} "
                      f"perfect={v['perfect']}/{v['n']}"
                      + (f" ({v['failures']} transcription failures)" if v["failures"] else ""),
                      flush=True)
            else:
                print(f"[epoch {epoch}] VAL unavailable "
                      f"({v['failures']} transcription failures)", flush=True)

        if (epoch + 1) % t.get("save_every_epochs", 5) == 0:
            _save(out_dir, epoch, global_step, net_g, net_d, net_mrd, opt_g, opt_d,
                  cfg, run["mel"] / n, wer)


def _resolve_resume(resume, out_dir: Path):
    if not resume:
        return None
    if resume != "auto":
        p = Path(resume)
        if not p.exists():
            raise SystemExit(f"--resume points at a missing file: {p}")
        return p
    ck = sorted((out_dir / "checkpoints").glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not ck:
        # Fail loudly rather than silently starting from random weights and
        # discarding however many hours are already on disk.
        raise SystemExit("--resume auto, but no checkpoint found. "
                         "Drop --resume to start fresh.")
    return ck[-1]


def _save(out_dir, epoch, step, net_g, net_d, net_mrd, opt_g, opt_d, cfg, mel, wer=None):
    tag = f"epoch{epoch:04d}_step{step}_mel{mel:.4f}"
    if wer is not None:
        tag += f"_wer{wer:.4f}"
    p = out_dir / "checkpoints" / f"{tag}.pt"
    torch.save({
        "epoch": epoch, "global_step": step,
        "net_g": net_g.state_dict(), "net_d": net_d.state_dict(),
        "net_mrd": net_mrd.state_dict() if net_mrd else None,
        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        "config": cfg, "train_mel": mel, "val_wer": wer,
    }, p)
    # A straight file copy. This was torch.load(p) then torch.save(), which
    # deserialises and reserialises 834 MB to produce a file identical to the
    # one just written -- pure waste, and it doubles under per-epoch saving.
    latest = out_dir / "checkpoints" / "last.pt"
    shutil.copyfile(p, latest)
    print(f"saved {p}", flush=True)
    keep = cfg["train"].get("keep_best_checkpoints",
                            cfg["train"].get("keep_last_checkpoints", 10))
    _rotate(out_dir, keep)


# epoch / step / mel / optional validation wer, from the filename written above.
# The `_wer` suffix is why this cannot end at `mel([\d.]+)\.pt$`: a greedy
# `[\d.]+` stops at the underscore and the anchor then fails, which silently
# made every validated checkpoint unparseable elsewhere in this project.
_CKPT = re.compile(r"^epoch(\d+)_step(\d+)_mel([\d.]+?)(?:_wer([\d.]+))?\.pt$")


def _rotate(out_dir, keep: int) -> None:
    """Keep the `keep` BEST checkpoints by training mel -- not the newest ones.

    KEEPING THE NEWEST DISCARDS THE BEST MODEL. Rotation sorted by mtime, so a
    worse later epoch evicted a better earlier one and the best weights of a run
    could be deleted while it was still training. On the 167-epoch run the three
    lowest-loss epochs (165, 166, 167) were never written at all, because saving
    happened only every fifth epoch and the run ended before 169 -- so the best
    model of that run existed nowhere on disk. Saving every epoch fixes that
    half; ranking by loss fixes this half.

    Two things survive eviction regardless of rank:

    * `last.pt` -- the resume point; the glob never matches it anyway.
    * the best checkpoint carrying a validation WER. Validation runs only every
      `val_every_epochs`, so at most one checkpoint in ten has a measured WER;
      ranking purely on mel would delete every one of them and leave nothing
      whose intelligibility was ever actually checked.

    Each checkpoint is ~834 MB -- both discriminators and both optimiser states,
    not just the generator -- so `keep` is a real disk decision: 10 is ~8.3 GB.
    """
    if keep is None or keep <= 0:
        return
    scored, unparsed = [], []
    for f in (out_dir / "checkpoints").glob("epoch*.pt"):
        m = _CKPT.match(f.name)
        if m is None:
            # Never delete a file we cannot judge. Rotating on an unrecognised
            # name would mean deleting checkpoints for being unfamiliar.
            unparsed.append(f)
            continue
        scored.append({"path": f, "epoch": int(m.group(1)),
                       "mel": float(m.group(3)),
                       "wer": float(m.group(4)) if m.group(4) else None})
    if unparsed:
        print("  rotation left alone (unrecognised names): "
              + ", ".join(f.name for f in unparsed), flush=True)
    if len(scored) <= keep:
        return

    # Best loss first. Ties go to the later epoch -- same loss, more training.
    scored.sort(key=lambda c: (c["mel"], -c["epoch"]))
    protected = {id(c) for c in scored[:keep]}
    validated = [c for c in scored if c["wer"] is not None]
    if validated:
        protected.add(id(min(validated, key=lambda c: (c["wer"], c["mel"]))))

    for rank, c in enumerate(scored, 1):
        if id(c) in protected:
            continue
        try:
            c["path"].unlink()
            print(f"  rotated out {c['path'].name} "
                  f"(mel {c['mel']:.4f}, rank {rank} of {len(scored)})", flush=True)
        except OSError as exc:
            print(f"  could not remove {c['path'].name}: {exc}", flush=True)


if __name__ == "__main__":
    main()
