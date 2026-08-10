#!/usr/bin/env python3
"""
Predicted-MOS comparison: the distilled student against its own teacher.

Both engines speak the SAME 20 held-out sentences -- verified disjoint from all
4,104 training texts -- so the comparison is paired and every sentence appears
on both sides. Paired designs are what make an n of 20 worth reporting.

Scored with UTMOS22 (utmos22_strong via torch.hub, tarepan/SpeechMOS v1.2.0),
the VoiceMOS Challenge system. UTMOS is a PREDICTOR, not a measurement: it
assigns ~3.3 to pure silence, so absolute values carry little meaning and the
teacher-student delta is the quantity of interest.

Synthesis is stochastic (noise_w > 0), so the student is generated REPEATS
times and we report the spread across runs as well as across sentences.

    python3 mos_eval.py            # writes mos_eval_dgx_spark.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

# Paths resolve against this checkout, with environment overrides, so nothing is
# pinned to the machine this was first run on.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PIPELINE = REPO / "4_voice_pipeline"
EVIDENCE = REPO / "3_evidence"

ENGINE_DIR = Path(os.environ.get("VOICEYOG_ENGINE", PIPELINE))
SENTENCES = Path(os.environ.get("VOICEYOG_SENTENCES", PIPELINE / "tts" / "eval_sentences.txt"))
# The held-out check needs the training metadata, which is NOT in this repository
# -- it is an artifact of your own training run. Point at it, or the script says
# so and stops rather than quietly scoring on text the model may have seen.
TRAIN_META = Path(os.environ.get("VOICEYOG_TRAIN_META",
                                 Path.home() / ".voiceyog" / "datasets" / "metadata.csv"))
OUT_DIR = Path(os.environ.get("VOICEYOG_MOS_WORKDIR", HERE / "mos_audio"))


def _find_student() -> Path:
    if os.environ.get("VOICEYOG_MODEL"):
        return Path(os.environ["VOICEYOG_MODEL"])
    best = None
    for root in (Path.home() / ".voiceyog" / "models", REPO / "models"):
        if not root.is_dir():
            continue
        for c in root.rglob("*.onnx"):
            if "encoder_prefix" in c.name:
                continue
            if best is None or c.stat().st_mtime > best.stat().st_mtime:
                best = c
    if best is None:
        sys.exit("  fail  no installed model found. Run: bash manage.sh install\n"
                 "        or set VOICEYOG_MODEL=/path/to/model.onnx")
    return best


STUDENT_ONNX = _find_student()
REPEATS = 3
TEACHER_VOICE = "af_heart"

sys.path.insert(0, str(ENGINE_DIR))


def read_sentences() -> list[str]:
    return [l.strip() for l in SENTENCES.read_text(encoding="utf-8").splitlines() if l.strip()]


def assert_held_out(sents: list[str]) -> int:
    """Refuse to score on text the student was trained on."""
    train = {r.split("|")[1].strip()
             for r in TRAIN_META.read_text(encoding="utf-8").splitlines()
             if "|" in r}
    overlap = [s for s in sents if s in train]
    if overlap:
        raise SystemExit(f"  ABORT: {len(overlap)} eval sentences appear in training text")
    return len(train)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def gen_student(sents: list[str], run: int) -> list[Path]:
    from tts.engine import TTSModel
    m = TTSModel(str(STUDENT_ONNX))
    out = []
    for i, s in enumerate(sents):
        p = OUT_DIR / f"student_r{run}_{i:02d}.wav"
        write_wav(p, m.synthesize(s), m.sample_rate)
        out.append(p)
    return out


def gen_teacher(sents: list[str]) -> list[Path]:
    # Kokoro is deterministic for a fixed voice, and the device does not change
    # the samples, so the teacher is generated once on the GPU for speed.
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a", device="cuda")
    out = []
    for i, s in enumerate(sents):
        chunks = [(x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x))
                  for _, _, x in pipe(s, voice=TEACHER_VOICE)]
        p = OUT_DIR / f"teacher_{i:02d}.wav"
        write_wav(p, np.concatenate(chunks), 24000)
        out.append(p)
    return out


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """16-bit PCM reader.

    torchaudio 2.11 routes load() through TorchCodec, which is not installed
    here and which we are not going to add to somebody's working environment
    for a read we can do with the standard library. Every file scored below was
    written by write_wav() above, so the format is known: mono, 16-bit, PCM.
    """
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        ch = f.getnchannels()
        raw = f.readframes(f.getnframes())
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def score(paths: list[Path]) -> list[float]:
    import torch, torchaudio
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor.eval()
    scores = []
    with torch.no_grad():
        for p in paths:
            a, sr = read_wav(p)
            wav = torch.from_numpy(a).unsqueeze(0)      # [1, T], mono
            if sr != 16000:                             # UTMOS expects 16 kHz
                wav = torchaudio.functional.resample(wav, sr, 16000)
            scores.append(float(predictor(wav, sr=16000).squeeze()))
    return scores


def ci95(x: np.ndarray) -> tuple[float, float]:
    """Mean and half-width of the 95% CI (t-interval)."""
    from scipy import stats
    n = len(x)
    if n < 2:
        return float(x.mean()), float("nan")
    half = float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return float(x.mean()), half


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sents = read_sentences()
    n_train = assert_held_out(sents)
    print(f"\n  {len(sents)} held-out sentences, disjoint from {n_train} training texts")
    print(f"  student: {STUDENT_ONNX}")
    print(f"  teacher: Kokoro-82M, voice={TEACHER_VOICE}\n")

    teacher_paths = gen_teacher(sents)
    t = np.array(score(teacher_paths))

    runs = []
    for r in range(REPEATS):
        s = np.array(score(gen_student(sents, r)))
        runs.append(s)
        print(f"    student run {r + 1}: mean UTMOS {s.mean():.3f}")
    S = np.vstack(runs)
    s_mean_per_sentence = S.mean(axis=0)          # average the stochasticity out

    t_m, t_ci = ci95(t)
    s_m, s_ci = ci95(s_mean_per_sentence)

    from scipy import stats
    w = stats.wilcoxon(s_mean_per_sentence, t)     # paired, same sentences
    d = s_mean_per_sentence - t

    print(f"\n  teacher  UTMOS {t_m:.3f} +/- {t_ci:.3f}   (95% CI, n={len(t)})")
    print(f"  student  UTMOS {s_m:.3f} +/- {s_ci:.3f}")
    print(f"  delta    {d.mean():+.3f}  (student - teacher)")
    print(f"  Wilcoxon signed-rank: W={w.statistic:.1f}  p={w.pvalue:.4f}")
    print(f"  student run-to-run spread: {S.mean(axis=1).std(ddof=1):.4f}\n")

    rec = {
        "what": "UTMOS22 predicted MOS, distilled student vs its teacher, paired on identical held-out text",
        "predictor": {
            "name": "UTMOS22 (utmos22_strong)",
            "source": "torch.hub tarepan/SpeechMOS:v1.2.0",
            "caveat": "Neural predictor, not a human listening test. Scores ~3.3 on silence, "
                      "so absolute values are weak evidence; the paired delta is the quantity of interest.",
        },
        "text": {
            "n_sentences": len(sents),
            "held_out": True,
            "overlap_with_training_text": 0,
            "training_texts_checked": n_train,
        },
        "student": {
            "model": str(STUDENT_ONNX), "bytes": STUDENT_ONNX.stat().st_size,
            "repeats": REPEATS,
            "note": "synthesis is stochastic (noise_w > 0); per-sentence scores averaged over repeats",
            "utmos_mean": s_m, "utmos_ci95_halfwidth": s_ci,
            "per_run_means": [float(x) for x in S.mean(axis=1)],
        },
        "teacher": {
            "model": "Kokoro-82M", "voice": TEACHER_VOICE,
            "utmos_mean": t_m, "utmos_ci95_halfwidth": t_ci,
        },
        "paired_delta_student_minus_teacher": {
            "mean": float(d.mean()),
            "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
        },
        "per_sentence": [
            {"text": s, "teacher": float(a), "student": float(b)}
            for s, a, b in zip(sents, t, s_mean_per_sentence)
        ],
    }
    out = Path("/tmp/mos_eval_dgx_spark.json")
    out.write_text(json.dumps(rec, indent=2))
    print(f"  -> {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
