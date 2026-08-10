#!/usr/bin/env python3
"""
Intelligibility of the shipped model, measured rather than asserted.

Speed and size say nothing about whether the words survive. This synthesizes the
held-out sentences with the model these packages install, transcribes the audio
back with Whisper, and scores the transcript against the text that produced it.

Word error rate answers a different question from a MOS predictor. MOS asks how
natural it sounds; WER asks whether the sentence arrives. A distilled model can
give up polish and still be perfectly intelligible, and the two numbers together
say which of those happened.

    python3 8_wer_cer.py

That is the whole procedure. On first run it builds a private virtualenv beside
this file and installs Whisper into it, then re-executes itself there -- your
system Python and the serving environment are left exactly as they were. Nothing
is written outside this repository except the pip cache.

Writes ../3_evidence/wer_cer_<platform>.json, which verify_claims.py reads, so
the figure in the write-up is checked against this run rather than typed in.

    python3 8_wer_cer.py --whisper base.en     # faster, slightly less accurate
    python3 8_wer_cer.py --no-bootstrap        # use the current interpreter
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import unicodedata
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVIDENCE = REPO / "3_evidence"
PIPELINE = REPO / "4_voice_pipeline"
SENTENCES = PIPELINE / "tts" / "eval_sentences.txt"
VENV = HERE / ".venv-wer"
BOOTSTRAPPED = "VOICEYOG_WER_BOOTSTRAPPED"


# --------------------------------------------------------------------------
# Bootstrap: one command, no instructions to follow
# --------------------------------------------------------------------------

def bootstrap() -> None:
    """Build a venv with Whisper in it and re-exec there.

    --system-site-packages on purpose: torch is a multi-gigabyte download on
    aarch64 and a DGX Spark almost always has one already. If it does, pip
    reuses it and this takes a minute; if not, pip fetches it and takes longer.
    Either way nothing is installed into an environment that was already there.
    """
    py = VENV / "bin" / "python"
    if not py.exists():
        print(f"  building {VENV.name} (first run only)")
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(VENV)],
                       check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    # Everything the run needs: Whisper to transcribe, onnxruntime and numpy to
    # synthesize. Inheriting site-packages covers these on a machine that already
    # serves the model, but a bare interpreter has neither -- and discovering that
    # after Whisper has downloaded is a poor way to find out.
    needed = {"whisper": "openai-whisper", "onnxruntime": "onnxruntime", "numpy": "numpy"}
    missing = [pkg for mod, pkg in needed.items()
               if subprocess.run([str(py), "-c", f"import {mod}"],
                                 capture_output=True).returncode != 0]
    if missing:
        print(f"  installing {', '.join(missing)} (this is the slow part)")
        subprocess.run([str(py), "-m", "pip", "install", "-q", *missing], check=True)

    env = dict(os.environ, **{BOOTSTRAPPED: "1"})
    os.execve(str(py), [str(py), str(Path(__file__).resolve()), *sys.argv[1:]], env)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def normalize(s: str) -> str:
    """Fold away everything an ASR system cannot be expected to reproduce.

    Whisper chooses its own casing and punctuation, and neither is part of what
    a TTS model got right or wrong. So: lowercase, strip accents, drop
    punctuation except the apostrophes that separate real words ("were" from
    "we're"), collapse whitespace. Deliberately simple, and written into the
    evidence file, because a WER means nothing unless the reader knows which
    normalizer produced it.
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def levenshtein(a: list, b: list) -> int:
    """Edit distance, so this stage needs no jiwer and no wheel that might not
    build on aarch64."""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def words(s: str) -> list[str]:
    return normalize(s).split()


def chars(s: str) -> list[str]:
    return list(normalize(s).replace(" ", ""))


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------

def _ort_version() -> str:
    try:
        import onnxruntime
        return onnxruntime.__version__
    except Exception:
        return "unknown"


def write_wav(path: Path, audio, sr: int) -> None:
    import numpy as np
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def find_model(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    # Whatever manage.sh installed, newest first. encoder_prefix.onnx is the
    # split-graph half, not a servable model, so it never qualifies.
    best = None
    for root in (Path.home() / ".voiceyog" / "models", REPO / "models"):
        if not root.is_dir():
            continue
        for p in root.rglob("*.onnx"):
            if "encoder_prefix" in p.name:
                continue
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
    if best is None:
        sys.exit("  fail  no installed model found. Run: bash manage.sh install")
    return best


def synthesize(model_path: Path, sentences: list[str], out_dir: Path):
    sys.path.insert(0, str(PIPELINE))
    try:
        from tts.engine import TTSModel
    except ImportError as e:
        sys.exit(f"  fail  cannot import tts.engine ({e})")
    m = TTSModel(str(model_path))
    paths = []
    for i, s in enumerate(sentences):
        p = out_dir / f"utt_{i:02d}.wav"
        write_wav(p, m.synthesize(s), m.sample_rate)
        paths.append(p)
    return paths, m.sample_rate


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whisper", default="small.en", help="ASR model (default small.en)")
    ap.add_argument("--model", default=None, help="path to the .onnx to score")
    ap.add_argument("--out", default=None, help="evidence json to write")
    ap.add_argument("--audio-dir", default=str(HERE / "wer_audio"))
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="use the current interpreter instead of building a venv")
    args = ap.parse_args()

    if not args.no_bootstrap and not os.environ.get(BOOTSTRAPPED):
        try:
            import whisper  # noqa: F401
        except ImportError:
            bootstrap()     # never returns

    try:
        import whisper
    except ImportError:
        sys.exit("  fail  whisper unavailable; drop --no-bootstrap to install it")

    sentences = [l.strip() for l in SENTENCES.read_text(encoding="utf-8").splitlines() if l.strip()]
    model_path = find_model(args.model)
    audio_dir = Path(args.audio_dir); audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  model      {model_path}")
    print(f"  sentences  {len(sentences)} held-out")
    print(f"  whisper    {args.whisper}\n")

    wavs, sr = synthesize(model_path, sentences, audio_dir)
    asr = whisper.load_model(args.whisper)

    rows, exact = [], 0
    for text, wav in zip(sentences, wavs):
        # Language pinned, temperature 0: transcription should be a measurement,
        # not a sample from a distribution.
        hyp = asr.transcribe(str(wav), language="en", temperature=0.0,
                             fp16=False)["text"].strip()
        we = levenshtein(words(text), words(hyp))
        ce = levenshtein(chars(text), chars(hyp))
        if normalize(text) == normalize(hyp):
            exact += 1
        rows.append({"reference": text, "heard_as": hyp,
                     "word_errors": we, "ref_words": len(words(text)),
                     "char_errors": ce, "ref_chars": len(chars(text)),
                     "wer": we / max(len(words(text)), 1)})
        print(f"    {we / max(len(words(text)), 1):6.4f}  {hyp[:62]}")

    # Corpus WER: total edits over total reference words, NOT the mean of
    # per-sentence rates. Averaging rates lets a three-word sentence outweigh a
    # twenty-word one, which is how a good-looking number gets manufactured.
    wer = sum(r["word_errors"] for r in rows) / sum(r["ref_words"] for r in rows)
    cer = sum(r["char_errors"] for r in rows) / sum(r["ref_chars"] for r in rows)

    plat = ("dgx_spark" if (platform.system() == "Linux" and platform.machine() == "aarch64")
            else "m1_max" if platform.system() == "Darwin" else platform.machine())
    out = Path(args.out) if args.out else EVIDENCE / f"wer_cer_{plat}.json"
    out.write_text(json.dumps({
        "what": "ASR intelligibility of the shipped model: synthesize held-out text, transcribe it back, score the transcript",
        "asr": {"engine": "openai-whisper", "model": args.whisper,
                "language": "en", "temperature": 0.0},
        "normalizer": "NFKD, strip accents, lowercase, keep [a-z0-9'] and spaces, collapse whitespace",
        "wer_definition": "total word edits / total reference words (corpus WER, not the mean of per-sentence rates)",
        "model": {"path": str(model_path), "bytes": model_path.stat().st_size,
                  "sample_rate": sr, "onnxruntime": _ort_version()},
        "text": {"n_sentences": len(sentences), "held_out": True,
                 "source": "4_voice_pipeline/tts/eval_sentences.txt"},
        "wer": wer, "cer": cer, "exact_transcripts": exact, "n": len(rows),
        "per_sentence": rows,
        "platform": {"machine": platform.machine(), "system": platform.system()},
    }, indent=2))

    print(f"\n  WER {wer:.4f}   CER {cer:.4f}   exact {exact}/{len(rows)}")
    print(f"  -> {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
