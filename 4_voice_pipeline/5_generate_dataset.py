#!/usr/bin/env python3
"""
Turn the prompt corpus into a training dataset, using a teacher model.

This is the step that decides WHOSE voice you end up with, and it is the only
step that changes between the two things this pipeline is for.

    # an AI voice -- Kokoro-82M speaks the corpus
    python3 5_generate_dataset.py --teacher kokoro --voice af_heart --hours 3.0

    # your own voice -- a clone service speaks the corpus in your voice,
    # conditioned on one recording of you
    python3 5_generate_dataset.py --teacher clone \\
        --ref my-voice.wav --api http://127.0.0.1:8005 --hours 3.0

Both write the same thing, because nothing downstream cares which teacher
produced the audio:

    output/wavs/000123.wav    24 kHz mono 16-bit
    output/metadata.csv       <wav>|<text>
    output/manifest.jsonl     per-clip detail, including duration

Resumable. Clips already on disk are skipped, so an interrupted run continues
where it stopped rather than starting again.

TEACHERS

  kokoro   Runs in-process from the `kokoro` pip package. Any of its voices
           works -- pass --voice. This is how the released model was built.

  clone    POSTs to a voice-cloning service that takes one reference recording
           and speaks arbitrary text in that voice. Qwen3-TTS is a free model
           that does this and is supported on DGX Spark; install it and follow
           the Qwen TTS guide on Hugging Face for the service itself. The
           reference recording is sent only to the URL you pass, which is
           expected to be a service you run.

WHY A HEAVY TEACHER MAKES A LIGHT STUDENT

The teacher runs once, on a GPU, to produce a few hours of audio. The student
trained on that audio is 68.5 MB and runs on a CPU forever after. You are
paying GPU time once to avoid needing a GPU ever again.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "output" / "corpus.jsonl"

SAMPLE_RATE = 24000
MAX_CLIP_SEC = 6.25          # the trainer's ceiling; longer clips teach it to stop mid-phrase


def die(msg: str) -> None:
    print(f"\n  fail  {msg}\n", file=sys.stderr)
    sys.exit(1)


def load_corpus() -> list[dict]:
    if not CORPUS.is_file():
        die(f"{CORPUS} not found.\n        Run:  python3 4_build_corpus.py")
    rows = []
    with CORPUS.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        die(f"{CORPUS} is empty")
    return rows


def write_wav(path: Path, samples, rate: int = SAMPLE_RATE) -> float:
    """float32 in [-1, 1] -> 16-bit PCM. Returns duration in seconds."""
    import numpy as np
    a = np.asarray(samples, dtype="float32").reshape(-1)
    pcm = (np.clip(a, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return len(a) / rate


# --- teachers ---------------------------------------------------------------
class KokoroTeacher:
    """Kokoro-82M, in-process. No server to stand up."""

    def __init__(self, voice: str, device: str):
        try:
            from kokoro import KPipeline
        except ImportError:
            die("the `kokoro` package is not installed.\n"
                "        pip install kokoro     (and espeak-ng on the system)")
        self.voice = voice
        self.pipe = KPipeline(lang_code="a", device=device)
        self.rate = 24000

    def say(self, text: str, dest: Path) -> float:
        import numpy as np
        chunks = []
        for _, _, audio in self.pipe(text, voice=self.voice):
            if audio is None:
                continue
            a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else \
                (audio.numpy() if hasattr(audio, "numpy") else audio)
            if getattr(a, "size", 0):
                chunks.append(a.reshape(-1))
        if not chunks:
            return 0.0
        return write_wav(dest, np.concatenate(chunks), self.rate)


class CloneTeacher:
    """A voice-cloning HTTP service: one reference recording in, any text out.

    curl rather than a Python HTTP client because this uploads a multipart wav
    on every call and curl is present on every machine this runs on, where
    `requests` may not be.
    """

    def __init__(self, api: str, ref: Path, language: str):
        if not ref.is_file():
            die(f"reference recording not found: {ref}")
        self.api, self.ref, self.language = api.rstrip("/"), ref, language
        try:
            subprocess.run(["curl", "-sf", "-m", "10", f"{self.api}/health"],
                           check=True, capture_output=True)
        except Exception:
            die(f"no clone service answering at {self.api}/health\n"
                "        Qwen3-TTS is a free model that provides this and is supported\n"
                "        on DGX Spark -- install it and follow the Qwen TTS guide on\n"
                "        Hugging Face, then pass its URL with --api.")

    def say(self, text: str, dest: Path) -> float:
        r = subprocess.run(
            ["curl", "-s", "-m", "900", "-X", "POST", f"{self.api}/clone",
             "-F", f"ref_audio=@{self.ref};type=audio/wav",
             "-F", f"text={text}",
             "-F", "ref_text=",
             # "English", not "en" -- the service rejects the short form.
             "-F", f"language={self.language}",
             "-o", str(dest), "-w", "%{http_code}"],
            capture_output=True, text=True)
        if r.stdout.strip() != "200":
            dest.unlink(missing_ok=True)
            return 0.0
        try:
            with wave.open(str(dest)) as w:
                return w.getnframes() / w.getframerate()
        except Exception:
            dest.unlink(missing_ok=True)
            return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", choices=["kokoro", "clone"], default="kokoro")
    ap.add_argument("--voice", default="af_heart", help="kokoro voice id")
    ap.add_argument("--device", default="cuda", help="kokoro device: cuda or cpu")
    ap.add_argument("--ref", type=Path, help="clone: one reference recording of the voice")
    ap.add_argument("--api", default="http://127.0.0.1:8005", help="clone: service base URL")
    ap.add_argument("--language", default="English", help="clone: language name")
    ap.add_argument("--hours", type=float, default=3.0, help="stop once this much audio exists")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (smoke test)")
    # A smoke test writing into the same directory as a real dataset silently
    # replaces its metadata.csv, and the next training run then trains on
    # twelve clips and produces a checkpoint that looks fine until you read the
    # loss. Separate output directories, by default one per teacher.
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: output/<teacher>)")
    args = ap.parse_args()

    out = args.out or (HERE / "output" / args.teacher)
    wav_dir = out / "wavs"
    metadata = out / "metadata.csv"
    manifest = out / "manifest.jsonl"

    corpus = load_corpus()
    wav_dir.mkdir(parents=True, exist_ok=True)

    if args.teacher == "kokoro":
        teacher = KokoroTeacher(args.voice, args.device)
        who = f"kokoro:{args.voice} on {args.device}"
    else:
        if not args.ref:
            die("--teacher clone needs --ref <a recording of the voice>")
        teacher = CloneTeacher(args.api, args.ref, args.language)
        who = f"clone via {args.api} from {args.ref.name}"

    print()
    print(f"  teacher   {who}")
    print(f"  corpus    {len(corpus)} sentences")
    print(f"  target    {args.hours:.2f} h" + (f", max {args.limit} clips" if args.limit else ""))
    print()

    rows, total, made, skipped, failed = [], 0.0, 0, 0, 0
    t0 = time.perf_counter()
    for i, row in enumerate(corpus, 1):
        if total >= args.hours * 3600 or (args.limit and made + skipped >= args.limit):
            break
        text = (row.get("text") or "").strip()
        if not text:
            continue
        name = f"{i:06d}.wav"
        dest = wav_dir / name

        if dest.is_file():                      # resumable
            try:
                with wave.open(str(dest)) as w:
                    secs = w.getnframes() / w.getframerate()
                rows.append({"wav": name, "text": text, "seconds": round(secs, 3)})
                total += secs; skipped += 1
                continue
            except Exception:
                dest.unlink(missing_ok=True)

        secs = teacher.say(text, dest)
        if secs <= 0:
            failed += 1
            continue
        if secs > MAX_CLIP_SEC:
            # Kept out of metadata rather than deleted, so it is obvious what
            # happened. The trainer's ceiling is not a suggestion: longer clips
            # teach the model to stop mid-phrase.
            dest.rename(dest.with_suffix(".toolong"))
            failed += 1
            continue

        rows.append({"wav": name, "text": text, "seconds": round(secs, 3)})
        total += secs; made += 1
        if made % 25 == 0:
            rate = made / max(time.perf_counter() - t0, 1e-9)
            print(f"    {made:>5} new · {total/3600:.2f} h · {rate:.1f} clips/s", flush=True)

    if not rows:
        die("no usable clips were produced")

    with metadata.open("w", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for r in rows:
            w.writerow([r["wav"], r["text"]])
    with manifest.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print()
    print(f"  {len(rows)} clips · {total/3600:.2f} h "
          f"({made} new, {skipped} already there, {failed} rejected)")
    print(f"  {metadata}")
    print()
    print(f"  next:  python3 6_validate_dataset.py {out}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
