"""
Synthesizes the corpus with Kokoro-82M to build a distillation dataset.

    python3 1_SyntheticAudioDataset/5_generate_audio.py [target_hours] [voice]

    python3 1_SyntheticAudioDataset/5_generate_audio.py 3.0 af_heart   # the release
    python3 1_SyntheticAudioDataset/5_generate_audio.py 3.0 am_michael # any other voice

THIS IS THE STEP THAT MAKES THE PIPELINE REUSABLE. Change `voice` and you get a
dataset for a different speaker; train on it and you have your own single-voice
model. Nothing downstream is specific to af_heart.

Kokoro runs IN-PROCESS from the `kokoro` pip package -- there is no TTS server
to stand up. An earlier version of this called a local HTTP service, which
meant the pipeline only ran on the machine that happened to host it.

Resumable: clips already present are skipped, so an interrupted run continues.

Outputs
    output/wavs/000123.wav   24 kHz mono 16-bit
    output/metadata.csv      <wav>|<text>, the format the trainer reads
    output/manifest.jsonl    per-clip detail, including duration
"""
import csv
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "output" / "corpus.jsonl"
WAV_DIR = HERE / "output" / "wavs"
MANIFEST = HERE / "output" / "manifest.jsonl"
METADATA = HERE / "output" / "metadata.csv"

SAMPLE_RATE = 24000
# Clips longer than this are dropped, not truncated. The model is trained on
# whole utterances; a cut-off sentence teaches it to stop mid-phrase.
MAX_CLIP_SEC = 6.25


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return len(pcm) / sample_rate


def load_corpus() -> list[dict]:
    if not CORPUS.exists():
        sys.exit(f"{CORPUS} not found. Run 4_build_corpus.py first.")
    with open(CORPUS) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_done() -> tuple[set[int], float]:
    if not MANIFEST.exists():
        return set(), 0.0
    ids, total = set(), 0.0
    with open(MANIFEST) as f:
        for line in f:
            r = json.loads(line)
            ids.add(r["id"])
            total += r["duration_sec"]
    return ids, total


def generate(target_hours: float = 3.0, voice: str = "af_heart") -> float:
    try:
        from kokoro import KPipeline          # Apache-2.0
    except ImportError:
        sys.exit("kokoro is not installed:  pip install kokoro")

    corpus = load_corpus()
    done, cumulative = load_done()
    target_sec = target_hours * 3600
    if done:
        print(f"[resume] {len(done)} clips present, {cumulative/3600:.2f} h so far")

    pipe = KPipeline(lang_code="a", device="cpu")
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(MANIFEST, "a")

    t0, made, skipped = time.time(), 0, 0
    for row in corpus:
        if cumulative >= target_sec:
            break
        if row["id"] in done:
            continue
        try:
            chunks = [np.asarray(a, dtype=np.float32) for _, _, a in
                      pipe(row["text"], voice=voice)]
            audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
        except Exception as exc:
            print(f"[warn] id={row['id']} failed: {exc}")
            continue
        if audio.size == 0:
            continue

        dur = audio.shape[0] / SAMPLE_RATE
        if dur > MAX_CLIP_SEC:
            skipped += 1
            continue

        path = WAV_DIR / f"{row['id']:06d}.wav"
        write_wav(path, audio)
        fh.write(json.dumps({"id": row["id"], "text": row["text"],
                             "source": row["source"], "audio_path": path.name,
                             "duration_sec": dur, "voice": voice}) + "\n")
        fh.flush()
        cumulative += dur
        made += 1
        if made % 100 == 0:
            print(f"[progress] {made} this run, {cumulative/3600:.2f} h "
                  f"/ {target_hours} h, {(time.time()-t0)/60:.1f} min elapsed",
                  flush=True)
    fh.close()

    # metadata.csv is what 2_TrainOnDGX reads: <wav>|<text>, one clip per line.
    rows = [json.loads(l) for l in open(MANIFEST) if l.strip()]
    with open(METADATA, "w", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for r in rows:
            w.writerow([r["audio_path"], r["text"]])

    total = sum(r["duration_sec"] for r in rows)
    print(f"\n=== done ===")
    print(f"clips: {len(rows)} total ({made} this run"
          + (f", {skipped} dropped for exceeding {MAX_CLIP_SEC}s" if skipped else "") + ")")
    print(f"audio: {total/3600:.3f} h   voice: {voice}")
    print(f"wavs:     {WAV_DIR}")
    print(f"metadata: {METADATA}   <- point the trainer at this")
    return total


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    v = sys.argv[2] if len(sys.argv) > 2 else "af_heart"
    generate(hours, v)
