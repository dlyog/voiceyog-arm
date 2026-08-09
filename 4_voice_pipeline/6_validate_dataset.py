"""
Validate that a directory is a usable training dataset, BEFORE training starts.

    python3 1_SyntheticAudioDataset/6_validate_dataset.py <dir>

Training a bad dataset fails slowly and confusingly -- often thousands of steps
in, or worse, it trains happily on 40 clips and produces noise. Every check
here is one that has actually gone wrong: a metadata.csv with the wrong
delimiter, wav paths that include the folder prefix, a mix of sample rates,
stereo files, or clips long enough to teach the model to stop mid-phrase.

Expected layout -- the same one 5_generate_audio.py writes and the released
Hugging Face dataset uses:

    <dir>/metadata.csv        <wav filename>|<transcript>, one per line
    <dir>/wavs/000001.wav     24 kHz mono 16-bit PCM

Exit code 0 if usable, 1 if not. Importable as validate(dir) -> dict.
"""
from __future__ import annotations

import csv
import json
import sys
import wave
from pathlib import Path

SAMPLE_RATE = 24000
MAX_CLIP_SEC = 6.25
MIN_CLIPS = 50           # below this, training cannot produce anything usable
SAMPLE_CHECK = 200       # how many wavs to open for format/duration checks


def validate(root: str | Path) -> dict:
    root = Path(root).expanduser()
    errors: list[str] = []
    warnings: list[str] = []
    info: dict = {"path": str(root)}

    if not root.exists():
        return {**info, "ok": False, "errors": [f"{root} does not exist."],
                "warnings": [], "hint": "Pick a directory that exists."}
    if not root.is_dir():
        return {**info, "ok": False, "errors": [f"{root} is a file, not a directory."],
                "warnings": [], "hint": "Select the folder that contains metadata.csv."}

    csv_path = root / "metadata.csv"
    wav_dir = root / "wavs"

    if not csv_path.exists():
        found = [p.name for p in root.glob("*.csv")][:5]
        errors.append(
            "metadata.csv is missing." +
            (f" Found instead: {', '.join(found)}. Rename one to metadata.csv."
             if found else " The folder has no .csv file at all."))
    if not wav_dir.is_dir():
        stray = len(list(root.glob("*.wav")))
        errors.append(
            "There is no wavs/ subfolder." +
            (f" {stray} .wav files sit directly in this folder -- move them into "
             "wavs/, because metadata.csv holds bare filenames and the trainer "
             "resolves them against wavs/." if stray else ""))
    if errors:
        return {**info, "ok": False, "errors": errors, "warnings": warnings,
                "hint": "Expected: <dir>/metadata.csv and <dir>/wavs/*.wav"}

    # --- metadata.csv -------------------------------------------------------
    rows: list[tuple[str, str]] = []
    bad_lines: list[int] = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|")
            if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
                bad_lines.append(i)
                continue
            rows.append((parts[0].strip(), parts[1].strip()))

    if not rows:
        peek = csv_path.read_text(errors="replace").splitlines()[:1]
        errors.append(
            "metadata.csv has no usable rows. Expected '<wav>|<text>' per line" +
            (f"; the first line looks like: {peek[0][:90]!r}" if peek else "."))
        if peek and "," in peek[0] and "|" not in peek[0]:
            errors.append("It looks comma-separated. This format uses a PIPE '|' "
                          "separator, because transcripts contain commas.")
        return {**info, "ok": False, "errors": errors, "warnings": warnings,
                "hint": "One line per clip:  000001.wav|The text spoken."}

    if bad_lines:
        warnings.append(f"{len(bad_lines)} malformed line(s) skipped "
                        f"(first at line {bad_lines[0]}).")

    # --- do the referenced files exist? -------------------------------------
    prefixed = sum(1 for n, _ in rows[:50] if "/" in n)
    missing = [n for n, _ in rows if not (wav_dir / n).is_file()]
    if prefixed:
        errors.append(
            "metadata.csv contains paths like 'wavs/000001.wav'. It must hold "
            "BARE filenames ('000001.wav') -- the trainer joins them onto wavs/ "
            "itself, so a prefix resolves to wavs/wavs/...")
    elif missing:
        errors.append(
            f"{len(missing)} of {len(rows)} referenced files are missing from "
            f"wavs/ (e.g. {', '.join(missing[:3])}).")

    n_wavs = sum(1 for _ in wav_dir.glob("*.wav"))
    info.update(rows=len(rows), wavs_on_disk=n_wavs, missing=len(missing))

    if len(rows) < MIN_CLIPS:
        errors.append(f"Only {len(rows)} clips. Training needs at least "
                      f"{MIN_CLIPS} to produce anything, and ~3 hours to be good.")

    # --- audio format -------------------------------------------------------
    rates, chans, widths, durs, unreadable = set(), set(), set(), [], []
    # SPREAD THE SAMPLE OVER THE WHOLE CORPUS, NOT ITS FIRST PAGE.
    #
    # rows[:SAMPLE_CHECK] read the first 200 clips, and a corpus does not start
    # the way it continues: the openers are the word bank and CMU ARCTIC
    # prompts, which run longer than the LLM sentences that follow. Extrapolating
    # their mean over 6,760 clips reported ~4.932 h for a corpus holding 3.769 h
    # of audio -- a 31% overestimate, on the screen that says "ready to train".
    # An even stride costs nothing and sees the whole file.
    stride = max(1, len(rows) // SAMPLE_CHECK)
    for name, _ in rows[::stride][:SAMPLE_CHECK]:
        f = wav_dir / name
        if not f.is_file():
            continue
        try:
            with wave.open(str(f)) as w:
                rates.add(w.getframerate())
                chans.add(w.getnchannels())
                widths.add(w.getsampwidth())
                durs.append(w.getnframes() / w.getframerate())
        except Exception as exc:
            unreadable.append(f"{name}: {exc}")

    if unreadable:
        errors.append(f"{len(unreadable)} file(s) are not readable WAV audio "
                      f"(e.g. {unreadable[0]}).")
    if len(rates) > 1:
        errors.append(f"Mixed sample rates: {sorted(rates)}. Every clip must be "
                      f"the same rate; the trainer does not resample.")
    elif rates and next(iter(rates)) != SAMPLE_RATE:
        r = next(iter(rates))
        warnings.append(f"Sample rate is {r} Hz, not {SAMPLE_RATE}. Training will "
                        f"work, but set data.sample_rate to {r} in "
                        f"2_TrainOnDGX/config.json or the audio will be misread.")
    if chans - {1}:
        errors.append(f"Not all clips are mono (found channel counts {sorted(chans)}).")
    if widths - {2}:
        warnings.append(f"Expected 16-bit PCM; found sample widths {sorted(widths)} bytes.")

    if durs:
        total = sum(durs) / len(durs) * len(rows)     # extrapolate from the sample
        info.update(sample_rate=next(iter(rates)) if len(rates) == 1 else None,
                    est_hours=round(total / 3600, 3),
                    longest_sec=round(max(durs), 2),
                    mean_sec=round(sum(durs) / len(durs), 2))
        long = [d for d in durs if d > MAX_CLIP_SEC]
        if long:
            warnings.append(
                f"{len(long)} of the {len(durs)} sampled clips exceed "
                f"{MAX_CLIP_SEC}s (longest {max(durs):.2f}s). The released model "
                f"never saw anything longer; expect artifacts past that length.")
        if total < 1800:
            warnings.append(f"Only about {total/3600:.2f} h of audio. Quality "
                            f"tracks dataset size; the release used 3.0 h.")

    ok = not errors
    return {**info, "ok": ok, "errors": errors, "warnings": warnings,
            "hint": ("Dataset validated — ready to start training."
                     if ok else "Fix the errors above, then validate again.")}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    d = validate(sys.argv[1])
    print(json.dumps(d, indent=2))
    print()
    if d["ok"]:
        print(f"OK — {d.get('rows')} clips, ~{d.get('est_hours')} h, "
              f"{d.get('sample_rate')} Hz. Ready to train.")
    else:
        print("NOT USABLE:")
        for e in d["errors"]:
            print(f"  - {e}")
    for w in d["warnings"]:
        print(f"  warning: {w}")
    return 0 if d["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
