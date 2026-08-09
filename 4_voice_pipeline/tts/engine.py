"""
GPL-free inference for the af_heart ONNX model.

Why this exists
---------------
The rest of this project used piper1-gpl's `PiperVoice` for inference. piper is
GPL-3.0, and a Python `import` is linking, so any code that imports it is
subject to GPL terms. That is fine for training (which is not distributed) but
not for the inference code that ships.

This module replaces `PiperVoice` with ~150 lines depending only on:

  onnxruntime   MIT
  numpy         BSD-3-Clause
  espeak-ng     GPL-3.0, but invoked as a SEPARATE PROCESS

That last point matters and is easy to get wrong. GPL propagates through
linking, not through running a program. Calling the `espeak-ng` binary via
subprocess is the standard "separate programs" case and does not make this
module a derivative work. What must be avoided is the Python bindings
(piper-phonemize, espeakng-loader, phonemizer) that load espeak's shared
library into this process -- those DO link.

Everything else the model needs is data we own: `af_heart.onnx.json` carries
the sample rate, the phoneme-to-id map and the inference scales.

Equivalence with piper is not assumed -- `4_VerifyCorrectness/compare_with_piper.py` checks
phoneme ids and audio length on the fixed eval set. Three real bugs were found
that way and are called out inline below. Each of them still produced audio
that sounded like speech, which is exactly why "it runs" is not evidence of
correctness here.

Usage
-----
    from tts.engine import TTSModel

    tts = TTSModel("af_heart.onnx", "af_heart.onnx.json")
    tts.synthesize_to_wav("Hello there.", "out.wav")

    for audio in tts.stream("First sentence. Second sentence."):
        play(audio)          # first sentence plays while the rest generates
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
import wave
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import onnxruntime as ort

__all__ = ["TTSModel", "Phonemizer", "split_sentences", "write_wav"]

# piper's convention, carried in the voice config's phoneme_id_map.
_PAD = "_"
_BOS = "^"
_EOS = "$"

# BUG 1 (found by QA, not by inspection): espeak marks ties with different
# characters depending on version -- combining ties U+0361 / U+035C, and ZERO
# WIDTH JOINER U+200D for tied affricates and diphthongs. None are in the
# phoneme map. Missing U+200D cost a phoneme per tie, producing a shorter id
# sequence than piper's for the same text.
_TIE_CHARS = "͜͡‍‌️"

# Punctuation the model was trained with. espeak strips it from its phoneme
# output, but it is present in the phoneme_id_map and carries intonation.
_KEPT_PUNCT = ".,!?;:"
# Capturing split, so the punctuation survives in the parts list and can be
# spliced back at its original position.
_PUNCT_SPLIT = re.compile(r"([" + re.escape(_KEPT_PUNCT) + r"])")


class Phonemizer:
    """Text to espeak-ng IPA phonemes, via subprocess.

    A process per sentence is wasteful for very long text but keeps the GPL
    dependency at arm's length, which is the entire point of this module.
    """

    def __init__(self, voice: str = "en-us", binary: str | None = None):
        # piper links espeak-ng 1.52; distro packages are often 1.51, and the
        # two disagree on liaison and function-word reduction. ESPEAK_NG_BIN
        # lets a matching build be used without changing code.
        self.voice = voice
        self.binary = binary or os.environ.get("ESPEAK_NG_BIN", "espeak-ng")
        self._env = dict(os.environ)
        lib = os.environ.get("ESPEAK_NG_LIB")
        if lib:
            self._env["LD_LIBRARY_PATH"] = lib + os.pathsep + self._env.get("LD_LIBRARY_PATH", "")
        self._check()

    def _check(self) -> None:
        try:
            subprocess.run([self.binary, "--version"], capture_output=True,
                           check=True, env=self._env)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"'{self.binary}' not found or not runnable. Install it:\n"
                "  apt-get install espeak-ng"
            ) from exc

    def _run(self, text: str) -> str:
        proc = subprocess.run(
            [self.binary, "-q", "-x", "--ipa=3", "-v", self.voice],
            input=text, capture_output=True, text=True, check=True, env=self._env,
        )
        out = " ".join(proc.stdout.split())
        # espeak can emit (lang) switch markers around foreign words. They are
        # not phonemes and are not in the id map; piper strips them too.
        return re.sub(r"\([^)]+\)", "", out)

    def _phonemize_one(self, sentence: str) -> str:
        """Phonemize a single sentence, preserving punctuation in place.

        BUG 2 (found by QA): espeak DELETES punctuation from its phoneme
        output, but the model was trained with it -- the punctuation
        characters are in the phoneme_id_map and carry intonation. Restoring
        only the sentence-final mark is not enough: an internal comma matters
        too. piper produces

            wˈaʊ, aɪ dɪdnˌɑːt ...

        for "Wow, I did not ...", so the comma sits directly after the
        preceding phoneme.

        There is no espeak flag that does this. `--punct` was tested and
        speaks the punctuation as a word ("kˈɑːmə"), which is worse than
        dropping it. So the text is split on punctuation, each fragment is
        phonemized, and the punctuation is spliced back at its original
        position.
        """
        parts = _PUNCT_SPLIT.split(sentence)
        out: list[str] = []
        for part in parts:
            if not part:
                continue
            if part in _KEPT_PUNCT:
                out.append(part)
                continue
            stripped = part.strip()
            if not stripped:
                continue
            # Preserve whether there was whitespace before this fragment, so
            # "wˈaʊ, aɪ" keeps its space and "wˈaʊ," keeps none.
            if out and part[0].isspace():
                out.append(" ")
            out.append(self._run(stripped))
        return "".join(out)

    def phonemize(self, text: str) -> list[str]:
        """One IPA string per sentence, with punctuation preserved in place."""
        sentences = split_sentences(text)
        if not sentences:
            return []
        # One espeak call per fragment rather than one per document: espeak
        # can fold or drop lines when given several at once, and a silent
        # misalignment would pair one sentence's text with another's audio.
        return [self._phonemize_one(s) for s in sentences]


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries.

    Not cosmetic: the model was trained on clips of at most ~6.25 s and
    produces audible artifacts past that in a single pass. Synthesis is done
    per sentence and the audio concatenated.
    """
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def _performance_core_count() -> int | None:
    """Count the fast cores on an asymmetric Arm CPU.

    Uses Linux's `cpu_capacity`, which is the signal the scheduler itself
    uses for big.LITTLE-style layouts. Measured on GB10:

        Cortex-A725  capacity 718-731   max 2.81 GHz   (efficiency)
        Cortex-X925  capacity 997-1024  max 3.90 GHz   (performance)

    An earlier version of this function counted cores per MIDR and took the
    largest group as "performance". That was wrong in principle and only gave
    the right answer here by luck -- this machine has exactly 10 of each, so
    max() picked one arbitrarily. Arm Performix reporting the real topology
    (5+5 of each type across two clusters) is what exposed it.

    Falls back to max frequency, then to None (let ONNX Runtime decide) on
    a kernel that exposes neither.
    """
    import glob

    def _read_int(path: str) -> int | None:
        try:
            return int(Path(path).read_text().strip())
        except Exception:
            return None

    caps = {}
    for f in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpu_capacity"):
        v = _read_int(f)
        if v:
            caps[f] = v
    if not caps:
        for f in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
            v = _read_int(f)
            if v:
                caps[f] = v
    if len(caps) < 2:
        return None

    top = max(caps.values())
    # Within 10% of the fastest counts as a performance core: capacities are
    # not identical across clusters (997 vs 1024 here) even for the same core.
    return sum(1 for v in caps.values() if v >= top * 0.9)


def _tuned_thread_count() -> int | None:
    """Threads to use for ONNX Runtime on this machine.

    Measured on GB10 (10x Cortex-X925 + 10x Cortex-A725), sweeping 1..20
    threads on the real model:

        default (ORT picks)  182.21 ms
        6 threads             99.78 ms
        9 threads             92.07 ms   <- best, 1.98x over default
        10 threads           128.41 ms
        20 threads (all)     155.83 ms

    Using every core is 1.7x WORSE than using nine. A batch finishes when its
    slowest thread does, so spilling onto the efficiency cluster makes
    everything wait. ONNX Runtime's default does not know this.

    One below the performance-core count leaves a core for the OS and the
    phonemizer subprocess, which is what measured best.

    Override with num_threads= or ORT_INTRA_OP_THREADS.
    """
    override = os.environ.get("ORT_INTRA_OP_THREADS")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    perf = _performance_core_count()
    if perf and perf >= 2:
        return perf - 1
    return None


class TTSModel:
    """ONNX inference for a piper-format VITS voice, without piper."""

    def __init__(
        self,
        onnx_path: str | Path,
        config_path: str | Path | None = None,
        num_threads: int | None = None,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        auto_threads: bool = True,
    ):
        onnx_path = Path(onnx_path)
        config_path = Path(config_path) if config_path else Path(f"{onnx_path}.json")
        cfg = json.loads(config_path.read_text())

        self.sample_rate: int = cfg["audio"]["sample_rate"]
        self.phoneme_id_map: dict[str, list[int]] = cfg["phoneme_id_map"]
        inf = cfg.get("inference", {})
        self.noise_scale: float = float(inf.get("noise_scale", 0.667))
        self.length_scale: float = float(inf.get("length_scale", 1.0))
        self.noise_w: float = float(inf.get("noise_w", 0.8))

        so = ort.SessionOptions()
        if num_threads is None and auto_threads:
            num_threads = _tuned_thread_count()
        if num_threads:
            so.intra_op_num_threads = num_threads
        self.num_threads = num_threads
        self.session = ort.InferenceSession(str(onnx_path), so, providers=list(providers))
        self.phonemizer = Phonemizer(cfg.get("espeak", {}).get("voice", "en-us"))

    def phonemes_to_ids(self, phonemes: str) -> list[int]:
        """Interleave PAD between every token, VITS-style.

        BUG 3 (found by QA): the exact layout is

            BOS, PAD, p1, PAD, p2, PAD, ..., pN, PAD, EOS

        Note the PAD immediately AFTER BOS. Omitting it shifts every token by
        one position and yields a sequence two shorter than piper's -- which
        still runs and still sounds like speech, just subtly wrong. This is
        why the QA script compares ids rather than only checking that audio
        comes out.
        """
        pad = self.phoneme_id_map[_PAD]
        ids: list[int] = list(self.phoneme_id_map[_BOS])
        ids.extend(pad)
        # NFD decomposition, matching piper: it splits precomposed characters
        # into base + combining marks, and the id map contains the marks as
        # separate entries. Without this, an accented phoneme is looked up as
        # one composed character, missed, and silently dropped.
        for ch in unicodedata.normalize("NFD", phonemes):
            if ch in _TIE_CHARS:
                continue
            mapped = self.phoneme_id_map.get(ch)
            if mapped is None:
                # Unknown phoneme: skip rather than crash. espeak's inventory
                # can drift slightly from the map the model was trained with.
                continue
            ids.extend(mapped)
            ids.extend(pad)
        ids.extend(self.phoneme_id_map[_EOS])
        return ids

    def _infer(self, ids: list[int]) -> np.ndarray:
        arr = np.expand_dims(np.array(ids, dtype=np.int64), 0)
        lengths = np.array([arr.shape[1]], dtype=np.int64)
        scales = np.array(
            [self.noise_scale, self.length_scale, self.noise_w], dtype=np.float32
        )
        audio = self.session.run(
            None, {"input": arr, "input_lengths": lengths, "scales": scales}
        )[0]
        return np.asarray(audio).squeeze()

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Yield float32 audio per sentence, as soon as each is ready.

        This is what makes time-to-first-audio a fraction of total synthesis
        time: the caller can play sentence one while sentence two generates.
        Total compute is unchanged.
        """
        for phonemes in self.phonemizer.phonemize(text):
            ids = self.phonemes_to_ids(phonemes)
            if len(ids) <= 3:
                continue
            audio = self._infer(ids)
            if audio.size:
                yield audio.astype(np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        chunks = list(self.stream(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def synthesize_to_wav(self, text: str, out_path: str | Path) -> float:
        audio = self.synthesize(text)
        write_wav(out_path, audio, self.sample_rate)
        return audio.shape[-1] / self.sample_rate


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python3 tts/engine.py <model.onnx> <out.wav> [text]")
        raise SystemExit(1)
    model, out = sys.argv[1], sys.argv[2]
    txt = sys.argv[3] if len(sys.argv) > 3 else "The weather changed suddenly this afternoon."
    tts = TTSModel(model)
    dur = tts.synthesize_to_wav(txt, out)
    print(f"{dur:.2f}s of audio -> {out}  (no GPL code in this path)")
