"""
Dataset for piper-format corpora, with GPL-free phonemization.

The VITS reference ships `data_utils.py` expecting its own filelist format and
a linked phonemizer. This replaces it with:

  * piper's `metadata.csv`  ->  <wav filename>|<transcript>
  * espeak-ng via subprocess (see tts.engine.Phonemizer) -- a separate process,
    so no linking and no GPL propagation
  * an on-disk cache, because phonemizing 18,853 clips through subprocesses
    once per epoch would dominate training time

Phoneme ids come from the voice config's `phoneme_id_map`, so a model trained
here is interchangeable with one trained by piper: same symbol inventory, same
BOS/PAD/EOS layout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from scipy.io.wavfile import read as wavread

from ..engine import Phonemizer
from .mel_processing import spectrogram_torch


def load_wav(path: str, expected_sr: int) -> torch.FloatTensor:
    sr, data = wavread(path)
    if sr != expected_sr:
        raise ValueError(f"{path}: sample rate {sr} != expected {expected_sr}")
    # 16-bit PCM -> [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    return torch.from_numpy(data)


class TextAudioDataset(torch.utils.data.Dataset):
    """Returns (phoneme_ids, spectrogram, waveform) per clip.

    Spectrograms and phoneme ids are cached to disk on first use. The cache
    key includes every parameter that changes their value, so a config change
    cannot silently reuse stale features -- that class of bug is expensive
    here because it produces a model that trains on the wrong targets.
    """

    def __init__(self, csv_path, audio_dir, config_path, cache_dir,
                 sample_rate=24000, filter_length=1024, hop_length=256,
                 win_length=1024, espeak_voice="en-us",
                 min_seconds=0.5, max_seconds=12.0):
        self.audio_dir = Path(audio_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length

        cfg = json.loads(Path(config_path).read_text())
        self.phoneme_id_map = cfg["phoneme_id_map"]
        self.phonemizer = Phonemizer(espeak_voice)

        # Anything that changes the cached tensors must be in the key -- and
        # that INCLUDES WHICH DATASET THIS IS.
        #
        # An earlier version keyed only on audio parameters, so two different
        # corpora sharing a sample rate and hop length produced the same tag.
        # Entries are named <tag>_<row index>, so row 0 of the second dataset
        # silently loaded row 0 of the first: an entire training run completed
        # on the wrong audio, with a plausible loss curve and no error at any
        # point. Identity is now part of the key, and including the size and
        # mtime of metadata.csv means editing a corpus in place also
        # invalidates it.
        # The AUDIO is fingerprinted too, not just the csv. Regenerating a
        # corpus in place -- same sentences, same filenames, different voice --
        # leaves metadata.csv byte-identical and its mtime untouched, so a
        # csv-only key would serve the previous voice's cached tensors and
        # train on it silently. That is the original bug wearing a new hat, and
        # Dataset Studio makes it routine: preview a voice, dislike it, pick
        # another candidate, regenerate into the same folder.
        #
        # Count, total bytes and newest mtime over the wavs -- one directory
        # scan, no file reads. It cannot detect a same-size same-mtime edit,
        # which no realistic generator produces.
        csv = Path(csv_path)
        try:
            st = csv.stat()
            ident = (f"{csv.resolve()}|{st.st_size}|{int(st.st_mtime)}"
                     f"|{Path(audio_dir).resolve()}")
        except OSError:
            ident = f"{csv}|{audio_dir}"
        try:
            n = 0
            total = 0
            newest = 0
            for w in Path(audio_dir).glob("*.wav"):
                ws = w.stat()
                n += 1
                total += ws.st_size
                newest = max(newest, int(ws.st_mtime))
            ident += f"|wavs:{n}:{total}:{newest}"
        except OSError:
            # A missing audio dir is a real error, but it belongs to the
            # loading code with a useful message -- not to key construction.
            ident += "|wavs:unavailable"
        self.cache_tag = hashlib.sha1(
            f"{sample_rate}|{filter_length}|{hop_length}|{win_length}|"
            f"{espeak_voice}|{len(self.phoneme_id_map)}|{ident}".encode()
        ).hexdigest()[:12]

        rows = []
        for line in Path(csv_path).read_text().splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            wav, text = line.split("|", 1)
            rows.append((wav.strip(), text.strip()))

        # Length filtering keeps a pathological clip from blowing up memory
        # and keeps the batch shapes sane.
        self.rows = rows
        self.min_samples = int(min_seconds * sample_rate)
        self.max_samples = int(max_seconds * sample_rate)
        self._lengths: list[int] | None = None

    # -- phonemes ---------------------------------------------------------
    def _phoneme_ids(self, text: str) -> list[int]:
        pad = self.phoneme_id_map["_"]
        ids = list(self.phoneme_id_map["^"])
        ids.extend(pad)
        for ph in self.phonemizer.phonemize(text):
            for ch in ph:
                mapped = self.phoneme_id_map.get(ch)
                if mapped is None:
                    continue
                ids.extend(mapped)
                ids.extend(pad)
        ids.extend(self.phoneme_id_map["$"])
        return ids

    def _cache_path(self, idx: int) -> Path:
        return self.cache_dir / f"{self.cache_tag}_{idx:06d}.pt"

    def __getitem__(self, idx: int):
        cache = self._cache_path(idx)
        if cache.exists():
            try:
                d = torch.load(cache, weights_only=True)
                return d["ids"], d["spec"], d["wav"]
            except Exception:
                cache.unlink(missing_ok=True)  # corrupt: regenerate

        wav_name, text = self.rows[idx]
        wav = load_wav(str(self.audio_dir / wav_name), self.sample_rate)
        wav = wav.unsqueeze(0)
        spec = spectrogram_torch(
            wav, self.filter_length, self.sample_rate,
            self.hop_length, self.win_length, center=False,
        ).squeeze(0)
        ids = torch.LongTensor(self._phoneme_ids(text))

        torch.save({"ids": ids, "spec": spec, "wav": wav}, cache)
        return ids, spec, wav

    def __len__(self) -> int:
        return len(self.rows)

    def audio_lengths(self) -> list[int]:
        """Frame counts, for length-bucketed batching."""
        if self._lengths is None:
            self._lengths = []
            for wav_name, _ in self.rows:
                p = self.audio_dir / wav_name
                # WAV header: bytes 40..44 hold the data chunk size.
                try:
                    with open(p, "rb") as f:
                        f.seek(40)
                        n = int.from_bytes(f.read(4), "little") // 2
                except OSError:
                    n = 0
                self._lengths.append(n // self.hop_length)
        return self._lengths


class TextAudioCollate:
    """Pad a batch to the longest member, longest-first.

    Longest-first ordering is required by the reference model's use of
    packed sequences; getting it wrong produces a shape error rather than a
    silent bug, but it is worth stating.
    """

    def __call__(self, batch):
        order = torch.argsort(
            torch.LongTensor([x[1].size(1) for x in batch]), descending=True
        )

        max_ids = max(len(x[0]) for x in batch)
        max_spec = max(x[1].size(1) for x in batch)
        max_wav = max(x[2].size(1) for x in batch)

        ids_len = torch.LongTensor(len(batch))
        spec_len = torch.LongTensor(len(batch))
        wav_len = torch.LongTensor(len(batch))

        ids_pad = torch.LongTensor(len(batch), max_ids).zero_()
        spec_pad = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec).zero_()
        wav_pad = torch.FloatTensor(len(batch), 1, max_wav).zero_()

        for i, src in enumerate(order):
            ids, spec, wav = batch[src]
            ids_pad[i, : ids.size(0)] = ids
            ids_len[i] = ids.size(0)
            spec_pad[i, :, : spec.size(1)] = spec
            spec_len[i] = spec.size(1)
            wav_pad[i, :, : wav.size(1)] = wav
            wav_len[i] = wav.size(1)

        return ids_pad, ids_len, spec_pad, spec_len, wav_pad, wav_len


class LengthBucketSampler(torch.utils.data.Sampler):
    """Group similar-length clips into batches.

    Without this, a batch pads to its longest member and most of the compute
    goes into padding. On this corpus (clips 1.45-6.25 s) that is roughly a
    30% waste, which is the difference between finishing overnight and not.
    """

    def __init__(self, lengths, batch_size, shuffle=True, seed=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self._build()

    def _build(self):
        order = np.argsort(self.lengths, kind="stable")
        self.batches = [
            order[i : i + self.batch_size].tolist()
            for i in range(0, len(order), self.batch_size)
        ]
        # Drop a trailing partial batch: BatchNorm-free here, but uneven final
        # batches interact badly with the segment-slicing below.
        if self.batches and len(self.batches[-1]) < self.batch_size:
            self.batches.pop()

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)
