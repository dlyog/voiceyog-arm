"""
In-loop validation: synthesize held-out sentences, transcribe, score WER.

Why WER rather than a predicted MOS
-----------------------------------
The piper runs tracked `val_mos` (UTMOS, a learned MOS predictor). That metric
proved unreliable here: it saturates near the top, is documented to rank
strong models poorly, and was trained on human recordings while this model
produces distilled synthetic speech -- out of its domain. At one point it fell
(4.3155 -> 4.2708) while every other signal improved.

WER via an ASR round-trip is objective and has no such failure mode: the model
either produced words the recogniser can read back or it did not. It measures
intelligibility only -- it will happily score buzzy audio 0.00 -- so it is a
floor check, not a quality verdict. Combined with the training losses it is
enough to see whether an epoch helped.

The sentences are the same fixed, hand-written, held-out set used everywhere
else in this project, so numbers here are directly comparable to the ones
measured for the shipped model.
"""
from __future__ import annotations

import io
import re
import wave

import numpy as np
import requests
import torch

# Empty by default: in-loop WER validation is optional and the previous
# default pointed at the author's own server, which nobody else can reach.
DEFAULT_WHISPER_URL = ""


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, divided by reference length."""
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(prev[j - 1] if r == h else 1 + min(prev[j], cur[j - 1], prev[j - 1]))
        prev = cur
    return prev[-1] / len(ref)


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcribe(wav: bytes, url: str, timeout: int = 60) -> str:
    resp = requests.post(
        url,
        headers={"Authorization": "Bearer dummy_api_key"},
        files={"file": ("clip.wav", wav, "audio/wav")},
        data={"model": "base", "language": "en", "response_format": "verbose_json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["text"]


@torch.no_grad()
def validate_wer(net_g, phonemizer, phoneme_id_map, sentences, sample_rate,
                 device, url: str = DEFAULT_WHISPER_URL,
                 noise_scale: float = 0.0, length_scale: float = 1.0,
                 noise_w: float = 0.6) -> dict:
    """Synthesize each sentence, transcribe it, return WER statistics.

    Runs in eval mode and restores training mode afterwards -- forgetting that
    leaves dropout disabled for the rest of the run, which trains a subtly
    different model than intended.
    """
    was_training = net_g.training
    net_g.eval()

    pad = phoneme_id_map["_"]
    results, rows, failures = [], [], 0

    for text in sentences:
        ids = list(phoneme_id_map["^"]) + list(pad)
        for ph in phonemizer.phonemize(text):
            for ch in ph:
                mapped = phoneme_id_map.get(ch)
                if mapped is None:
                    continue
                ids.extend(mapped)
                ids.extend(pad)
        ids.extend(phoneme_id_map["$"])

        x = torch.LongTensor(ids).unsqueeze(0).to(device)
        x_len = torch.LongTensor([x.size(1)]).to(device)
        audio = net_g.infer(x, x_len, noise_scale=noise_scale,
                            length_scale=length_scale,
                            noise_scale_w=noise_w)[0]
        audio = audio.squeeze().float().cpu().numpy()

        try:
            heard = transcribe(_wav_bytes(audio, sample_rate), url)
        except Exception:
            # A validation outage must not take the training run down with it.
            failures += 1
            continue
        # KEEP WHAT WAS HEARD, not just the rate.
        #
        # This appended the bare number and dropped `heard` on the floor, so a
        # WER that sits at exactly 0.0361 for thirty epochs is unexplainable:
        # nobody can tell whether three sentences are genuinely mispronounced or
        # whether the recogniser writes "350" where the prompt says "three
        # hundred fifty" and `_normalize` above counts that as three word
        # errors. Those are opposite conclusions -- one says keep training, the
        # other says the metric bottomed out and the number is an artifact.
        rate = word_error_rate(text, heard)
        results.append(rate)
        rows.append({"said": text, "heard": heard.strip(), "wer": round(rate, 4)})

    if was_training:
        net_g.train()

    if not results:
        return {"mean_wer": None, "n": 0, "perfect": 0, "failures": failures,
                "rows": rows}
    return {
        "mean_wer": sum(results) / len(results),
        "n": len(results),
        "perfect": sum(1 for r in results if r == 0.0),
        "failures": failures,
        # Worst first: the ones worth reading are the ones that failed.
        "rows": sorted(rows, key=lambda r: -r["wer"]),
    }
