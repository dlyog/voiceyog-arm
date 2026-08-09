"""
Optional post-processing for synthesised audio.

WHAT THIS IS FOR

A vocoder's output can carry broadband hiss and low-frequency rumble that sit
underneath otherwise clean speech. This module applies conventional signal
processing to that output. It changes nothing about the model -- it is a filter
on the waveform the model already produced.

MEASUREMENT INTEGRITY

Post-processing time is measured and reported SEPARATELY from synthesis time,
and RTF continues to be computed from synthesis alone. Folding a denoiser into
the synthesis timer would quietly inflate every benchmark number in this
project, and those numbers are the submission.

STAGES, IN ORDER, EACH INDEPENDENTLY DISABLEABLE

  1. high-pass      remove rumble below ~70 Hz, under the vocal range
  2. denoise        spectral gating (noisereduce), estimating the noise floor
                    from the signal itself
  3. de-ess         gentle shelf on 5-9 kHz sibilance, off by default
  4. normalise      peak-normalise to a fixed headroom so A/B comparison is not
                    confounded by loudness -- a louder clip is reliably judged
                    "better" regardless of quality

HONEST LIMITS

Spectral gating removes STATIONARY noise: hiss, hum, a constant noise floor. It
does not remove artefacts that are correlated with the speech itself -- vocoder
buzz, metallic ringing on voiced frames -- because those are not separable from
the signal by a stationary noise estimate. If what remains after denoising still
sounds wrong, the problem is in the model or its training audio, and no amount
of filtering here will fix it.

Aggressive settings introduce their own damage: musical noise (isolated
short-lived tones) and a hollow, underwater timbre. Defaults here are
deliberately conservative.
"""
from __future__ import annotations

import time

import numpy as np

# noisereduce is optional. Its absence disables one stage rather than breaking
# synthesis -- this is an enhancement, never a dependency of the audio path.
try:
    import noisereduce as _nr
    HAVE_NR = True
except Exception:                                    # pragma: no cover
    _nr = None
    HAVE_NR = False

try:
    from scipy.signal import butter, sosfilt, sosfiltfilt
    HAVE_SCIPY = True
except Exception:                                    # pragma: no cover
    HAVE_SCIPY = False


DEFAULTS = {
    # Values below were chosen by listening test on this model, not by taste.
    # Each was A/B'd against its neighbours; see the notes per line.
    "enabled": False,        # opt-in: the unprocessed path stays the default
    "mode": "auto",          # "auto" = gate on measurement; "always" = force

    # 130 Hz, not the 70 Hz first used. The vocoder's noise measures loudest at
    # 0-300 Hz (-28 dB, only 15 dB below speech) and falls away above 2 kHz.
    # 130 sits under a 156 Hz male fundamental, so it cuts rumble not voice.
    "highpass": True,
    "highpass_hz": 130.0,
    "highpass_auto": True,   # place the cutoff from the measured F0

    # Strength 1.0 is safe ONLY because the profile is learned (below). With
    # blind estimation this value produced musical noise and a hollow timbre.
    "denoise": True,
    "strength": 1.0,
    "stationary": True,
    "noise_print": True,     # learn from a silent stretch -- the Audacity/RX method

    "deess": False,

    # Trim the vocoder's run-up rather than fading it. -35 dB / 300 ms measured
    # better than the gentler -45 dB / 150 ms: speech starts immediately and the
    # onset artefact is removed rather than attenuated.
    "edge_fade": True,
    "fade_ms": 8.0,
    "trim_db": -35.0,
    "max_trim_ms": 300.0,

    # 0.1 is the documented compromise for HiFi-GAN-style bias subtraction.
    # 1.0 and 2.0 were tried first and were both too aggressive to be useful.
    "debias": True,
    "debias_strength": 0.1,

    # RMS, not peak. Peak normalisation left clips sounding muted because peak
    # tracks one sample while loudness follows RMS.
    "normalize": True,
    "peak_dbfs": -1.0,
    "rms_dbfs": -20.0,
}


def _highpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    """Zero-phase 2nd-order high-pass.

    filtfilt rather than a causal filter: a one-shot TTS clip is not streaming,
    and zero phase avoids the group-delay smearing a causal filter would add to
    plosive onsets.
    """
    if not HAVE_SCIPY or cutoff <= 0:
        return x
    nyq = sr / 2.0
    if cutoff >= nyq:
        return x
    sos = butter(2, cutoff / nyq, btype="highpass", output="sos")
    # filtfilt needs more samples than the filter's padding length.
    if x.size < 32:
        return sosfilt(sos, x).astype(np.float32)
    return sosfiltfilt(sos, x).astype(np.float32)


def _deess(x: np.ndarray, sr: int, amount: float = 0.35) -> np.ndarray:
    """Attenuate 5-9 kHz, where vocoder sibilance artefacts concentrate.

    A band-reject shelf rather than a dynamic de-esser: simpler, and at these
    depths the difference is inaudible on speech this short.
    """
    if not HAVE_SCIPY or amount <= 0:
        return x
    nyq = sr / 2.0
    lo, hi = 5000.0 / nyq, min(9000.0 / nyq, 0.99)
    if lo >= hi:
        return x
    sos = butter(2, [lo, hi], btype="bandpass", output="sos")
    band = sosfiltfilt(sos, x)
    return (x - float(np.clip(amount, 0.0, 1.0)) * band).astype(np.float32)


def _edge_fade(x: np.ndarray, sr: int, fade_ms: float = 5.0) -> np.ndarray:
    """Cosine fade at both ends.

    A waveform that starts or stops at non-zero amplitude produces a step
    discontinuity, heard as a click or pop. A few milliseconds of taper removes
    it without touching anything audible in the speech.

    Cosine rather than linear: linear leaves a slope discontinuity at the join,
    which is itself a (much quieter) click.
    """
    n = int(sr * fade_ms / 1000.0)
    if n < 2 or x.size <= 2 * n:
        return x
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))).astype(np.float32)
    x = x.copy()
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


# --- vocoder bias removal ----------------------------------------------------
# A HiFi-GAN generator accumulates bias through its transposed convolutions, so
# it emits audio even when fed an all-zero latent. Measured on this checkpoint:
#
#     zeros in      -> -50.37 dBFS out, peak 0.058
#     1e-3 noise in -> -50.29 dBFS out          (identical: constant, not signal-driven)
#
# Against a -20 dBFS signal that floor sits ~30 dB down, which is where hiss
# becomes audible, and it is present in EVERY output regardless of content.
#
# This is why blind spectral gating failed on it. noisereduce estimates the
# noise floor from the signal itself and cannot separate a component that is
# correlated with the speech. Here the profile can be MEASURED exactly -- feed
# the vocoder silence and look at what comes out -- and subtracted.


def median_f0(x: np.ndarray, sr: int, lo_hz: float = 70.0,
              hi_hz: float = 350.0) -> float | None:
    """Median fundamental frequency, by autocorrelation over voiced frames.

    Used to place the high-pass automatically. The cutoff must sit below the
    speaker's fundamental or it removes the voice: 130 Hz was hand-picked for a
    156 Hz male voice and would be wrong for a 195 Hz female one. Measuring the
    voice removes the last hand-tuned constant in this module.

    Returns None when there is too little voiced audio to be confident, in
    which case the caller keeps its configured cutoff.
    """
    win, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / hi_hz), int(sr / lo_hz)
    if x.size < win * 4 or lo < 2:
        return None
    vals = []
    for i in range(0, x.size - win, hop):
        f = x[i:i + win]
        if float(np.sqrt(np.mean(np.square(f)))) < 0.02:
            continue
        f = f - f.mean()
        c = np.correlate(f, f, "full")[win - 1:]
        if c[0] <= 0:
            continue
        c = c / c[0]
        seg = c[lo:hi]
        if seg.size == 0:
            continue
        pk = int(np.argmax(seg)) + lo
        if c[pk] > 0.35:
            vals.append(sr / pk)
    return float(np.median(vals)) if len(vals) >= 20 else None


def auto_highpass_hz(f0: float | None, fallback: float = 130.0,
                     ratio: float = 0.8) -> float:
    """Cutoff placed just under the fundamental.

    0.8 x F0 leaves the fundamental intact while removing the rumble beneath
    it: 125 Hz for a 156 Hz voice, 156 Hz for a 195 Hz one. Clamped so a bad
    F0 estimate can never produce a cutoff that eats speech.
    """
    if not f0 or f0 <= 0:
        return fallback
    return float(np.clip(ratio * f0, 60.0, 200.0))


# Gate threshold, in dB. Measured across all four engines on the same sentence:
#
#     mine_cpu     -27.1 dB    <- audibly noisy, cleanup helps
#     our_cpu      -37.2 dB    <- clean; cleanup AUDIBLY ALTERS the voice
#     kokoro_cpu   -37.4 dB
#     our_hybrid   -37.1 dB
#
# -32 dB sits in the 10 dB gap between the two groups.
#
# Note this is the LOW-BAND ratio, not broadband. Broadband separation ranks the
# engines in the opposite order -- mine_cpu measures the CLEANEST broadband at
# -42.4 dB while being the one that sounds noisy -- so a broadband gate would
# process exactly the wrong models. The audible noise lives below the
# fundamental, so that is what has to be measured.
NOISY_THRESHOLD_DB = -32.0


def assess(x: np.ndarray, sr: int) -> dict:
    """Decide whether this audio needs cleaning up.

    Returns the measurements and a recommendation. Cleanup is not free: on a
    source that is already clean it removes voice along with the little noise
    there is, which is why this is gated rather than always applied.
    """
    out: dict = {"f0_hz": None, "low_band_db": None, "noisy": False,
                 "reason": "not assessed"}
    if not HAVE_SCIPY or x.size < sr // 2:
        out["reason"] = "clip too short to assess"
        return out
    f0 = median_f0(x, sr)
    npr = noise_print(x, sr)
    if f0 is None or npr is None:
        out["reason"] = "no pitch or no quiet stretch found"
        return out
    from scipy.signal import stft

    def band_db(sig, lo, hi):
        fr, _, S = stft(sig, fs=sr, nperseg=1024, noverlap=768, window="hann")
        m = np.mean(np.abs(S), axis=1)
        sel = (fr >= lo) & (fr < hi)
        if not sel.any():
            return None
        return 20.0 * np.log10(max(float(np.mean(m[sel])), 1e-12))

    hi = max(60.0, f0 * 0.85)
    n_lo, s_lo = band_db(npr, 20.0, hi), band_db(x, 20.0, hi)
    if n_lo is None or s_lo is None:
        out["reason"] = "band measurement failed"
        return out
    ratio = n_lo - s_lo
    out.update(f0_hz=round(f0, 1), low_band_db=round(ratio, 1),
               noisy=bool(ratio > NOISY_THRESHOLD_DB),
               reason=(f"low-band noise {ratio:.1f} dB "
                       f"{'above' if ratio > NOISY_THRESHOLD_DB else 'below'} "
                       f"the {NOISY_THRESHOLD_DB:.0f} dB threshold"))
    return out


# Edge energy, relative to the chunk's own peak, below which a source is left
# alone entirely.
#
# This is NOT an audibility threshold, though an earlier version of this
# constant claimed to be one. It was -60 dB, derived from a six-chunk suite
# where audible chunks measured -52 to -57 and inaudible ones -69 to -70. That
# did not generalise: on other text, chunks measuring -60 to -70 were still
# audibly noisy. Two metrics were built on that reasoning and both were
# contradicted by listening tests -- do not reintroduce a per-chunk audibility
# gate here without a listening test to back it.
#
# What it actually does is tell SOURCES apart, which measurement supports well:
#
#     private training checkpoints   -47 to  -70 dB   emit a floor, need work
#     released af_heart model        -81 to  -87 dB   clean, judged better raw
#     Kokoro-82M                          -149 dB    no floor at all
#
# -75 sits in the 11 dB gap between the first two groups. Deliberately a coarse
# source check: processing a source that does not need it can only remove
# speech, which is what a listening test found on the released model.
NO_FLOOR_DB = -75.0

# Kept for callers that still import the old name.
LEAD_IN_AUDIBLE_DB = NO_FLOOR_DB


def edge_db(x: np.ndarray, sr: int, win_ms: float = 50.0,
            tail: bool = False) -> float | None:
    """Energy of one edge of the chunk, in dB relative to its own peak.

    `tail=True` measures the last win_ms instead of the first. Both edges must
    be measured: on real text the TRAILING edge is routinely the worse of the
    two -- measured at -35.5 dB on a sentence whose lead-in read -47.4 dB --
    and a gate that looks only at the lead-in skips those chunks entirely.
    """
    n = int(sr * win_ms / 1000.0)
    if x.size < n or n < 1:
        return None
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        return None
    seg = x[-n:] if tail else x[:n]
    rms_ = float(np.sqrt(np.mean(np.square(seg))))
    return 20.0 * np.log10(max(rms_, 1e-12)) - 20.0 * np.log10(peak)


def lead_in_db(x: np.ndarray, sr: int, win_ms: float = 50.0) -> float | None:
    """Energy of the chunk's lead-in, in dB relative to its own peak.

    This is the measurement that predicts the sentence-boundary artefact, and
    it is deliberately RELATIVE: a quiet lead-in only matters in proportion to
    the speech that follows it, so an absolute dBFS figure does not separate
    the audible cases from the inaudible ones. This one does, cleanly.

    It does not replace assess(), it answers a different question. assess()
    measures the band BELOW the fundamental, which is right for low-frequency
    rumble and wrong here -- the boundary artefact is 1-6 kHz, so a chunk can
    read clean on the low band while carrying 30 dB of excess in the band that
    is actually audible.

    Returns None when the chunk is too short to measure.
    """
    n = int(sr * win_ms / 1000.0)
    if x.size < n or n < 1:
        return None
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        return None
    lead = float(np.sqrt(np.mean(np.square(x[:n]))))
    return 20.0 * np.log10(max(lead, 1e-12)) - 20.0 * np.log10(peak)


def chunk_is_noisy(x: np.ndarray, sr: int,
                   threshold_db: float = NO_FLOOR_DB) -> bool:
    """Whether this chunk comes from a source that emits a noise floor.

    Deliberately near-unconditional. Every attempt to be selective here -- to
    process only the chunks that "needed" it -- made the result worse by ear,
    because the artefact is not reliably predicted by how loud a chunk's edges
    are. Processing every chunk is what a listening test says is clean.

    So this answers only the safety question: does this source have a floor at
    all? Our vocoder does; Kokoro does not, and denoising Kokoro could only
    take speech away. See NO_FLOOR_DB for the measurements.

    Judged on the worse of the two edges, since the trailing edge is routinely
    the louder one.
    """
    a = edge_db(x, sr, tail=False)
    b = edge_db(x, sr, tail=True)
    vals = [v for v in (a, b) if v is not None]
    return bool(vals) and max(vals) > threshold_db


def noise_print(x: np.ndarray, sr: int, win_ms: float = 300.0,
                prefer_lead: bool = True) -> np.ndarray | None:
    """Find the quietest window in the clip -- the "noise print".

    This is the standard audio-restoration workflow (Audacity's Get Noise
    Profile, iZotope RX's Learn): pick a stretch of the SAME recording that
    contains only the noise, profile it, subtract that profile from the whole
    file. Learning from silence in the actual recording is far more accurate
    than estimating a floor from material that also contains speech.

    An earlier version called the denoiser with no noise sample at all, leaving
    it to guess the profile from the mixed signal -- which is the weak mode and
    is why gentle settings did nothing while strong ones damaged the speech.

    Returns None when the clip is too short or has no quiet stretch, in which
    case the caller falls back to blind estimation.
    """
    n = int(sr * win_ms / 1000.0)
    if x.size < 3 * n or n < 128:
        return None

    # Prefer the lead-in when it is quiet: the residual noise a listener
    # notices is at the START, and the vocoder's cold-start artefact lives
    # there. A mid-clip pause has a different profile and profiling it leaves
    # the onset untouched.
    if prefer_lead:
        lead = x[:n]
        rest_peak = float(np.max(np.abs(x[n:]))) if x.size > n else 0.0
        if rest_peak > 0 and float(np.max(np.abs(lead))) < 0.25 * rest_peak:
            return lead.copy()

    hop = max(1, n // 4)
    starts = range(0, x.size - n, hop)
    energies = [(float(np.mean(np.square(x[i:i + n]))), i) for i in starts]
    if not energies:
        return None
    quietest_e, i = min(energies)
    loudest_e = max(e for e, _ in energies)
    # If the quietest stretch is not meaningfully quieter than the loudest, the
    # clip has no pause and any "print" would contain speech -- subtracting
    # that would gouge the voice.
    if loudest_e <= 0 or quietest_e > loudest_e * 0.05:
        return None
    return x[i:i + n].copy()


def _stft(x: np.ndarray, n_fft: int, hop: int):
    from scipy.signal import stft
    _, _, S = stft(x, nperseg=n_fft, noverlap=n_fft - hop,
                   window="hann", boundary="zeros", padded=True)
    return S


def _istft(S, n_fft: int, hop: int, length: int) -> np.ndarray:
    from scipy.signal import istft
    _, y = istft(S, nperseg=n_fft, noverlap=n_fft - hop,
                 window="hann", boundary=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size < length:
        y = np.pad(y, (0, length - y.size))
    return y[:length]


def bias_spectrum(decoder, device: str = "cuda", channels: int = 192,
                  frames: int = 200, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Measure the vocoder's output for an all-zero latent.

    Returns a per-frequency magnitude profile, averaged over time. Averaging is
    valid precisely because the artefact is constant -- that is what the zeros
    vs tiny-noise measurement above establishes.
    """
    import torch
    with torch.no_grad():
        z = torch.zeros(1, channels, frames, device=device)
        y = decoder(z).float().squeeze().detach().cpu().numpy().astype(np.float32)
    S = _stft(y.reshape(-1), n_fft, hop)
    return np.mean(np.abs(S), axis=-1, keepdims=True).astype(np.float32)


def subtract_bias(x: np.ndarray, bias_mag: np.ndarray, strength: float = 1.0,
                  n_fft: int = 1024, hop: int = 256, floor: float = 0.02) -> np.ndarray:
    """Spectral subtraction against a KNOWN noise profile.

    Phase is preserved from the original: phase carries the speech structure and
    the artefact is being removed in magnitude only.

    `floor` keeps a small fraction of the original magnitude instead of clamping
    to zero. Hard-zeroing bins is what produces musical noise -- isolated
    surviving bins ringing as tones -- which is the damage the aggressive
    noisereduce settings caused.
    """
    if bias_mag is None or x.size < n_fft:
        return x
    S = _stft(x.reshape(-1), n_fft, hop)
    mag, phase = np.abs(S), np.angle(S)
    b = bias_mag
    if b.shape[0] != mag.shape[0]:
        return x                          # profile from a different FFT size
    reduced = np.maximum(mag - float(strength) * b, float(floor) * mag)
    y = _istft(reduced * np.exp(1j * phase), n_fft, hop, x.size)
    return y.astype(np.float32)


def edge_clean(x: np.ndarray, sr: int, fade_ms: float = 8.0,
               trim_db: float = -45.0, max_trim_ms: float = 150.0) -> np.ndarray:
    """Trim the near-silent lead-in/out of ONE synthesized chunk, then fade.

    Measured on this model: the first ~100 ms of a chunk carries a noise floor
    at -54 to -68 dBFS before speech starts -- roughly 35-45 dB below the
    signal, which is where hiss becomes audible in a quiet lead-in. It is a
    boundary artefact of the vocoder's transposed convolutions, not speech.

    This MUST run per chunk, before concatenation. The engine synthesizes one
    chunk per sentence and joins them, so an utterance of three sentences has
    three of these onsets. Post-processing the concatenated waveform only ever
    reaches the first and last edge and leaves every internal join untouched --
    which is exactly the bug this function exists to fix.

    Trimming is bounded by max_trim_ms so a quiet but genuine speech onset can
    never be eaten; beyond that limit the fade alone handles it.
    """
    if x.size < 64:
        return x
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        return x

    win = max(1, int(sr * 0.005))                        # 5 ms detector
    n = (x.size // win) * win
    frames = np.abs(x[:n]).reshape(-1, win).max(axis=1)
    thresh = peak * (10.0 ** (trim_db / 20.0))
    loud = np.nonzero(frames > thresh)[0]
    if loud.size:
        cap = int(max_trim_ms / 1000.0 * sr)
        start = min(int(loud[0]) * win, cap)
        end = max(x.size - min(x.size - (int(loud[-1]) + 1) * win, cap), start + win)
        x = x[start:end]

    return _edge_fade(x, sr, fade_ms)


def _normalize(x: np.ndarray, peak_dbfs: float, rms_dbfs: float | None = None,
               sr_hint: int = 24000) -> np.ndarray:
    """Normalise level so an A/B is not decided by loudness.

    PEAK normalisation alone is not enough and was the reason the first set of
    comparison clips sounded quiet: peak tracks the single loudest sample, so a
    clip with a few sharp transients and a low average level still reads as
    "full scale" while sounding muted. Perceived loudness follows RMS.

    So: normalise RMS to the target, then apply a peak ceiling only if that
    would clip. This raises quiet material properly and never distorts.
    """
    if x.size == 0:
        return x
    ceiling = 10.0 ** (peak_dbfs / 20.0)

    if rms_dbfs is None:                               # peak-only mode
        peak = float(np.max(np.abs(x)))
        return (x * (ceiling / peak)).astype(np.float32) if peak > 1e-9 else x

    rms = float(np.sqrt(np.mean(np.square(x))))
    if rms > 1e-9:
        x = (x * (10.0 ** (rms_dbfs / 20.0) / rms)).astype(np.float32)

    # Peaks are LIMITED, not rescaled. Scaling the whole signal back down to fit
    # the loudest sample exactly cancels the loudness gain -- verified: a clip at
    # -36 dBFS RMS with one transient came back out at -36 dBFS, which is why the
    # first comparison clips sounded muted. A limiter reduces gain only where and
    # when it is needed.
    peak = float(np.max(np.abs(x)))
    if peak <= ceiling:
        return x.astype(np.float32)

    need = np.minimum(1.0, ceiling / np.maximum(np.abs(x), 1e-9)).astype(np.float32)
    # Look-ahead: take the running minimum so gain is already down before the
    # transient arrives, rather than clipping its leading edge.
    w = max(1, int(sr_hint * 0.001))                   # 1 ms
    pad = np.pad(need, (w, w), mode="edge")
    env = np.minimum.reduceat(pad, np.arange(0, pad.size, 1)[: need.size])
    for i in range(1, w + 1):                          # cheap running min
        env = np.minimum(env, pad[i : i + need.size])
    # One-pole release so gain returns smoothly instead of pumping.
    rel = float(np.exp(-1.0 / (sr_hint * 0.050)))      # 50 ms
    g = np.empty_like(env)
    cur = 1.0
    for i, v in enumerate(env):
        cur = v if v < cur else cur * rel + v * (1.0 - rel)
        g[i] = cur
    return np.clip(x * g, -ceiling, ceiling).astype(np.float32)


def process(audio: np.ndarray, sample_rate: int, opts: dict | None = None) -> tuple[np.ndarray, dict]:
    """Apply the enabled stages. Returns (audio, report).

    The report names every stage that ran and what it cost, so the UI can show
    what was actually done rather than implying a black box.
    """
    o = dict(DEFAULTS)
    if opts:
        o.update({k: v for k, v in opts.items() if v is not None})

    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    report: dict = {"applied": [], "ms": 0.0, "available": {"noisereduce": HAVE_NR,
                                                            "scipy": HAVE_SCIPY}}
    if not o["enabled"] or x.size == 0:
        return x, report

    t0 = time.perf_counter()
    peak_in = float(np.max(np.abs(x))) if x.size else 0.0

    # "auto": measure the source and skip everything if it is already clean.
    # Loudness normalisation still runs -- it is not noise removal and never
    # alters timbre.
    if o.get("mode", "always") == "auto":
        a = assess(x, sample_rate)
        report["assessment"] = a
        if not a["noisy"]:
            report["applied"].append(f"skipped -- source is clean ({a['reason']})")
            if o["normalize"]:
                rms_t = o.get("rms_dbfs")
                x = _normalize(x, float(o["peak_dbfs"]),
                               None if rms_t is None else float(rms_t), sample_rate)
                report["applied"].append("loudness only")
            report["ms"] = round((time.perf_counter() - t0) * 1000, 2)
            return np.clip(x, -1.0, 1.0).astype(np.float32), report
        o["_f0"] = a["f0_hz"]

    if o["highpass"] and HAVE_SCIPY:
        cut = float(o["highpass_hz"])
        if o.get("highpass_auto", True):
            f0 = o.get("_f0") or median_f0(x, sample_rate)
            cut = auto_highpass_hz(f0, cut)
            if f0:
                report["f0_hz"] = round(f0, 1)
        x = _highpass(x, sample_rate, cut)
        report["applied"].append(f"high-pass {cut:.0f} Hz"
                                 + (" (auto, from F0)" if o.get("highpass_auto", True) else ""))
        report["highpass_hz"] = round(cut, 1)

    # Bias subtraction runs FIRST, before any blind denoiser: it removes a
    # component whose profile is known exactly, so anything left for the
    # spectral gate afterwards is genuinely unknown noise.
    bias = o.get("_bias_mag")
    if o.get("debias", True) and bias is not None:
        before = float(np.sqrt(np.mean(np.square(x))))
        x = subtract_bias(x, bias, float(o.get("debias_strength", 1.0)))
        after = float(np.sqrt(np.mean(np.square(x))))
        report["applied"].append(
            f"vocoder de-bias (strength {float(o.get('debias_strength', 1.0)):.2f})")
        report["debias_db"] = round(20.0 * np.log10(max(after, 1e-12) / max(before, 1e-12)), 2)

    if o["denoise"] and HAVE_NR:
        strength = float(np.clip(o["strength"], 0.0, 1.0))
        try:
            # Learn the profile from a silent stretch of THIS clip when one
            # exists -- the Audacity/RX workflow -- and only fall back to
            # blind estimation when the clip has no pause.
            print_clip = noise_print(x, sample_rate) if o.get("noise_print", True) else None
            kw = {"y_noise": print_clip} if print_clip is not None else {}
            x = _nr.reduce_noise(
                y=x, sr=sample_rate,
                stationary=bool(o["stationary"]),
                prop_decrease=strength, **kw,
            ).astype(np.float32)
            report["applied"].append(
                f"spectral subtraction ({'learned noise print' if print_clip is not None else 'blind estimate'}"
                f", strength {strength:.2f})")
            report["noise_print"] = bool(print_clip is not None)
        except Exception as exc:
            # Never fail synthesis because an enhancement misbehaved.
            report["error"] = f"denoise skipped: {exc}"
    elif o["denoise"] and not HAVE_NR:
        report["error"] = "noisereduce not installed; denoise skipped"

    if o["deess"]:
        x = _deess(x, sample_rate)
        report["applied"].append("de-ess 5-9 kHz")

    if o.get("edge_fade", True):
        x = _edge_fade(x, sample_rate, float(o.get("fade_ms", 5.0)))
        report["applied"].append(f"edge fade {o.get('fade_ms', 5.0):.0f} ms")

    # Energy change is measured HERE, before normalisation. Measuring it after
    # would report the make-up gain rather than what the filters removed -- a
    # normalised clip reads as energy GAINED even when the chain stripped noise.
    e_in = float(np.mean(np.square(np.asarray(audio, dtype=np.float32).reshape(-1))))
    e_filt = float(np.mean(np.square(x)))
    if e_in > 0:
        report["energy_removed_db"] = round(
            10.0 * np.log10(max(e_filt, 1e-12) / e_in), 2)

    if o["normalize"]:
        rms_t = o.get("rms_dbfs")
        x = _normalize(x, float(o["peak_dbfs"]),
                       None if rms_t is None else float(rms_t), sample_rate)
        report["applied"].append(
            f"loudness {rms_t:.0f} dBFS RMS, ceiling {o['peak_dbfs']:.1f} dBFS"
            if rms_t is not None else f"peak normalise {o['peak_dbfs']:.1f} dBFS")
        report["rms_out_dbfs"] = round(
            20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(x)))), 1e-12)), 2)
        report["makeup_gain_db"] = round(
            10.0 * np.log10(max(float(np.mean(np.square(x))), 1e-12) / max(e_filt, 1e-12)), 2)

    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    report["ms"] = round((time.perf_counter() - t0) * 1000, 2)
    report["peak_in"] = round(peak_in, 4)
    report["peak_out"] = round(float(np.max(np.abs(x))) if x.size else 0.0, 4)
    return x, report
