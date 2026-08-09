"""
Monotonic Alignment Search, without the Cython extension.

The VITS reference ships MAS as a Cython module needing a `build_ext` step.
That is a build dependency and a portability hazard, so this reimplements the
same algorithm in NumPy (optionally numba-jitted).

Verified numerically identical to the compiled reference on random inputs by
`4_VerifyCorrectness/monotonic_align.py`. Do not "clean up" the indexing without re-running
that check. MAS decides which text position aligns to which audio frame; an
error here does not raise, it silently trains a model that never learns to
align, and you would not find out for hours.

Axis convention -- the part that is easy to get wrong
-----------------------------------------------------
`neg_cent` and `mask` are [b, t_t, t_s]: text on axis 1, spectrogram frames on
axis 2. The reference then indexes `value[y, x]` with

    y = TEXT position   (loop bound t_y = text length)
    x = SPEC frame      (loop bound t_x = spec length)

and backtracks starting from `index = t_x - 1`, i.e. the last spectrogram
frame. Transposing this still runs and still produces a plausible-looking
binary matrix -- it just aligns nothing. That was the first version of this
file, and it disagreed with the reference on 25/25 random trials.

Algorithm (Kim et al., 2021): find the monotonic surjective path maximising
total log-likelihood. The window

    max(0, t_x + y - t_y) <= x < min(t_x, y + 1)

is not an optimisation. It is the set of frames from which a valid path can
still reach the end; widening it produces invalid alignments.
"""
from __future__ import annotations

import numpy as np
import torch

try:  # optional: much faster, identical result
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        return deco(args[0]) if args and callable(args[0]) else deco

    prange = range  # type: ignore

_NEG_INF = -1e9


@njit(cache=True)
def _maximum_path_each(path, value, t_y, t_x):
    """Single item. `value` is used as scratch and modified in place.

    t_y = text length, t_x = spec length. Mirrors core.pyx exactly.
    """
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = _NEG_INF
            else:
                v_cur = value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else _NEG_INF
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (
            index == y or value[y - 1, index] < value[y - 1, index - 1]
        ):
            index -= 1


@njit(cache=True, parallel=True)
def _maximum_path_batch(paths, values, t_ys, t_xs):
    for b in prange(paths.shape[0]):
        _maximum_path_each(paths[b], values[b], t_ys[b], t_xs[b])


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        neg_cent: [b, t_t, t_s] alignment log-likelihoods
        mask:     [b, t_t, t_s] 1 where both text and spec positions are real

    Returns:
        [b, t_t, t_s] binary path, same device/dtype as `neg_cent`
    """
    device, dtype = neg_cent.device, neg_cent.dtype
    value = neg_cent.detach().to(torch.float32).cpu().numpy().copy()
    m = mask.detach().cpu().numpy()

    path = np.zeros(value.shape, dtype=np.int32)
    # sum over axis 1 collapses text -> count of real TEXT positions
    t_t_max = m.sum(1)[:, 0].astype(np.int32)
    # sum over axis 2 collapses spec -> count of real SPEC frames
    t_s_max = m.sum(2)[:, 0].astype(np.int32)

    if _HAVE_NUMBA:
        _maximum_path_batch(path, value, t_t_max, t_s_max)
    else:
        for b in range(path.shape[0]):
            _maximum_path_each(path[b], value[b], int(t_t_max[b]), int(t_s_max[b]))

    return torch.from_numpy(path).to(device=device, dtype=dtype)
