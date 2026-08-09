"""
GPL-free VITS, adapted from MIT-licensed sources.

  commons.py, modules.py, attentions.py, transforms.py, losses.py,
  mel_processing.py, models.py
      adapted from jaywalnut310/vits        (MIT) -- LICENSE-vits
      generator/resblocks from jik876/hifi-gan (MIT) -- LICENSE-hifigan

  monotonic_align.py   rewritten to remove the Cython build dependency;
                       verified identical to the compiled reference
  discriminators.py    MRD, written on torch primitives
  data.py              piper-format corpus + espeak-ng via subprocess

No GPL code is imported anywhere in this package.
"""
