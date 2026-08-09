"""
GPL-free TTS for piper-format voices.

  engine.py       TTSModel / Phonemizer -- ONNX inference, espeak-ng as a
                  SEPARATE PROCESS (no linking, so no GPL propagation)
  hybrid.py       HybridEngine -- Arm CPU prefix -> GPU decoder
  vits/           model definitions, adapted from MIT sources
  export_onnx.py  checkpoint -> full ONNX
  export_split.py checkpoint -> CPU-half ONNX for the cooperative pipeline

No GPL code is imported anywhere in this package.
"""
from .engine import Phonemizer, TTSModel, split_sentences, write_wav  # noqa: F401
