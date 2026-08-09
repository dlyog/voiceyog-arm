#!/usr/bin/env python3
"""
Export a trained checkpoint to the ONNX graph that ships.

    python3 7_export_onnx.py --checkpoint runs/<name>/checkpoints/best.pt \
                             --out my_voice.onnx

The real exporter lives inside the vendored engine at `tts/export_onnx.py` and
uses package-relative imports, so it must run as a module rather than as a
loose file. This wrapper does that, and exists so every step of the pipeline is
a numbered script in one directory.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv[0] = "tts.export_onnx"
runpy.run_module("tts.export_onnx", run_name="__main__")
