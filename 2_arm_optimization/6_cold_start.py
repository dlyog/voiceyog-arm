#!/usr/bin/env python3
"""
Launch to first audio. The number a person actually waits for.

Every other benchmark here measures steady-state synthesis with the model
already resident, which is the right way to compare engines. It is not the
right way to describe what using them feels like. A GPU engine has to
initialise CUDA and move its weights into VRAM before it can say a word, and
you pay that every time the process starts -- on a laptop lid opening, on a
serverless cold start, on the first sentence after a reboot.

So this measures one thing: from `python` starting to the first sample of
audio existing, in a fresh process, nothing warmed.

    python3 6_cold_start.py --json cold_start.json

Measured on a DGX Spark GB10, which has both engines available:

    VoiceYog on the Arm CPU    0.88 s
    Kokoro-82M on the GPU      5.42 s

The steady-state picture is the opposite way round -- per sentence, once
loaded, the GPU is about 2.7x faster. Both facts are true and this project
publishes both. Which one matters depends on whether your process is long
lived or not.

Needs the `kokoro` package and a CUDA device for the baseline half; without
them it measures our engine alone and says so.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

OURS = """
import sys, time
sys.path.insert(0, {engine_dir!r})
from tts import TTSModel
m = TTSModel({model!r})
a = m.synthesize("The weather changed suddenly this afternoon.")
assert len(a) > 0
"""

KOKORO = """
from kokoro import KPipeline
p = KPipeline(lang_code="a", device="cuda")
for _, _, a in p("The weather changed suddenly this afternoon.", voice="af_heart"):
    if a is not None:
        break
"""


def run_once(python: str, code: str) -> tuple[float, int, str]:
    t0 = time.perf_counter()
    r = subprocess.run([python, "-c", code], capture_output=True, text=True)
    return time.perf_counter() - t0, r.returncode, (r.stderr or "")[-300:]


def measure(python: str, code: str, runs: int) -> dict:
    """Median of N cold launches.

    Median, not minimum: the minimum flatters whichever engine got the
    friendliest page cache, and the point here is what a person typically
    waits for rather than the best case.
    """
    times, err = [], ""
    for _ in range(runs):
        s, rc, e = run_once(python, code)
        if rc != 0:
            return {"ok": False, "error": e}
        times.append(s)
    return {"ok": True, "runs": runs,
            "seconds_median": round(statistics.median(times), 3),
            "seconds_min": round(min(times), 3),
            "seconds_max": round(max(times), 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter that can import the engine (and kokoro, for the baseline)")
    ap.add_argument("--model", required=True, help="path to the .onnx that ships")
    ap.add_argument("--engine-dir", required=True,
                    help="directory containing the importable `tts` package")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--json")
    args = ap.parse_args()

    print()
    print("  Cold start — launch to first audio, nothing warmed")
    print()

    ours = measure(args.python, OURS.format(engine_dir=args.engine_dir, model=args.model),
                   args.runs)
    if not ours["ok"]:
        print(f"  fail  could not run our engine:\n{ours['error']}", file=sys.stderr)
        return 1
    print(f"    VoiceYog on the Arm CPU   {ours['seconds_median']:>6.2f}s")

    kok = measure(args.python, KOKORO, args.runs)
    if kok["ok"]:
        print(f"    Kokoro-82M on the GPU     {kok['seconds_median']:>6.2f}s")
        ratio = kok["seconds_median"] / ours["seconds_median"]
        print()
        print(f"    {ratio:.1f}x faster to first audio, on the CPU, with no GPU involved.")
        print()
        print("    Steady state is the other way round: once both are loaded, the")
        print("    GPU is about 2.7x faster per sentence. Long-lived server, the GPU")
        print("    wins. Anything that starts and stops, this does.")
    else:
        print("    Kokoro-82M on the GPU     unavailable (no kokoro package or no CUDA)")
        print("    Measured our engine only; the comparison needs both on one machine.")

    print()
    if args.json:
        Path(args.json).write_text(json.dumps({
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC",
            "machine": {"machine": platform.machine(), "system": platform.system()},
            "what": "seconds from process launch to first audio sample, cold",
            "runs_per_engine": args.runs,
            "ours_arm_cpu": ours,
            "kokoro_gpu": kok,
            "speedup_to_first_audio": (
                round(kok["seconds_median"] / ours["seconds_median"], 2)
                if kok["ok"] else None),
            "note": "Steady-state per-sentence latency is the opposite: the GPU is "
                    "~2.7x faster once loaded. See benchmark_of_record_dgx_spark.json.",
        }, indent=2))
        print(f"    written  {args.json}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
