#!/usr/bin/env python3
"""
The optimization, measured on YOUR machine in about a minute.

This is the whole claim in one runnable file: on an asymmetric Arm CPU, the
thread count ONNX Runtime picks for itself is the wrong one, and handing it
every core is worse still. The right count comes from the core topology
(1_core_topology.py), and this script proves or disproves that here, on the
hardware in front of you, against the exact model the package ships.

Why this is an Arm optimization and not a generic knob
------------------------------------------------------
On a symmetric x86 server, "use all the cores" is a defensible default and
the tuning is worth little. On the Arm parts we target it is actively
harmful, because both are heterogeneous by design:

    DGX Spark GB10   10x Cortex-X925 @ 3.90 GHz  +  10x Cortex-A725 @ 2.81 GHz
    Apple M1 Max      8 performance cores        +   2 efficiency cores

ONNX Runtime parallelises an operator by splitting it across intra-op
threads and joining. That join is a barrier: the operator finishes when its
SLOWEST thread finishes. Put one thread on a core that is 28% slower and
every other thread waits for it. Adding efficiency cores adds work capacity
and subtracts latency, and for an interactive TTS server latency is the
product.

So the technique is standard -- set intra_op_num_threads -- and the value is
hardware-specific, derived at runtime from cpu_capacity on Linux and
hw.perflevel0 on macOS. That is the whole point: the same binary, the same
ONNX graph, tuned per Arm part with no recompilation.

What it measures
----------------
For each thread count: build a fresh ONNX Runtime session, warm it, then
synthesize the held-out eval sentences and record per-sentence latency.
Sessions are rebuilt per configuration because intra_op_num_threads is fixed
at session construction -- reusing one session would measure the first
setting repeatedly.

Reported as MEDIAN, not mean. One scheduler hiccup skews a mean of twenty
samples and would make a rerun disagree with this one for no reason.

Run:
    # from inside an installed bundle
    .venv/bin/python3 2_thread_sweep.py --bundle .

    # or point at one
    /path/to/bundle/.venv/bin/python3 2_thread_sweep.py \
        --bundle /path/to/bundle --json sweep.json
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


def find_bundle(explicit: str | None) -> Path:
    """Locate an installed bundle: the thing that holds models/ and the engine."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    here = Path(__file__).resolve().parent
    candidates.append(here)
    candidates.append(here.parent)
    # 1_download/voiceyog-local-tts-<model>-<target>-<version>/
    dl = here.parent / "1_download"
    if dl.is_dir():
        candidates.extend(sorted(dl.glob("voiceyog-local-tts-*")))
    for c in candidates:
        if (c / "voiceyog_local_tts" / "_engine" / "engine.py").is_file() and (c / "models").is_dir():
            return c
    raise SystemExit(
        "Could not find an installed bundle.\n"
        "Unzip one of the packages in 1_download/, run its install.sh, then:\n"
        "    <bundle>/.venv/bin/python3 2_thread_sweep.py --bundle <bundle>")


def load_engine(bundle: Path):
    sys.path.insert(0, str(bundle / "voiceyog_local_tts"))
    try:
        from _engine.engine import TTSModel  # noqa: E402
    except ImportError as e:
        raise SystemExit(
            f"Could not import the inference engine ({e}).\n"
            f"Run this with the BUNDLE's interpreter, which has onnxruntime:\n"
            f"    {bundle}/.venv/bin/python3 {Path(__file__).name} --bundle {bundle}")
    return TTSModel


def model_paths(bundle: Path) -> tuple[Path, Path]:
    onnx = sorted(p for p in (bundle / "models").glob("*.onnx")
                  if not p.name.endswith("_prefix.onnx"))
    if not onnx:
        raise SystemExit(f"no model in {bundle / 'models'}")
    m = onnx[0]
    cfg = m.with_suffix(".onnx.json")
    return m, (cfg if cfg.is_file() else None)


def sentences(bundle: Path, n: int) -> list[str]:
    f = bundle / "voiceyog_local_tts" / "_engine" / "eval_sentences.txt"
    if f.is_file():
        lines = [l.strip() for l in f.read_text().splitlines() if l.strip()]
    else:
        # The bundle always carries these. This branch exists so the script
        # still runs against a hand-assembled model directory.
        lines = ["The weather changed suddenly this afternoon.",
                 "Could you please pass me that book?",
                 "She walked quietly across the empty room."]
    return lines[:n] if n > 0 else lines


def topology_hint() -> tuple[int | None, str]:
    """The thread count the topology argues for. Imported if available."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "topo", Path(__file__).resolve().parent / "1_core_topology.py")
        topo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(topo)
        t = topo.topology()
        n, why = topo.recommended_threads(t)
        return n, why
    except Exception:
        return None, ""


def cpu_total() -> int:
    return os.cpu_count() or 4


def plan(total: int) -> list[int | None]:
    """Thread counts worth measuring. None means 'let ONNX Runtime decide'.

    Always includes 1 (the serial floor), the topology's recommendation, and
    the total core count -- those three are the ones the story turns on.
    """
    want = {1, 2, 4, 6, 8, 9, 10, 12, 16, 20, total}
    hint, _ = topology_hint()
    if hint:
        want.add(hint)
        want.add(hint + 1)
    steps = sorted(n for n in want if 1 <= n <= total)
    return [None] + steps


def measure(TTSModel, model, cfg, texts, threads, warmup=2) -> dict:
    """One configuration, one fresh session."""
    t0 = time.perf_counter()
    m = TTSModel(model, cfg,
                 num_threads=threads,
                 providers=("CPUExecutionProvider",),
                 auto_threads=(threads is not None))
    load_ms = (time.perf_counter() - t0) * 1000

    for s in texts[:warmup]:
        m.synthesize(s)

    lat, audio_s = [], 0.0
    for s in texts:
        t = time.perf_counter()
        audio = m.synthesize(s)
        lat.append((time.perf_counter() - t) * 1000)
        audio_s += len(audio) / float(m.sample_rate)

    total_ms = sum(lat)
    return {
        "threads": threads,
        "effective_threads": getattr(m, "num_threads", None),
        "n": len(lat),
        "latency_ms_median": round(statistics.median(lat), 2),
        "latency_ms_mean": round(statistics.fmean(lat), 2),
        "latency_ms_min": round(min(lat), 2),
        "latency_ms_max": round(max(lat), 2),
        "rtf_median": round(statistics.median(lat) / 1000 /
                            (audio_s / len(lat)), 5) if audio_s else None,
        "audio_sec": round(audio_s, 2),
        "session_load_ms": round(load_ms, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="an installed bundle directory")
    ap.add_argument("--sentences", type=int, default=20,
                    help="how many eval sentences per configuration (default 20)")
    ap.add_argument("--threads", help="comma-separated list, overrides the plan")
    ap.add_argument("--json", help="write full results here")
    args = ap.parse_args()

    bundle = find_bundle(args.bundle)
    TTSModel = load_engine(bundle)
    model, cfg = model_paths(bundle)
    texts = sentences(bundle, args.sentences)
    total = cpu_total()

    if args.threads:
        steps: list[int | None] = [None] + [int(x) for x in args.threads.split(",") if x.strip()]
    else:
        steps = plan(total)

    hint, hint_why = topology_hint()

    print()
    print("  Thread sweep — the same ONNX graph, the same sentences, one knob")
    print()
    print(f"    machine          {platform.machine()} · {platform.system()} · {total} logical cpus")
    print(f"    model            {model.name}  ({model.stat().st_size / 1e6:.1f} MB)")
    print(f"    sentences        {len(texts)} held-out, {2} warmup per configuration")
    if hint:
        print(f"    topology says    {hint} threads — {hint_why}")
    print(f"    measuring {len(steps)} configurations", end="", flush=True)

    rows = []
    for th in steps:
        rows.append(measure(TTSModel, model, cfg, texts, th))
        print(".", end="", flush=True)
    print()

    best = min(rows, key=lambda r: r["latency_ms_median"])
    default = next(r for r in rows if r["threads"] is None)
    allcores = next((r for r in rows if r["threads"] == total), None)

    print()
    print(f"    {'threads':>12}  {'median ms':>10}  {'RTF':>8}  {'vs best':>8}")
    print(f"    {'':->12}  {'':->10}  {'':->8}  {'':->8}")
    for r in rows:
        label = "ORT default" if r["threads"] is None else str(r["threads"])
        ratio = r["latency_ms_median"] / best["latency_ms_median"]
        mark = "  <- best" if r is best else ""
        print(f"    {label:>12}  {r['latency_ms_median']:>10.2f}  "
              f"{(r['rtf_median'] or 0):>8.5f}  {ratio:>7.2f}x{mark}")

    print()
    bl = "ORT default" if best["threads"] is None else f"{best['threads']} threads"
    print(f"    best             {bl} at {best['latency_ms_median']:.2f} ms "
          f"(RTF {best['rtf_median']:.5f})")
    print(f"    vs ORT default   {default['latency_ms_median'] / best['latency_ms_median']:.2f}x faster")
    if allcores and allcores is not best:
        print(f"    vs all {total} cores  "
              f"{allcores['latency_ms_median'] / best['latency_ms_median']:.2f}x faster")
    if hint:
        got = next((r for r in rows if r["threads"] == hint), None)
        if got:
            agree = got is best or got["latency_ms_median"] <= best["latency_ms_median"] * 1.03
            print(f"    topology said    {hint} threads — "
                  f"{'confirmed' if agree else 'NOT confirmed on this machine'} "
                  f"({got['latency_ms_median']:.2f} ms vs best {best['latency_ms_median']:.2f} ms)")
    print()

    if args.json:
        out = {
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC",
            "machine": {"machine": platform.machine(), "system": platform.system(),
                        "release": platform.release(), "logical_cpus": total},
            "model": {"file": model.name, "bytes": model.stat().st_size},
            "n_sentences": len(texts),
            "topology_recommendation": hint,
            "rows": rows,
            "best_threads": best["threads"],
            "speedup_vs_ort_default": round(
                default["latency_ms_median"] / best["latency_ms_median"], 3),
            "speedup_vs_all_cores": (
                round(allcores["latency_ms_median"] / best["latency_ms_median"], 3)
                if allcores else None),
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"    written          {args.json}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
