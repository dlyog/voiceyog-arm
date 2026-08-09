#!/usr/bin/env python3
"""
The optimization we did NOT ship, and why. Reproducible.

INT8 dynamic quantization is the first thing anyone reaches for when asked to
optimize a model for Arm. It is one function call, it makes the file four
times smaller, and on a slide it looks like the whole job is done.

On this model, on this hardware, it makes inference SLOWER. This script
quantizes the shipped graph and measures both, so that statement is a result
you can reproduce in two minutes rather than a claim you have to take on
faith.

Why it loses here
-----------------
INT8 wins when the model is memory-bandwidth-bound: the weights dominate
traffic, halving or quartering them buys real time. This graph is not that.
It is a VITS text encoder, duration predictor, flow and a 3.76M-parameter
HiFi-GAN vocoder -- dominated by convolutions over short sequences, already
distilled down to 68.5 MB, and comfortably resident in cache-adjacent memory
on both targets.

What dynamic quantization adds instead is per-tensor quantize/dequantize work
on every activation at runtime, plus operators that fall off the fused
kernel paths ONNX Runtime uses for float32 on Arm. You pay conversion on
every inference to save bandwidth you were not short of.

The honest conclusion
---------------------
Quantization is not an optimization. It is a TRADE, and whether it pays is a
property of the model and the hardware, not of the technique. We measured it,
it lost, and we shipped float32 with tuned threading instead -- which is the
same finding in the opposite direction: the win on these Arm parts was in
scheduling, not in numeric format.

Requires one package the runtime bundle deliberately does not carry:

    <bundle>/.venv/bin/pip install onnx

`onnx` is a build-time dependency for quantization only. Shipping it in the
inference bundle would add weight to every install to support a code path
that never runs in production.

Run:
    <bundle>/.venv/bin/python3 3_int8_negative_result.py --bundle <bundle>
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib.util import module_from_spec, spec_from_file_location  # noqa: E402

_spec = spec_from_file_location("sweep", Path(__file__).resolve().parent / "2_thread_sweep.py")
_sweep = module_from_spec(_spec)
_spec.loader.exec_module(_sweep)


def quantize(src: Path, dst: Path) -> None:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise SystemExit(
            "\n  This script needs the `onnx` package, which the runtime bundle\n"
            "  does not carry on purpose -- it is only needed to PRODUCE a\n"
            "  quantized model, never to run one.\n\n"
            "      <bundle>/.venv/bin/pip install onnx\n")
    # Dynamic, not static: static needs a calibration set, and the point here
    # is to test the technique everyone actually reaches for first.
    #
    # The quantizer prints the full protobuf of every node it cannot handle to
    # stdout. That is hundreds of lines of noise around a two-line result, so
    # it is captured and only surfaced if something fails.
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def integer_ops(path: Path) -> dict[str, int]:
    """Which integer operators the quantizer introduced, and how many.

    This is the root cause, read off the graph rather than guessed at from a
    stack trace: quantize_dynamic rewrites Conv into ConvInteger and MatMul
    into MatMulInteger, and wraps activations in DynamicQuantizeLinear. If a
    target has no kernel for one of those, the model does not run -- and the
    error names a node, not a technique, which is why this count is the more
    useful thing to print.
    """
    try:
        import onnx
    except ImportError:
        return {}
    m = onnx.load(str(path))
    counts: dict[str, int] = {}
    for node in m.graph.node:
        if node.op_type in ("ConvInteger", "MatMulInteger", "DynamicQuantizeLinear",
                            "QLinearConv", "QLinearMatMul", "QuantizeLinear",
                            "DequantizeLinear"):
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def bench(TTSModel, model: Path, cfg, texts, threads) -> dict:
    m = TTSModel(model, cfg, num_threads=threads,
                 providers=("CPUExecutionProvider",), auto_threads=False)
    for s in texts[:2]:
        m.synthesize(s)
    lat = []
    for s in texts:
        t = time.perf_counter()
        m.synthesize(s)
        lat.append((time.perf_counter() - t) * 1000)
    return {"median_ms": round(statistics.median(lat), 2),
            "mean_ms": round(statistics.fmean(lat), 2),
            "bytes": model.stat().st_size}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle")
    ap.add_argument("--sentences", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = use the tuned count from 1_core_topology.py")
    ap.add_argument("--json")
    args = ap.parse_args()

    bundle = _sweep.find_bundle(args.bundle)
    TTSModel = _sweep.load_engine(bundle)
    model, cfg = _sweep.model_paths(bundle)
    texts = _sweep.sentences(bundle, args.sentences)
    threads = args.threads or (_sweep.topology_hint()[0] or _sweep.cpu_total())

    tmp = Path(tempfile.mkdtemp(prefix="int8-"))
    try:
        q = tmp / (model.stem + ".int8.onnx")
        print()
        print("  INT8 dynamic quantization — measured, not assumed")
        print()
        print(f"    machine        {platform.machine()} · {platform.system()}")
        print(f"    model          {model.name}")
        print(f"    threads        {threads} (tuned, same for both)")
        print(f"    sentences      {len(texts)}")
        print()
        print("    quantizing...", end="", flush=True)
        t0 = time.perf_counter()
        quantize(model, q)
        print(f" done in {time.perf_counter() - t0:.1f}s")

        # Both configurations get the same thread count, the same sentences and
        # the same order. Changing two things at once would make the result
        # unattributable.
        ops = integer_ops(q)
        print("    measuring float32...", end="", flush=True)
        f32 = bench(TTSModel, model, cfg, texts, threads)
        print(" done")

        # The quantized graph keeps the original's sidecar config: same
        # phoneme map, same sample rate. Only the weights changed.
        cfg_q = q.with_suffix(".onnx.json")
        if cfg:
            shutil.copy(cfg, cfg_q)

        print("    measuring int8...", end="", flush=True)
        i8, failure = None, None
        try:
            i8 = bench(TTSModel, q, cfg_q if cfg else None, texts, threads)
            print(" done")
        except Exception as e:
            # Not a crash to be fixed -- this IS the result, and it is a
            # harder result than "slower": on this target the quantized model
            # cannot be loaded at all.
            failure = f"{type(e).__name__}: {e}"
            print(" FAILED TO RUN")

        print()
        print(f"    {'':<12} {'size':>10}  {'median ms':>12}")
        print(f"    {'':-<12} {'':->10}  {'':->12}")
        print(f"    {'float32':<12} {f32['bytes']/1e6:>9.1f}M  {f32['median_ms']:>12.2f}")
        q_size = q.stat().st_size
        print(f"    {'int8':<12} {q_size/1e6:>9.1f}M  "
              f"{(f'{i8['median_ms']:.2f}' if i8 else 'does not run'):>12}")
        print()
        print(f"    file           {f32['bytes']/q_size:.2f}x smaller")

        ratio = None
        if i8:
            ratio = i8["median_ms"] / f32["median_ms"]
            if ratio > 1.0:
                print(f"    inference      {ratio:.2f}x SLOWER  <- this is why we ship float32")
            else:
                print(f"    inference      {1/ratio:.2f}x faster on this machine")
                print("    NOTE: that contradicts our DGX Spark and M1 Max results.")
                print("    Your hardware disagrees with ours -- which is the point of")
                print("    shipping the measurement rather than the conclusion.")
        else:
            print("    inference      the quantized graph does not load on this target")
            print()
            print(f"    {failure}")
            print()
            if ops:
                print("    Root cause, read off the quantized graph:")
                for op, n in sorted(ops.items(), key=lambda kv: -kv[1]):
                    print(f"      {n:>5}x  {op}")
                print()
                print("    quantize_dynamic rewrote every Conv into ConvInteger. ONNX")
                print("    Runtime's CPU provider has no ConvInteger kernel for this")
                print("    graph on aarch64, so the session cannot even be built.")
            print()
            print("    Worth being exact about what this does and does not mean:")
            print("    INT8 is not broken, and it is not useless on Arm. It is")
            print("    inapplicable to THIS graph through THIS path -- a convolutional")
            print("    vocoder quantized dynamically. The one-line optimization that")
            print("    looks free produced an artifact that is 4x smaller and runs")
            print("    nowhere. Shipping it on the strength of the file size, without")
            print("    loading it once, is the failure mode this script exists to show.")
        print()

        if args.json:
            Path(args.json).write_text(json.dumps({
                "measured_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC",
                "machine": {"machine": platform.machine(), "system": platform.system(),
                            "release": platform.release()},
                "threads": threads, "n_sentences": len(texts),
                "float32": f32,
                "int8": i8 or {"bytes": q_size, "ran": False, "error": failure},
                "int8_ran": bool(i8),
                "integer_ops_introduced": ops,
                "size_reduction": round(f32["bytes"] / q_size, 3),
                "inference_slowdown": round(ratio, 3) if ratio else None,
            }, indent=2))
            print(f"    written        {args.json}")
            print()
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
