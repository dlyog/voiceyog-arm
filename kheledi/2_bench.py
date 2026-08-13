#!/usr/bin/env python3
"""
One benchmark, run unchanged on a DGX Spark and on a Mac.

The comparison this directory exists for only means something if both machines
run the same code over the same bytes. So:

  * The input fixture is FROZEN and shipped in inputs/inputs.npz, not
    generated per machine. It has to be. The DGX Spark has espeak-ng 1.51 and
    this Mac has 1.52, and the two disagree on liaison and function-word
    reduction -- regenerating locally would silently give each platform a
    different phoneme sequence and quietly invalidate the comparison.
  * noise_scale and noise_w are zero. This graph has two RandomNormalLike
    nodes, and through the stochastic duration predictor they change how many
    samples come out. Left on, per-iteration latency would vary because the
    MODEL varied, which sits right on top of the effect being measured.
  * Thread count is derived from the core topology by 1_platform.py rather
    than hard-coded, because the right answer differs per part (9 on GB10,
    8 on M1 Max) and hard-coding either one would penalise the other machine.

Everything that identifies the run -- CPU, ISA, runtime version, KleidiAI
symbol count, thread count -- is written into the result file, so no reader
has to take the configuration on trust.

Self-contained: imports nothing from the rest of this repository.

    python3 2_bench.py                       # auto-detects everything
    python3 2_bench.py --threads 8 --iters 60
    python3 2_bench.py --model /path/to.onnx --label my_run
"""
import argparse
import glob
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "inputs", "inputs.npz")
EVIDENCE = os.path.join(HERE, "evidence")
SAMPLE_RATE = 24000

_spec = importlib.util.spec_from_file_location("kh_platform", os.path.join(HERE, "1_platform.py"))
kh_platform = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kh_platform)


def find_model(explicit=None):
    """Locate kokoro-heart-new.onnx without assuming a layout.

    Checked in order of how likely each is to be the model actually being
    served, so a machine with both an installed copy and a package copy
    benchmarks the installed one.
    """
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"model not found: {explicit}")
        return explicit
    pats = [
        os.path.expanduser("~/.voiceyog/models/kokoro-heart-new/v3/kokoro-heart-new.onnx"),
        os.path.expanduser("~/.voiceyog/models/*/*/kokoro-heart-new.onnx"),
        os.path.join(HERE, "..", "1_packages", "*", "models", "kokoro-heart-new.onnx"),
        os.path.expanduser("~/voiceyog-local-tts-*/models/kokoro-heart-new.onnx"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return os.path.realpath(hits[-1])
    sys.exit("could not find kokoro-heart-new.onnx -- pass --model")


def kleidi_symbols(ort):
    """Count KleidiAI micro-kernels linked into this runtime.

    strings rather than nm -D: KleidiAI is statically linked into the pybind
    module, so it is absent from the dynamic symbol table even when present.
    """
    capi = os.path.join(os.path.dirname(ort.__file__), "capi")
    try:
        sos = [f for f in os.listdir(capi)
               if f.startswith("onnxruntime_pybind11_state") and f.endswith(".so")]
        if not sos:
            return {"kai_total": 0, "note": "no pybind .so found"}
        out = subprocess.run(["strings", os.path.join(capi, sos[0])],
                             capture_output=True, text=True).stdout
        names = sorted({l for l in out.splitlines() if l.startswith("kai_run_")})
        return {
            "kai_total": len(names),
            "sme_kernels": len([n for n in names if "sme" in n]),
            "non_sme_kernels": len([n for n in names if "sme" not in n]),
            "sample": names[:4],
        }
    except Exception as e:
        return {"kai_total": 0, "error": str(e)}


def loadavg():
    try:
        return list(os.getloadavg())
    except Exception:
        return None


def contention_check(info, load, quiet=False):
    """Refuse to present a number that is really a measurement of something else.

    A latency benchmark on a busy machine measures the queue, not the model.
    This matters more here than usual, because the whole point of the exercise
    is comparing two machines: if one of them was compiling something at the
    time, the comparison is worthless and nothing in the output would say so.

    The threshold is deliberately generous -- a 1-minute load above a third of
    the core count is already enough to distort tail latency badly.
    """
    if not load:
        return None
    cores = info.get("total_cores") or 1
    ratio = load[0] / cores
    verdict = "idle" if ratio < 0.15 else ("busy" if ratio < 0.35 else "CONTENDED")
    if verdict != "idle" and not quiet:
        print(f"\n  !! load average {load[0]:.2f} on {cores} cores ({ratio*100:.0f}% busy) "
              f"-- {verdict}")
        print("  !! latency here reflects competition for cores, not the model.")
        print("  !! re-run on an idle machine before quoting these figures.\n")
    return {"load_1m": load[0], "load_5m": load[1], "load_15m": load[2],
            "cores": cores, "load_per_core": round(ratio, 3), "verdict": verdict}


class Sampler(threading.Thread):
    """CPU% and RSS while the loop runs. psutil if present, else ps(1)."""

    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval, self.stop_flag = interval, False
        self.cpu, self.rss = [], []
        try:
            import psutil
            self.proc = psutil.Process()
        except Exception:
            self.proc = None

    def run(self):
        if self.proc is not None:
            self.proc.cpu_percent(None)
        pid = os.getpid()
        while not self.stop_flag:
            time.sleep(self.interval)
            if self.proc is not None:
                self.cpu.append(self.proc.cpu_percent(None))
                self.rss.append(self.proc.memory_info().rss)
            else:
                try:
                    o = subprocess.run(["ps", "-o", "%cpu=,rss=", "-p", str(pid)],
                                       capture_output=True, text=True).stdout.split()
                    if len(o) >= 2:
                        self.cpu.append(float(o[0]))
                        self.rss.append(float(o[1]) * 1024)
                except Exception:
                    pass

    def stop(self):
        self.stop_flag = True
        self.join(timeout=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="defaults to the hostname")
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=None, help="default: derived from topology")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pin", action="store_true",
                    help="pin to the performance cores (Linux only; macOS has no "
                         "affinity API, and the finding below is why this exists)")
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort

    info = kh_platform.probe()
    threads = args.threads or info["tuned_intra_op_threads"]
    model = find_model(args.model)
    label = args.label or info["hostname"]

    if not os.path.exists(NPZ):
        sys.exit(f"frozen inputs missing: {NPZ}")
    d = np.load(NPZ)
    scales = d["scales_det"]
    n_sent = sum(1 for k in d.files if k.startswith("input_"))
    inputs = [(d[f"input_{i}"], d[f"len_{i}"]) for i in range(n_sent)]

    # Pinning is not a micro-optimisation here.
    #
    # Measured on GB10: onnxruntime 1.20.1 unpinned is stable at 59-61 ms p50
    # across every run. onnxruntime 1.28.0 unpinned is bimodal -- usually
    # 58-60 ms, but repeatedly observed at ~120 ms, a 2x regression, with the
    # machine idle and nothing else running. Pinned to the performance cores
    # it is stable at 59.6 ms. The slow mode is the scheduler placing threads
    # on Cortex-A725 efficiency cores; the intra-op join then waits on them.
    pinned = []
    if args.pin:
        ids = info.get("performance_core_ids")
        if not ids:
            print("  !! --pin: no performance core ids on this platform, ignoring")
        else:
            try:
                os.sched_setaffinity(0, set(ids))
                pinned = sorted(os.sched_getaffinity(0))
                print(f"  pinned to performance cores {pinned}")
            except AttributeError:
                print("  !! --pin: os.sched_setaffinity is Linux-only, ignoring")
            except Exception as e:
                print(f"  !! --pin failed: {e}")

    load_before = loadavg()
    contention_check(info, load_before)

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    t0 = time.perf_counter()
    sess = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    session_init_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(args.warmup):
        for arr, ln in inputs:
            sess.run(None, {"input": arr, "input_lengths": ln, "scales": scales})
    warmup_s = time.perf_counter() - t0

    sampler = Sampler()
    lat, samples = [], []
    sampler.start()
    t_start = time.perf_counter()
    for _ in range(args.iters):
        for arr, ln in inputs:
            t = time.perf_counter()
            out = sess.run(None, {"input": arr, "input_lengths": ln, "scales": scales})
            lat.append((time.perf_counter() - t) * 1000.0)
            samples.append(out[0].shape[-1])
    wall = time.perf_counter() - t_start
    sampler.stop()

    lat.sort()
    n = len(lat)

    def pct(p):
        return lat[min(n - 1, int(round(p / 100.0 * (n - 1))))]

    audio_s = sum(samples) / SAMPLE_RATE

    result = {
        "label": label,
        "measured_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "platform": info,
        "runtime": {
            "onnxruntime": ort.__version__,
            "providers_available": ort.get_available_providers(),
            "provider_used": sess.get_providers(),
            "python": platform.python_version(),
            "kleidiai": kleidi_symbols(ort),
        },
        "model": {
            "path": model,
            "size_bytes": os.path.getsize(model),
            "size_mb": round(os.path.getsize(model) / 1e6, 1),
        },
        "config": {
            "intra_op_num_threads": threads,
            "inter_op_num_threads": 1,
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "execution_mode": "ORT_SEQUENTIAL",
            "scales": scales.tolist(),
            "deterministic": True,
            "pinned": bool(pinned),
            "cpu_affinity": pinned or None,
        },
        "workload": {
            "sentences": n_sent,
            "iters": args.iters,
            "warmup_iters": args.warmup,
            "total_inferences": n,
            "inputs_npz": os.path.relpath(NPZ, HERE),
        },
        "timing": {
            "session_init_s": round(session_init_s, 4),
            "warmup_total_s": round(warmup_s, 4),
            "wall_s": round(wall, 4),
            "latency_ms": {
                "mean": round(statistics.mean(lat), 3),
                "stdev": round(statistics.pstdev(lat), 3),
                "min": round(lat[0], 3),
                "p50": round(pct(50), 3),
                "p90": round(pct(90), 3),
                "p95": round(pct(95), 3),
                "p99": round(pct(99), 3),
                "max": round(lat[-1], 3),
            },
            "throughput_inferences_per_s": round(n / wall, 3),
            "audio_seconds_produced": round(audio_s, 3),
            "rtf": round(wall / audio_s, 6) if audio_s else None,
        },
        "resources": {
            "cpu_percent_mean": round(statistics.mean(sampler.cpu), 1) if sampler.cpu else None,
            "peak_rss_mb": round(max(sampler.rss) / 1e6, 1) if sampler.rss else None,
            "sampler": "psutil" if sampler.proc is not None else "ps(1)",
        },
        "machine_state": {
            "before": contention_check(info, load_before, quiet=True),
            "after": contention_check(info, loadavg(), quiet=True),
        },
    }
    st = result["machine_state"]["before"]
    result["valid"] = bool(st and st["verdict"] == "idle")
    result["validity_note"] = (
        "machine was idle; figures are usable" if result["valid"] else
        "machine was NOT idle during the run -- these figures measure "
        "contention as well as the model, and should not be quoted")

    os.makedirs(EVIDENCE, exist_ok=True)
    out = args.out or os.path.join(EVIDENCE, f"bench_{label}.json")
    json.dump(result, open(out, "w"), indent=2)

    isa = info["isa"]
    t = result["timing"]["latency_ms"]
    print(f"  host        {label}  ·  {info['cpu']}")
    print(f"  os          {info['os']} {info.get('os_version','')}")
    print(f"  onnxruntime {ort.__version__}   provider {sess.get_providers()}")
    print(f"  kleidiai    {result['runtime']['kleidiai'].get('kai_total',0)} kernels linked   "
          f"reachable: {'YES' if info['kleidiai_reachable'] else 'NO'} "
          f"(sme={isa.get('sme')} sme2={isa.get('sme2')})")
    print(f"  threads     {threads} of {info['total_cores']} cores "
          f"({info['performance_cores']}P/{info['efficiency_cores']}E)")
    print(f"  model       {result['model']['size_mb']} MB")
    print(f"  init        {session_init_s*1000:.1f} ms    warmup {warmup_s:.2f} s")
    print(f"  latency ms  mean={t['mean']}  p50={t['p50']}  p95={t['p95']}  p99={t['p99']}")
    print(f"  throughput  {result['timing']['throughput_inferences_per_s']} inf/s   "
          f"RTF={result['timing']['rtf']}")
    print(f"  cpu%        {result['resources']['cpu_percent_mean']}   "
          f"peak RSS {result['resources']['peak_rss_mb']} MB")
    st = result["machine_state"]["before"]
    print(f"  machine     load {st['load_1m']:.2f} on {st['cores']} cores -> {st['verdict']}"
          if st else "  machine     load unavailable")
    print(f"  VALID       {'yes' if result['valid'] else 'NO -- rerun on an idle machine'}")
    print(f"  written     {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
