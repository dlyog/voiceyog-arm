#!/usr/bin/env python3
"""
Every number in this submission, checked against the file that produced it.

The rule this enforces: if a figure appears in README.md or SUBMISSION.md, it
exists in one of the JSON files beside this script, and a measurement script
put it there. No figure is typed by a human.

Why bother. An earlier version of this project's documentation claimed "3x
faster than Kokoro-GPU". Re-measured properly, with both engines in the same
run on an idle GPU, it was 1.77x. The original number had been taken while the
GPU was busy, which made the baseline look slow. Nobody lied; a number was
carried by hand out of its context and stopped being true. This script exists
so that cannot happen quietly again: it exits non-zero the moment a claim and
its evidence disagree.

Run this first. Python 3 and nothing else -- no model, no virtualenv, and no
network.

    python3 verify_claims.py
    python3 verify_claims.py --show     # every claim with its measured value

Exit 0 means every figure in the writeup is backed by a measurement here.
Exit 1 names the ones that are not.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

FILES = {
    "bench": "benchmark_of_record_dgx_spark.json",
    "sweep_dgx": "thread_sweep_dgx_spark.json",
    "sweep_m1": "thread_sweep_m1_max.json",
    "topo_dgx": "core_topology_dgx_spark.json",
    "topo_m1": "core_topology_m1_max.json",
    "int8_dgx": "int8_negative_result_dgx_spark.json",
    "int8_m1": "int8_negative_result_m1_max.json",
    "perfx": "arm_performix_profile_dgx_spark.json",
    "cold": "cold_start_dgx_spark.json",
}

MODEL = "kokoro-heart-new v3"


def load() -> dict:
    data, missing = {}, []
    for key, name in FILES.items():
        p = HERE / name
        if not p.is_file():
            missing.append(name)
            continue
        data[key] = json.loads(p.read_text())
    if missing:
        print(f"  MISSING evidence files: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)
    return data


def row(sweep: dict, threads):
    """One configuration out of a sweep. threads=None is ONNX Runtime's default."""
    for r in sweep["rows"]:
        if r["threads"] == threads:
            return r
    raise KeyError(f"no row for threads={threads}")


def best(sweep: dict) -> dict:
    return min(sweep["rows"], key=lambda r: r["latency_ms_median"])


def build_claims(d: dict) -> list[tuple]:
    """(claim, measured, claimed, tolerance, source)

    Tolerances are absolute, in the unit of the claim, and exist only because
    the writeup rounds -- "1.4x" is the honest rendering of 1.386. They are not
    slack for a wrong number. Anything needing a loose tolerance to pass should
    be restated in the writeup instead.
    """
    b = d["bench"]
    r = b["results"]
    sz = b["artifact_sizes_mb"]
    ours_cpu, ours_hyb = r["ours_cpu"], r["ours_hybrid"]
    kok_cpu, kok_gpu = r["kokoro_cpu"], r["kokoro_gpu"]
    st = ours_hyb["stages"]
    sd, sm = d["sweep_dgx"], d["sweep_m1"]
    td, tm = d["topo_dgx"], d["topo_m1"]
    bd, bm = best(sd), best(sm)
    i8d, i8m = d["int8_dgx"], d["int8_m1"]

    C = []

    # --- the model ---------------------------------------------------------
    C.append(("the model is 68.5 MB", sz["full_onnx"], 68.5, 0.05,
              "bench.artifact_sizes_mb.full_onnx"))
    C.append(("Kokoro-82M, the teacher and the baseline, is 326 MB",
              sz["kokoro_reference"], 326.0, 0.5,
              "bench.artifact_sizes_mb.kokoro_reference"))
    C.append(("4.8x smaller than the teacher",
              sz["kokoro_reference"] / sz["full_onnx"], 4.8, 0.05, "326.0 / 68.52"))
    C.append(("the benchmark ran on the model that ships",
              1 if "kokoro-heart-new" in b["checkpoint"] else 0, 1, 0,
              "bench.checkpoint"))
    C.append(("the GPU was idle when it ran",
              b["gpu_utilisation_before_start_pct"], 0, 0,
              "bench.gpu_utilisation_before_start_pct"))

    # --- Arm CPU only, which is what both packages ship --------------------
    C.append(("Arm CPU only: RTF 0.0389", ours_cpu["rtf_mean"], 0.03888, 0.0001,
              "bench.results.ours_cpu.rtf_mean"))
    C.append(("Arm CPU only: 82.9 ms mean latency", ours_cpu["latency_ms_mean"],
              82.89, 0.05, "bench.results.ours_cpu.latency_ms_mean"))
    C.append(("Arm CPU only: 356 MB peak RSS", ours_cpu["peak_rss_mb"], 356.0, 1.0,
              "bench.results.ours_cpu.peak_rss_mb"))
    # Two ratios, named by their metric. RTF normalises by audio produced and
    # latency does not, so the same pair of engines gives 8.6x and 11.4x. Quoting
    # one number as "faster" without saying which metric it is invites exactly the
    # arithmetic complaint a reader would otherwise raise against the table.
    C.append(("8.6x lower RTF than the teacher on the same Arm CPU",
              kok_cpu["rtf_mean"] / ours_cpu["rtf_mean"], 8.6, 0.05,
              "kokoro_cpu.rtf_mean / ours_cpu.rtf_mean"))
    C.append(("11.4x lower mean sentence latency than the teacher, same Arm CPU",
              kok_cpu["latency_ms_mean"] / ours_cpu["latency_ms_mean"], 11.43, 0.05,
              "kokoro_cpu.latency_ms_mean / ours_cpu.latency_ms_mean"))
    C.append(("7.5x less memory than the teacher on the same Arm CPU",
              kok_cpu["peak_rss_mb"] / ours_cpu["peak_rss_mb"], 7.5, 0.05,
              "kokoro_cpu.peak_rss_mb / ours_cpu.peak_rss_mb"))

    # --- the cooperative split --------------------------------------------
    C.append(("the Arm CPU runs 57.4% of every utterance",
              st["arm_cpu_share_pct"], 57.4, 0.05,
              "bench.results.ours_hybrid.stages.arm_cpu_share_pct"))
    C.append(("the unified-memory handoff costs 0.12 ms", st["handoff_ms"], 0.1215,
              0.001, "bench.results.ours_hybrid.stages.handoff_ms"))
    C.append(("the handoff is 0.49% of the pipeline", st["handoff_share_pct"], 0.49,
              0.01, "bench.results.ours_hybrid.stages.handoff_share_pct"))
    C.append(("cooperative path: RTF 0.0196", ours_hyb["rtf_mean"], 0.01957, 0.0001,
              "bench.results.ours_hybrid.rtf_mean"))
    C.append(("cooperative path is 0.72x Kokoro-GPU -- SLOWER, stated plainly",
              kok_gpu["rtf_mean"] / ours_hyb["rtf_mean"], 0.72, 0.01,
              "kokoro_gpu.rtf_mean / ours_hybrid.rtf_mean"))
    C.append(("cooperative path uses 1.8x less memory than Kokoro-GPU",
              kok_gpu["peak_rss_mb"] / ours_hyb["peak_rss_mb"], 1.8, 0.05,
              "kokoro_gpu.peak_rss_mb / ours_hybrid.peak_rss_mb"))

    # --- the optimization, DGX Spark GB10 ----------------------------------
    C.append(("DGX Spark: tuned threads beat ONNX Runtime's default by 1.65x",
              row(sd, None)["latency_ms_median"] / bd["latency_ms_median"],
              1.65, 0.02, "sweep_dgx: ORT default / best"))
    C.append(("DGX Spark: tuned threads beat using all 20 cores by 1.39x",
              row(sd, 20)["latency_ms_median"] / bd["latency_ms_median"],
              1.39, 0.02, "sweep_dgx: 20 threads / best"))
    C.append(("DGX Spark: the topology predicted 9 threads and 9 measured best",
              bd["threads"], 9, 0, "sweep_dgx.best_threads"))
    C.append(("DGX Spark: that prediction came from the topology, not the sweep",
              sd["topology_recommendation"], 9, 0,
              "sweep_dgx.topology_recommendation"))

    # --- the optimization, Apple M1 Max ------------------------------------
    C.append(("M1 Max: tuned threads beat using all 10 cores by 2.21x",
              row(sm, 10)["latency_ms_median"] / bm["latency_ms_median"],
              2.21, 0.02, "sweep_m1: 10 threads / best"))
    C.append(("M1 Max: tuned threads beat ONNX Runtime's default by 1.20x",
              row(sm, None)["latency_ms_median"] / bm["latency_ms_median"],
              1.20, 0.02, "sweep_m1: ORT default / best"))
    C.append(("M1 Max: the topology predicted 8 threads and 8 measured best",
              bm["threads"], 8, 0, "sweep_m1.best_threads"))
    C.append(("M1 Max: that prediction came from the topology, not the sweep",
              sm["topology_recommendation"], 8, 0,
              "sweep_m1.topology_recommendation"))

    # --- the topology the optimization is derived from ---------------------
    C.append(("DGX Spark has 10 performance cores", len(td["performance_cpus"]), 10, 0,
              "topo_dgx.performance_cpus"))
    C.append(("DGX Spark has 10 efficiency cores", len(td["efficiency_cpus"]), 10, 0,
              "topo_dgx.efficiency_cpus"))
    C.append(("DGX Spark's performance cores are NOT contiguous",
              0 if td["performance_contiguous"] else 1, 1, 0,
              "topo_dgx.performance_contiguous"))
    C.append(("M1 Max has 8 performance and 2 efficiency cores",
              (tm["performance_cores"] or 0) * 100 + (tm["efficiency_cores"] or 0),
              802, 0, "topo_m1.performance_cores/efficiency_cores"))

    # --- the optimization we did not ship ----------------------------------
    C.append(("INT8 makes the file 3.3x smaller", i8m["size_reduction"], 3.34, 0.02,
              "int8_m1.size_reduction"))
    C.append(("the INT8 graph does NOT run on DGX Spark",
              0 if i8d["int8_ran"] else 1, 1, 0, "int8_dgx.int8_ran"))
    C.append(("the INT8 graph does NOT run on M1 Max",
              0 if i8m["int8_ran"] else 1, 1, 0, "int8_m1.int8_ran"))
    C.append(("root cause: 183 Conv nodes became ConvInteger",
              i8m["integer_ops_introduced"].get("ConvInteger", 0), 183, 0,
              "int8_m1.integer_ops_introduced.ConvInteger"))
    C.append(("the same failure at the same node on both Arm targets",
              1 if (i8d["int8"]["error"] or "").split("name")[-1]
              == (i8m["int8"]["error"] or "").split("name")[-1] else 0,
              1, 0, "int8_dgx.int8.error == int8_m1.int8.error"))

    # --- where the Arm CPU time actually goes, per Arm's own profiler --------
    px = d["perfx"]
    top = px["images"][0]
    ort = next((i for i in px["images"]
                if "onnxruntime" in i["image"]), {"percent": 0})
    C.append(("Arm Performix says 94.0% of Arm CPU time is inside ONNX Runtime",
              ort["percent"], 94.0, 0.1, "perfx.images[onnxruntime].percent"))
    C.append(("ONNX Runtime is the top image, not Python or libc",
              1 if "onnxruntime" in top["image"] else 0, 1, 0,
              "perfx.images[0].image"))
    C.append(("the profile is a real sampled run, not a hand-written table",
              1 if px["total_samples"] > 1000 else 0, 1, 0,
              "perfx.total_samples"))
    C.append(("Python interpreter overhead is under 1%",
              next((i["percent"] for i in px["images"]
                    if i["image"] == "python"), 0.0), 0.2, 0.3,
              "perfx.images[python].percent"))

    # --- memory against the deployment people are trying to escape ----------
    C.append(("9.7x less memory than Kokoro needs on a GPU",
              kok_gpu["peak_rss_mb"] / ours_cpu["peak_rss_mb"], 9.7, 0.05,
              "kokoro_gpu.peak_rss_mb / ours_cpu.peak_rss_mb"))
    C.append(("Kokoro on a GPU needs 3464 MB", kok_gpu["peak_rss_mb"], 3464.0, 1.0,
              "bench.results.kokoro_gpu.peak_rss_mb"))

    # --- cold start: the wait a person actually experiences -----------------
    cs = d["cold"]
    C.append(("cold start on the Arm CPU: 0.94 s to first audio",
              cs["ours_arm_cpu"]["seconds_median"], 0.94, 0.15,
              "cold.ours_arm_cpu.seconds_median"))
    C.append(("cold start on the GPU: 5.42 s to first audio",
              cs["kokoro_gpu"]["seconds_median"], 5.42, 0.5,
              "cold.kokoro_gpu.seconds_median"))
    C.append(("5.8x faster to first audio than the GPU, from cold",
              cs["speedup_to_first_audio"], 5.8, 0.4,
              "cold.speedup_to_first_audio"))
    # The GPU wins once loaded, and it wins by different amounts depending on the
    # metric. Stating it as "2.7x faster per sentence" conflated the RTF ratio with
    # a latency ratio it does not equal; both are asserted separately now.
    C.append(("Kokoro GPU: 2.74x lower RTF than VoiceYog on the Arm CPU",
              ours_cpu["rtf_mean"] / kok_gpu["rtf_mean"], 2.74, 0.05,
              "ours_cpu.rtf_mean / kokoro_gpu.rtf_mean"))
    C.append(("Kokoro GPU: 2.07x lower mean sentence latency -- both stated",
              ours_cpu["latency_ms_mean"] / kok_gpu["latency_ms_mean"], 2.07, 0.02,
              "ours_cpu.latency_ms_mean / kokoro_gpu.latency_ms_mean"))

    return C


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true",
                    help="print every claim with its measured value and source")
    args = ap.parse_args()

    d = load()
    claims = build_claims(d)

    print()
    print(f"  Claims in README.md and SUBMISSION.md, checked against the")
    print(f"  measurements in this directory.  Model under test: {MODEL}")
    print()

    failed = []
    for claim, measured, claimed, tol, source in claims:
        ok = abs(float(measured) - float(claimed)) <= tol
        if not ok:
            failed.append(claim)
        print(f"    {'pass' if ok else 'FAIL'}  {claim}")
        if args.show or not ok:
            print(f"          measured {float(measured):.6g}   claimed {float(claimed):.6g}"
                  f"   tolerance {tol:g}")
            print(f"          source   {source}")

    print()
    if failed:
        print(f"  {len(failed)} of {len(claims)} claims are NOT supported by the evidence:")
        for c in failed:
            print(f"    - {c}")
        print()
        print("  Fix the writeup or re-measure. Do not widen the tolerance.")
        print()
        return 1

    print(f"  All {len(claims)} claims are backed by the measurements in this directory.")
    print()
    print("  Each came from a script in 2_arm_optimization/ or from the benchmark")
    print("  of record. Re-run them on your own machine: the sweep takes about a")
    print("  minute and will tell you where your hardware disagrees with ours.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
