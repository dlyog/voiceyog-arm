#!/usr/bin/env python3
"""
Put every machine's result next to every other machine's.

Reads whatever bench_*.json files are in evidence/ and prints one table. It
deliberately does not pick a winner: a DGX Spark and a laptop are not
competing, and the interesting content is that the same 68.5 MB model and the
same frozen inputs run on both without either being a special case.

    python3 3_compare.py
    python3 3_compare.py --markdown    # paste-ready for the README
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVIDENCE = os.path.join(HERE, "evidence")


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(EVIDENCE, "bench_*.json"))):
        try:
            rows.append(json.load(open(f)))
        except Exception as e:
            print(f"  skipping {os.path.basename(f)}: {e}", file=sys.stderr)
    return rows


def cell(r):
    p, t, rt = r["platform"], r["timing"], r["runtime"]
    return {
        "label": r["label"],
        "cpu": p["cpu"],
        "os": p["os"],
        "cores": f"{p['total_cores']} ({p['performance_cores']}P/{p['efficiency_cores']}E)",
        "threads": r["config"]["intra_op_num_threads"],
        "ort": rt["onnxruntime"],
        "kai_linked": rt["kleidiai"].get("kai_total", 0),
        "sme": p["isa"].get("sme", False),
        "kleidiai_usable": p["kleidiai_reachable"],
        "model_mb": r["model"]["size_mb"],
        "init_ms": round(t["session_init_s"] * 1000, 1),
        "p50": t["latency_ms"]["p50"],
        "p95": t["latency_ms"]["p95"],
        "p99": t["latency_ms"]["p99"],
        "inf_s": t["throughput_inferences_per_s"],
        "rtf": t["rtf"],
        "cpu_pct": r["resources"]["cpu_percent_mean"],
        "rss_mb": r["resources"]["peak_rss_mb"],
        "inferences": r["workload"]["total_inferences"],
        "machine": (r.get("machine_state", {}).get("before") or {}).get("verdict", "?"),
        "valid": r.get("valid", None),
    }


ROWS = [
    ("CPU", "cpu"), ("OS", "os"), ("cores", "cores"),
    ("intra-op threads", "threads"), ("onnxruntime", "ort"),
    ("KleidiAI kernels linked", "kai_linked"), ("SME present", "sme"),
    ("KleidiAI usable", "kleidiai_usable"), ("model on disk (MB)", "model_mb"),
    ("session init (ms)", "init_ms"), ("p50 latency (ms)", "p50"),
    ("p95 latency (ms)", "p95"), ("p99 latency (ms)", "p99"),
    ("throughput (inf/s)", "inf_s"), ("RTF", "rtf"),
    ("CPU utilisation (%)", "cpu_pct"), ("peak RSS (MB)", "rss_mb"),
    ("timed inferences", "inferences"),
    ("machine state", "machine"), ("figures usable", "valid"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    data = [cell(r) for r in load()]
    if not data:
        sys.exit(f"no bench_*.json in {EVIDENCE} -- run 2_bench.py on each machine first")

    labels = [d["label"] for d in data]

    def fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        return str(v)

    if args.markdown:
        print("| metric | " + " | ".join(labels) + " |")
        print("|---|" + "---|" * len(labels))
        for name, key in ROWS:
            print(f"| {name} | " + " | ".join(fmt(d[key]) for d in data) + " |")
    else:
        w = max(24, max(len(n) for n, _ in ROWS) + 2)
        cw = max([18] + [len(l) + 2 for l in labels]
                 + [len(fmt(d[k])) + 2 for d in data for _, k in ROWS])
        print()
        print("  " + "".ljust(w) + "".join(l.ljust(cw) for l in labels))
        print("  " + "-" * (w + cw * len(labels)))
        for name, key in ROWS:
            print("  " + name.ljust(w) + "".join(fmt(d[key]).ljust(cw) for d in data))
        print()

    if len(data) == 2 and all(d["p50"] for d in data):
        a, b = data
        ratio = b["p50"] / a["p50"]
        print(f"  p50 ratio  {b['label']} / {a['label']} = {ratio:.2f}x")
    if not any(d["kleidiai_usable"] for d in data):
        print("  KleidiAI: not usable on any machine measured here -- no SME on either part.")


if __name__ == "__main__":
    main()
