"""
Profile the Arm CPU stage with Arm Performix, and reduce the result to a
per-image / per-function breakdown.

    python3 5_BenchmarkOnDGX/profile_arm.py            # profile and report
    python3 5_BenchmarkOnDGX/profile_arm.py --run-id <id>   # re-parse a past run
    python3 5_BenchmarkOnDGX/profile_arm.py --list

Arm Performix: https://developer.arm.com/servers-and-cloud-computing/arm-performix

WHY THIS EXISTS

The thread-tuning result claims the Arm CPU prefix is real compute that
benefits from the performance cores. That is a claim about where cycles go, so
it should be measured with a profiler rather than inferred from wall-clock
timings. `code_hotspots` samples the running workload and attributes samples to
symbols; this script joins the three files Performix emits --

    symbols.json              symbol_id -> {name, image_name}
    call_tree_samples.json    call_frame_id -> symbol_id  (a tree)
    callpath_self_samples.json  call_frame_id -> self samples

-- into a flat, sorted breakdown, and writes profile.json beside results.json.

HONEST LIMITATION: ONNX Runtime ships stripped, so samples inside it resolve to
"<Unknown code in onnxruntime...>" rather than individual kernels. The image
attribution is still meaningful -- it shows how much time is ONNX Runtime's
fused kernels versus Python, libc or the phonemizer -- but do not expect
per-kernel names without a symbolised build.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "profile.json"

# The workload Performix launches: a short, pure Arm-CPU synthesis loop. Kept
# deliberately small -- profiling is about proportions, not throughput.
WORKLOAD = r'''
import sys, time
sys.path.insert(0, "3_ExportAndInferenceEngine")
from tts import TTSModel
m = TTSModel("models/af_heart.onnx")
sentences = [l.strip() for l in
             open("3_ExportAndInferenceEngine/tts/eval_sentences.txt") if l.strip()]
t0 = time.perf_counter()
for _ in range(3):
    for s in sentences:
        m.synthesize(s)
print("profiled %.1fs of synthesis" % (time.perf_counter() - t0))
'''


def _apx() -> str:
    exe = shutil.which("performix") or shutil.which("apx")
    if not exe:
        home_local = Path.home() / ".local" / "bin" / "performix"
        if home_local.exists():
            return str(home_local)
        sys.exit(
            "Arm Performix is not installed.\n"
            "  https://developer.arm.com/servers-and-cloud-computing/arm-performix\n"
            "Install the CLI, then re-run. Everything else in this project works "
            "without it; this script only adds profiler evidence.")
    return exe


def list_runs() -> None:
    subprocess.run([_apx(), "run", "list"], check=False)


def profile() -> str:
    """Launch a code_hotspots run over the CPU engine. Returns the run id."""
    apx = _apx()
    script = ROOT / ".profile_workload.py"
    script.write_text(WORKLOAD)
    try:
        cmd = [apx, "recipe", "run", "code_hotspots",
               "--workload", f"{sys.executable} {script}",
               "--working-dir", str(ROOT)]
        print(f"$ {' '.join(cmd)}\n")
        r = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            sys.stderr.write(r.stderr)
            sys.exit("Performix run failed.")
    finally:
        script.unlink(missing_ok=True)

    # Performix prints "Run ID: <hex>" on completion. Read it from there
    # rather than re-querying: 'run list' has no stable machine-readable form,
    # and picking "the newest" races anything else profiling on the box.
    for line in r.stdout.splitlines():
        if "run id:" in line.lower():
            return line.split(":", 1)[1].strip()
    sys.exit("Performix did not report a run id. Use --list, then --run-id <id>.")


def parse(run_id: str) -> dict:
    apx = _apx()
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([apx, "run", "export", run_id, td], check=True,
                       capture_output=True, text=True)
        zips = list(Path(td).glob("*.zip"))
        if not zips:
            sys.exit(f"Export produced no archive for run {run_id}.")
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(td)
        out = next(Path(td).rglob("callpath_self_samples.json")).parent

        symbols = {s["id"]: s for s in json.loads((out / "symbols.json").read_text())}
        tree = json.loads((out / "call_tree_samples.json").read_text())
        selfs = json.loads((out / "callpath_self_samples.json").read_text())

    frame2sym: dict[int, int] = {}

    def walk(node):
        frame2sym[node["id"]] = node["symbol_id"]
        for c in node.get("children", []):
            walk(c)

    walk(tree)

    by_fn: dict[tuple[str, str], int] = defaultdict(int)
    by_img: dict[str, int] = defaultdict(int)
    total = 0
    for row in selfs["rows"]:
        n = row["column_data"][0]
        if not n:
            continue
        sym = symbols.get(frame2sym.get(row["call_frame_id"], -1), {})
        by_fn[(sym.get("name", "?"), sym.get("image_name", "?"))] += n
        by_img[sym.get("image_name", "?")] += n
        total += n

    pct = lambda n: round(100 * n / total, 2) if total else 0.0
    return {
        "tool": "Arm Performix",
        "recipe": "code_hotspots",
        "run_id": run_id,
        "total_samples": total,
        "images": [{"image": i, "samples": n, "percent": pct(n)}
                   for i, n in sorted(by_img.items(), key=lambda x: -x[1])],
        "functions": [{"function": f, "image": i, "samples": n, "percent": pct(n)}
                      for (f, i), n in sorted(by_fn.items(), key=lambda x: -x[1])[:25]],
        "note": ("ONNX Runtime ships stripped, so samples inside it resolve to "
                 "'<Unknown code>' rather than individual kernels. Image-level "
                 "attribution is still meaningful."),
    }


def report(d: dict) -> None:
    print(f"\nArm Performix · {d['recipe']} · run {d['run_id']} · "
          f"{d['total_samples']:,} samples\n")
    print("  where the Arm CPU time went")
    for r in d["images"][:8]:
        print(f"    {r['percent']:5.1f}%  {r['image']}")
    print("\n  top functions")
    for r in d["functions"][:10]:
        print(f"    {r['percent']:5.1f}%  {r['function'][:50]:<50} {r['image'][:30]}")
    print(f"\n  {d['note']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", help="parse an existing run instead of profiling")
    ap.add_argument("--list", action="store_true", help="list previous runs")
    args = ap.parse_args()

    if args.list:
        list_runs()
        return 0

    run_id = args.run_id or profile()
    d = parse(run_id)
    OUT.write_text(json.dumps(d, indent=2))
    report(d)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
