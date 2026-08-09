#!/usr/bin/env python3
"""
Read the Arm core topology off the silicon. Do not guess it from a core count.

Every optimization in this submission depends on one fact: the Arm CPUs we
target are ASYMMETRIC. Some cores are fast, some are efficient, and the fast
ones are not where you would assume. This script reads that layout from the
hardware's own registers and prints it, so the thread-count decision made in
2_thread_sweep.py is grounded in a measurement rather than a guess.

Where the numbers come from
---------------------------
Linux (DGX Spark GB10):
    /sys/devices/system/cpu/cpuN/regs/identification/midr_el1
        MIDR_EL1 is an Arm architectural register. Bits [15:4] are the part
        number, which names the core design. Each core reports its OWN.
    /sys/devices/system/cpu/cpuN/cpu_capacity
        The scheduler's own relative-performance number for big.LITTLE-style
        layouts. This is the signal the kernel uses, so it is the signal we
        use.

macOS (Apple M-series):
    sysctl hw.perflevel0.logicalcpu   performance cores
    sysctl hw.perflevel1.logicalcpu   efficiency cores
        Apple exposes the split directly and does not expose MIDR to
        userspace, so this is the authoritative source on that platform.

The finding that mattered
-------------------------
On GB10 the performance cores are NOT CONTIGUOUS. The layout is interleaved
across two clusters:

    cpu0-4    Cortex-A725   efficiency
    cpu5-9    Cortex-X925   performance
    cpu10-14  Cortex-A725   efficiency
    cpu15-19  Cortex-X925   performance

So the obvious way to pin to "the fast half" -- taskset -c 0-9 -- lands on
five efficiency cores and five performance cores. It looks like an
optimization and is half a pessimization. Anything that pins must use the
IDs this script prints, not a range.

An earlier version of the parent project counted cores per MIDR and called
the largest group "performance". That is wrong in principle, and on this
machine it only ever gave the right answer by luck: there are exactly ten of
each, so max() picked one arbitrarily. Capacity, not population, decides.

Run:
    python3 1_core_topology.py
    python3 1_core_topology.py --json
"""
from __future__ import annotations

import glob
import json
import platform
import subprocess
import sys
from pathlib import Path

# Arm part numbers, MIDR_EL1 bits [15:4]. Only the ones we have actually seen
# on a machine are listed; an unknown part prints its hex rather than being
# silently mislabelled.
ARM_PARTS = {
    0xD85: "Cortex-X925",
    0xD87: "Cortex-A725",
    0xD4F: "Neoverse-V2",
    0xD0C: "Neoverse-N1",
    0xD49: "Neoverse-N2",
}


def _read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except Exception:
        return None


def _read_hex(path: str) -> int | None:
    try:
        return int(Path(path).read_text().strip(), 16)
    except Exception:
        return None


def _sysctl_int(key: str) -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, timeout=5)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _sysctl_str(key: str) -> str:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def linux_topology() -> dict:
    cores = []
    for d in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"),
                    key=lambda p: int(p.rsplit("cpu", 1)[1])):
        cid = int(d.rsplit("cpu", 1)[1])
        midr = _read_hex(f"{d}/regs/identification/midr_el1")
        part = (midr >> 4) & 0xFFF if midr is not None else None
        cores.append({
            "cpu": cid,
            "midr": f"0x{midr:016x}" if midr is not None else None,
            "part": f"0x{part:03x}" if part is not None else None,
            "name": ARM_PARTS.get(part, f"unknown part 0x{part:03x}"
                                  if part is not None else "unknown"),
            "capacity": _read_int(f"{d}/cpu_capacity"),
            "max_khz": _read_int(f"{d}/cpufreq/cpuinfo_max_freq"),
        })
    if not cores:
        return {"platform": "linux", "cores": [], "asymmetric": False,
                "note": "kernel exposes no per-cpu topology under /sys"}

    caps = [c["capacity"] for c in cores if c["capacity"]]
    if not caps:
        caps = [c["max_khz"] for c in cores if c["max_khz"]]
        for c in cores:
            c["capacity"] = c["max_khz"]
    top = max(caps) if caps else 0
    # Within 10% of the fastest counts as a performance core. Capacities are
    # not identical even within one cluster -- this machine reports 997, 1017
    # and 1024 for cores of the same design -- so an equality test would split
    # one cluster into three.
    for c in cores:
        c["class"] = ("performance" if c["capacity"] and c["capacity"] >= top * 0.9
                      else "efficiency")

    perf = [c["cpu"] for c in cores if c["class"] == "performance"]
    eff = [c["cpu"] for c in cores if c["class"] == "efficiency"]
    return {
        "platform": "linux",
        "machine": platform.machine(),
        "cores": cores,
        "performance_cpus": perf,
        "efficiency_cpus": eff,
        "asymmetric": bool(perf and eff),
        "performance_contiguous": perf == list(range(min(perf), max(perf) + 1)) if perf else True,
    }


def macos_topology() -> dict:
    perf_n = _sysctl_int("hw.perflevel0.logicalcpu")
    eff_n = _sysctl_int("hw.perflevel1.logicalcpu")
    total = _sysctl_int("hw.logicalcpu") or 0
    return {
        "platform": "macos",
        "machine": platform.machine(),
        "brand": _sysctl_str("machdep.cpu.brand_string"),
        "performance_cores": perf_n,
        "efficiency_cores": eff_n,
        "logical_cpus": total,
        "asymmetric": bool(perf_n and eff_n),
        # macOS does not let a process pin to a specific core. You ask for a
        # thread count and Quality-of-Service, and the scheduler places you.
        # So on Apple the lever is the COUNT, and only the count.
        "pinning_available": False,
        "performance_contiguous": None,
    }


def topology() -> dict:
    if platform.system() == "Darwin":
        return macos_topology()
    return linux_topology()


def recommended_threads(t: dict) -> tuple[int, str]:
    """The thread count this topology argues for, and why.

    One BELOW the performance-core count. The reasoning is that the ONNX
    graph is latency-bound, not throughput-bound: a batch finishes when its
    slowest thread does, so a thread that lands on an efficiency core holds
    up every other thread. Leaving one performance core free also gives the
    OS and the espeak-ng phonemizer subprocess somewhere to run that is not
    on top of an inference thread.

    This is the hypothesis. 2_thread_sweep.py tests it on your machine and
    will happily contradict it.
    """
    if t["platform"] == "macos":
        p = t.get("performance_cores") or 0
        if p >= 2:
            return p, f"{p} performance cores; macOS places threads itself"
        return t.get("logical_cpus") or 4, "no asymmetry reported"
    perf = t.get("performance_cpus") or []
    if len(perf) >= 2:
        return len(perf) - 1, (f"{len(perf)} performance cores, minus one for "
                               f"the OS and the phonemizer subprocess")
    return len(t.get("cores") or []) or 4, "no asymmetry reported"


def main() -> int:
    t = topology()
    if "--json" in sys.argv:
        t["recommended_threads"], t["recommended_because"] = recommended_threads(t)
        print(json.dumps(t, indent=2))
        return 0

    print()
    print("  Arm core topology — read from the hardware, not assumed")
    print()

    if t["platform"] == "macos":
        print(f"    processor          {t.get('brand') or platform.processor()}")
        print(f"    logical cpus       {t.get('logical_cpus')}")
        print(f"    performance cores  {t.get('performance_cores')}   (hw.perflevel0.logicalcpu)")
        print(f"    efficiency cores   {t.get('efficiency_cores')}   (hw.perflevel1.logicalcpu)")
        print()
        print("    macOS does not expose per-core pinning to userspace, so the")
        print("    only lever here is the thread COUNT. That is enough: see")
        print("    2_thread_sweep.py.")
    else:
        if not t["cores"]:
            print(f"    {t.get('note')}")
            return 1
        print(f"    {'cpu':>4}  {'core':<14} {'class':<12} {'capacity':>8}  {'max GHz':>8}")
        print(f"    {'':->4}  {'':-<14} {'':-<12} {'':->8}  {'':->8}")
        for c in t["cores"]:
            ghz = f"{c['max_khz']/1e6:.2f}" if c["max_khz"] else "-"
            print(f"    {c['cpu']:>4}  {c['name']:<14} {c['class']:<12} "
                  f"{str(c['capacity'] or '-'):>8}  {ghz:>8}")
        print()
        perf, eff = t["performance_cpus"], t["efficiency_cpus"]
        print(f"    performance cores  {len(perf):>2}   cpus {_compact(perf)}")
        print(f"    efficiency cores   {len(eff):>2}   cpus {_compact(eff)}")
        if perf and not t["performance_contiguous"]:
            print()
            print("    NOTE: the performance cores are NOT contiguous.")
            lo, hi = min(perf), min(perf) + len(perf) - 1
            naive = list(range(lo, hi + 1))
            wrong = [c for c in naive if c not in perf]
            print(f"    Pinning to a range -- taskset -c {lo}-{hi} -- would put")
            print(f"    {len(wrong)} of {len(naive)} threads on EFFICIENCY cores "
                  f"(cpus {_compact(wrong)}).")
            print("    Use the ids above, not a range.")

    n, why = recommended_threads(t)
    print()
    print(f"    threads to use     {n}")
    print(f"    because            {why}")
    print()
    print("    That is a hypothesis, not a result. Run 2_thread_sweep.py.")
    print()
    return 0


def _compact(ids: list[int]) -> str:
    """[5,6,7,8,9,15,16,17,18,19] -> '5-9,15-19'."""
    if not ids:
        return "-"
    out, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = i
    out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


if __name__ == "__main__":
    sys.exit(main())
